"""
analyze_dataset.py
==================
Global statistical analysis of the ik_train or ik_test datasets.

Two modes:
  1. Basic mode  — no model needed, reads original ik_train/ or ik_test/ 
                   (fast: completes in seconds)
  2. RR mode     — reads ik_train_rr/ or ik_test_rr/ (produced by create_rr_dataset.py)
                   for rhetorical-role label distribution analysis

Prints a comprehensive report to stdout and saves plots as PNG files.

Usage
-----
    python analyze_dataset.py train   # analyze ik_train  (+ ik_train_rr  if available)
    python analyze_dataset.py test    # analyze ik_test   (+ ik_test_rr   if available)
"""

import sys
import os
import json
import ast
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Optional matplotlib - degrade gracefully if not installed
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[WARNING] matplotlib not found — plots will be skipped.")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
INPUT_DIRS     = {"train": "ik_train",    "test": "ik_test"}
RR_DIRS        = {"train": "ik_train_rr", "test": "ik_test_rr"}
OUTPUT_DIRS    = {"train": "analysis_train", "test": "analysis_test"}

VALID_RR_LABELS = {
    "Fact", "RulingByLowerCourt", "Argument", "Statute",
    "RatioOfTheDecision", "RulingByPresentCourt", "Precedent"
}

# Colour palette for plots
LABEL_COLORS = {
    "Fact":                  "#4C72B0",
    "RulingByLowerCourt":    "#DD8452",
    "Argument":              "#55A868",
    "Statute":               "#C44E52",
    "RatioOfTheDecision":    "#8172B2",
    "RulingByPresentCourt":  "#937860",
    "Precedent":             "#DA8BC3",
}


# ──────────────────────────────────────────────────────────────────────────────
# Pretty printing helpers
# ──────────────────────────────────────────────────────────────────────────────
W = 70

def header(title: str):
    print("\n" + "─" * W)
    pad = (W - len(title) - 2) // 2
    print(f"{'─'*pad} {title} {'─'*pad}")
    print("─" * W)

def kv(label: str, value, width: int = 38):
    print(f"  {label:<{width}} {value}")


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_json_mapping(json_path: str) -> dict:
    """
    Load train.json / test.json.
    Returns {query_id: [candidate_id, ...], ...}
    """
    with open(json_path) as f:
        raw = json.load(f)
    mapping = {}
    key = list(raw.keys())[0]          # "Query Set" or similar
    for item in raw[key]:
        qid  = item.get("id") or item.get("query_name")
        cids = item.get("relevant candidates", [])
        mapping[qid] = cids
    return mapping


def load_csv_mapping(csv_path: str) -> dict:
    """Fallback: load from CSV if JSON not available."""
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("Query Case", "").strip()
            raw_cids = row.get("Cited Cases", "")
            try:
                cids = ast.literal_eval(raw_cids)
            except Exception:
                cids = []
            mapping[qid] = cids
    return mapping


def txt_files_in(dirpath: str) -> list[str]:
    if not os.path.isdir(dirpath):
        return []
    return sorted(
        f for f in os.listdir(dirpath)
        if f.endswith(".txt") and f != ".gitkeep"
    )


SAMPLE_SIZE = 100   # max documents to read fully for word/sentence stats

def word_count(text: str) -> int:
    return len(text.split())


def sentence_count(text: str) -> int:
    # Simple heuristic — split on period/exclamation/question mark
    return max(1, len(re.split(r"[.!?]+", text)))


# ──────────────────────────────────────────────────────────────────────────────
# Analysis functions
# ──────────────────────────────────────────────────────────────────────────────
def analyze_directory(dirpath: str, label: str = "docs") -> dict:
    """
    Compute per-file statistics for all .txt files in dirpath.
    - File sizes: computed for ALL files (fast, no I/O beyond stat())
    - Word/sentence counts: computed for a random SAMPLE of up to SAMPLE_SIZE docs
    """
    files = txt_files_in(dirpath)
    if not files:
        return {"files": [], "sizes_bytes": [], "word_counts": [], "sentence_counts": [], "sampled": 0}

    # All file sizes (fast)
    sizes = [os.path.getsize(os.path.join(dirpath, f)) for f in files]

    # Sample for detailed stats
    import random
    rng = random.Random(42)
    sample_files = rng.sample(files, min(SAMPLE_SIZE, len(files)))
    words, sents = [], []
    for fname in sample_files:
        fpath = os.path.join(dirpath, fname)
        try:
            with open(fpath, encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            text = ""
        words.append(word_count(text))
        sents.append(sentence_count(text))

    return {
        "files"          : files,
        "sizes_bytes"    : sizes,
        "word_counts"    : words,
        "sentence_counts": sents,
        "sampled"        : len(sample_files),
    }


def print_dir_stats(stats: dict, title: str):
    n = len(stats["files"])
    if n == 0:
        print(f"  (no files found in {title})")
        return
    sz  = np.array(stats["sizes_bytes"])
    wc  = np.array(stats["word_counts"]) if stats["word_counts"] else None
    sc  = np.array(stats["sentence_counts"]) if stats["sentence_counts"] else None
    s   = stats.get("sampled", n)
    samp_note = f"  (sampled {s}/{n} docs)" if s < n else ""
    print(f"\n  {title}  ({n} documents)")
    kv("Total documents",        n)
    kv("Total size (MB)",        f"{sz.sum()/1e6:.1f}")
    kv("Avg file size (KB)",     f"{sz.mean()/1e3:.1f}  (min {sz.min()/1e3:.1f} / max {sz.max()/1e3:.1f})")
    if wc is not None:
        kv(f"Avg words / document{samp_note}",   f"{wc.mean():.0f}  (min {wc.min()} / max {wc.max()})")
    if sc is not None:
        kv(f"Avg sentences / doc{samp_note}",    f"{sc.mean():.0f}  (min {sc.min()} / max {sc.max()})")
    if wc is not None:
        kv("Est. total words (all docs)",  f"{int(wc.mean() * n):,}")


def analyze_citation_graph(mapping: dict, candidate_set: set) -> dict:
    """Analyse the query→candidate citation graph."""
    n_citations = [len(v) for v in mapping.values()]
    cited_all   = [c for vs in mapping.values() for c in vs]
    covered     = [c for c in cited_all if c in candidate_set]
    return {
        "n_queries"     : len(mapping),
        "n_citations"   : n_citations,
        "total_cited"   : len(cited_all),
        "coverage"      : len(covered) / max(len(cited_all), 1),
        "unique_cited"  : len(set(cited_all)),
        "in_pool_unique": len(set(covered)),
    }


def print_graph_stats(gs: dict):
    nc = np.array(gs["n_citations"])
    kv("Total queries",          gs["n_queries"])
    kv("Avg citations / query",  f"{nc.mean():.1f}  (min {nc.min()} / max {nc.max()})")
    kv("Total cited references", gs["total_cited"])
    kv("Unique cited docs",      gs["unique_cited"])
    kv("Cited docs in pool",     f"{gs['in_pool_unique']}  ({gs['coverage']*100:.1f}% coverage)")


def analyze_rr_labels(rr_dir: str, subdir: str = "candidate") -> dict:
    """
    Read pre-annotated files (label\\tsentence per line) and count label distribution.
    Returns {label: count, ...}
    """
    target = os.path.join(rr_dir, subdir)
    files  = txt_files_in(target)
    label_counts    = Counter()
    doc_label_counts = defaultdict(Counter)   # per-doc label counts
    n_docs_processed = 0

    for fname in files:
        fpath = os.path.join(target, fname)
        try:
            with open(fpath, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if "\t" not in line:
                        continue
                    label, *_ = line.split("\t", 1)
                    if label in VALID_RR_LABELS:
                        label_counts[label] += 1
                        doc_label_counts[fname][label] += 1
            n_docs_processed += 1
        except Exception:
            pass

    return {
        "label_counts"     : dict(label_counts),
        "doc_label_counts" : dict(doc_label_counts),
        "n_docs_processed" : n_docs_processed,
    }


def print_rr_stats(rr: dict, title: str = ""):
    lc    = rr["label_counts"]
    total = sum(lc.values())
    if total == 0:
        print("  (no labelled sentences found)")
        return
    print(f"\n  {title}  — {rr['n_docs_processed']} documents | {total:,} labelled sentences")
    print(f"  {'Label':<28} {'Count':>8}  {'%':>6}")
    print(f"  {'-'*44}")
    for label in sorted(VALID_RR_LABELS):
        cnt = lc.get(label, 0)
        pct = 100 * cnt / total if total else 0
        print(f"  {label:<28} {cnt:>8,}  {pct:>5.1f}%")


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────
def save_plot(fig, out_dir: str, fname: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_doc_lengths(cand_stats: dict, query_stats: dict, out_dir: str, split: str):
    if not HAS_PLOT:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{split.upper()} — Document Length Distribution", fontsize=14)

    for ax, stats, label in zip(axes,
                                [cand_stats, query_stats],
                                ["Candidates", "Queries"]):
        if not stats["word_counts"]:
            ax.set_title(f"{label} — no data"); continue
        ax.hist(stats["word_counts"], bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.set_title(f"{label} (n={len(stats['files'])})")
        ax.set_xlabel("Word count")
        ax.set_ylabel("Number of documents")
        ax.axvline(np.mean(stats["word_counts"]), color="red", linestyle="--",
                   label=f"Mean {np.mean(stats['word_counts']):.0f}")
        ax.legend()

    save_plot(fig, out_dir, f"{split}_doc_length_distribution.png")


def plot_rr_distribution(rr_cand: dict, rr_query: dict, out_dir: str, split: str):
    if not HAS_PLOT:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{split.upper()} — Rhetorical Role Distribution", fontsize=14)

    for ax, rr, title in zip(axes,
                             [rr_cand, rr_query],
                             ["Candidates", "Queries"]):
        lc    = rr["label_counts"]
        total = sum(lc.values())
        if not lc or total == 0:
            ax.set_title(f"{title} — no data"); continue

        labels = sorted(VALID_RR_LABELS)
        counts = [lc.get(l, 0) for l in labels]
        pcts   = [100 * c / total for c in counts]
        colors = [LABEL_COLORS.get(l, "#888888") for l in labels]

        bars = ax.bar(range(len(labels)), pcts, color=colors, edgecolor="white")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l[:12] for l in labels], rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Percentage (%)")
        ax.set_title(f"{title} (n={rr['n_docs_processed']} docs)")
        for bar, pct in zip(bars, pcts):
            if pct > 1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f"{pct:.1f}%", ha="center", va="bottom", fontsize=7)

    save_plot(fig, out_dir, f"{split}_rr_distribution.png")


def plot_citation_distribution(mapping: dict, out_dir: str, split: str):
    if not HAS_PLOT:
        return
    n_cit = [len(v) for v in mapping.values()]
    if not n_cit:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(n_cit, bins=range(0, max(n_cit) + 2), color="#55A868", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Number of relevant candidates per query")
    ax.set_ylabel("Number of queries")
    ax.set_title(f"{split.upper()} — Citations per Query")
    ax.axvline(np.mean(n_cit), color="red", linestyle="--",
               label=f"Mean {np.mean(n_cit):.1f}")
    ax.legend()
    save_plot(fig, out_dir, f"{split}_citations_per_query.png")


def plot_rr_per_doc_box(rr_cand: dict, out_dir: str, split: str):
    """Box plot of sentences per RR label across documents."""
    if not HAS_PLOT or not rr_cand["doc_label_counts"]:
        return
    data = {l: [] for l in VALID_RR_LABELS}
    for doc_counters in rr_cand["doc_label_counts"].values():
        for label in VALID_RR_LABELS:
            data[label].append(doc_counters.get(label, 0))

    labels = sorted(VALID_RR_LABELS)
    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot([data[l] for l in labels], patch_artist=True,
                    medianprops=dict(color="white", linewidth=2))
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(LABEL_COLORS.get(label, "#888"))
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels([l[:14] for l in labels], rotation=30, ha="right")
    ax.set_ylabel("Sentences per document")
    ax.set_title(f"{split.upper()} Candidates — Sentences per RR Label per Document")
    save_plot(fig, out_dir, f"{split}_rr_per_doc_boxplot.png")


# ──────────────────────────────────────────────────────────────────────────────
# Cross-split overlap
# ──────────────────────────────────────────────────────────────────────────────
def compute_overlap():
    """How many candidate docs appear in both train and test?"""
    train_cands = set(txt_files_in("ik_train/candidate"))
    test_cands  = set(txt_files_in("ik_test/candidate"))
    train_querys = set(txt_files_in("ik_train/query"))
    test_querys  = set(txt_files_in("ik_test/query"))
    return {
        "cand_overlap"  : train_cands & test_cands,
        "query_overlap" : train_querys & test_querys,
        "train_cands"   : train_cands,
        "test_cands"    : test_cands,
        "train_querys"  : train_querys,
        "test_querys"   : test_querys,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "test"):
        print("Usage: python analyze_dataset.py <train|test>")
        print("  train  → analyzes ik_train  (+ ik_train_rr  if available)")
        print("  test   → analyzes ik_test   (+ ik_test_rr   if available)")
        sys.exit(1)

    split_key = sys.argv[1]
    in_dir    = INPUT_DIRS[split_key]
    rr_dir    = RR_DIRS[split_key]
    out_dir   = OUTPUT_DIRS[split_key]

    if not os.path.isdir(in_dir):
        print(f"[ERROR] Input directory not found: {in_dir}")
        sys.exit(1)

    has_rr = os.path.isdir(rr_dir)

    print("=" * W)
    print(f"  Dataset Analysis  —  {split_key.upper()}  ({in_dir}/)")
    if has_rr:
        print(f"  RR annotations   —  {rr_dir}/  ✅")
    else:
        print(f"  RR annotations   —  {rr_dir}/  ⚠️  not found (run create_rr_dataset.py {split_key} first)")
    print("=" * W)

    # ── Load mapping ─────────────────────────────────────────────────────────
    json_path = os.path.join(in_dir, f"{split_key}.json")
    csv_path  = os.path.join(in_dir, f"{split_key}.csv")
    if os.path.exists(json_path):
        mapping = load_json_mapping(json_path)
    elif os.path.exists(csv_path):
        mapping = load_csv_mapping(csv_path)
    else:
        print("[WARNING] No mapping file (train.json/train.csv) found. Citation analysis skipped.")
        mapping = {}

    # ── Candidate / Query stats ───────────────────────────────────────────────
    header("Document Corpus Statistics")
    cand_dir  = os.path.join(in_dir, "candidate")
    query_dir = os.path.join(in_dir, "query")

    cand_stats  = analyze_directory(cand_dir,  "Candidates")
    query_stats = analyze_directory(query_dir, "Queries")

    print_dir_stats(cand_stats,  "CANDIDATES")
    print_dir_stats(query_stats, "QUERIES")

    total_docs = len(cand_stats["files"]) + len(query_stats["files"])
    total_mb   = (sum(cand_stats["sizes_bytes"]) + sum(query_stats["sizes_bytes"])) / 1e6
    kv("\n  Combined total documents", total_docs)
    kv("  Combined total size (MB)",  f"{total_mb:.1f}")

    # ── Citation graph ────────────────────────────────────────────────────────
    if mapping:
        header("Citation Graph Analysis")
        candidate_set = set(cand_stats["files"])
        graph_stats   = analyze_citation_graph(mapping, candidate_set)
        print_graph_stats(graph_stats)

    # ── Cross-split overlap ───────────────────────────────────────────────────
    if os.path.isdir("ik_train") and os.path.isdir("ik_test"):
        header("Cross-Split Overlap (train ↔ test)")
        ov = compute_overlap()
        kv("Train candidates",           len(ov["train_cands"]))
        kv("Test candidates",            len(ov["test_cands"]))
        kv("Shared candidate docs",      f"{len(ov['cand_overlap'])}  ({100*len(ov['cand_overlap'])/max(len(ov['train_cands']),1):.1f}% of train)")
        kv("Train queries",              len(ov["train_querys"]))
        kv("Test queries",               len(ov["test_querys"]))
        kv("Shared query docs",          f"{len(ov['query_overlap'])}  ({100*len(ov['query_overlap'])/max(len(ov['train_querys']),1):.1f}% of train)")

    # ── RR label distribution ─────────────────────────────────────────────────
    rr_cand = rr_query = None
    if has_rr:
        header("Rhetorical Role Label Distribution")
        print("\n  Reading pre-annotated files (no model required)...")

        rr_cand  = analyze_rr_labels(rr_dir, "candidate")
        rr_query = analyze_rr_labels(rr_dir, "query")

        print_rr_stats(rr_cand,  title="CANDIDATES")
        print_rr_stats(rr_query, title="QUERIES")

        # Per-document sentence count summary
        header("Sentences per Document (from RR annotations)")
        for name, rr in [("CANDIDATES", rr_cand), ("QUERIES", rr_query)]:
            if not rr["doc_label_counts"]:
                continue
            per_doc_totals = [sum(c.values()) for c in rr["doc_label_counts"].values()]
            arr = np.array(per_doc_totals)
            print(f"\n  {name}")
            kv("  Avg sentences / doc",   f"{arr.mean():.1f}")
            kv("  Min sentences / doc",   arr.min())
            kv("  Max sentences / doc",   arr.max())
            kv("  Docs with 0 sentences", sum(1 for x in arr if x == 0))
    else:
        header("Rhetorical Role Distribution")
        print(f"\n  ⚠️  Skipped — run  python create_rr_dataset.py {split_key}  first.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if HAS_PLOT:
        header("Saving Plots")
        plot_doc_lengths(cand_stats, query_stats, out_dir, split_key)
        if mapping:
            plot_citation_distribution(mapping, out_dir, split_key)
        if has_rr and rr_cand and rr_query:
            plot_rr_distribution(rr_cand, rr_query, out_dir, split_key)
            plot_rr_per_doc_box(rr_cand, out_dir, split_key)
        print(f"\n  All plots saved to:  {out_dir}/")

    print(f"\n{'='*W}")
    print("  ✅  Analysis complete.")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
