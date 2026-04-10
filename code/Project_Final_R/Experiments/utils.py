"""
utils.py
--------
All shared utilities for PCR retrieval experiments.

Contents
--------
  1. STOPWORDS
  2. Text preprocessing  (clean_text, read_document)
  3. Data loading        (load_split)
  4. Citation extraction (extract_citations)
  5. Vocabulary helpers  (build_vocab, tokens_to_ids)
  6. Embedding helpers   (build_w2v, make_embed_matrix, mean_vec, tfidf_weighted_mean)
  7. Numpy cosine        (cosine_sim_matrix, cosine_sim_sparse)
  8. Z-normalisation     (z_norm)
  9. Evaluation metrics  (all 8 metrics × K values)
 10. Result I/O          (save_results, load_results_dir, print_results_table)
 11. GPU device          (get_device)
"""

import os
import re
import json
import csv
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. STOPWORDS
# ─────────────────────────────────────────────────────────────────────────────

STOPWORDS: Set[str] = {
    "a","an","the","and","or","but","if","in","on","at","to","for","of",
    "with","by","from","as","is","was","are","were","be","been","being",
    "have","has","had","do","does","did","will","would","could","should",
    "may","might","shall","can","not","no","nor","so","yet","both","either",
    "neither","each","few","more","most","other","some","such","than","too",
    "very","just","that","this","these","those","it","its","he","she","they",
    "we","you","i","me","him","her","us","them","my","your","his","our",
    "their","what","which","who","whom","when","where","why","how","all",
    "any","both","each","every","one","also","into","about","up","out",
    "then","there","here","over","under","again","further","once","only",
    "own","same","s","t","said","upon","per","re","vs","v","etc","ie",
    "eg","viz","mr","mrs","ms","dr","ltd","inc","corp","llc","co",
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(
    text: str,
    remove_stopwords: bool = True,
    ngram: int = 1,
    min_len: int = 2,
) -> List[str]:
    """
    Tokenise and clean raw legal text.

    Steps
    -----
    1. Strip XML-style anonymisation tags  (<n>, <ORG>, <CITATION_*>)
    2. Lowercase
    3. Keep only alphabetic characters
    4. Remove stopwords (optional) and short tokens
    5. Extend with n-grams when ngram > 1
       (returns unigrams + higher-order grams combined)

    Parameters
    ----------
    text             : raw document string
    remove_stopwords : whether to remove common stopwords
    ngram            : maximum n-gram order to add (1 = unigrams only)
    min_len          : minimum token length to keep
    """
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) >= min_len]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if ngram == 1:
        return tokens
    all_tokens = list(tokens)
    for n in range(2, ngram + 1):
        for i in range(len(tokens) - n + 1):
            all_tokens.append("_".join(tokens[i : i + n]))
    return all_tokens


def read_document(filepath: str) -> str:
    """Read a .txt file; return empty string on error."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except FileNotFoundError:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_split(
    base_dir: str,
    split: str = "train",
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    """
    Load one dataset split from the standard directory layout.

    Layout
    ------
    <base_dir>/
      ik_<split>/
        <split>.json   OR   <split>.csv
        query/         *.txt files
        candidate/     *.txt files

    Returns
    -------
    queries    : {doc_id -> raw_text}
    candidates : {doc_id -> raw_text}
    relevance  : {query_id -> [relevant_candidate_ids]}
    """
    split_dir = os.path.join(base_dir, f"ik_{split}")
    query_dir = os.path.join(split_dir, "query")
    cand_dir  = os.path.join(split_dir, "candidate")

    # ── ground truth ──────────────────────────────────────────────────────
    relevance: Dict[str, List[str]] = {}
    json_path = os.path.join(split_dir, f"{split}.json")
    csv_path  = os.path.join(split_dir, f"{split}.csv")

    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        for item in data.get("Query Set", []):
            qid = item["id"]
            relevance[qid] = item.get("relevant candidates", [])
    elif os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                qid   = row.get("id") or row.get("query_name", "")
                cands = row.get("relevant candidates", "")
                if isinstance(cands, str):
                    cands = [c.strip() for c in cands.split(",") if c.strip()]
                relevance[qid] = cands

    # ── documents ─────────────────────────────────────────────────────────
    def load_dir(d: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.endswith(".txt"):
                    out[fn] = read_document(os.path.join(d, fn))
        return out

    queries    = load_dir(query_dir)
    candidates = load_dir(cand_dir)

    print(
        f"[{split}] queries={len(queries)} | "
        f"candidates={len(candidates)} | "
        f"gt_queries={len(relevance)}"
    )
    return queries, candidates, relevance


# ─────────────────────────────────────────────────────────────────────────────
# 4. CITATION EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_citations(text: str) -> Set[str]:
    """Return set of numeric IDs from <CITATION_XXXXXXX> tags."""
    return set(re.findall(r"<CITATION_(\d+)>", text))


# ─────────────────────────────────────────────────────────────────────────────
# 5. VOCABULARY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab(
    all_token_lists: List[List[str]],
    min_freq: int = 2,
) -> Tuple[Dict[str, int], List[str]]:
    """
    Build word-to-index mapping from a list of token lists.

    Returns
    -------
    w2i   : {token -> index}   (index 0 = PAD, 1 = UNK)
    vocab : list of tokens in index order
    """
    freq: Dict[str, int] = defaultdict(int)
    for tokens in all_token_lists:
        for t in tokens:
            freq[t] += 1
    vocab = ["<PAD>", "<UNK>"] + [
        t for t, c in sorted(freq.items(), key=lambda x: -x[1])
        if c >= min_freq
    ]
    w2i = {w: i for i, w in enumerate(vocab)}
    return w2i, vocab


def tokens_to_ids(
    tokens: List[str],
    w2i: Dict[str, int],
    max_len: int,
) -> List[int]:
    """Convert tokens to integer IDs, pad/truncate to max_len."""
    unk  = w2i.get("<UNK>", 1)
    ids  = [w2i.get(t, unk) for t in tokens[:max_len]]
    ids += [0] * (max_len - len(ids))
    return ids


def tokens_to_hier_ids(
    tokens: List[str],
    w2i: Dict[str, int],
    max_sents: int,
    sent_len: int,
) -> List[List[int]]:
    """
    Convert flat token list → 2-D sequence (max_sents × sent_len).
    Used by hierarchical models.
    """
    unk    = w2i.get("<UNK>", 1)
    chunks = []
    for i in range(0, len(tokens), sent_len):
        chunk = tokens[i : i + sent_len]
        ids   = [w2i.get(t, unk) for t in chunk]
        ids  += [0] * (sent_len - len(ids))
        chunks.append(ids)
    while len(chunks) < max_sents:
        chunks.append([0] * sent_len)
    return [c[:sent_len] for c in chunks[:max_sents]]


# ─────────────────────────────────────────────────────────────────────────────
# 6. EMBEDDING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_w2v(
    all_token_lists: List[List[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    sg: int = 1,
    workers: int = 4,
    epochs: int = 5,
    seed: int = 42,
):
    """Train and return a gensim Word2Vec model."""
    from gensim.models import Word2Vec
    return Word2Vec(
        sentences=all_token_lists,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=workers,
        epochs=epochs,
        seed=seed,
    )


def make_embed_matrix(
    vocab: List[str],
    w2i: Dict[str, int],
    dim: int,
    w2v=None,
) -> np.ndarray:
    """
    Build embedding weight matrix of shape (|vocab|, dim).
    Pre-fills with W2V vectors where available; random init otherwise.
    Row 0 (PAD) is always zero.
    """
    mat = np.random.uniform(-0.25, 0.25, (len(vocab), dim)).astype(np.float32)
    mat[0] = 0.0
    if w2v is not None:
        for word, idx in w2i.items():
            if word in w2v.wv:
                mat[idx] = w2v.wv[word].astype(np.float32)
    return mat


def compute_idf(corpus: Dict[str, List[str]]) -> Dict[str, float]:
    """Compute smoothed IDF over a token corpus."""
    N  = len(corpus)
    df: Dict[str, int] = defaultdict(int)
    for tokens in corpus.values():
        for t in set(tokens):
            df[t] += 1
    return {t: math.log((N + 1) / (c + 1)) + 1.0 for t, c in df.items()}


def mean_vec(
    tokens: List[str],
    w2v,
    dim: int,
    idf: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Mean (or IDF-weighted mean) of W2V token vectors.
    Returns zero vector if no tokens found in vocabulary.
    """
    valid   = [(t, w2v.wv[t]) for t in tokens if t in w2v.wv]
    if not valid:
        return np.zeros(dim, dtype=np.float32)
    if idf:
        weights = np.array([idf.get(t, 1.0) for t, _ in valid], np.float32)
        weights /= weights.sum() + 1e-10
        return np.sum([w * v for (_, v), w in zip(valid, weights)], axis=0).astype(np.float32)
    return np.mean([v for _, v in valid], axis=0).astype(np.float32)


def embed_corpus_w2v(
    corpus: Dict[str, List[str]],
    w2v,
    dim: int,
    idf: Optional[Dict[str, float]] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Embed every document in corpus with W2V mean vectors.

    Returns
    -------
    doc_ids : list of document IDs in matrix row order
    matrix  : (N, dim) float32 array
    """
    doc_ids = list(corpus.keys())
    matrix  = np.stack(
        [mean_vec(corpus[did], w2v, dim, idf) for did in doc_ids],
        axis=0,
    )
    return doc_ids, matrix


# ─────────────────────────────────────────────────────────────────────────────
# 7. COSINE SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim_matrix(
    query_vecs: np.ndarray,
    cand_vecs: np.ndarray,
) -> np.ndarray:
    """
    Batch cosine similarity.

    Parameters
    ----------
    query_vecs : (Q, D)
    cand_vecs  : (C, D)

    Returns
    -------
    (Q, C) similarity matrix
    """
    qn = np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-10
    cn = np.linalg.norm(cand_vecs,  axis=1, keepdims=True) + 1e-10
    return (query_vecs / qn) @ (cand_vecs / cn).T


def cosine_sim_sparse(
    va: Dict[str, float],
    vb: Dict[str, float],
) -> float:
    """Cosine similarity between two sparse (dict) vectors."""
    if not va or not vb:
        return 0.0
    dot  = sum(va.get(t, 0.0) * vb.get(t, 0.0) for t in va)
    norm_a = math.sqrt(sum(x * x for x in va.values()))
    norm_b = math.sqrt(sum(x * x for x in vb.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Z-NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def z_norm(scores: Dict[str, float]) -> Dict[str, float]:
    """Z-score normalise a dict of scores (zero-variance → all zeros)."""
    vals = list(scores.values())
    if not vals:
        return scores
    mu  = float(np.mean(vals))
    std = float(np.std(vals))
    if std < 1e-10:
        return {k: 0.0 for k in scores}
    return {k: (v - mu) / std for k, v in scores.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 9. EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    hits = sum(1 for d in ranked[:k] if d in relevant)
    return hits / k if k else 0.0


def recall_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in relevant)
    return hits / len(relevant)


def f1_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    p = precision_at_k(ranked, relevant, k)
    r = recall_at_k(ranked, relevant, k)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def average_precision(ranked: List[str], relevant: Set[str]) -> float:
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranked):
        if d in relevant:
            hits  += 1
            total += hits / (i + 1)
    return total / len(relevant)


def ndcg_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    dcg  = sum(1.0 / math.log2(i + 2)
               for i, d in enumerate(ranked[:k]) if d in relevant)
    r    = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(r))
    return dcg / idcg if idcg else 0.0


def r_precision(ranked: List[str], relevant: Set[str]) -> float:
    r = len(relevant)
    return precision_at_k(ranked, relevant, r) if r else 0.0


def micro_f1_at_k(
    results: Dict[str, List[str]],
    relevance: Dict[str, List[str]],
    k: int,
) -> float:
    """Aggregate TP / FP / FN across ALL queries before computing F1.

    Matches pcr-eval.py: the query document itself is excluded from its
    own ranked list before taking the top-k slice (c != id filter).
    """
    tp = fp = fn = 0
    for qid, ranked in results.items():
        rel  = set(relevance.get(qid, []))
        # --- official eval: exclude the query doc from its own predictions ---
        ranked_filtered = [d for d in ranked if d != qid]
        top  = set(ranked_filtered[:k])
        tp  += len(top & rel)
        fp  += len(top - rel)
        fn  += len(rel - top)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def evaluate_all(
    results: Dict[str, List[str]],
    relevance: Dict[str, List[str]],
    k_values: Optional[List[int]] = None,
    label: str = "",
    verbose: bool = True,
) -> Dict:
    """
    Compute the full metric suite for a ranked retrieval result.

    Metrics computed
    ----------------
    MAP, MRR, R-Precision
    For each k in k_values:
      P@k, R@k, F1@k, NDCG@k  (macro-averaged over queries)
      MicroF1@k                (micro-averaged: aggregate TP/FP/FN first)

    Parameters
    ----------
    results   : {query_id -> ranked list of candidate ids}
    relevance : {query_id -> list of ground-truth relevant candidate ids}
    k_values  : K cutoffs (default [5, 10, 20, 50, 100])
    label     : string printed in the report header
    verbose   : whether to print the formatted report

    Returns
    -------
    dict of all metric values, plus 'model' key = label
    """
    if k_values is None:
        k_values = [5, 10, 20, 50, 100]

    aps: List[float] = []
    rps: List[float] = []
    mrrs: List[float] = []
    pk:    Dict[int, float] = defaultdict(float)
    rk:    Dict[int, float] = defaultdict(float)
    f1k:   Dict[int, float] = defaultdict(float)
    ndcgk: Dict[int, float] = defaultdict(float)
    apk:   Dict[int, float] = defaultdict(float)  # official AP@k accumulator
    n_valid = 0

    for qid, ranked in results.items():
        rel = set(relevance.get(qid, []))
        if not rel:
            continue
        # --- official eval (pcr-eval.py line 83): c != id ---
        # Remove the query doc itself from the ranked list before scoring.
        # Its identical twin always received cosine = 1.0 and would
        # unfairly occupy rank-1 without being in the relevant set.
        ranked = [d for d in ranked if d != qid]
        n_valid += 1
        aps.append(average_precision(ranked, rel))
        rps.append(r_precision(ranked, rel))
        mrrs.append(
            next((1.0 / (i + 1) for i, d in enumerate(ranked) if d in rel), 0.0)
        )
        for k in k_values:
            pk[k]    += precision_at_k(ranked, rel, k)
            rk[k]    += recall_at_k(ranked, rel, k)
            f1k[k]   += f1_at_k(ranked, rel, k)
            ndcgk[k] += ndcg_at_k(ranked, rel, k)
            # AP@k: official normalises by min(|relevant|, k), not |relevant|
            # so we track a separate per-k AP sum to match pcr-eval.py exactly.
            hits = [1 if d in rel else 0 for d in ranked[:k]]
            ap_k = 0.0
            hits_so_far = 0
            for i, h in enumerate(hits):
                if h:
                    hits_so_far += 1
                    ap_k += hits_so_far / (i + 1)
            denom = min(len(rel), k)
            apk[k] += ap_k / denom if denom else 0.0

    n = n_valid or 1
    metrics: Dict = {
        "model":       label,
        "MAP":         sum(aps)  / n,   # standard global AP (normalized by |rel|)
        "MRR":         sum(mrrs) / n,
        "R-Precision": sum(rps)  / n,
        "n_queries":   n_valid,
    }
    for k in k_values:
        metrics[f"P@{k}"]        = pk[k]    / n
        metrics[f"R@{k}"]        = rk[k]    / n
        metrics[f"F1@{k}"]       = f1k[k]   / n
        metrics[f"NDCG@{k}"]     = ndcgk[k] / n
        metrics[f"MAP@{k}"]      = apk[k]   / n   # official-style AP@k
        metrics[f"MicroF1@{k}"]  = micro_f1_at_k(results, relevance, k)

    if verbose:
        _print_single(metrics, k_values)

    return metrics


def _print_single(m: Dict, k_values: List[int]):
    """Print one result row in a consistent format."""
    W   = 88
    sep = "─" * W
    print(f"\n{sep}")
    print(f"  {m.get('model','?')}".center(W))
    print(sep)
    print(f"  queries={m['n_queries']}  "
          f"MAP={m['MAP']:.4f}  MRR={m['MRR']:.4f}  "
          f"R-P={m['R-Precision']:.4f}")
    hdr = f"  {'@K':<5} {'P':>7} {'R':>7} {'F1':>7} {'NDCG':>8} {'MAP@K':>8} {'MicF1':>8}"
    print(hdr)
    print(f"  {'─'*56}")
    for k in k_values:
        print(
            f"  @{k:<4} "
            f"{m[f'P@{k}']:>7.4f} "
            f"{m[f'R@{k}']:>7.4f} "
            f"{m[f'F1@{k}']:>7.4f} "
            f"{m[f'NDCG@{k}']:>8.4f} "
            f"{m[f'MAP@{k}']:>8.4f} "
            f"{m[f'MicroF1@{k}']:>8.4f}"
        )
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# 10. RESULT I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_results(rows: List[Dict], path: str) -> None:
    """
    Append results to a JSON file (creates new if not exists).
    Each call merges new rows with any previously saved rows,
    so re-running a subset doesn't lose other results.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # Load existing rows if file already present
    existing: List[Dict] = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except Exception:
            existing = []

    # Replace any row whose 'model' key matches a new row
    existing_by_name = {r.get("model", ""): r for r in existing}
    for r in rows:
        existing_by_name[r.get("model", "")] = r
    merged = list(existing_by_name.values())

    with open(path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\n  ✓  {len(rows)} config(s) saved → {path}  "
          f"(total in file: {len(merged)})")


def load_results(path: str) -> List[Dict]:
    """Load results from a single JSON file."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def load_results_dir(result_dir: str) -> List[Dict]:
    """Load and merge all result JSON files from a directory."""
    rows: List[Dict] = []
    for fn in sorted(os.listdir(result_dir)):
        if fn.endswith(".json"):
            rows.extend(load_results(os.path.join(result_dir, fn)))
    return rows


def print_results_table(
    rows: List[Dict],
    sort_by: str = "MAP",
    top_n: int = 40,
    title: str = "RESULTS",
) -> None:
    """
    Print a ranked comparison table.
    Called at the end of every individual method file.
    """
    if not rows:
        print("  (no results to display)")
        return

    valid = [r for r in rows if sort_by in r]
    valid.sort(key=lambda x: x.get(sort_by, 0.0), reverse=True)

    W   = 116
    sep = "═" * W
    print(f"\n{sep}")
    print(f"  {title}  —  sorted by {sort_by}  "
          f"(showing top {min(top_n, len(valid))} of {len(valid)})".center(W))
    print(sep)

    hdr = (
        f"{'Model':<56} "
        f"{'MAP':>7} {'MRR':>7} "
        f"{'P@10':>7} {'R@10':>7} {'F1@10':>7} "
        f"{'NDCG@10':>9} {'MicF1@10':>9}"
    )
    print(hdr)
    print("─" * W)

    for r in valid[:top_n]:
        name = str(r.get("model", "?"))[:55]
        print(
            f"{name:<56} "
            f"{r.get('MAP',0):>7.4f} "
            f"{r.get('MRR',0):>7.4f} "
            f"{r.get('P@10',0):>7.4f} "
            f"{r.get('R@10',0):>7.4f} "
            f"{r.get('F1@10',0):>7.4f} "
            f"{r.get('NDCG@10',0):>9.4f} "
            f"{r.get('MicroF1@10',0):>9.4f}"
        )
    print(sep)

    # Best overall
    best = valid[0] if valid else {}
    if best:
        print(f"\n  ★  Best {sort_by} = {best.get(sort_by,0):.4f}  "
              f"→  {best.get('model','?')}")


def save_results_csv(rows: List[Dict], path: str) -> None:
    """Save all results to CSV for spreadsheet analysis."""
    if not rows:
        return
    keys = ["model"] + sorted(
        {k for r in rows for k in r if k not in ("model", "n_queries")},
        key=lambda x: (x.split("@")[0], int(x.split("@")[1]) if "@" in x else 0)
    ) + ["n_queries"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  ✓  CSV saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. GPU DEVICE
# ─────────────────────────────────────────────────────────────────────────────

def get_device():
    """Return the best available torch device: CUDA > MPS > CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            d = torch.device("cuda")
            print(f"  Using device: {d}  "
                  f"({torch.cuda.get_device_name(0)})")
            return d
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            d = torch.device("mps")
            print(f"  Using device: {d}")
            return d
    except ImportError:
        pass
    print("  Using device: cpu")
    return "cpu" if True else None   # always returns a usable value
