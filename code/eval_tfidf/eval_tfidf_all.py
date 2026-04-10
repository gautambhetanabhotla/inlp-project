#!/usr/bin/env python3
"""
eval_tfidf_all.py
=================
Exhaustive TF-IDF evaluation across legal case retrieval benchmarks.

Datasets
--------
  IL-PCR     Indian Legal Prior Case Retrieval  (IL-TUR benchmark)
  COLIEE     Competition on Legal Information Extraction/Entailment (2022/2023)
  LeCaRD     Chinese Legal Case Retrieval Dataset
  CLERC      Case Law Evaluation Retrieval Collection (US case law)
  ECtHR      European Court of Human Rights (LexGLUE → retrieval)

Sweep
-----
  n-gram ∈ {1 .. max_ngram}, TF schemes: log / raw / binary / augmented

Metrics (combined from all papers)
----------------------------------
  Standard IR : MAP, MRR, R-Precision
  Per-K       : P@K, R@K, F1@K, NDCG@K, MAP@K
  IL-TUR      : MicroF1@K  (official IL-PCR metric)
  CaseGNN     : MicroP@K, MicroR@K, MicroF1@K, MacroP@K, MacroR@K, MacroF1@K
  PromptCase  : NDCG@5, MRR, MAP, P@5
  CLERC       : MAP, NDCG@10, R@100, R@1000

Usage
-----
  python eval_tfidf_all.py                                    # IL-PCR only (data ready)
  python eval_tfidf_all.py --datasets ilpcr,coliee2023        # select datasets
  python eval_tfidf_all.py --datasets all --download          # all + auto-download
  python eval_tfidf_all.py --max_ngram 5 --schemes log raw    # quick sweep
"""

import os, sys, re, json, csv, argparse, time, math, gc, gzip, subprocess
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tfidf_utils import clean_text

# ═════════════════════════════════════════════════════════════════════════════
# Paths
# ═════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
CASEGNN_DIR = os.path.join(PROJECT_ROOT, "code/casegnn/CaseGNN-main")
PROMPTCASE_DIR = os.path.join(PROJECT_ROOT, "code/casegnn/PromptCase-main")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

K_VALUES = [5, 6, 7, 9, 10, 11, 15, 18, 20]
ALL_SCHEMES = ["log", "raw", "binary", "augmented"]

DATASET_NAMES = ["ilpcr", "coliee2022", "coliee2023", "lecard", "clerc", "ecthr"]


# ═════════════════════════════════════════════════════════════════════════════
# Dataset container
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Dataset:
    name: str
    queries: Dict[str, str]                         # id → text
    candidates: Dict[str, str]                      # id → text
    relevance: Dict[str, Set[str]]                  # qid → set of relevant cids
    candidate_pool: Optional[Dict[str, List[str]]]  # qid → candidate pool (None = all)
    lang: str = "en"
    official_metric: str = "MAP"
    notes: str = ""


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _load_txt_dir(directory: str) -> Dict[str, str]:
    docs = {}
    if not os.path.isdir(directory):
        return docs
    for fn in sorted(os.listdir(directory)):
        if fn.endswith(".txt"):
            with open(os.path.join(directory, fn), encoding="utf-8", errors="replace") as f:
                docs[fn] = f.read()
    return docs


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# Dataset loaders
# ═════════════════════════════════════════════════════════════════════════════

def load_ilpcr() -> Optional[Dataset]:
    """IL-PCR (Indian Legal Prior Case Retrieval)."""
    test_dir = os.path.join(DATA_ROOT, "ik_test")
    json_path = os.path.join(test_dir, "test.json")
    if not os.path.exists(json_path):
        print(f"  ✗ IL-PCR: {json_path} not found")
        return None

    with open(json_path) as f:
        data = json.load(f)

    relevance = {}
    for item in data["Query Set"]:
        qid = item["id"]
        relevance[qid] = set(item.get("relevant candidates", []))

    queries = _load_txt_dir(os.path.join(test_dir, "query"))
    candidates = _load_txt_dir(os.path.join(test_dir, "candidate"))

    print(f"  ✓ IL-PCR: {len(queries)} queries, {len(candidates)} candidates, "
          f"{len(relevance)} gt queries")
    return Dataset("IL-PCR", queries, candidates, relevance, None,
                   lang="en", official_metric="MicroF1@10",
                   notes="IL-TUR Benchmark · Indian Supreme Court")


def load_coliee(year: str = "2023", download: bool = False) -> Optional[Dataset]:
    """COLIEE Task 1 (case retrieval with year-filtered candidate pool)."""
    # --- text files ---
    if year == "2023":
        txt_dir = os.path.join(PROMPTCASE_DIR,
                               "COLIEE/task1_test_2023/summary_test_2023_txt")
        note = "ChatGPT-summary texts"
    else:
        txt_dir = os.path.join(CASEGNN_DIR,
                               f"task1_test_{year}/task1_test_files_{year}")
        note = "full case texts"

    # --- yearfilter (candidate pool) ---
    yf_path = os.path.join(CASEGNN_DIR,
                           f"label/test_{year}_candidate_with_yearfilter.json")

    # --- ground-truth labels ---
    label_candidates = [
        os.path.join(CASEGNN_DIR, f"label/task1_test_labels_{year}.json"),
        os.path.join(PROMPTCASE_DIR,
                     f"COLIEE/task1_test_{year}/task1_test_labels_{year}.json"),
    ]
    label_path = next((p for p in label_candidates if os.path.exists(p)), None)

    if not os.path.isdir(txt_dir):
        print(f"  ✗ COLIEE {year}: texts not found at {txt_dir}")
        print(f"    → download from https://sites.ualberta.ca/~rabelo/COLIEE{year}/")
        return None
    if label_path is None:
        print(f"  ✗ COLIEE {year}: label file not found. Expected one of:")
        for p in label_candidates:
            print(f"      {p}")
        return None
    if not os.path.exists(yf_path):
        print(f"  ✗ COLIEE {year}: yearfilter not found at {yf_path}")
        return None

    with open(label_path) as f:
        raw_labels = json.load(f)
    # label format: {"query.txt": ["rel1.txt", …]}
    relevance: Dict[str, Set[str]] = {}
    if isinstance(raw_labels, dict):
        for qid, rels in raw_labels.items():
            relevance[qid] = set(rels) if isinstance(rels, list) else {rels}
    else:  # list of dicts
        for item in raw_labels:
            qid = item.get("id", "")
            relevance[qid] = set(item.get("relevant", item.get("noticed_cases", [])))

    with open(yf_path) as f:
        candidate_pool = json.load(f)

    all_docs = _load_txt_dir(txt_dir)

    query_ids = set(relevance.keys()) | set(candidate_pool.keys())
    all_cand_ids: Set[str] = set()
    for cands in candidate_pool.values():
        all_cand_ids.update(cands)

    queries = {k: v for k, v in all_docs.items() if k in query_ids}
    candidates = {k: v for k, v in all_docs.items() if k in all_cand_ids}

    # filter to available texts
    relevance = {q: r for q, r in relevance.items() if q in queries}
    candidate_pool = {q: [c for c in cs if c in candidates]
                      for q, cs in candidate_pool.items() if q in queries}

    n_miss_q = len(query_ids - set(queries))
    n_miss_c = len(all_cand_ids - set(candidates))
    if n_miss_q:
        print(f"    ⚠ {n_miss_q} queries have no text")
    if n_miss_c:
        print(f"    ⚠ {n_miss_c} candidates have no text")

    print(f"  ✓ COLIEE {year}: {len(queries)} q, {len(candidates)} c, "
          f"{len(relevance)} gt  ({note})")
    return Dataset(f"COLIEE-{year}", queries, candidates, relevance,
                   candidate_pool, lang="en", official_metric="Micro-F1@5",
                   notes=f"COLIEE Task 1 · {note}")


# ── LeCaRD ────────────────────────────────────────────────────────────────────

def _download_lecard(target_dir: str) -> bool:
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        return True
    print("  ↓ Cloning LeCaRD from GitHub …")
    _ensure_dir(os.path.dirname(target_dir))
    r = subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/myx666/LeCaRD.git", target_dir],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ✗ git clone failed: {r.stderr[:200]}")
        return False
    for zf in ["candidates1.zip", "candidates2.zip"]:
        zp = os.path.join(target_dir, "data/candidates", zf)
        if os.path.exists(zp):
            subprocess.run(["unzip", "-qo", zp, "-d",
                            os.path.join(target_dir, "data/candidates")])
    return True


def _clean_text_zh(text: str) -> List[str]:
    """Tokenize Chinese text — jieba if available, else char-level."""
    text = re.sub(r"[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\s]", " ", text)
    try:
        import jieba
        tokens = list(jieba.cut(text))
    except ImportError:
        tokens = list(text)
    return [t.strip() for t in tokens if t.strip()]


def load_lecard(download: bool = False) -> Optional[Dataset]:
    """LeCaRD (Chinese Legal Case Retrieval Dataset, 107 queries)."""
    lecard_dir = os.path.join(DATA_ROOT, "LeCaRD")
    if download and not os.path.isdir(lecard_dir):
        if not _download_lecard(lecard_dir):
            return None

    query_path = os.path.join(lecard_dir, "data/query/query.json")
    cand_base  = os.path.join(lecard_dir, "data/candidates")

    # label file: prefer golden_labels.json, fall back to label_top30_dict.json
    golden_path = os.path.join(lecard_dir, "data/label/golden_labels.json")
    graded_path = os.path.join(lecard_dir, "data/label/label_top30_dict.json")

    if not os.path.exists(query_path):
        print(f"  ✗ LeCaRD: not found at {lecard_dir}")
        if not download:
            print("    → run with --download or clone https://github.com/myx666/LeCaRD")
        return None

    # queries
    queries: Dict[str, str] = {}
    with open(query_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            queries[str(item["ridx"])] = item["q"]

    # relevance labels
    relevance: Dict[str, Set[str]] = {}
    if os.path.exists(golden_path):
        with open(golden_path, encoding="utf-8") as f:
            raw_labels = json.load(f)
        for qid, cids in raw_labels.items():
            relevance[str(qid)] = {str(c) for c in cids}
    elif os.path.exists(graded_path):
        with open(graded_path, encoding="utf-8") as f:
            raw_labels = json.load(f)
        for qid, cand_scores in raw_labels.items():
            rel = {str(cid) for cid, score in cand_scores.items()
                   if int(score) >= 2}
            if rel:
                relevance[str(qid)] = rel
    else:
        print(f"  ✗ LeCaRD: no label file found")
        return None

    # candidate texts — per-query dirs, may be in candidates/ or candidates1/
    candidates: Dict[str, str] = {}
    candidate_pool: Dict[str, List[str]] = {}
    for qid in queries:
        # try candidates/{qid}/ then candidates1/{qid}/
        cdir = None
        for sub in ["", "candidates1", "candidates2"]:
            trial = os.path.join(cand_base, sub, str(qid))
            if os.path.isdir(trial):
                cdir = trial
                break
        if cdir is None:
            continue
        pool = []
        for fn in os.listdir(cdir):
            if not fn.endswith(".json"):
                continue
            cid = fn.replace(".json", "")
            pool.append(cid)
            if cid not in candidates:
                with open(os.path.join(cdir, fn), encoding="utf-8",
                          errors="replace") as f:
                    try:
                        doc = json.load(f)
                        candidates[cid] = doc.get("ajjbqk", doc.get("qw", ""))
                    except json.JSONDecodeError:
                        candidates[cid] = ""
        candidate_pool[qid] = pool

    print(f"  ✓ LeCaRD: {len(queries)} q, {len(candidates)} c, "
          f"{len(relevance)} gt")
    return Dataset("LeCaRD", queries, candidates, relevance,
                   candidate_pool, lang="zh", official_metric="NDCG@5",
                   notes="Chinese criminal case retrieval")


# ── CLERC ─────────────────────────────────────────────────────────────────────

def _download_clerc(target_dir: str) -> bool:
    _ensure_dir(target_dir)
    base = "https://huggingface.co/datasets/jhu-clsp/CLERC/resolve/main"
    files = [
        "qrels/qrels-doc.test.direct.tsv",
        "queries/test.all-removed.direct.tsv",
        "collection/collection.doc.tsv.gz",
    ]
    for fp in files:
        local = os.path.join(target_dir, fp)
        if os.path.exists(local):
            continue
        _ensure_dir(os.path.dirname(local))
        url = f"{base}/{fp}"
        print(f"    ↓ {fp} …")
        for cmd in [["wget", "-q", "--show-progress", "-O", local, url],
                    ["curl", "-L", "-o", local, url]]:
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                break
        else:
            print(f"    ✗ failed: {fp}")
            return False
    return True


def load_clerc(download: bool = False) -> Optional[Dataset]:
    """CLERC (US Case Law Evaluation Retrieval Collection)."""
    clerc_dir = os.path.join(DATA_ROOT, "CLERC")

    if download and not os.path.isdir(clerc_dir):
        print("  ↓ Downloading CLERC (collection.doc.tsv.gz ≈ 10 GB) …")
        if not _download_clerc(clerc_dir):
            return None

    qrels_path = os.path.join(clerc_dir, "qrels/qrels-doc.test.direct.tsv")
    queries_path = os.path.join(clerc_dir, "queries/test.all-removed.direct.tsv")
    coll_path = os.path.join(clerc_dir, "collection/collection.doc.tsv.gz")

    for p, label in [(qrels_path, "qrels"), (queries_path, "queries"),
                     (coll_path, "collection")]:
        if not os.path.exists(p):
            print(f"  ✗ CLERC: {label} not found → {p}")
            if not download:
                print("    → run with --download")
            return None

    # queries (TSV: qid \t text)
    queries: Dict[str, str] = {}
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]

    # qrels  (TSV: qid \t 0 \t did \t rel)
    relevance: Dict[str, Set[str]] = defaultdict(set)
    with open(qrels_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and int(parts[3]) > 0:
                relevance[parts[0]].add(parts[2])
    relevance = dict(relevance)

    # figure out which doc ids we actually need
    needed = set()
    for rels in relevance.values():
        needed.update(rels)
    needed.update(queries.keys())

    print(f"    Loading collection (filtering to {len(needed)} needed docs) …")
    candidates: Dict[str, str] = {}
    with gzip.open(coll_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2 and parts[0] in needed:
                candidates[parts[0]] = parts[1]
                if len(candidates) >= len(needed):
                    break  # found all we need

    print(f"  ✓ CLERC: {len(queries)} q, {len(candidates)} c, "
          f"{len(relevance)} gt")
    return Dataset("CLERC", queries, candidates, relevance, None,
                   lang="en", official_metric="MAP",
                   notes="US Case Law from CourtListener")


# ── ECtHR ─────────────────────────────────────────────────────────────────────

def _download_ecthr(target_dir: str) -> bool:
    _ensure_dir(target_dir)
    cache = os.path.join(target_dir, "ecthr_cases.json")
    if os.path.exists(cache):
        return True
    try:
        from datasets import load_dataset
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets"],
                       capture_output=True)
        from datasets import load_dataset

    print("    ↓ ECtHR cases from HuggingFace …")
    try:
        ds = load_dataset("coastalcph/lex_glue", "ecthr_a",
                          trust_remote_code=True)
        all_cases = []
        for split in ["train", "validation", "test"]:
            if split in ds:
                for item in ds[split]:
                    text = (" ".join(item["text"]) if isinstance(item["text"], list)
                            else item["text"])
                    all_cases.append({"text": text,
                                      "labels": item["labels"],
                                      "split": split})
        with open(cache, "w") as f:
            json.dump(all_cases, f)
        return True
    except Exception as e:
        print(f"    ✗ download failed: {e}")
        return False


def load_ecthr(download: bool = False) -> Optional[Dataset]:
    """
    ECtHR converted to retrieval: test cases are queries,
    train+val cases are candidates, relevance = shared violated ECHR articles.
    """
    ecthr_dir = os.path.join(DATA_ROOT, "ECtHR")
    if download:
        _download_ecthr(ecthr_dir)
    cache = os.path.join(ecthr_dir, "ecthr_cases.json")
    if not os.path.exists(cache):
        print(f"  ✗ ECtHR: data not found → {cache}")
        if not download:
            print("    → run with --download")
        return None

    with open(cache) as f:
        all_cases = json.load(f)

    train_cases = [c for c in all_cases if c["split"] in ("train", "validation")]
    test_cases  = [c for c in all_cases if c["split"] == "test"]

    queries: Dict[str, str] = {}
    candidates: Dict[str, str] = {}
    relevance: Dict[str, Set[str]] = {}

    for i, c in enumerate(train_cases):
        candidates[f"c_{i:05d}"] = c["text"]

    for i, c in enumerate(test_cases):
        qid = f"q_{i:04d}"
        queries[qid] = c["text"]
        q_labels = set(c["labels"])
        if not q_labels:
            continue
        rel = set()
        for j, tc in enumerate(train_cases):
            if q_labels & set(tc["labels"]):
                rel.add(f"c_{j:05d}")
        if rel:
            relevance[qid] = rel

    print(f"  ✓ ECtHR: {len(queries)} q, {len(candidates)} c, "
          f"{len(relevance)} gt  (shared-article retrieval)")
    return Dataset("ECtHR", queries, candidates, relevance, None,
                   lang="en", official_metric="MicroF1@10",
                   notes="LexGLUE ECtHR → retrieval by shared violated articles")


# ═════════════════════════════════════════════════════════════════════════════
# Comprehensive metric computation (combined from all papers)
# ═════════════════════════════════════════════════════════════════════════════

def precision_at_k(ranked, rel, k):
    return sum(1 for d in ranked[:k] if d in rel) / k if k else 0.0

def recall_at_k(ranked, rel, k):
    return sum(1 for d in ranked[:k] if d in rel) / len(rel) if rel else 0.0

def f1_at_k(ranked, rel, k):
    p, r = precision_at_k(ranked, rel, k), recall_at_k(ranked, rel, k)
    return 2*p*r/(p+r) if (p+r) else 0.0

def average_precision(ranked, rel):
    if not rel:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranked):
        if d in rel:
            hits += 1
            total += hits / (i + 1)
    return total / len(rel)

def ndcg_at_k(ranked, rel, k):
    dcg  = sum(1.0/math.log2(i+2) for i, d in enumerate(ranked[:k]) if d in rel)
    r    = min(len(rel), k)
    idcg = sum(1.0/math.log2(i+2) for i in range(r))
    return dcg/idcg if idcg else 0.0

def r_precision(ranked, rel):
    r = len(rel)
    return precision_at_k(ranked, rel, r) if r else 0.0


def evaluate_comprehensive(
    results: Dict[str, List[str]],
    relevance: Dict[str, Set[str]],
    k_values: List[int],
    label: str = "",
) -> Dict:
    """
    Full metric suite combining IL-TUR, CaseGNN, PromptCase, CLERC metrics.

    Per-query (macro-averaged):
        MAP, MRR, R-Precision, P@K, R@K, F1@K, NDCG@K, MAP@K

    Micro-aggregated (CaseGNN / IL-TUR style):
        MicroP@K, MicroR@K, MicroF1@K

    Macro-aggregated (CaseGNN style):
        MacroP@K, MacroR@K, MacroF1@K
    """
    aps, rps, mrrs = [], [], []
    pk   = defaultdict(float)
    rk   = defaultdict(float)
    f1k  = defaultdict(float)
    ndcg = defaultdict(float)
    apk  = defaultdict(float)
    n_valid = 0

    for qid, ranked in results.items():
        rel = relevance.get(qid)
        if not rel:
            continue
        # remove self from ranked list (standard in PCR eval)
        ranked = [d for d in ranked if d != qid]
        n_valid += 1
        aps.append(average_precision(ranked, rel))
        rps.append(r_precision(ranked, rel))
        mrrs.append(
            next((1.0/(i+1) for i, d in enumerate(ranked) if d in rel), 0.0))
        for k in k_values:
            pk[k]   += precision_at_k(ranked, rel, k)
            rk[k]   += recall_at_k(ranked, rel, k)
            f1k[k]  += f1_at_k(ranked, rel, k)
            ndcg[k] += ndcg_at_k(ranked, rel, k)
            # AP@K (normalised by min(|rel|,k) as in pcr-eval.py)
            hits = [1 if d in rel else 0 for d in ranked[:k]]
            h, ap = 0, 0.0
            for i, flag in enumerate(hits):
                if flag:
                    h += 1
                    ap += h / (i + 1)
            denom = min(len(rel), k)
            apk[k] += ap / denom if denom else 0.0

    n = n_valid or 1
    m: Dict = {
        "model": label, "n_queries": n_valid,
        "MAP": sum(aps)/n, "MRR": sum(mrrs)/n, "R-Precision": sum(rps)/n,
    }
    for k in k_values:
        m[f"P@{k}"]    = pk[k] / n
        m[f"R@{k}"]    = rk[k] / n
        m[f"F1@{k}"]   = f1k[k] / n
        m[f"NDCG@{k}"] = ndcg[k] / n
        m[f"MAP@{k}"]  = apk[k] / n

    # ── Micro P / R / F1  (CaseGNN & IL-TUR style) ───────────────────
    for k in k_values:
        tp = fp = fn = 0
        for qid, ranked in results.items():
            rel = relevance.get(qid)
            if not rel:
                continue
            ranked_f = [d for d in ranked if d != qid]
            top = set(ranked_f[:k])
            tp += len(top & rel)
            fp += len(top - rel)
            fn += len(rel - top)
        mic_p = tp / (tp + fp) if (tp + fp) else 0.0
        mic_r = tp / (tp + fn) if (tp + fn) else 0.0
        mic_f = 2 * mic_p * mic_r / (mic_p + mic_r) if (mic_p + mic_r) else 0.0
        m[f"MicroP@{k}"]  = mic_p
        m[f"MicroR@{k}"]  = mic_r
        m[f"MicroF1@{k}"] = mic_f

    # ── Macro P / R / F1  (CaseGNN style) ────────────────────────────
    for k in k_values:
        mac_p_sum = mac_r_sum = 0.0
        cnt = 0
        for qid, ranked in results.items():
            rel = relevance.get(qid)
            if not rel:
                continue
            ranked_f = [d for d in ranked if d != qid]
            cnt += 1
            hits = sum(1 for d in ranked_f[:k] if d in rel)
            mac_p_sum += hits / k if k else 0.0
            mac_r_sum += hits / len(rel) if rel else 0.0
        mac_p = mac_p_sum / cnt if cnt else 0.0
        mac_r = mac_r_sum / cnt if cnt else 0.0
        mac_f = 2 * mac_p * mac_r / (mac_p + mac_r) if (mac_p + mac_r) else 0.0
        m[f"MacroP@{k}"]  = mac_p
        m[f"MacroR@{k}"]  = mac_r
        m[f"MacroF1@{k}"] = mac_f

    return m


# ═════════════════════════════════════════════════════════════════════════════
# TF-IDF sweep engine  (supports per-query candidate pools)
# ═════════════════════════════════════════════════════════════════════════════

def tokenize_doc(text: str, lang: str, ngram: int) -> List[str]:
    if lang == "zh":
        tokens = _clean_text_zh(text)
        if ngram == 1:
            return tokens
        all_tokens = list(tokens)
        for n in range(2, ngram + 1):
            for i in range(len(tokens) - n + 1):
                all_tokens.append("_".join(tokens[i:i+n]))
        return all_tokens
    else:
        return clean_text(text, remove_stopwords=True, ngram=ngram)


def run_tfidf_sweep(
    ds: Dataset,
    schemes: List[str],
    max_ngram: int,
    min_df: int,
    max_df: float,
) -> List[Dict]:
    """Run exhaustive TF-IDF sweep over a single dataset."""

    cand_ids = sorted(ds.candidates.keys())
    query_ids = sorted(ds.queries.keys())
    n_cands, n_queries = len(cand_ids), len(query_ids)

    # map id → index for fast lookup
    cand_idx = {cid: i for i, cid in enumerate(cand_ids)}

    results_list = []
    token_cache_c: Dict[int, List[List[str]]] = {}
    token_cache_q: Dict[int, List[List[str]]] = {}

    NGRAMS = list(range(1, max_ngram + 1))
    total = len(NGRAMS) * len(schemes)
    done = 0
    t0 = time.time()

    print(f"\n  Running {total} configs ({len(NGRAMS)} n-grams × {len(schemes)} schemes) "
          f"on {ds.name} …\n")

    for n in NGRAMS:
        # tokenize once per n-gram (pass token lists directly, no join)
        if n not in token_cache_c:
            print(f"    Tokenising (n_gram={n}) …", flush=True)
            token_cache_c[n] = [tokenize_doc(ds.candidates[cid], ds.lang, n)
                                for cid in cand_ids]
            token_cache_q[n] = [tokenize_doc(ds.queries[qid], ds.lang, n)
                                for qid in query_ids]

        cand_tokens = token_cache_c.pop(n)
        query_tokens = token_cache_q.pop(n)

        for si, scheme in enumerate(schemes):
            done += 1
            label = (f"{ds.name}  tfidf  n={n:<2}  scheme={scheme:<9}  "
                     f"min_df={min_df}  max_df={max_df}")

            try:
                norm_c, norm_q = _build_tfidf(
                    cand_tokens, query_tokens, scheme, min_df, max_df)
            except ValueError as e:
                print(f"    [{done:>3}/{total}]  {label}  ⚠ {e}")
                continue

            # cosine scores  (n_cands, n_queries)
            scores_mat = (norm_c @ norm_q.T).toarray()

            # build ranked results respecting candidate pool
            ranked_results: Dict[str, List[str]] = {}
            for qi, qid in enumerate(query_ids):
                if ds.candidate_pool is not None:
                    pool = ds.candidate_pool.get(qid)
                    if pool is None:
                        continue
                    pool_idxs = [cand_idx[c] for c in pool if c in cand_idx]
                    if not pool_idxs:
                        continue
                    pool_cids = [cand_ids[idx] for idx in pool_idxs]
                    pool_scores = scores_mat[pool_idxs, qi]
                    order = np.argsort(pool_scores)[::-1]
                    ranked_results[qid] = [pool_cids[j] for j in order]
                else:
                    scores = scores_mat[:, qi]
                    top_idx = np.argsort(scores)[::-1]
                    ranked_results[qid] = [cand_ids[i] for i in top_idx]

            m = evaluate_comprehensive(ranked_results, ds.relevance,
                                       K_VALUES, label=label)
            m["dataset"] = ds.name
            m["n"] = n
            m["scheme"] = scheme
            results_list.append(m)

            # progress — show MicF1 at several K values
            micf1_parts = []
            for k in K_VALUES:
                key = f"MicroF1@{k}"
                if key in m:
                    micf1_parts.append(f"@{k}={m[key]*100:.2f}%")
            print(f"    [{done:>3}/{total}]  n={n:<2} {scheme:<9}  "
                  f"MAP={m['MAP']*100:5.2f}%  MRR={m['MRR']*100:5.2f}%  "
                  f"MicF1[{' '.join(micf1_parts)}]")

            del norm_c, norm_q, scores_mat

        # free token lists after all schemes for this n-gram
        del cand_tokens, query_tokens
        gc.collect()

    elapsed = time.time() - t0
    print(f"\n  {ds.name}: {len(results_list)} configs in {elapsed:.0f}s")
    return results_list


MAX_FEATURES = 500_000  # cap vocabulary to limit memory


def _identity_analyzer(doc):
    """Analyzer that returns pre-tokenized document as-is."""
    return doc


def _build_tfidf(cand_tokens, query_tokens, scheme, min_df, max_df):
    """Build normalised TF-IDF matrices for a given scheme.

    Args:
        cand_tokens:  list of token-lists (one per candidate doc)
        query_tokens: list of token-lists (one per query doc)
    """
    if scheme == "augmented":
        cv = CountVectorizer(
            analyzer=_identity_analyzer,
            min_df=min_df, max_df=max_df,
            max_features=MAX_FEATURES)
        raw_c = cv.fit_transform(cand_tokens).astype(float)
        raw_q = cv.transform(query_tokens).astype(float)

        def augment(mat):
            mat = mat.tocsr()
            for i in range(mat.shape[0]):
                s, e = mat.indptr[i], mat.indptr[i+1]
                if e > s:
                    mx = mat.data[s:e].max()
                    if mx > 0:
                        mat.data[s:e] = 0.5 + 0.5 * mat.data[s:e] / mx
            return mat

        aug_c, aug_q = augment(raw_c.copy()), augment(raw_q.copy())
        del raw_c, raw_q
        N = aug_c.shape[0]
        df = np.diff(aug_c.tocsc().indptr)
        idf = np.log((1 + N) / (1 + df)) + 1.0
        norm_c = normalize(aug_c.multiply(idf), norm="l2")
        norm_q = normalize(aug_q.multiply(idf), norm="l2")
        del aug_c, aug_q
    else:
        vec = TfidfVectorizer(
            analyzer=_identity_analyzer,
            min_df=min_df, max_df=max_df,
            max_features=MAX_FEATURES,
            sublinear_tf=(scheme == "log"),
            binary=(scheme == "binary"),
            use_idf=True, norm="l2")
        norm_c = vec.fit_transform(cand_tokens)
        norm_q = vec.transform(query_tokens)
        del vec

    return norm_c, norm_q


# ═════════════════════════════════════════════════════════════════════════════
# Result I/O
# ═════════════════════════════════════════════════════════════════════════════

def save_all_results(all_results: List[Dict], tag: str):
    _ensure_dir(RESULTS_DIR)
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(RESULTS_DIR, f"tfidf_{tag}_{ts}")

    # JSON
    jpath = base + ".json"
    with open(jpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results → {jpath}")

    # CSV
    if not all_results:
        return
    all_keys = set()
    for r in all_results:
        all_keys.update(r.keys())
    # order columns sensibly
    base_cols = ["dataset", "model", "n", "scheme", "n_queries",
                 "MAP", "MRR", "R-Precision"]
    metric_cols = sorted(all_keys - set(base_cols),
                         key=lambda x: (x.split("@")[0],
                                        int(x.split("@")[1]) if "@" in x else 0))
    cols = [c for c in base_cols if c in all_keys] + metric_cols

    cpath = base + ".csv"
    with open(cpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_results:
            row = {}
            for k, v in r.items():
                if isinstance(v, float) and k not in ("n",):
                    row[k] = round(v * 100, 4)
                else:
                    row[k] = v
            w.writerow(row)
    print(f"  Results → {cpath}")


def print_summary(all_results: List[Dict]):
    if not all_results:
        return

    datasets = sorted(set(r["dataset"] for r in all_results))
    W = 150

    for ds_name in datasets:
        ds_rows = [r for r in all_results if r["dataset"] == ds_name]
        ds_rows.sort(key=lambda x: x.get("MicroF1@10", x.get("MAP", 0)),
                     reverse=True)

        print(f"\n{'═'*W}")
        print(f"  {ds_name}  —  {len(ds_rows)} configs  "
              f"(sorted by MicroF1@10 / MAP)".center(W))
        print(f"{'═'*W}")
        print(f"  {'n':>2} {'scheme':<10} "
              f"{'MAP':>7} {'MRR':>7} {'R-P':>7} "
              f"{'P@5':>7} {'R@5':>7} {'F1@5':>7} "
              f"{'P@10':>7} {'R@10':>7} {'F1@10':>7} "
              f"{'NDCG@5':>8} {'NDCG@10':>8} "
              f"{'MicF1@5':>8} {'MicF1@10':>8} "
              f"{'MacF1@5':>8} {'MacF1@10':>8}")
        print(f"  {'─'*(W-2)}")
        for r in ds_rows[:20]:
            print(
                f"  {r.get('n',''):>2} {r.get('scheme',''):<10} "
                f"{r.get('MAP',0)*100:>6.2f}% "
                f"{r.get('MRR',0)*100:>6.2f}% "
                f"{r.get('R-Precision',0)*100:>6.2f}% "
                f"{r.get('P@5',0)*100:>6.2f}% "
                f"{r.get('R@5',0)*100:>6.2f}% "
                f"{r.get('F1@5',0)*100:>6.2f}% "
                f"{r.get('P@10',0)*100:>6.2f}% "
                f"{r.get('R@10',0)*100:>6.2f}% "
                f"{r.get('F1@10',0)*100:>6.2f}% "
                f"{r.get('NDCG@5',0)*100:>7.2f}% "
                f"{r.get('NDCG@10',0)*100:>7.2f}% "
                f"{r.get('MicroF1@5',0)*100:>7.2f}% "
                f"{r.get('MicroF1@10',0)*100:>7.2f}% "
                f"{r.get('MacroF1@5',0)*100:>7.2f}% "
                f"{r.get('MacroF1@10',0)*100:>7.2f}%")
        print(f"{'═'*W}")

        # best per scheme
        print(f"\n  Best per scheme ({ds_name}):")
        for scheme in ALL_SCHEMES:
            rows = [r for r in ds_rows if r.get("scheme") == scheme]
            if not rows:
                continue
            best = rows[0]
            print(f"    {scheme:<10}  n={best.get('n','')}  "
                  f"MAP={best['MAP']*100:.2f}%  "
                  f"MicF1@10={best.get('MicroF1@10',0)*100:.2f}%  "
                  f"NDCG@5={best.get('NDCG@5',0)*100:.2f}%")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Exhaustive TF-IDF evaluation across legal retrieval benchmarks")
    parser.add_argument("--datasets", default="ilpcr",
                        help="Comma-separated: " + ",".join(DATASET_NAMES) + " or 'all'")
    parser.add_argument("--max_ngram", type=int, default=10)
    parser.add_argument("--schemes", nargs="+", default=ALL_SCHEMES)
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--max_df", type=float, default=0.95)
    parser.add_argument("--download", action="store_true",
                        help="Auto-download missing datasets")
    args = parser.parse_args()

    requested = (DATASET_NAMES if args.datasets.lower() == "all"
                 else [s.strip().lower() for s in args.datasets.split(",")])

    print(f"\n{'═'*70}")
    print(f"  TF-IDF Exhaustive Evaluation")
    print(f"  datasets : {', '.join(requested)}")
    print(f"  n-grams  : 1 .. {args.max_ngram}")
    print(f"  schemes  : {', '.join(args.schemes)}")
    print(f"  min_df   : {args.min_df}   max_df : {args.max_df}")
    print(f"{'═'*70}\n")

    # ── Load datasets ─────────────────────────────────────────────────
    print("Loading datasets …")
    loaders = {
        "ilpcr":      lambda: load_ilpcr(),
        "coliee2022": lambda: load_coliee("2022", args.download),
        "coliee2023": lambda: load_coliee("2023", args.download),
        "lecard":     lambda: load_lecard(args.download),
        "clerc":      lambda: load_clerc(args.download),
        "ecthr":      lambda: load_ecthr(args.download),
    }

    # common typo aliases
    _aliases = {
        "collie2022": "coliee2022", "collie2023": "coliee2023",
        "coliee": "coliee2023", "collie": "coliee2023",
        "etchr": "ecthr", "echr": "ecthr",
        "il-pcr": "ilpcr", "iltur": "ilpcr",
    }

    datasets: List[Dataset] = []
    for name in requested:
        resolved = _aliases.get(name, name)
        if resolved != name:
            print(f"  (resolved '{name}' → '{resolved}')")
        if resolved not in loaders:
            print(f"  ✗ Unknown dataset: {name}")
            continue
        ds = loaders[resolved]()
        if ds is not None:
            datasets.append(ds)

    if not datasets:
        print("\nNo datasets loaded. Exiting.")
        sys.exit(1)

    print(f"\n  {len(datasets)} dataset(s) loaded: "
          f"{', '.join(d.name for d in datasets)}\n")

    # ── Run sweeps ────────────────────────────────────────────────────
    all_results: List[Dict] = []
    for ds in datasets:
        ds_results = run_tfidf_sweep(ds, args.schemes, args.max_ngram,
                                     args.min_df, args.max_df)
        all_results.extend(ds_results)

    # ── Save & display ────────────────────────────────────────────────
    tag = "_".join(d.name.lower().replace("-", "") for d in datasets)
    save_all_results(all_results, tag)
    print_summary(all_results)

    # ── Cross-dataset comparison on shared metrics ────────────────────
    if len(datasets) > 1:
        print(f"\n{'═'*100}")
        print("  CROSS-DATASET BEST RESULTS (best MicroF1@10 per dataset)")
        print(f"{'═'*100}")
        print(f"  {'Dataset':<15} {'Config':<25} "
              f"{'MAP':>7} {'MRR':>7} {'NDCG@10':>8} "
              f"{'MicF1@5':>8} {'MicF1@10':>9} {'MacF1@10':>9}")
        print(f"  {'─'*94}")
        for ds in datasets:
            rows = [r for r in all_results if r["dataset"] == ds.name]
            if not rows:
                continue
            best = max(rows, key=lambda x: x.get("MicroF1@10", x.get("MAP", 0)))
            cfg = f"n={best.get('n','')} {best.get('scheme','')}"
            print(
                f"  {ds.name:<15} {cfg:<25} "
                f"{best['MAP']*100:>6.2f}% "
                f"{best['MRR']*100:>6.2f}% "
                f"{best.get('NDCG@10',0)*100:>7.2f}% "
                f"{best.get('MicroF1@5',0)*100:>7.2f}% "
                f"{best.get('MicroF1@10',0)*100:>8.2f}% "
                f"{best.get('MacroF1@10',0)*100:>8.2f}%")
        print(f"{'═'*100}")


if __name__ == "__main__":
    main()
