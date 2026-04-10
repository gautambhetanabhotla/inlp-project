"""
svm_retrieval.py
================
SVM-based Prior Case Retrieval with a 12-dimensional feature vector.
Adapted from Jackson et al. (2001/2003).

Run
---
    python3 svm_retrieval.py --data_dir /path/to/dataset/

Trains on ik_train, evaluates on ik_test.

Feature vector (12 dimensions per query-candidate pair)
---------------------------------------------------------
  1.  TF-IDF cosine similarity
  2.  BM25 score (normalised 0-1 per query)
  3.  Citation overlap flag (binary)
  4.  Citation Jaccard similarity
  5.  Shared citation count (normalised)
  6.  Query document length (log-normalised)
  7.  Candidate document length (log-normalised)
  8.  Length ratio (min/max)
  9.  Vocabulary Jaccard (token sets)
  10. BM25 rank (1 - rank/N)
  11. TF-IDF rank (1 - rank/N)
  12. IDF-weighted citation cosine

Parameter grid
--------------
  Edit CONFIGS below.  Comment out any line to skip that config.
"""

import os
import math
import argparse
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler

from utils import (
    load_split, clean_text, extract_citations, evaluate_all,
    save_results, print_results_table, save_results_csv,
    cosine_sim_sparse, compute_idf,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "/home/raghavgrover/Desktop/Sem6/INLP/Project_Final/Experiments"
OUTPUT   = "results/svm_results.json"
K_VALUES = [5, 10, 20, 50, 100]

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGS  ← comment out any line to skip
# Each entry: (kernel, C, neg_ratio, gamma2)
#   kernel    : "linear" | "rbf"
#   C         : SVM regularisation
#   neg_ratio : negative : positive training pairs ratio (Jackson uses 5)
#   gamma2    : relative threshold for Decision Maker (0 = disabled)
#               Only top candidates scoring >= gamma2 * max_score are kept.
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS: List[Tuple] = [
    # ── Linear SVM ────────────────────────────────────────────────────────────
    ("linear", 0.01, 5,  0.0),    # no Decision Maker
    ("linear", 0.1,  5,  0.0),
    ("linear", 1.0,  5,  0.0),
    ("linear", 10.0, 5,  0.0),
    # with Decision Maker γ2
    ("linear", 0.01, 5,  0.5),
    ("linear", 0.1,  5,  0.5),
    ("linear", 1.0,  5,  0.5),
    ("linear", 1.0,  5,  0.7),
    ("linear", 1.0,  5,  0.9),
    ("linear", 10.0, 5,  0.5),
    # vary neg ratio
    ("linear", 1.0,  1,  0.0),
    ("linear", 1.0,  3,  0.0),
    ("linear", 1.0,  10, 0.0),
    ("linear", 1.0,  1,  0.5),
    ("linear", 1.0,  3,  0.5),
    ("linear", 1.0,  10, 0.5),

    # ── RBF SVM ───────────────────────────────────────────────────────────────
    ("rbf",    0.1,  5,  0.0),
    ("rbf",    1.0,  5,  0.0),
    ("rbf",    10.0, 5,  0.0),
    ("rbf",    1.0,  5,  0.5),
    ("rbf",    1.0,  5,  0.7),
    ("rbf",    10.0, 5,  0.5),
    ("rbf",    1.0,  3,  0.0),
    ("rbf",    1.0,  10, 0.0),
    ("rbf",    1.0,  3,  0.5),
]


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_tfidf(corpus: Dict[str, List[str]]):
    """Return (vectors, idf_map) — L2-normalised log-TF × IDF."""
    N  = len(corpus)
    df: Dict[str, int] = defaultdict(int)
    raws: Dict[str, Dict[str, int]] = {}
    for did, toks in corpus.items():
        tf: Dict[str, int] = defaultdict(int)
        for t in toks: tf[t] += 1
        raws[did] = dict(tf)
        for t in tf: df[t] += 1
    idf = {t: math.log((N+1)/(c+1))+1.0 for t, c in df.items()}
    vecs: Dict[str, Dict[str, float]] = {}
    for did, tf in raws.items():
        v    = {t: (1+math.log(c))*idf[t] for t, c in tf.items() if t in idf}
        norm = math.sqrt(sum(x*x for x in v.values()))
        vecs[did] = {t: x/norm for t, x in v.items()} if norm else {}
    return vecs, idf


def _bm25_scores(qtoks, cand_ids, doc_freqs, doc_lens, avgdl, df_map, N,
                 k1=1.5, b=0.75):
    def _idf(t):
        d = df_map.get(t, 0)
        return math.log((N - d + 0.5) / (d + 0.5) + 1)
    raw: Dict[str, float] = {}
    for cid in cand_ids:
        sc = 0.0
        for t in set(qtoks):
            f = doc_freqs[cid].get(t, 0)
            if f == 0: continue
            sc += _idf(t)*f*(k1+1)/(f+k1*(1-b+b*doc_lens[cid]/avgdl))
        raw[cid] = sc
    mx = max(raw.values()) or 1.0
    return {cid: s/mx for cid, s in raw.items()}


def _cite_idf_cosine(a: Set[str], b: Set[str], idf: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(idf.get(c, 1.0)**2 for c in a & b)
    na  = math.sqrt(sum(idf.get(c, 1.0)**2 for c in a))
    nb  = math.sqrt(sum(idf.get(c, 1.0)**2 for c in b))
    return dot / (na * nb) if na * nb else 0.0


def build_features(qid, cid, q_tok, c_tok, q_vecs, c_vecs,
                   bm25_sc, bm25_rnk, tfidf_rnk,
                   q_cites, c_cites, cite_idf, N_cands):
    qtoks = q_tok.get(qid, [])
    ctoks = c_tok.get(cid, [])
    qc    = q_cites.get(qid, set())
    cc    = c_cites.get(cid, set())

    tfidf_cos = cosine_sim_sparse(q_vecs.get(qid, {}), c_vecs.get(cid, {}))
    bm25_s    = bm25_sc.get(qid, {}).get(cid, 0.0)
    cite_flag = 1.0 if (qc & cc) else 0.0
    cite_j    = len(qc & cc) / len(qc | cc) if (qc | cc) else 0.0
    cite_n    = len(qc & cc) / max(len(qc), len(cc)) if (qc or cc) else 0.0
    q_len     = math.log(len(qtoks) + 1) / 10.0
    c_len     = math.log(len(ctoks) + 1) / 10.0
    len_ratio = min(len(qtoks)+1, len(ctoks)+1) / max(len(qtoks)+1, len(ctoks)+1)
    vocab_j   = len(set(qtoks) & set(ctoks)) / len(set(qtoks) | set(ctoks)) \
                if (set(qtoks) | set(ctoks)) else 0.0
    bm25_rnk_ = 1.0 - bm25_rnk.get(qid, {}).get(cid, N_cands) / N_cands
    tfidf_rk_ = 1.0 - tfidf_rnk.get(qid, {}).get(cid, N_cands) / N_cands
    cite_idf_ = _cite_idf_cosine(qc, cc, cite_idf)

    return [tfidf_cos, bm25_s, cite_flag, cite_j, cite_n,
            q_len, c_len, len_ratio, vocab_j, bm25_rnk_, tfidf_rk_, cite_idf_]


def decision_maker(scored: List[Tuple[str, float]], gamma2: float) -> List[str]:
    """Keep only candidates with score >= gamma2 * top_score."""
    if not scored or gamma2 <= 0.0:
        return [d for d, _ in scored]
    top = max(s for _, s in scored)
    return [d for d, s in scored if s >= gamma2 * top]


def _prepare(queries, candidates, relevance):
    """Build all shared structures needed for feature extraction."""
    c_tok_ = {cid: clean_text(t, True, 1) for cid, t in candidates.items()}
    q_tok_ = {qid: clean_text(t, True, 1) for qid, t in queries.items()}

    c_vecs_, _ = _build_tfidf(c_tok_)
    q_vecs_, _ = _build_tfidf(q_tok_)

    cand_ids = list(candidates.keys())
    N = len(cand_ids)

    # BM25 scores + ranks
    df_map: Dict[str, int] = defaultdict(int)
    doc_freqs: Dict[str, Dict[str, int]] = {}
    doc_lens:  Dict[str, int] = {}
    for cid, toks in c_tok_.items():
        tf: Dict[str, int] = defaultdict(int)
        for t in toks: tf[t] += 1
        doc_freqs[cid] = dict(tf); doc_lens[cid] = len(toks)
        for t in tf: df_map[t] += 1
    avgdl = sum(doc_lens.values()) / N if N else 1.0

    bm25_sc_:   Dict[str, Dict[str, float]] = {}
    bm25_rnk_:  Dict[str, Dict[str, int]]   = {}
    tfidf_rnk_: Dict[str, Dict[str, int]]   = {}

    for qid in relevance:
        bm25_sc_[qid] = _bm25_scores(
            q_tok_.get(qid, []), cand_ids, doc_freqs, doc_lens,
            avgdl, df_map, N)
        s_bm25 = sorted(cand_ids, key=lambda c: bm25_sc_[qid][c], reverse=True)
        bm25_rnk_[qid] = {c: i for i, c in enumerate(s_bm25)}

        tfidf_raw = {cid: cosine_sim_sparse(q_vecs_.get(qid,{}), c_vecs_.get(cid,{}))
                     for cid in cand_ids}
        s_tfidf = sorted(cand_ids, key=lambda c: tfidf_raw[c], reverse=True)
        tfidf_rnk_[qid] = {c: i for i, c in enumerate(s_tfidf)}

    q_cites_ = {qid: extract_citations(t) for qid, t in queries.items()}
    c_cites_ = {cid: extract_citations(t) for cid, t in candidates.items()}
    cite_df_: Dict[str, int] = defaultdict(int)
    for cset in c_cites_.values():
        for c in cset: cite_df_[c] += 1
    cite_idf_ = {c: math.log((N+1)/(d+1))+1.0 for c, d in cite_df_.items()}

    return (cand_ids, N, c_tok_, q_tok_, c_vecs_, q_vecs_,
            bm25_sc_, bm25_rnk_, tfidf_rnk_,
            q_cites_, c_cites_, cite_idf_)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--output",   default=OUTPUT)
    args = parser.parse_args()

    print("Loading train split ...")
    tr_q, tr_c, tr_rel = load_split(args.data_dir, "train")
    print("Loading test split ...")
    te_q, te_c, te_rel = load_split(args.data_dir, "test")

    # Evaluate on test if available, else train
    ev_q   = te_q   if te_q   else tr_q
    ev_c   = te_c   if te_c   else tr_c
    ev_rel = te_rel if te_rel else tr_rel

    print("\nPreparing train features ...")
    tr = _prepare(tr_q, tr_c, tr_rel)
    (tr_cids, tr_N, tr_ctok, tr_qtok, tr_cvecs, tr_qvecs,
     tr_bm25, tr_bm25rnk, tr_tfrnk, tr_qcites, tr_ccites, tr_cidf) = tr

    print("Preparing eval features ...")
    ev = _prepare(ev_q, ev_c, ev_rel)
    (ev_cids, ev_N, ev_ctok, ev_qtok, ev_cvecs, ev_qvecs,
     ev_bm25, ev_bm25rnk, ev_tfrnk, ev_qcites, ev_ccites, ev_cidf) = ev

    all_results = []

    for (kernel, C, neg_ratio, gamma2) in CONFIGS:
        dm_str = f"_DM={gamma2}" if gamma2 > 0 else ""
        name   = f"SVM_{kernel}_C={C}_neg={neg_ratio}{dm_str}"
        print(f"\n{'─'*60}\n  {name}\n{'─'*60}")

        # ── Build training data ──────────────────────────────────────────────
        X_tr, y_tr = [], []
        for qid, rel in tr_rel.items():
            rel_set  = set(rel)
            neg_pool = [c for c in tr_cids if c not in rel_set]
            for pos in rel:
                if pos not in tr_c: continue
                X_tr.append(build_features(
                    qid, pos, tr_qtok, tr_ctok, tr_qvecs, tr_cvecs,
                    tr_bm25, tr_bm25rnk, tr_tfrnk,
                    tr_qcites, tr_ccites, tr_cidf, tr_N))
                y_tr.append(1)
            n_neg = min(len(rel) * neg_ratio, len(neg_pool))
            for neg in random.sample(neg_pool, n_neg):
                X_tr.append(build_features(
                    qid, neg, tr_qtok, tr_ctok, tr_qvecs, tr_cvecs,
                    tr_bm25, tr_bm25rnk, tr_tfrnk,
                    tr_qcites, tr_ccites, tr_cidf, tr_N))
                y_tr.append(0)

        print(f"  Train: {len(X_tr)} samples  "
              f"(pos={y_tr.count(1)}, neg={y_tr.count(0)})")

        scaler = StandardScaler()
        X_np   = scaler.fit_transform(X_tr)

        cw = {1: neg_ratio, 0: 1}
        if kernel == "linear":
            clf = LinearSVC(C=C, class_weight=cw, max_iter=2000, random_state=42)
            clf.fit(X_np, y_tr)
            def score_fn(fv_scaled): return clf.decision_function([fv_scaled])[0]
        else:
            clf = SVC(kernel=kernel, C=C, class_weight=cw,
                      probability=True, random_state=42)
            clf.fit(X_np, y_tr)
            def score_fn(fv_scaled): return clf.predict_proba([fv_scaled])[0][1]

        # ── Evaluate ──────────────────────────────────────────────────────────
        results: Dict[str, List[str]] = {}
        for qid in ev_rel:
            if qid not in ev_qtok: continue
            scored: List[Tuple[str, float]] = []
            for cid in ev_cids:
                fv = build_features(
                    qid, cid, ev_qtok, ev_ctok, ev_qvecs, ev_cvecs,
                    ev_bm25, ev_bm25rnk, ev_tfrnk,
                    ev_qcites, ev_ccites, ev_cidf, ev_N)
                fv_s = scaler.transform([fv])[0]
                scored.append((cid, score_fn(fv_s)))
            scored.sort(key=lambda x: x[1], reverse=True)
            results[qid] = decision_maker(scored, gamma2)

        m = evaluate_all(results, ev_rel, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MAP",
                        title="SVM — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
