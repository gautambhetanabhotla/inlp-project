"""
ensemble_retrieval.py
=====================
Z-normalised weighted ensemble retrieval for Prior Case Retrieval.
Combines lexical (BM25, TF-IDF) and semantic (W2V) scores.
Adapted from IL-PCSR ensemble approach.

Run
---
    python3 ensemble_retrieval.py --data_dir /path/to/dataset/ --split train

Ensemble formula
----------------
  score(q,c) = Σ_i  weight_i × Z-norm(score_i(q,c))

Z-normalisation applied per-query so scores are comparable across methods.

Parameter grid
--------------
  Edit PAIRWISE_CONFIGS and TRIPLE_CONFIGS below.
  Comment out any line to skip that configuration.
"""

import os
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from utils import (
    load_split, clean_text, evaluate_all,
    save_results, print_results_table, save_results_csv,
    cosine_sim_matrix, compute_idf, z_norm,
    build_w2v, embed_corpus_w2v, mean_vec,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS  (edit here)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "/home/raghavgrover/Desktop/Sem6/INLP/Project_Final/Experiments"
SPLIT    = "train"
TOP_K    = 1000
OUTPUT   = "results/ensemble_results.json"
K_VALUES = [5, 6, 7, 8, 9, 10, 11, 15, 20]

WORKERS  = 4

# ─────────────────────────────────────────────────────────────────────────────
# PAIRWISE_CONFIGS  ← comment out any line to skip
# Each entry: (method_A, method_B, alpha)
#   method_A / method_B : keys from SCORE_REGISTRY (built at runtime)
#   alpha               : weight of A  (1-alpha = weight of B)
#
# Available method keys (see SCORE_REGISTRY below):
#   BM25_ng1_k1.2     BM25_ng2_k1.2     BM25_ng5_k1.2
#   BM25_ng1_k1.5     BM25_ng5_k1.5
#   TFIDF_ng1         TFIDF_ng2         TFIDF_ng5
#   W2V_sg_100        W2V_sg_200        W2V_cbow_100
#   W2V_tfidf_sg_100  W2V_tfidf_sg_200
# ─────────────────────────────────────────────────────────────────────────────

PAIRWISE_CONFIGS: List[Tuple] = [
    # ── BM25 (5-gram) + TF-IDF ────────────────────────────────────────────────
    ("BM25_ng5_k1.2", "TFIDF_ng1",  0.1),
    ("BM25_ng5_k1.2", "TFIDF_ng1",  0.3),
    ("BM25_ng5_k1.2", "TFIDF_ng1",  0.5),
    ("BM25_ng5_k1.2", "TFIDF_ng1",  0.7),
    ("BM25_ng5_k1.2", "TFIDF_ng1",  0.9),

    # ── BM25 (1-gram) + BM25 (5-gram) ─────────────────────────────────────────
    ("BM25_ng1_k1.2", "BM25_ng5_k1.2", 0.3),
    ("BM25_ng1_k1.2", "BM25_ng5_k1.2", 0.5),
    ("BM25_ng1_k1.2", "BM25_ng5_k1.2", 0.7),

    # ── TF-IDF (1-gram) + TF-IDF (2-gram) ────────────────────────────────────
    ("TFIDF_ng1", "TFIDF_ng2",  0.3),
    ("TFIDF_ng1", "TFIDF_ng2",  0.5),
    ("TFIDF_ng1", "TFIDF_ng2",  0.7),

    # ── BM25 (5-gram) + BM25 (2-gram) ─────────────────────────────────────────
    ("BM25_ng5_k1.2", "BM25_ng2_k1.2", 0.5),
    ("BM25_ng5_k1.2", "BM25_ng2_k1.2", 0.7),

    # ── BM25 + W2V (skip-gram) ────────────────────────────────────────────────
    ("BM25_ng5_k1.2", "W2V_sg_100",       0.5),
    ("BM25_ng5_k1.2", "W2V_sg_100",       0.7),
    ("BM25_ng5_k1.2", "W2V_sg_200",       0.5),
    ("BM25_ng5_k1.2", "W2V_sg_200",       0.7),
    ("BM25_ng5_k1.2", "W2V_tfidf_sg_100", 0.5),
    ("BM25_ng5_k1.2", "W2V_tfidf_sg_200", 0.5),
    ("BM25_ng5_k1.2", "W2V_tfidf_sg_200", 0.7),

    # ── TF-IDF + W2V ──────────────────────────────────────────────────────────
    ("TFIDF_ng1", "W2V_sg_100",       0.5),
    ("TFIDF_ng1", "W2V_sg_200",       0.5),
    ("TFIDF_ng1", "W2V_tfidf_sg_100", 0.5),
    ("TFIDF_ng1", "W2V_tfidf_sg_200", 0.5),
    ("TFIDF_ng1", "W2V_tfidf_sg_200", 0.7),

    # ── BM25 (1-gram) + W2V ───────────────────────────────────────────────────
    ("BM25_ng1_k1.2", "W2V_sg_200",       0.5),
    ("BM25_ng1_k1.2", "W2V_tfidf_sg_200", 0.5),
]

# ─────────────────────────────────────────────────────────────────────────────
# TRIPLE_CONFIGS  ← comment out any line to skip
# Each entry: (method_A, method_B, method_C, wA, wB, wC)
#   wA+wB+wC need not sum to 1 (scores are Z-normalised first)
# ─────────────────────────────────────────────────────────────────────────────

TRIPLE_CONFIGS: List[Tuple] = [
    # BM25(1g) + BM25(5g) + TFIDF
    ("BM25_ng1_k1.2", "BM25_ng5_k1.2", "TFIDF_ng1", 0.2, 0.5, 0.3),
    ("BM25_ng1_k1.2", "BM25_ng5_k1.2", "TFIDF_ng1", 0.3, 0.4, 0.3),
    ("BM25_ng1_k1.2", "BM25_ng5_k1.2", "TFIDF_ng1", 0.1, 0.6, 0.3),
    ("BM25_ng1_k1.2", "BM25_ng5_k1.2", "TFIDF_ng1", 0.3, 0.5, 0.2),
    ("BM25_ng1_k1.2", "BM25_ng5_k1.2", "TFIDF_ng1", 0.4, 0.4, 0.2),
    # BM25(5g) + TFIDF + W2V
    ("BM25_ng5_k1.2", "TFIDF_ng1", "W2V_sg_200",       0.5, 0.3, 0.2),
    ("BM25_ng5_k1.2", "TFIDF_ng1", "W2V_sg_200",       0.4, 0.3, 0.3),
    ("BM25_ng5_k1.2", "TFIDF_ng1", "W2V_tfidf_sg_200", 0.5, 0.3, 0.2),
    ("BM25_ng5_k1.2", "TFIDF_ng1", "W2V_tfidf_sg_200", 0.4, 0.4, 0.2),
    # BM25(1g) + BM25(5g) + W2V
    ("BM25_ng1_k1.2", "BM25_ng5_k1.2", "W2V_sg_200",       0.2, 0.6, 0.2),
    ("BM25_ng1_k1.2", "BM25_ng5_k1.2", "W2V_tfidf_sg_200", 0.2, 0.5, 0.3),
]


# ─────────────────────────────────────────────────────────────────────────────
# SCORE COMPUTATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _bm25_scores(q_tok, c_tok, k1=1.2, b=0.75):
    ids = list(c_tok.keys()); N = len(ids)
    df  = defaultdict(int); tfs = {}; lens = {}
    for cid, toks in c_tok.items():
        tf = defaultdict(int)
        for t in toks: tf[t] += 1
        tfs[cid] = dict(tf); lens[cid] = len(toks)
        for t in tf: df[t] += 1
    avgdl = sum(lens.values()) / N if N else 1.0

    def idf(t):
        d = df.get(t, 0)
        return math.log((N - d + 0.5) / (d + 0.5) + 1)

    out: Dict[str, Dict[str, float]] = {}
    for qid, qtoks in q_tok.items():
        sc = {}
        for cid in ids:
            s = sum(idf(t) * tfs[cid].get(t, 0) * (k1+1) / (
                    tfs[cid].get(t, 0) +
                    k1*(1-b+b*lens[cid]/avgdl))
                    for t in set(qtoks) if tfs[cid].get(t, 0))
            sc[cid] = s
        out[qid] = sc
    return out


def _tfidf_scores(q_tok, c_tok):
    N   = len(c_tok)
    df  = defaultdict(int); raw = {}
    for cid, toks in c_tok.items():
        tf = defaultdict(int)
        for t in toks: tf[t] += 1
        raw[cid] = dict(tf)
        for t in tf: df[t] += 1
    idf  = {t: math.log((N+1)/(c+1))+1.0 for t, c in df.items()}

    def vec(tf_dict):
        v = {t: (1+math.log(c))*idf[t] for t, c in tf_dict.items() if t in idf}
        n = math.sqrt(sum(x*x for x in v.values()))
        return {t: x/n for t, x in v.items()} if n else {}

    c_vecs = {cid: vec(raw[cid]) for cid in c_tok}

    def cosine(va, vb):
        if not va or not vb: return 0.0
        dot = sum(va.get(t,0)*vb.get(t,0) for t in va)
        na  = math.sqrt(sum(x*x for x in va.values()))
        nb  = math.sqrt(sum(x*x for x in vb.values()))
        return dot/(na*nb) if na*nb else 0.0

    out: Dict[str, Dict[str, float]] = {}
    for qid, qtoks in q_tok.items():
        tf_q = defaultdict(int)
        for t in qtoks: tf_q[t] += 1
        qvec = vec(dict(tf_q))
        out[qid] = {cid: cosine(qvec, cv) for cid, cv in c_vecs.items()}
    return out


def _w2v_scores(q_tok, c_tok, w2v, dim, idf=None):
    cand_ids, cand_mat = embed_corpus_w2v(c_tok, w2v, dim, idf)
    out: Dict[str, Dict[str, float]] = {}
    for qid, qtoks in q_tok.items():
        qvec = mean_vec(qtoks, w2v, dim, idf).reshape(1, -1)
        sims = cosine_sim_matrix(qvec, cand_mat)[0]
        out[qid] = {cand_ids[i]: float(sims[i]) for i in range(len(cand_ids))}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--split",    default=SPLIT)
    parser.add_argument("--top_k",    type=int, default=TOP_K)
    parser.add_argument("--output",   default=OUTPUT)
    parser.add_argument("--workers",  type=int, default=WORKERS)
    args = parser.parse_args()

    queries, candidates, relevance = load_split(args.data_dir, args.split)
    all_results = []

    # ── Pre-compute all base scores ──────────────────────────────────────────
    print("\nPre-computing base retrieval scores ...")

    # Tokenise for multiple n-gram orders
    def tok(corpus, ng):
        return {k: clean_text(v, True, ng) for k, v in corpus.items()}

    c1 = tok(candidates, 1);  q1 = tok(queries, 1)
    c2 = tok(candidates, 2);  q2 = tok(queries, 2)
    c5 = tok(candidates, 5);  q5 = tok(queries, 5)

    idf1 = compute_idf(c1)

    # Train W2V models
    print("  Training Word2Vec models ...")
    all1 = list(c1.values()) + list(q1.values())
    w2v_sg_100  = build_w2v(all1, vector_size=100, sg=1, workers=args.workers)
    w2v_sg_200  = build_w2v(all1, vector_size=200, sg=1, workers=args.workers)
    w2v_cbow_100 = build_w2v(all1, vector_size=100, sg=0, workers=args.workers)

    # Compute all base scores
    SCORE_REGISTRY: Dict[str, Dict[str, Dict[str, float]]] = {}

    print("  BM25 scores ...")
    SCORE_REGISTRY["BM25_ng1_k1.2"] = _bm25_scores(q1, c1, 1.2, 0.75)
    SCORE_REGISTRY["BM25_ng1_k1.5"] = _bm25_scores(q1, c1, 1.5, 0.75)
    SCORE_REGISTRY["BM25_ng2_k1.2"] = _bm25_scores(q2, c2, 1.2, 0.75)
    SCORE_REGISTRY["BM25_ng5_k1.2"] = _bm25_scores(q5, c5, 1.2, 0.75)
    SCORE_REGISTRY["BM25_ng5_k1.5"] = _bm25_scores(q5, c5, 1.5, 0.75)

    print("  TF-IDF scores ...")
    SCORE_REGISTRY["TFIDF_ng1"] = _tfidf_scores(q1, c1)
    SCORE_REGISTRY["TFIDF_ng2"] = _tfidf_scores(q2, c2)
    SCORE_REGISTRY["TFIDF_ng5"] = _tfidf_scores(q5, c5)

    print("  Word2Vec scores ...")
    SCORE_REGISTRY["W2V_sg_100"]       = _w2v_scores(q1, c1, w2v_sg_100,  100)
    SCORE_REGISTRY["W2V_sg_200"]       = _w2v_scores(q1, c1, w2v_sg_200,  200)
    SCORE_REGISTRY["W2V_cbow_100"]     = _w2v_scores(q1, c1, w2v_cbow_100,100)
    SCORE_REGISTRY["W2V_tfidf_sg_100"] = _w2v_scores(q1, c1, w2v_sg_100,  100, idf1)
    SCORE_REGISTRY["W2V_tfidf_sg_200"] = _w2v_scores(q1, c1, w2v_sg_200,  200, idf1)

    cand_ids = list(candidates.keys())

    def combine_and_rank(qid, method_weights):
        """Z-norm each method's scores then weighted sum."""
        combined: Dict[str, float] = defaultdict(float)
        for method, weight in method_weights:
            raw = SCORE_REGISTRY[method].get(qid, {})
            znorm = z_norm(raw)
            for cid in cand_ids:
                combined[cid] += weight * znorm.get(cid, 0.0)
        return sorted(cand_ids, key=lambda c: combined[c], reverse=True)

    # ── Pairwise ensembles ────────────────────────────────────────────────────
    for (mA, mB, alpha) in PAIRWISE_CONFIGS:
        name = f"Ensemble_{mA}+{mB}_a={alpha}"
        print(f"\n{'─'*64}\n  {name}\n{'─'*64}")

        results: Dict[str, List[str]] = {}
        for qid in relevance:
            ranked = combine_and_rank(qid, [(mA, alpha), (mB, 1-alpha)])
            results[qid] = ranked[:args.top_k]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    # ── Triple ensembles ─────────────────────────────────────────────────────
    for (mA, mB, mC, wA, wB, wC) in TRIPLE_CONFIGS:
        name = f"Ensemble3_{mA}+{mB}+{mC}_w={wA},{wB},{wC}"
        print(f"\n{'─'*64}\n  {name}\n{'─'*64}")

        results: Dict[str, List[str]] = {}
        for qid in relevance:
            ranked = combine_and_rank(qid, [(mA,wA),(mB,wB),(mC,wC)])
            results[qid] = ranked[:args.top_k]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MAP",
                        title="Ensemble (Z-norm) — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
