"""
citation_reranker.py
=====================
Two-stage retrieval: 5-gram TF-IDF retrieval → citation-boosted reranking.

Motivation
----------
  • 5-gram TF-IDF is the best first-stage retriever (MicroF1@10=0.360)
  • Citation overlap (BC, Jaccard, IDF-cosine) was tested standalone
    but never as a RERANKER on top of 5-gram TF-IDF results
  • Combining: stage-1 retrieves top-N with 5-gram TF-IDF,
               stage-2 reranks using citation similarity + TF-IDF score

Formula (linear interpolation, Z-normalised):
    final_score(q,d) = α · Z(tfidf_score) + (1-α) · Z(citation_score)

Also implements a multiplicative boost:
    final_score = tfidf_score × (1 + β × citation_overlap)

This is the approach from IL-PCSR (sequential multi-task) applied
to citations as a complementary signal.

Run
---
    python3 citation_reranker.py --data_dir /path/to/dataset/ --split train

Parameter grid
--------------
  Edit RERANK_CONFIGS below.  Comment out any line to skip.
"""

import os
import re
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import (
    load_split, clean_text, extract_citations, evaluate_all,
    save_results, print_results_table, save_results_csv, z_norm,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS  (edit here)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "./"
SPLIT    = "train"
TOP_K    = 1000
OUTPUT   = "results/citation_reranker_results.json"
K_VALUES = [5, 6, 7, 8, 9, 10, 11, 15, 20]

# ─────────────────────────────────────────────────────────────────────────────
# RERANK_CONFIGS  ← comment out any line to skip
# (cite_method, alpha, beta, rerank_depth)
#
#   cite_method   : "BC" | "Jaccard" | "Dice" | "IDFCosine" | "overlap_count"
#   alpha         : weight of TF-IDF score in linear combination (0–1)
#                   (1-alpha) = weight of citation score
#                   alpha=1.0 means citation is ignored (baseline)
#   beta          : multiplicative boost factor for citation overlap
#                   final = tfidf × (1 + beta × cite_sim)
#                   beta=0.0 means no boost (baseline)
#   rerank_depth  : how many top TF-IDF candidates to rerank
#                   (rest are kept in original order)
# ─────────────────────────────────────────────────────────────────────────────

RERANK_CONFIGS: List[Tuple] = [
    # ── Baseline: pure 5-gram TF-IDF (alpha=1, beta=0) ────────────────────────
    ("BC",         1.0, 0.0, 1000),   # no citation signal

    # ── Linear combination: Z-normalised TF-IDF + citation ────────────────────
    ("BC",         0.9, 0.0,  100),
    ("BC",         0.8, 0.0,  100),
    ("BC",         0.7, 0.0,  100),
    ("BC",         0.6, 0.0,  100),
    ("BC",         0.5, 0.0,  100),
    ("Jaccard",    0.9, 0.0,  100),
    ("Jaccard",    0.8, 0.0,  100),
    ("Jaccard",    0.7, 0.0,  100),
    ("Jaccard",    0.6, 0.0,  100),
    ("IDFCosine",  0.9, 0.0,  100),
    ("IDFCosine",  0.8, 0.0,  100),
    ("IDFCosine",  0.7, 0.0,  100),
    ("IDFCosine",  0.6, 0.0,  100),

    # ── Vary rerank depth ─────────────────────────────────────────────────────
    ("IDFCosine",  0.8, 0.0,   50),
    ("IDFCosine",  0.8, 0.0,  200),
    ("IDFCosine",  0.8, 0.0,  500),
    ("BC",         0.8, 0.0,   50),
    ("BC",         0.8, 0.0,  200),
    ("BC",         0.8, 0.0,  500),

    # ── Multiplicative boost ───────────────────────────────────────────────────
    ("BC",         1.0, 0.5, 1000),
    ("BC",         1.0, 1.0, 1000),
    ("BC",         1.0, 2.0, 1000),
    ("BC",         1.0, 5.0, 1000),
    ("IDFCosine",  1.0, 0.5, 1000),
    ("IDFCosine",  1.0, 1.0, 1000),
    ("IDFCosine",  1.0, 2.0, 1000),
    ("IDFCosine",  1.0, 5.0, 1000),
    ("Jaccard",    1.0, 1.0, 1000),
    ("Jaccard",    1.0, 2.0, 1000),

    # ── Combined linear + multiplicative ──────────────────────────────────────
    ("IDFCosine",  0.8, 1.0,  100),
    ("IDFCosine",  0.8, 2.0,  100),
    ("BC",         0.8, 1.0,  100),
    ("BC",         0.8, 2.0,  100),
]


# ─────────────────────────────────────────────────────────────────────────────
# CITATION SIMILARITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def bc(a, b):
    if not a or not b: return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))

def jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0

def dice(a, b):
    d = len(a) + len(b)
    return 2 * len(a & b) / d if d else 0.0

def idf_cosine(a, b, idf):
    if not a or not b: return 0.0
    dot = sum(idf.get(c, 1.0)**2 for c in a & b)
    na  = math.sqrt(sum(idf.get(c, 1.0)**2 for c in a))
    nb  = math.sqrt(sum(idf.get(c, 1.0)**2 for c in b))
    return dot / (na * nb) if na * nb else 0.0

def overlap_count(a, b):
    return float(len(a & b))

CITE_FNS = {
    "BC":           bc,
    "Jaccard":      jaccard,
    "Dice":         dice,
    "IDFCosine":    None,   # needs idf arg
    "overlap_count": overlap_count,
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--split",    default=SPLIT)
    parser.add_argument("--top_k",    type=int, default=TOP_K)
    parser.add_argument("--output",   default=OUTPUT)
    args = parser.parse_args()

    queries, candidates, relevance = load_split(args.data_dir, args.split)
    cand_ids = list(candidates.keys())
    all_results = []

    # ── Build 5-gram TF-IDF index (best first-stage) ─────────────────────────
    print("\nBuilding 5-gram TF-IDF index ...")
    sw_c = [" ".join(clean_text(candidates[c], True, 1)) for c in cand_ids]
    sw_q = {qid: " ".join(clean_text(queries[qid], True, 1)) for qid in queries}

    vec = TfidfVectorizer(ngram_range=(5,5), min_df=2, max_df=0.95,
                          sublinear_tf=True, norm="l2")
    C = vec.fit_transform(sw_c)

    print("  Computing TF-IDF scores for all queries ...")
    tfidf_scores: Dict[str, Dict[str, float]] = {}
    tfidf_ranked: Dict[str, List[str]]        = {}
    for qid in queries:
        q    = vec.transform([sw_q[qid]])
        sims = cosine_similarity(q, C)[0]
        tfidf_scores[qid] = {cand_ids[i]: float(sims[i]) for i in range(len(cand_ids))}
        order = np.argsort(-sims)[:args.top_k]
        tfidf_ranked[qid] = [cand_ids[i] for i in order]

    # ── Extract citations ─────────────────────────────────────────────────────
    print("Extracting citations ...")
    q_cites: Dict[str, Set[str]] = {
        qid: extract_citations(t) for qid, t in queries.items()
    }
    c_cites: Dict[str, Set[str]] = {
        cid: extract_citations(t) for cid, t in candidates.items()
    }

    cite_df: Dict[str, int] = defaultdict(int)
    for cset in c_cites.values():
        for c in cset: cite_df[c] += 1
    N = len(c_cites)
    cite_idf = {c: math.log((N+1)/(df+1))+1.0 for c, df in cite_df.items()}

    n_q_cites = sum(1 for c in q_cites.values() if c)
    n_c_cites = sum(1 for c in c_cites.values() if c)
    print(f"  Queries with citations: {n_q_cites}/{len(q_cites)}")
    print(f"  Cands with citations:   {n_c_cites}/{len(c_cites)}")

    # ── Rerank ────────────────────────────────────────────────────────────────
    for (cite_method, alpha, beta, rerank_depth) in RERANK_CONFIGS:
        name = (f"CiteRerank_{cite_method}_a={alpha}_b={beta}_"
                f"depth={rerank_depth}")
        print(f"\n{'─'*64}\n  {name}\n{'─'*64}")

        results: Dict[str, List[str]] = {}
        for qid in relevance:
            if qid not in tfidf_ranked: continue

            qc    = q_cites.get(qid, set())
            pool  = tfidf_ranked[qid]          # already sorted by TF-IDF
            top_n = pool[:rerank_depth]         # candidates to rerank
            rest  = pool[rerank_depth:]         # pass-through

            # Compute citation scores for pool
            cite_scores: Dict[str, float] = {}
            for cid in top_n:
                cc = c_cites.get(cid, set())
                if cite_method == "IDFCosine":
                    cite_scores[cid] = idf_cosine(qc, cc, cite_idf)
                else:
                    cite_scores[cid] = CITE_FNS[cite_method](qc, cc)

            if alpha == 1.0 and beta == 0.0:
                # Baseline: pure TF-IDF order
                results[qid] = pool
                continue

            if beta > 0.0:
                # Multiplicative boost
                reranked = sorted(
                    top_n,
                    key=lambda c: tfidf_scores[qid].get(c, 0.0) *
                                  (1 + beta * cite_scores.get(c, 0.0)),
                    reverse=True
                )
            else:
                # Linear Z-normalised combination
                tfidf_z = z_norm({c: tfidf_scores[qid].get(c, 0.0) for c in top_n})
                cite_z  = z_norm(cite_scores)
                combined = {c: alpha * tfidf_z.get(c, 0.0) +
                               (1 - alpha) * cite_z.get(c, 0.0)
                            for c in top_n}
                reranked = sorted(top_n, key=lambda c: combined[c], reverse=True)

            results[qid] = reranked + rest

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MicroF1@10",
                        title="Citation Reranker — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
