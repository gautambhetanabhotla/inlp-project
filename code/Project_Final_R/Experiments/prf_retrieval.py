"""
prf_retrieval.py
================
Pseudo-Relevance Feedback (PRF) for Prior Case Retrieval.

Why this is novel here
----------------------
All previous methods use a fixed query representation.
PRF assumes the top-K results from 5-gram TF-IDF are relevant,
extracts their most distinctive terms/n-grams, adds them to the
query vector, and re-retrieves against the full candidate pool.

This is proven to give +10-20% MAP in TREC Legal Track (2006-2011).

Why it should work for Indian legal documents
---------------------------------------------
Legal judgements discussing the same prior case share specific
quoted phrases: the name of the cited case, the principles it
established, and the statutory provisions it interpreted.
The top-K TF-IDF results share these phrases reliably — so
PRF-expanded queries find MORE related cases that quote the same.

Two PRF algorithms implemented
-------------------------------
1. Rocchio (1971) — classic additive query expansion:
     q_new = α·q_orig + β·(mean of top-K doc vectors) - γ·(mean of bottom-M doc vectors)
   γ is usually 0 in IR (no negative feedback).

2. RM3 (Lavrenko & Croft, 2001) — relevance model:
     P(t|q_RM) = (1-λ)·P(t|q_MLE) + λ·Σ_{d∈top-K} P(d|q)·P(t|d)
   Expands with terms ranked by P(t|q_RM), interpolated back with original.
   RM3 is the standard PRF baseline in most modern IR evaluations.

Run
---
    python3 prf_retrieval.py --data_dir /path/to/dataset/ --split train

Parameter grid
--------------
  Edit CONFIGS below.  Comment out any line to skip.
"""

import os
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import (
    load_split, clean_text, evaluate_all,
    save_results, print_results_table, save_results_csv,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS  (edit here)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = "./"
SPLIT      = "train"
TOP_K      = 1000
OUTPUT     = "results/prf_results.json"
K_VALUES   = [5, 6, 7, 8, 9, 10, 11, 15, 20]

# First-stage retriever settings (the 5-gram winner)
BASE_NGRAM  = 5
BASE_MINDF  = 2
BASE_MAXDF  = 0.95

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGS  ← comment out any line to skip
# Each entry: (prf_method, fb_docs, fb_terms, alpha, beta, lambda_rm3)
#
#   prf_method  : "rocchio" | "rm3" | "rm3_ngram"
#   fb_docs     : number of top documents assumed relevant (feedback pool)
#   fb_terms    : number of feedback terms added to query
#   alpha       : original query weight (Rocchio only)
#   beta        : feedback document weight (Rocchio only)
#   lambda_rm3  : interpolation weight for original query in RM3
#                 (0 = pure RM3 expansion, 1 = no expansion)
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS: List[Tuple] = [
    # ── Rocchio on 5-gram TF-IDF ──────────────────────────────────────────────
    ("rocchio", 3,  50,  1.0, 0.8, 0.0),
    ("rocchio", 5,  50,  1.0, 0.8, 0.0),
    ("rocchio", 10, 50,  1.0, 0.8, 0.0),
    ("rocchio", 3,  100, 1.0, 0.8, 0.0),
    ("rocchio", 5,  100, 1.0, 0.8, 0.0),
    ("rocchio", 10, 100, 1.0, 0.8, 0.0),
    ("rocchio", 3,  200, 1.0, 0.8, 0.0),
    ("rocchio", 5,  200, 1.0, 0.8, 0.0),
    ("rocchio", 3,  50,  1.0, 0.5, 0.0),
    ("rocchio", 5,  50,  1.0, 0.5, 0.0),
    ("rocchio", 10, 50,  1.0, 0.5, 0.0),
    ("rocchio", 3,  50,  1.0, 1.0, 0.0),
    ("rocchio", 5,  100, 1.0, 1.0, 0.0),
    ("rocchio", 10, 100, 1.0, 1.2, 0.0),
    ("rocchio", 5,  50,  0.8, 0.8, 0.0),
    ("rocchio", 5,  50,  0.6, 0.8, 0.0),

    # ── RM3 on 5-gram TF-IDF ──────────────────────────────────────────────────
    ("rm3",     3,  50,  0.0, 0.0, 0.5),
    ("rm3",     5,  50,  0.0, 0.0, 0.5),
    ("rm3",     10, 50,  0.0, 0.0, 0.5),
    ("rm3",     3,  100, 0.0, 0.0, 0.5),
    ("rm3",     5,  100, 0.0, 0.0, 0.5),
    ("rm3",     10, 100, 0.0, 0.0, 0.5),
    ("rm3",     3,  50,  0.0, 0.0, 0.3),
    ("rm3",     5,  50,  0.0, 0.0, 0.3),
    ("rm3",     10, 50,  0.0, 0.0, 0.3),
    ("rm3",     3,  50,  0.0, 0.0, 0.7),
    ("rm3",     5,  50,  0.0, 0.0, 0.7),
    ("rm3",     5,  200, 0.0, 0.0, 0.5),
    ("rm3",     10, 200, 0.0, 0.0, 0.5),
    ("rm3",     3,  50,  0.0, 0.0, 0.1),   # mostly expansion
    ("rm3",     5,  100, 0.0, 0.0, 0.1),

    # ── RM3 on 1-gram then re-score with 5-gram ────────────────────────────────
    # (prf from unigram retrieval, re-rank with 5-gram expansion)
    ("rm3_ngram", 5,  50,  0.0, 0.0, 0.5),
    ("rm3_ngram", 10, 100, 0.0, 0.0, 0.5),
    ("rm3_ngram", 5,  100, 0.0, 0.0, 0.3),
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf(cand_texts, ngram, min_df, max_df):
    vec = TfidfVectorizer(
        ngram_range=(ngram, ngram),
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        norm="l2",
    )
    C = vec.fit_transform(cand_texts)
    return vec, C


def rocchio_expand(q_vec, C_top_k, alpha, beta, top_terms):
    """
    Rocchio:  q_new = alpha*q + beta * mean(top_k_docs)
    Returns a new dense query vector in the TF-IDF space.
    """
    q_dense = np.asarray(q_vec.todense()).flatten()
    fb_mean = np.asarray(C_top_k.mean(axis=0)).flatten()
    q_new   = alpha * q_dense + beta * fb_mean

    # Zero out all but top_terms highest-weight dimensions
    if top_terms < len(q_new):
        thresh = np.partition(q_new, -top_terms)[-top_terms]
        q_new[q_new < thresh] = 0.0

    # Re-normalise
    norm = np.linalg.norm(q_new)
    return q_new / norm if norm > 0 else q_new


def rm3_expand(q_vec, C_top_k, q_tfidf_dense,
               top_terms, lambda_rm3, cand_scores_top_k):
    """
    RM3:
      P(t|R) ∝ Σ_{d∈top-K} P(d|q) * tf(t,d)/|d|
      q_new = lambda*P(t|q_MLE) + (1-lambda)*P(t|R)
    """
    # P(d|q) proportional to TF-IDF cosine (already positive)
    scores = np.array(cand_scores_top_k, dtype=np.float32)
    if scores.sum() > 0:
        scores /= scores.sum()

    # Build RM expansion vector in TF-IDF space
    C_dense = np.asarray(C_top_k.todense())       # (K, V)
    # Weighted average of document vectors by P(d|q)
    rm_vec  = (scores[:, None] * C_dense).sum(axis=0)  # (V,)

    # Normalise RM vector
    rm_norm = rm_vec.sum()
    if rm_norm > 0:
        rm_vec /= rm_norm

    # Original query vector (MLE)
    q_mle = q_tfidf_dense.copy()
    q_mle_norm = q_mle.sum()
    if q_mle_norm > 0:
        q_mle /= q_mle_norm

    # Interpolate
    q_rm3 = lambda_rm3 * q_mle + (1 - lambda_rm3) * rm_vec

    # Keep top_terms
    if top_terms < len(q_rm3):
        thresh = np.partition(q_rm3, -top_terms)[-top_terms]
        q_rm3[q_rm3 < thresh] = 0.0

    norm = np.linalg.norm(q_rm3)
    return q_rm3 / norm if norm > 0 else q_rm3


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

    # ── Pre-tokenise ────────────────────────────────────────────────────────
    print("\nTokenising corpus (unigrams) ...")
    c_str1 = {cid: " ".join(clean_text(t, True, 1)) for cid, t in candidates.items()}
    q_str1 = {qid: " ".join(clean_text(t, True, 1)) for qid, t in queries.items()}

    print(f"Tokenising corpus ({BASE_NGRAM}-grams) ...")
    c_str5 = {cid: " ".join(clean_text(t, True, BASE_NGRAM)) for cid, t in candidates.items()}
    q_str5 = {qid: " ".join(clean_text(t, True, BASE_NGRAM)) for qid, t in queries.items()}

    c_texts5 = [c_str5[c] for c in cand_ids]
    c_texts1 = [c_str1[c] for c in cand_ids]

    # ── Build 5-gram and 1-gram TF-IDF indexes ───────────────────────────────
    print(f"Building {BASE_NGRAM}-gram TF-IDF index ...")
    vec5, C5 = build_tfidf(c_texts5, BASE_NGRAM, BASE_MINDF, BASE_MAXDF)

    print("Building 1-gram TF-IDF index (for rm3_ngram) ...")
    vec1, C1 = build_tfidf(c_texts1, 1, BASE_MINDF, BASE_MAXDF)

    # ── Pre-compute initial 5-gram rankings ─────────────────────────────────
    print("Computing first-stage rankings ...")
    stage1_ranked5: Dict[str, List[Tuple[str, float]]] = {}
    for qid in relevance:
        if qid not in q_str5: continue
        qvec = vec5.transform([q_str5[qid]])
        sims = cosine_similarity(qvec, C5)[0]
        order = np.argsort(-sims)
        stage1_ranked5[qid] = [(cand_ids[i], float(sims[i])) for i in order]

    stage1_ranked1: Dict[str, List[Tuple[str, float]]] = {}
    for qid in relevance:
        if qid not in q_str1: continue
        qvec = vec1.transform([q_str1[qid]])
        sims = cosine_similarity(qvec, C1)[0]
        order = np.argsort(-sims)
        stage1_ranked1[qid] = [(cand_ids[i], float(sims[i])) for i in order]

    # ── Run each PRF config ───────────────────────────────────────────────────
    for (prf_method, fb_docs, fb_terms, alpha, beta, lambda_rm3) in CONFIGS:
        if prf_method == "rocchio":
            name = f"PRF_Rocchio_fb={fb_docs}_ft={fb_terms}_a={alpha}_b={beta}"
        elif prf_method == "rm3":
            name = f"PRF_RM3_fb={fb_docs}_ft={fb_terms}_lam={lambda_rm3}"
        else:
            name = f"PRF_RM3ng_fb={fb_docs}_ft={fb_terms}_lam={lambda_rm3}"

        print(f"\n{'─'*64}\n  {name}\n{'─'*64}")

        results: Dict[str, List[str]] = {}

        for qid in relevance:
            if qid not in stage1_ranked5: continue

            stage1 = stage1_ranked5[qid]

            if prf_method == "rm3_ngram":
                # PRF from 1-gram retrieval → expand in 5-gram space
                stage1_for_fb = stage1_ranked1.get(qid, stage1)
            else:
                stage1_for_fb = stage1

            # Get indices into C of the feedback documents
            fb_cids    = [cid for cid, _ in stage1_for_fb[:fb_docs]]
            fb_indices = [cand_ids.index(cid) for cid in fb_cids
                          if cid in cand_ids]

            # Operate in 5-gram space
            qvec5  = vec5.transform([q_str5[qid]])
            q_dense = np.asarray(qvec5.todense()).flatten()

            if prf_method == "rocchio":
                if not fb_indices: continue
                C_top = C5[fb_indices]
                q_new = rocchio_expand(qvec5, C_top, alpha, beta, fb_terms)
                sims  = C5.dot(q_new)

            else:  # rm3 or rm3_ngram
                if not fb_indices: continue
                C_top = C5[fb_indices]
                fb_scores = [s for _, s in stage1_for_fb[:fb_docs]]
                q_new = rm3_expand(
                    qvec5, C_top, q_dense,
                    fb_terms, lambda_rm3, fb_scores
                )
                sims = C5.dot(q_new)

            if hasattr(sims, 'A1'):
                sims = sims.A1  # sparse matrix → 1-D

            order = np.argsort(-sims)[:args.top_k]
            results[qid] = [cand_ids[i] for i in order]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MicroF1@10",
                        title="Pseudo-Relevance Feedback — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
