"""
word2vec_retrieval.py
=====================
Word2Vec document embeddings for Prior Case Retrieval.
Trained on the corpus itself (no external downloads).

Run
---
    python3 word2vec_retrieval.py --data_dir /path/to/dataset/ --split train

Representations
---------------
  • Mean of token vectors
  • TF-IDF weighted mean of token vectors

Parameter grid
--------------
  Edit the CONFIGS list below.  Comment out any line to skip that config.
"""

import os
import argparse
from typing import Dict, List, Tuple

import numpy as np

from utils import (
    load_split, clean_text, evaluate_all,
    save_results, print_results_table, save_results_csv,
    build_w2v, embed_corpus_w2v, compute_idf,
    cosine_sim_matrix,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "/home/raghavgrover/Desktop/Sem6/INLP/Project_Final/Experiments"
SPLIT    = "train"
TOP_K    = 1000
OUTPUT   = "results/word2vec_results.json"
WORKERS  = 4
K_VALUES = [5, 10, 20, 50, 100]

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGS  ← comment out any line to skip
# Each entry: (architecture, vector_size, window, min_count, weighting, ngram)
#   architecture : "skipgram" (sg=1) | "cbow" (sg=0)
#   vector_size  : embedding dimension
#   window       : context window
#   min_count    : minimum token frequency to include
#   weighting    : "mean" | "tfidf"
#   ngram        : tokenisation order (1 = unigrams, 2 = bigrams)
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS: List[Tuple] = [
    # ── Skip-gram ─────────────────────────────────────────────────────────────
    ("skipgram", 100, 5,  2, "mean",  1),
    ("skipgram", 200, 5,  2, "mean",  1),
    ("skipgram", 300, 5,  2, "mean",  1),
    ("skipgram", 100, 10, 2, "mean",  1),
    ("skipgram", 200, 10, 2, "mean",  1),
    ("skipgram", 100, 5,  5, "mean",  1),
    ("skipgram", 200, 5,  5, "mean",  1),
    # TF-IDF weighted
    ("skipgram", 100, 5,  2, "tfidf", 1),
    ("skipgram", 200, 5,  2, "tfidf", 1),
    ("skipgram", 300, 5,  2, "tfidf", 1),
    ("skipgram", 200, 10, 2, "tfidf", 1),
    ("skipgram", 100, 5,  5, "tfidf", 1),
    # bigrams
    ("skipgram", 100, 5,  2, "mean",  2),
    ("skipgram", 200, 5,  2, "mean",  2),
    ("skipgram", 100, 5,  2, "tfidf", 2),
    ("skipgram", 200, 5,  2, "tfidf", 2),

    # ── CBOW ──────────────────────────────────────────────────────────────────
    ("cbow", 100, 5,  2, "mean",  1),
    ("cbow", 200, 5,  2, "mean",  1),
    ("cbow", 300, 5,  2, "mean",  1),
    ("cbow", 100, 10, 2, "mean",  1),
    ("cbow", 200, 10, 2, "mean",  1),
    ("cbow", 100, 5,  5, "mean",  1),
    # TF-IDF weighted
    ("cbow", 100, 5,  2, "tfidf", 1),
    ("cbow", 200, 5,  2, "tfidf", 1),
    ("cbow", 300, 5,  2, "tfidf", 1),
    ("cbow", 200, 10, 2, "tfidf", 1),
    # bigrams
    ("cbow", 100, 5,  2, "mean",  2),
    ("cbow", 200, 5,  2, "tfidf", 2),
]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--split",    default=SPLIT)
    parser.add_argument("--top_k",    type=int, default=TOP_K)
    parser.add_argument("--workers",  type=int, default=WORKERS)
    parser.add_argument("--output",   default=OUTPUT)
    args = parser.parse_args()

    queries, candidates, relevance = load_split(args.data_dir, args.split)

    tok_cache_c: Dict[int, Dict] = {}
    tok_cache_q: Dict[int, Dict] = {}
    w2v_cache: Dict[Tuple, object] = {}   # (arch, dim, win, mc, ngram) → model

    all_results = []

    for (arch, dim, window, min_count, weighting, ngram) in CONFIGS:
        name = (f"W2V_{arch}_d={dim}_w={window}_mc={min_count}_"
                f"{weighting}_ng={ngram}")
        print(f"\n{'─'*60}\n  {name}\n{'─'*60}")

        # Tokenise
        if ngram not in tok_cache_c:
            tok_cache_c[ngram] = {
                cid: clean_text(t, True, ngram)
                for cid, t in candidates.items()
            }
            tok_cache_q[ngram] = {
                qid: clean_text(t, True, ngram)
                for qid, t in queries.items()
            }

        cand_tok  = tok_cache_c[ngram]
        query_tok = tok_cache_q[ngram]

        # Train W2V (cache by model params + ngram)
        w2v_key = (arch, dim, window, min_count, ngram)
        if w2v_key not in w2v_cache:
            print(f"  Training Word2Vec ({arch}, dim={dim}) ...")
            all_sents = [t for t in
                         list(cand_tok.values()) + list(query_tok.values())
                         if t]
            sg = 1 if arch == "skipgram" else 0
            w2v_cache[w2v_key] = build_w2v(
                all_sents, vector_size=dim, window=window,
                min_count=min_count, sg=sg,
                workers=args.workers, epochs=5, seed=42
            )
        w2v = w2v_cache[w2v_key]

        idf = compute_idf(cand_tok) if weighting == "tfidf" else None

        # Embed candidates
        cand_ids, cand_mat = embed_corpus_w2v(cand_tok, w2v, dim, idf)

        # Retrieve
        results: Dict[str, List[str]] = {}
        for qid in relevance:
            if qid not in query_tok:
                continue
            from utils import mean_vec
            qvec = mean_vec(query_tok[qid], w2v, dim, idf).reshape(1, -1)
            sims  = cosine_sim_matrix(qvec, cand_mat)[0]
            order = np.argsort(-sims)[: args.top_k]
            results[qid] = [cand_ids[i] for i in order]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MAP",
                        title="Word2Vec — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
