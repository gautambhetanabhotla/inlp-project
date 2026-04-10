"""
glove_retrieval.py
==================
GloVe embeddings trained on the legal corpus using `mittens`.
No external downloads required.

Run
---
    python3 glove_retrieval.py --data_dir /path/to/dataset/ --split train

Parameter grid
--------------
  Edit CONFIGS below.  Comment out any line to skip that config.
"""

import os
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
from mittens import GloVe as MittensGloVe

from utils import (
    load_split, clean_text, evaluate_all,
    save_results, print_results_table, save_results_csv,
    cosine_sim_matrix, compute_idf,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "/home/raghavgrover/Desktop/Sem6/INLP/Project_Final/Experiments"
SPLIT    = "train"
TOP_K    = 1000
OUTPUT   = "results/glove_results.json"
K_VALUES = [5, 10, 20, 50, 100]

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGS  ← comment out any line to skip
# Each entry: (n_components, max_iter, window, weighting, ngram)
#   n_components : GloVe embedding dimension
#   max_iter     : optimisation iterations
#   window       : co-occurrence window (tokens on each side)
#   weighting    : "mean" | "tfidf"
#   ngram        : tokenisation order
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS: List[Tuple] = [
    # ── Unigrams / mean ───────────────────────────────────────────────────────
    (50,  100,  5, "mean",  1),
    (100, 100,  5, "mean",  1),
    (200, 100,  5, "mean",  1),
    (300, 100,  5, "mean",  1),
    (100, 200,  5, "mean",  1),
    (200, 200,  5, "mean",  1),
    (300, 200,  5, "mean",  1),
    (100, 100, 10, "mean",  1),
    (200, 100, 10, "mean",  1),
    (300, 100, 10, "mean",  1),
    (200, 200, 10, "mean",  1),

    # ── Unigrams / TF-IDF weighted ────────────────────────────────────────────
    (50,  100,  5, "tfidf", 1),
    (100, 100,  5, "tfidf", 1),
    (200, 100,  5, "tfidf", 1),
    (300, 100,  5, "tfidf", 1),
    (100, 200,  5, "tfidf", 1),
    (200, 200,  5, "tfidf", 1),
    (300, 200,  5, "tfidf", 1),
    (100, 100, 10, "tfidf", 1),
    (200, 100, 10, "tfidf", 1),
    (200, 200, 10, "tfidf", 1),

    # ── Bigrams ───────────────────────────────────────────────────────────────
    (100, 100,  5, "mean",  2),
    (200, 100,  5, "mean",  2),
    (100, 200,  5, "mean",  2),
    (100, 100,  5, "tfidf", 2),
    (200, 100,  5, "tfidf", 2),
    (200, 200,  5, "tfidf", 2),
]


# ─────────────────────────────────────────────────────────────────────────────
# CO-OCCURRENCE MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def build_cooccurrence(
    corpus_tokens: List[List[str]],
    vocab: Dict[str, int],
    window: int = 5,
) -> np.ndarray:
    """Symmetric, distance-weighted co-occurrence matrix."""
    V    = len(vocab)
    cooc = np.zeros((V, V), dtype=np.float32)
    for tokens in tqdm(corpus_tokens, desc="  Co-occurrence", leave=False):
        ids = [vocab[t] for t in tokens if t in vocab]
        for i, wi in enumerate(ids):
            lo = max(0, i - window)
            hi = min(len(ids), i + window + 1)
            for j in range(lo, hi):
                if i == j:
                    continue
                cooc[wi, ids[j]] += 1.0 / abs(i - j)
    return cooc


def embed_docs(
    corpus: Dict[str, List[str]],
    embeddings: Dict[str, np.ndarray],
    dim: int,
    idf: Optional[Dict[str, float]] = None,
) -> Tuple[List[str], np.ndarray]:
    """Return (doc_ids, matrix). Uses IDF-weighted mean when idf given."""
    doc_ids = list(corpus.keys())
    mat = np.zeros((len(doc_ids), dim), dtype=np.float32)
    for i, did in enumerate(doc_ids):
        toks  = corpus[did]
        valid = [(t, embeddings[t]) for t in toks if t in embeddings]
        if not valid:
            continue
        if idf:
            w = np.array([idf.get(t, 1.0) for t, _ in valid], np.float32)
            w /= w.sum() + 1e-10
            mat[i] = np.sum([wt * v for (_, v), wt in zip(valid, w)], axis=0)
        else:
            mat[i] = np.mean([v for _, v in valid], axis=0)
    return doc_ids, mat


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
    all_results = []

    tok_cache_c: Dict[int, Dict] = {}
    tok_cache_q: Dict[int, Dict] = {}
    cooc_cache:  Dict[Tuple, np.ndarray]     = {}   # (ngram, window) → cooc
    vocab_cache: Dict[Tuple, Dict[str, int]] = {}
    idf_cache:   Dict[int, Dict[str, float]] = {}   # ngram → idf

    for (n_comp, max_iter, window, weighting, ngram) in CONFIGS:
        name = (f"GloVe_dim={n_comp}_iter={max_iter}_"
                f"win={window}_{weighting}_ng={ngram}")
        print(f"\n{'─'*60}\n  {name}\n{'─'*60}")

        # Tokenise
        if ngram not in tok_cache_c:
            tok_cache_c[ngram] = {
                cid: clean_text(t, True, ngram) for cid, t in candidates.items()
            }
            tok_cache_q[ngram] = {
                qid: clean_text(t, True, ngram) for qid, t in queries.items()
            }
            idf_cache[ngram] = compute_idf(tok_cache_c[ngram])

        cand_tok = tok_cache_c[ngram]
        quer_tok = tok_cache_q[ngram]
        idf_map  = idf_cache[ngram]

        # Build vocab + co-occurrence (cache by ngram + window)
        cooc_key = (ngram, window)
        if cooc_key not in cooc_cache:
            print(f"  Building vocab + co-occurrence (ng={ngram}, win={window}) ...")
            df: Dict[str, int] = defaultdict(int)
            for toks in cand_tok.values():
                for t in set(toks): df[t] += 1
            vocab = {t: i for i, (t, _) in
                     enumerate((k, v) for k, v in df.items() if v >= 2)}
            vocab_cache[cooc_key] = vocab
            print(f"  Vocab size: {len(vocab)}")
            cooc_cache[cooc_key] = build_cooccurrence(
                list(cand_tok.values()), vocab, window=window
            )

        vocab = vocab_cache[cooc_key]
        cooc  = cooc_cache[cooc_key]

        # Train GloVe
        print(f"  Training GloVe (dim={n_comp}, iter={max_iter}) ...")
        glove = MittensGloVe(n=n_comp, max_iter=max_iter)
        E     = glove.fit(cooc)  # (|vocab|, n_comp)
        embeds: Dict[str, np.ndarray] = {
            t: E[i].astype(np.float32) for t, i in vocab.items()
        }

        use_idf = (weighting == "tfidf")

        # Embed candidates
        cand_ids, cand_mat = embed_docs(cand_tok, embeds, n_comp,
                                        idf=idf_map if use_idf else None)

        # Retrieve
        results: Dict[str, List[str]] = {}
        for qid in relevance:
            if qid not in quer_tok: continue
            valid = [(t, embeds[t]) for t in quer_tok[qid] if t in embeds]
            if not valid: continue
            if use_idf:
                w = np.array([idf_map.get(t, 1.0) for t, _ in valid], np.float32)
                w /= w.sum() + 1e-10
                qvec = np.sum([wt * v for (_, v), wt in zip(valid, w)], axis=0)
            else:
                qvec = np.mean([v for _, v in valid], axis=0)
            sims  = cosine_sim_matrix(qvec.reshape(1, -1), cand_mat)[0]
            order = np.argsort(-sims)[: args.top_k]
            results[qid] = [cand_ids[i] for i in order]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MAP",
                        title="GloVe — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
