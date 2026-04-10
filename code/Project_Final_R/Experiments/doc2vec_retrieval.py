"""
doc2vec_retrieval.py
====================
Doc2Vec (PV-DM / PV-DBOW) document embeddings for Prior Case Retrieval.
Trained on the corpus itself.

Run
---
    python3 doc2vec_retrieval.py --data_dir /path/to/dataset/ --split train

Parameter grid
--------------
  Edit the CONFIGS list below.  Comment out any line to skip that config.
"""

import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from utils import (
    load_split, clean_text, evaluate_all,
    save_results, print_results_table, save_results_csv,
    cosine_sim_matrix,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "/home/raghavgrover/Desktop/Sem6/INLP/Project_Final/Experiments"
SPLIT    = "train"
TOP_K    = 1000
OUTPUT   = "results/doc2vec_results.json"
WORKERS  = 4
K_VALUES = [5, 10, 20, 50, 100]

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGS  ← comment out any line to skip
# Each entry: (architecture, vector_size, window, min_count, infer_epochs, ngram)
#   architecture  : "pvdm" (dm=1) | "pvdbow" (dm=0)
#   vector_size   : embedding dimension
#   window        : context window (PV-DM)
#   min_count     : minimum token frequency
#   infer_epochs  : epochs for query vector inference at test time
#   ngram         : tokenisation order
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS: List[Tuple] = [
    # ── PV-DM (dm=1) ──────────────────────────────────────────────────────────
    ("pvdm",   100, 5,  2,  50, 1),
    ("pvdm",   200, 5,  2,  50, 1),
    ("pvdm",   300, 5,  2,  50, 1),
    ("pvdm",   100, 10, 2,  50, 1),
    ("pvdm",   200, 10, 2,  50, 1),
    ("pvdm",   100, 5,  5,  50, 1),
    ("pvdm",   200, 5,  5,  50, 1),
    # more infer epochs
    ("pvdm",   100, 5,  2, 200, 1),
    ("pvdm",   200, 5,  2, 200, 1),
    ("pvdm",   300, 5,  2, 200, 1),
    ("pvdm",   200, 10, 2, 200, 1),
    # bigrams
    ("pvdm",   100, 5,  2,  50, 2),
    ("pvdm",   200, 5,  2,  50, 2),
    ("pvdm",   200, 5,  2, 200, 2),

    # ── PV-DBOW (dm=0) ────────────────────────────────────────────────────────
    ("pvdbow", 100, 5,  2,  50, 1),
    ("pvdbow", 200, 5,  2,  50, 1),
    ("pvdbow", 300, 5,  2,  50, 1),
    ("pvdbow", 100, 5,  5,  50, 1),
    ("pvdbow", 200, 5,  5,  50, 1),
    ("pvdbow", 100, 5,  2, 200, 1),
    ("pvdbow", 200, 5,  2, 200, 1),
    ("pvdbow", 300, 5,  2, 200, 1),
    # bigrams
    ("pvdbow", 100, 5,  2,  50, 2),
    ("pvdbow", 200, 5,  2, 200, 2),
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
    model_cache: Dict[Tuple, Doc2Vec] = {}   # (arch, dim, win, mc, ngram) → model

    all_results = []
    cand_ids_cache: Dict[int, List[str]] = {}

    for (arch, dim, window, min_count, infer_ep, ngram) in CONFIGS:
        name = (f"D2V_{arch}_d={dim}_w={window}_mc={min_count}_"
                f"inf={infer_ep}_ng={ngram}")
        print(f"\n{'─'*60}\n  {name}\n{'─'*60}")

        # Tokenise
        if ngram not in tok_cache_c:
            tok_cache_c[ngram] = {
                cid: clean_text(t, True, ngram) for cid, t in candidates.items()
            }
            tok_cache_q[ngram] = {
                qid: clean_text(t, True, ngram) for qid, t in queries.items()
            }
            cand_ids_cache[ngram] = list(tok_cache_c[ngram].keys())

        cand_tok  = tok_cache_c[ngram]
        query_tok = tok_cache_q[ngram]
        cand_ids  = cand_ids_cache[ngram]

        # Train Doc2Vec (cache by model params)
        model_key = (arch, dim, window, min_count, ngram)
        if model_key not in model_cache:
            print(f"  Training Doc2Vec ({arch}, dim={dim}) ...")
            dm = 1 if arch == "pvdm" else 0
            tagged = [
                TaggedDocument(words=cand_tok[cid], tags=[i])
                for i, cid in enumerate(cand_ids)
            ]
            # Add query docs for vocab coverage (inferred at test time)
            extra = [
                TaggedDocument(words=query_tok[qid], tags=[len(cand_ids) + j])
                for j, qid in enumerate(query_tok.keys())
                if query_tok[qid]
            ]
            d2v = Doc2Vec(
                documents=tagged + extra,
                vector_size=dim,
                window=window,
                min_count=min_count,
                dm=dm,
                workers=args.workers,
                epochs=10,
                seed=42,
            )
            model_cache[model_key] = d2v

        d2v = model_cache[model_key]

        # Candidate vectors are fixed (trained)
        cand_mat = np.stack([d2v.dv[i] for i in range(len(cand_ids))]).astype(np.float32)

        # Retrieve
        results: Dict[str, List[str]] = {}
        for qid in relevance:
            if qid not in query_tok or not query_tok[qid]:
                continue
            qvec  = d2v.infer_vector(query_tok[qid], epochs=infer_ep).reshape(1, -1)
            sims  = cosine_sim_matrix(qvec, cand_mat)[0]
            order = np.argsort(-sims)[: args.top_k]
            results[qid] = [cand_ids[i] for i in order]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MAP",
                        title="Doc2Vec — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
