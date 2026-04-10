"""
tfidf_retrieval.py
==================
TF-IDF Cosine Similarity Prior Case Retrieval — multiple schemes.

Run
---
    python3 tfidf_retrieval.py --data_dir /path/to/dataset/ --split train

TF weighting schemes
--------------------
  raw      : tf
  log      : 1 + log(tf)
  binary   : 1 if tf > 0 else 0
  augmented: 0.5 + 0.5 * tf / max_tf  (normalised term frequency)

Parameter grid
--------------
  Edit the CONFIGS list below.  Comment out any line to skip that config.
"""

import os
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

from utils import (
    load_split, clean_text, evaluate_all,
    save_results, print_results_table, save_results_csv,
    cosine_sim_sparse, compute_idf,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "/home/raghavgrover/Desktop/Sem6/INLP/Project_Final/Experiments"
SPLIT    = "test"
TOP_K    = 1000
OUTPUT   = "results/tfidf_results.json"
K_VALUES = [5, 6, 7, 8, 9, 10, 11, 15, 20]

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGS  ← comment out any line to skip
# Each entry: (tf_scheme, ngram, min_df, max_df_frac)
#   tf_scheme   : "raw" | "log" | "binary" | "augmented"
#   ngram       : 1-5
#   min_df      : minimum document frequency to keep a term
#   max_df_frac : drop terms appearing in more than this fraction of docs
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS: List[Tuple] = [
    # ── Unigrams ─────────────────────────────────────────────────────────────
    ("raw",       1, 1, 1.00),
    ("raw",       1, 2, 1.00),
    ("raw",       1, 2, 0.95),
    ("log",       1, 1, 1.00),
    ("log",       1, 2, 1.00),
    ("log",       1, 2, 0.95),
    ("log",       1, 5, 0.95),
    ("log",       1, 2, 0.90),
    ("binary",    1, 1, 1.00),
    ("binary",    1, 2, 0.95),
    ("augmented", 1, 1, 1.00),
    ("augmented", 1, 2, 0.95),
    ("augmented", 1, 2, 0.90),

    # ── Bigrams ──────────────────────────────────────────────────────────────
    ("log",       2, 1, 1.00),
    ("log",       2, 2, 0.95),
    ("log",       2, 2, 1.00),
    ("binary",    2, 2, 0.95),
    ("augmented", 2, 2, 0.95),

    # ── Trigrams ─────────────────────────────────────────────────────────────
    ("log",       3, 1, 1.00),
    ("log",       3, 2, 0.95),
    ("augmented", 3, 2, 0.95),

    # ── 5-grams ──────────────────────────────────────────────────────────────
    ("log",       5, 1, 1.00),
    ("log",       5, 2, 0.95),
    ("augmented", 5, 2, 0.95),
]


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF INDEX
# ─────────────────────────────────────────────────────────────────────────────

def _tf_raw(count: int, max_tf: int) -> float:
    return float(count)

def _tf_log(count: int, max_tf: int) -> float:
    return (1.0 + math.log(count)) if count > 0 else 0.0

def _tf_binary(count: int, max_tf: int) -> float:
    return 1.0 if count > 0 else 0.0

def _tf_augmented(count: int, max_tf: int) -> float:
    return 0.5 + 0.5 * count / max_tf if max_tf > 0 else 0.0

_TF_FNS = {
    "raw":       _tf_raw,
    "log":       _tf_log,
    "binary":    _tf_binary,
    "augmented": _tf_augmented,
}


class TFIDFIndex:
    def __init__(self, tf_scheme="log", min_df=1, max_df_frac=1.0):
        self.tf_fn      = _TF_FNS[tf_scheme]
        self.min_df     = min_df
        self.max_df_frac = max_df_frac
        self.vocab: set         = set()
        self.idf: Dict[str, float] = {}
        self.vectors: Dict[str, Dict[str, float]] = {}
        self.doc_ids: List[str] = []

    def fit(self, corpus: Dict[str, List[str]]):
        self.doc_ids = list(corpus.keys())
        N            = len(self.doc_ids)
        max_df_abs   = int(self.max_df_frac * N)

        df: Dict[str, int]              = defaultdict(int)
        raw_tfs: Dict[str, Dict[str, int]] = {}

        for did in self.doc_ids:
            tf: Dict[str, int] = defaultdict(int)
            for t in corpus[did]:
                tf[t] += 1
            raw_tfs[did] = dict(tf)
            for t in tf:
                df[t] += 1

        self.vocab = {
            t for t, cnt in df.items()
            if self.min_df <= cnt <= max_df_abs
        }
        self.idf = {
            t: math.log((N + 1) / (df[t] + 1)) + 1.0
            for t in self.vocab
        }

        for did in self.doc_ids:
            tf   = raw_tfs[did]
            maxt = max(tf.values()) if tf else 1
            vec  = {
                t: self.tf_fn(cnt, maxt) * self.idf[t]
                for t, cnt in tf.items()
                if t in self.vocab
            }
            norm = math.sqrt(sum(v * v for v in vec.values()))
            if norm > 0:
                vec = {t: v / norm for t, v in vec.items()}
            self.vectors[did] = vec

    def _query_vec(self, tokens: List[str]) -> Dict[str, float]:
        tf: Dict[str, int] = defaultdict(int)
        for t in tokens:
            if t in self.vocab:
                tf[t] += 1
        maxt = max(tf.values()) if tf else 1
        vec  = {
            t: self.tf_fn(cnt, maxt) * self.idf.get(t, 0.0)
            for t, cnt in tf.items()
        }
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            vec = {t: v / norm for t, v in vec.items()}
        return vec

    def retrieve(self, qtoks: List[str], top_k: int = 1000) -> List[Tuple[str, float]]:
        qvec = self._query_vec(qtoks)
        if not qvec:
            return []
        scores = [
            (did, cosine_sim_sparse(qvec, self.vectors[did]))
            for did in self.doc_ids
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


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

    tok_cache_c: Dict[int, Dict] = {}
    tok_cache_q: Dict[int, Dict] = {}

    all_results = []

    for (tf_scheme, ngram, min_df, max_df_frac) in CONFIGS:
        name = (f"TFIDF_{tf_scheme}_ng={ngram}_"
                f"mindf={min_df}_maxdf={max_df_frac}")
        print(f"\n{'─'*60}\n  {name}\n{'─'*60}")

        if ngram not in tok_cache_c:
            tok_cache_c[ngram] = {
                cid: clean_text(t, remove_stopwords=True, ngram=ngram)
                for cid, t in candidates.items()
            }
            tok_cache_q[ngram] = {
                qid: clean_text(t, remove_stopwords=True, ngram=ngram)
                for qid, t in queries.items()
            }

        idx = TFIDFIndex(tf_scheme, min_df, max_df_frac)
        idx.fit(tok_cache_c[ngram])

        results: Dict[str, List[str]] = {}
        for qid in relevance:
            if qid not in tok_cache_q[ngram]:
                continue
            ranked    = idx.retrieve(tok_cache_q[ngram][qid], top_k=args.top_k)
            results[qid] = [d for d, _ in ranked]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MAP",
                        title="TF-IDF — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
