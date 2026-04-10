"""
bm25_retrieval.py
=================
BM25 Prior Case Retrieval — three variants, full parameter sweep.

Run
---
    python3 bm25_retrieval.py --data_dir /path/to/dataset/ --split train

All results are printed and saved to results/bm25_results.json.

Variants
--------
  • Okapi BM25   — standard Robertson BM25
  • BM25L        — long-document variant (lower-bound floor δ)
  • BM25+        — additive floor in term-freq numerator

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
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS  (edit here)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR   = "/home/raghavgrover/Desktop/Sem6/INLP/Project_Final/Experiments"
SPLIT      = "test"        # "train" or "test"
TOP_K      = 1000           # max candidates per query
OUTPUT     = "results/bm25_results.json"
K_VALUES   = [5, 6, 7, 8, 9, 10, 11, 15, 20]

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGS  ← comment out any line to skip
# Each entry: (variant, ngram, k1, b, delta)
#   variant : "okapi" | "bm25l" | "bm25plus"
#   ngram   : 1 = unigrams, 2 = bigrams, etc.
#   delta   : only used by bm25l and bm25plus
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS: List[Tuple] = [
    # ── Okapi BM25 ──────────────────────────────────────────────────────────
    ("okapi",   1,  0.9,  0.40, 0),    # ATIRE default
    ("okapi",   1,  1.2,  0.75, 0),    # LeCoPCR default
    ("okapi",   1,  1.5,  0.75, 0),    # Modified ATIRE / IL-PCSR
    ("okapi",   1,  2.0,  0.75, 0),
    ("okapi",   1,  1.2,  0.50, 0),
    ("okapi",   1,  1.5,  0.50, 0),
    ("okapi",   2,  1.2,  0.75, 0),    # bigram
    ("okapi",   2,  1.5,  0.75, 0),
    ("okapi",   3,  1.2,  0.75, 0),    # trigram
    ("okapi",   3,  1.5,  0.75, 0),
    ("okapi",   5,  1.2,  0.75, 0),    # 5-gram  (IL-PCSR best)
    ("okapi",   5,  1.5,  0.75, 0),
    ("okapi",   5,  1.2,  0.50, 0),

    # ── BM25L ───────────────────────────────────────────────────────────────
    ("bm25l",   1,  1.5,  0.75, 0.5),  # Turkish paper default
    ("bm25l",   1,  1.2,  0.75, 0.5),
    ("bm25l",   1,  1.5,  0.50, 0.5),
    ("bm25l",   1,  1.5,  0.75, 1.0),
    ("bm25l",   2,  1.5,  0.75, 0.5),
    ("bm25l",   5,  1.5,  0.75, 0.5),
    ("bm25l",   5,  1.2,  0.75, 0.5),

    # ── BM25+ ───────────────────────────────────────────────────────────────
    ("bm25plus", 1, 1.2,  0.75, 1.0),
    ("bm25plus", 1, 1.5,  0.75, 1.0),
    ("bm25plus", 1, 1.2,  0.75, 0.5),
    ("bm25plus", 1, 1.5,  0.50, 1.0),
    ("bm25plus", 2, 1.2,  0.75, 1.0),
    ("bm25plus", 5, 1.2,  0.75, 1.0),
    ("bm25plus", 5, 1.5,  0.75, 1.0),
]


# ─────────────────────────────────────────────────────────────────────────────
# BM25 IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

class _BM25Base:
    """Shared fit / doc-freq logic."""

    def fit(self, corpus: Dict[str, List[str]]):
        self.doc_ids   = list(corpus.keys())
        self.N         = len(self.doc_ids)
        self.doc_freqs = []
        self.doc_lens  = []
        self.df: Dict[str, int] = defaultdict(int)

        for did in self.doc_ids:
            tf: Dict[str, int] = defaultdict(int)
            for t in corpus[did]:
                tf[t] += 1
            self.doc_freqs.append(dict(tf))
            self.doc_lens.append(len(corpus[did]))
            for t in tf:
                self.df[t] += 1

        self.avgdl = sum(self.doc_lens) / self.N if self.N else 1.0

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def retrieve(self, qtoks: List[str], top_k: int = 1000) -> List[Tuple[str, float]]:
        scores = [(self.doc_ids[i], self._score(qtoks, i))
                  for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class OkapiBM25(_BM25Base):
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1; self.b = b

    def _score(self, qtoks, idx):
        tf_d = self.doc_freqs[idx]
        dl   = self.doc_lens[idx]
        sc   = 0.0
        for t in set(qtoks):
            f = tf_d.get(t, 0)
            if f == 0:
                continue
            sc += self._idf(t) * f * (self.k1 + 1) / (
                f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return sc


class BM25L(_BM25Base):
    def __init__(self, k1=1.5, b=0.75, delta=0.5):
        self.k1 = k1; self.b = b; self.delta = delta

    def _score(self, qtoks, idx):
        tf_d = self.doc_freqs[idx]
        dl   = self.doc_lens[idx]
        norm = 1 - self.b + self.b * dl / self.avgdl
        sc   = 0.0
        for t in set(qtoks):
            f = tf_d.get(t, 0)
            if f == 0:
                continue
            ctd  = f / norm
            sc  += self._idf(t) * (self.k1 + 1) * (ctd + self.delta) / (
                self.k1 + ctd + self.delta)
        return sc


class BM25Plus(_BM25Base):
    def __init__(self, k1=1.2, b=0.75, delta=1.0):
        self.k1 = k1; self.b = b; self.delta = delta

    def _score(self, qtoks, idx):
        tf_d = self.doc_freqs[idx]
        dl   = self.doc_lens[idx]
        sc   = 0.0
        for t in set(qtoks):
            f = tf_d.get(t, 0)
            if f == 0:
                continue
            num = (self.k1 + 1) * f + self.delta
            den = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            sc += self._idf(t) * num / den
        return sc


_VARIANT_MAP = {
    "okapi":   OkapiBM25,
    "bm25l":   BM25L,
    "bm25plus": BM25Plus,
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

    # Token cache keyed by ngram size — tokenise once, reuse
    tok_cache_c: Dict[int, Dict[str, List[str]]] = {}
    tok_cache_q: Dict[int, Dict[str, List[str]]] = {}

    all_results = []

    for (variant, ngram, k1, b, delta) in CONFIGS:
        # Build descriptive name
        if variant == "okapi":
            name = f"BM25_okapi_ng={ngram}_k1={k1}_b={b}"
        elif variant == "bm25l":
            name = f"BM25L_ng={ngram}_k1={k1}_b={b}_d={delta}"
        else:
            name = f"BM25+_ng={ngram}_k1={k1}_b={b}_d={delta}"

        print(f"\n{'─'*60}\n  {name}\n{'─'*60}")

        # Tokenise (only if not already cached for this ngram)
        if ngram not in tok_cache_c:
            tok_cache_c[ngram] = {
                cid: clean_text(t, remove_stopwords=True, ngram=ngram)
                for cid, t in candidates.items()
            }
            tok_cache_q[ngram] = {
                qid: clean_text(t, remove_stopwords=True, ngram=ngram)
                for qid, t in queries.items()
            }

        cand_tok  = tok_cache_c[ngram]
        query_tok = tok_cache_q[ngram]

        # Build and fit model
        cls_kwargs = dict(k1=k1, b=b)
        if variant in ("bm25l", "bm25plus"):
            cls_kwargs["delta"] = delta
        model = _VARIANT_MAP[variant](**cls_kwargs)
        model.fit(cand_tok)

        # Retrieve for every query that has ground truth
        results: Dict[str, List[str]] = {}
        for qid in relevance:
            if qid not in query_tok:
                continue
            ranked = model.retrieve(query_tok[qid], top_k=args.top_k)
            results[qid] = [d for d, _ in ranked]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    # ── Save + report ────────────────────────────────────────────────────────
    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MAP",
                        title="BM25 — ALL CONFIGURATIONS")
    save_results_csv(all_results,
                     args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
