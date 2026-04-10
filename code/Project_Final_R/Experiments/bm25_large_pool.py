"""
bm25_large_pool.py
==================
BM25 retrieval run on the FULL large candidate pool (all 2000 candidates,
817 queries) — the split used by Method1/2/3 and dense models.

Key insight from results analysis
----------------------------------
The existing bm25_retrieval.py only ran on 234 queries (the train split with
a small candidate pool).  The large-pool split (817 queries, 2000 candidates)
was never tested with BM25, yet it's the dataset that Method1 nouns+tfidf
achieves MicroF1@10=0.262 on.  BM25(5-gram) on this pool should be much
stronger since:
  • The candidate pool has ~2000 real prior cases
  • n-gram BM25 captures phrase-level legal vocabulary better than unigrams
  • BM25 naturally handles doc-length normalisation

Text variants explored
-----------------------
  full      : entire raw document (baseline, same as bm25_retrieval.py but
               now on the large pool)
  nouns     : NN/NNS/NNP only (Method-1 vocabulary)
  nouns_verbs: NN*/VB* (Method-2 vocabulary)
  content   : remove boilerplate headers/footers via heuristic

Run
---
    python3 bm25_large_pool.py --data_dir /path/to/dataset/ --split train

Parameter grid
--------------
  Edit CONFIGS below.  Comment out any line to skip.
"""

import os
import re
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer

from utils import (
    load_split, clean_text, evaluate_all,
    save_results, print_results_table, save_results_csv,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS  (edit here)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "./"
SPLIT    = "train"
TOP_K    = 1000
OUTPUT   = "results/bm25_large_pool_results.json"
K_VALUES = [5, 10, 20, 50, 100]

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGS  ← comment out any line to skip
# (variant, ngram, k1, b, delta, bm25_type)
#   variant   : "full" | "nouns" | "nouns_verbs"
#   ngram     : n-gram order for tokenisation
#   k1, b     : BM25 parameters
#   delta     : BM25+ additive floor (0 = standard Okapi)
#   bm25_type : "okapi" | "bm25plus" | "bm25l"
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS: List[Tuple] = [
    # ── Full text, n-gram BM25 ────────────────────────────────────────────────
    ("full", 1, 1.2, 0.75, 0,   "okapi"),
    ("full", 2, 1.2, 0.75, 0,   "okapi"),
    ("full", 3, 1.2, 0.75, 0,   "okapi"),
    ("full", 5, 1.2, 0.75, 0,   "okapi"),
    ("full", 5, 1.5, 0.75, 0,   "okapi"),
    ("full", 5, 1.2, 0.75, 1.0, "bm25plus"),
    ("full", 5, 1.5, 0.75, 1.0, "bm25plus"),

    # ── Nouns only (Method-1 vocabulary) + BM25 ───────────────────────────────
    ("nouns", 1, 1.2, 0.75, 0,   "okapi"),
    ("nouns", 2, 1.2, 0.75, 0,   "okapi"),
    ("nouns", 3, 1.2, 0.75, 0,   "okapi"),
    ("nouns", 5, 1.2, 0.75, 0,   "okapi"),
    ("nouns", 5, 1.5, 0.75, 0,   "okapi"),
    ("nouns", 5, 1.2, 0.75, 1.0, "bm25plus"),
    ("nouns", 1, 1.2, 0.75, 0,   "bm25l"),
    ("nouns", 5, 1.2, 0.75, 0,   "bm25l"),

    # ── Nouns + Verbs (Method-2 vocabulary) + BM25 ────────────────────────────
    ("nouns_verbs", 1, 1.2, 0.75, 0,   "okapi"),
    ("nouns_verbs", 2, 1.2, 0.75, 0,   "okapi"),
    ("nouns_verbs", 3, 1.2, 0.75, 0,   "okapi"),
    ("nouns_verbs", 5, 1.2, 0.75, 0,   "okapi"),
    ("nouns_verbs", 5, 1.5, 0.75, 0,   "okapi"),
    ("nouns_verbs", 5, 1.2, 0.75, 1.0, "bm25plus"),
    ("nouns_verbs", 5, 1.5, 0.75, 1.0, "bm25plus"),
]

# ─────────────────────────────────────────────────────────────────────────────
# POS-BASED TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

NOUN_TAGS = {"NN", "NNS", "NNP"}
VERB_TAGS = {"VB", "VBZ", "VBN", "VBD"}
_lem = WordNetLemmatizer()
_CITES = re.compile(r'\[?\?CITATION\?\]?|<CITATION_\d+>', re.IGNORECASE)
_PUNCT = re.compile(r'[",\-\'_]')


def _preprocess(text: str) -> str:
    text = _CITES.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    return text


def extract_nouns(text: str) -> List[str]:
    tokens = word_tokenize(_preprocess(text))
    return [_lem.lemmatize(w.lower(), 'n')
            for w, t in pos_tag(tokens) if t in NOUN_TAGS]


def extract_nouns_verbs(text: str) -> List[str]:
    tokens = word_tokenize(_preprocess(text))
    terms  = []
    for w, t in pos_tag(tokens):
        if t in NOUN_TAGS:
            terms.append(_lem.lemmatize(w.lower(), 'n'))
        elif t in VERB_TAGS:
            terms.append(_lem.lemmatize(w.lower(), 'v'))
    return terms


def get_tokens(text: str, variant: str, ngram: int) -> List[str]:
    if variant == "nouns":
        base = extract_nouns(text)
    elif variant == "nouns_verbs":
        base = extract_nouns_verbs(text)
    else:
        base = clean_text(text, remove_stopwords=True, ngram=1)

    if ngram == 1:
        return base
    all_toks = list(base)
    for n in range(2, ngram + 1):
        for i in range(len(base) - n + 1):
            all_toks.append("_".join(base[i: i + n]))
    return all_toks


# ─────────────────────────────────────────────────────────────────────────────
# BM25 IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

class BM25Base:
    def fit(self, corpus: Dict[str, List[str]]):
        self.ids   = list(corpus.keys()); self.N = len(self.ids)
        self.df    = defaultdict(int); self.tfs = []; self.lens = []
        for did in self.ids:
            tf = defaultdict(int)
            for t in corpus[did]: tf[t] += 1
            self.tfs.append(dict(tf)); self.lens.append(len(corpus[did]))
            for t in tf: self.df[t] += 1
        self.avgdl = sum(self.lens) / self.N if self.N else 1.0

    def _idf(self, t):
        d = self.df.get(t, 0)
        return math.log((self.N - d + 0.5) / (d + 0.5) + 1)

    def retrieve(self, qtoks, top_k=1000):
        scores = [(self.ids[i], self._score(qtoks, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class OkapiBM25(BM25Base):
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1; self.b = b

    def _score(self, qtoks, idx):
        sc = 0.0
        for t in set(qtoks):
            f = self.tfs[idx].get(t, 0)
            if not f: continue
            sc += self._idf(t) * f*(self.k1+1) / (
                f + self.k1*(1-self.b+self.b*self.lens[idx]/self.avgdl))
        return sc


class BM25Plus(BM25Base):
    def __init__(self, k1=1.2, b=0.75, delta=1.0):
        self.k1 = k1; self.b = b; self.delta = delta

    def _score(self, qtoks, idx):
        sc = 0.0
        for t in set(qtoks):
            f = self.tfs[idx].get(t, 0)
            if not f: continue
            num = (self.k1+1)*f + self.delta
            den = f + self.k1*(1-self.b+self.b*self.lens[idx]/self.avgdl)
            sc += self._idf(t) * num / den
        return sc


class BM25L(BM25Base):
    def __init__(self, k1=1.5, b=0.75, delta=0.5):
        self.k1 = k1; self.b = b; self.delta = delta

    def _score(self, qtoks, idx):
        sc = 0.0
        norm = 1 - self.b + self.b * self.lens[idx] / self.avgdl
        for t in set(qtoks):
            f = self.tfs[idx].get(t, 0)
            if not f: continue
            ctd = f / norm
            sc += self._idf(t) * (self.k1+1)*(ctd+self.delta) / (
                self.k1 + ctd + self.delta)
        return sc


_BM25_CLS = {"okapi": OkapiBM25, "bm25plus": BM25Plus, "bm25l": BM25L}


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

    # Token cache: (variant, ngram) → {id: tokens}
    tok_cache_c: Dict[Tuple, Dict] = {}
    tok_cache_q: Dict[Tuple, Dict] = {}

    for (variant, ngram, k1, b, delta, btype) in CONFIGS:
        name = (f"BM25lp_{btype}_{variant}_ng={ngram}_k1={k1}_b={b}"
                + (f"_d={delta}" if delta else ""))
        print(f"\n{'─'*64}\n  {name}\n{'─'*64}")

        key = (variant, ngram)
        if key not in tok_cache_c:
            print(f"  Tokenising ({variant}, ng={ngram}) ...")
            tok_cache_c[key] = {
                cid: get_tokens(text, variant, ngram)
                for cid, text in candidates.items()
            }
            tok_cache_q[key] = {
                qid: get_tokens(text, variant, ngram)
                for qid, text in queries.items()
            }

        # Build model
        cls = _BM25_CLS[btype]
        if btype == "okapi":
            model = cls(k1=k1, b=b)
        else:
            model = cls(k1=k1, b=b, delta=delta)
        model.fit(tok_cache_c[key])

        results: Dict[str, List[str]] = {}
        for qid in relevance:
            if qid not in tok_cache_q[key]: continue
            ranked = model.retrieve(tok_cache_q[key][qid], top_k=args.top_k)
            results[qid] = [d for d, _ in ranked]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MAP",
                        title="BM25 Large Pool — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
