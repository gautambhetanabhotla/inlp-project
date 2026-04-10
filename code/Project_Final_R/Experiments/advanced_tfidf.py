"""
advanced_tfidf.py
=================
Advanced TF-IDF weighting schemes for Prior Case Retrieval.
All run on the FULL candidate pool (817-query dataset).

Schemes implemented
--------------------

1. Pivot Normalised TF-IDF  (Singhal et al., 1996)
   TF_piv(t,d) = (1 + log(1 + log(tf))) / (1 - slope + slope × |d|/avgdl)
   IDF: standard log IDF
   → Specifically designed for long documents.  The pivot normalisation
     avoids over-penalising long documents (a known BM25 shortcoming).

2. BM25-style term saturation applied to TF-IDF vectors
   tf_sat(t,d) = tf / (tf + k)   for k ∈ {1, 2, 5, 10}
   → Soft TF saturation without full BM25 length normalisation.
     Useful when document lengths are roughly uniform.

3. Log-logistic TF  (studied in TREC legal track)
   tf_ll(t,d) = log(1 + log(1 + tf))
   → Double-log dampening, more aggressive than single log.

4. Okapi-style within a TF-IDF framework
   Combined with explicit IDF rather than BM25 IDF.

5. Augmented TF + max_df trim (already best on small pool; now on large pool)
   aug_tf = 0.5 + 0.5 × tf / max_tf_in_doc

All schemes optionally filter vocabulary to:
  (a) nouns only (Method-1 vocabulary)
  (b) nouns + verbs (Method-2 vocabulary)
  (c) full text

Run
---
    python3 advanced_tfidf.py --data_dir /path/to/dataset/ --split train

Parameter grid
--------------
  Edit CONFIGS below.  Comment out any line to skip.
"""

import os
import re
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Callable

import numpy as np
import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer

from utils import (
    load_split, clean_text, evaluate_all,
    save_results, print_results_table, save_results_csv,
    cosine_sim_sparse,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS  (edit here)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "./"
SPLIT    = "train"
TOP_K    = 1000
OUTPUT   = "results/advanced_tfidf_results.json"
K_VALUES = [5, 10, 20, 50, 100]

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGS  ← comment out any line to skip
# (tf_scheme, vocab, ngram, min_df, max_df, extra_param)
#
#   tf_scheme   : "pivot" | "saturation" | "loglog" | "augmented" | "sublinear"
#   vocab       : "full" | "nouns" | "nouns_verbs"
#   ngram       : n-gram tokenisation order
#   min_df      : minimum document frequency
#   max_df      : maximum document frequency fraction
#   extra_param : scheme-specific param
#                 pivot     → slope s  (0.0–1.0, lower = less length normalisation)
#                 saturation→ k (TF saturation constant)
#                 loglog    → unused (pass 0)
#                 augmented → unused (pass 0)
#                 sublinear → unused (pass 0)
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS: List[Tuple] = [
    # ── Pivot normalised TF-IDF ───────────────────────────────────────────────
    ("pivot",      "full",        1, 2, 0.95, 0.20),
    ("pivot",      "full",        1, 2, 0.95, 0.30),
    ("pivot",      "full",        1, 2, 0.95, 0.40),
    ("pivot",      "full",        1, 2, 0.95, 0.50),
    ("pivot",      "full",        5, 2, 0.95, 0.20),
    ("pivot",      "full",        5, 2, 0.95, 0.30),
    ("pivot",      "full",        5, 2, 0.95, 0.40),
    ("pivot",      "nouns",       1, 2, 0.95, 0.20),
    ("pivot",      "nouns",       1, 2, 0.95, 0.30),
    ("pivot",      "nouns",       1, 2, 0.95, 0.40),
    ("pivot",      "nouns",       3, 2, 0.95, 0.30),
    ("pivot",      "nouns_verbs", 1, 2, 0.95, 0.30),
    ("pivot",      "nouns_verbs", 3, 2, 0.95, 0.30),

    # ── TF Saturation ─────────────────────────────────────────────────────────
    ("saturation", "full",        1, 2, 0.95,  1),
    ("saturation", "full",        1, 2, 0.95,  2),
    ("saturation", "full",        1, 2, 0.95,  5),
    ("saturation", "full",        5, 2, 0.95,  1),
    ("saturation", "full",        5, 2, 0.95,  2),
    ("saturation", "nouns",       1, 2, 0.95,  1),
    ("saturation", "nouns",       1, 2, 0.95,  2),
    ("saturation", "nouns",       3, 2, 0.95,  1),
    ("saturation", "nouns_verbs", 1, 2, 0.95,  1),
    ("saturation", "nouns_verbs", 3, 2, 0.95,  1),

    # ── Log-log TF ────────────────────────────────────────────────────────────
    ("loglog",     "full",        1, 2, 0.95,  0),
    ("loglog",     "full",        5, 2, 0.95,  0),
    ("loglog",     "nouns",       1, 2, 0.95,  0),
    ("loglog",     "nouns",       3, 2, 0.95,  0),
    ("loglog",     "nouns_verbs", 1, 2, 0.95,  0),
    ("loglog",     "nouns_verbs", 3, 2, 0.95,  0),

    # ── Augmented TF on large pool ────────────────────────────────────────────
    ("augmented",  "full",        1, 2, 0.95,  0),
    ("augmented",  "full",        3, 2, 0.95,  0),
    ("augmented",  "full",        5, 2, 0.95,  0),
    ("augmented",  "nouns",       1, 2, 0.95,  0),
    ("augmented",  "nouns",       3, 2, 0.95,  0),
    ("augmented",  "nouns_verbs", 1, 2, 0.95,  0),
    ("augmented",  "nouns_verbs", 3, 2, 0.95,  0),

    # ── Sublinear TF on large pool ────────────────────────────────────────────
    ("sublinear",  "full",        1, 2, 0.95,  0),
    ("sublinear",  "full",        3, 2, 0.95,  0),
    ("sublinear",  "full",        5, 2, 0.95,  0),
    ("sublinear",  "nouns",       1, 2, 0.95,  0),
    ("sublinear",  "nouns",       3, 2, 0.95,  0),
    ("sublinear",  "nouns_verbs", 1, 2, 0.95,  0),
    ("sublinear",  "nouns_verbs", 3, 2, 0.95,  0),
]

# ─────────────────────────────────────────────────────────────────────────────
# POS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

NOUN_TAGS = {"NN","NNS","NNP"}
VERB_TAGS = {"VB","VBZ","VBN","VBD"}
_lem  = WordNetLemmatizer()
_CITE = re.compile(r'\[?\?CITATION\?\]?|<CITATION_\d+>', re.IGNORECASE)
_PUN  = re.compile(r'[",\-\'_]')

def _prep(t): return _PUN.sub(" ", _CITE.sub(" ", t))

def _nouns(text):
    toks = word_tokenize(_prep(text))
    return [_lem.lemmatize(w.lower(),'n') for w,t in pos_tag(toks)
            if t in NOUN_TAGS]

def _nouns_verbs(text):
    toks=word_tokenize(_prep(text)); out=[]
    for w,t in pos_tag(toks):
        if t in NOUN_TAGS: out.append(_lem.lemmatize(w.lower(),'n'))
        elif t in VERB_TAGS: out.append(_lem.lemmatize(w.lower(),'v'))
    return out

def tokenise(text, vocab, ngram):
    if vocab == "nouns":
        base = _nouns(text)
    elif vocab == "nouns_verbs":
        base = _nouns_verbs(text)
    else:
        base = clean_text(text, remove_stopwords=True, ngram=1)
    if ngram == 1:
        return base
    out = list(base)
    for n in range(2, ngram+1):
        for i in range(len(base)-n+1):
            out.append("_".join(base[i:i+n]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM TF-IDF INDEX (pure Python — no sklearn)
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedTFIDF:
    """
    Custom TF-IDF index supporting multiple TF weighting schemes,
    built without sklearn so we can apply any TF function directly.
    """

    def __init__(self, tf_scheme, min_df, max_df_frac, extra_param):
        self.tf_scheme   = tf_scheme
        self.min_df      = min_df
        self.max_df_frac = max_df_frac
        self.extra       = extra_param
        self.vocab:   set = set()
        self.idf:     Dict[str, float] = {}
        self.vectors: Dict[str, Dict[str, float]] = {}
        self.doc_ids: List[str] = []

    # ── TF functions ──────────────────────────────────────────────────────────

    def _tf(self, count: int, max_count: int, doc_len: int,
            avgdl: float) -> float:
        s = self.extra   # scheme-specific param
        if self.tf_scheme == "sublinear":
            return (1 + math.log(count)) if count > 0 else 0.0
        if self.tf_scheme == "augmented":
            return 0.5 + 0.5 * count / max_count if max_count > 0 else 0.0
        if self.tf_scheme == "loglog":
            return math.log(1 + math.log(1 + count)) if count > 0 else 0.0
        if self.tf_scheme == "saturation":
            k = s if s > 0 else 1
            return count / (count + k)
        if self.tf_scheme == "pivot":
            # Singhal 1996: (1+log(1+log(tf))) / (1-s+s*len/avgdl)
            slope = s if s > 0 else 0.3
            norm  = 1 - slope + slope * doc_len / avgdl
            raw   = (1 + math.log(1 + math.log(count))) if count > 0 else 0.0
            return raw / norm
        return float(count)  # raw TF fallback

    def fit(self, corpus: Dict[str, List[str]]):
        self.doc_ids = list(corpus.keys())
        N            = len(self.doc_ids)
        max_df_abs   = int(self.max_df_frac * N)

        df:     Dict[str, int]              = defaultdict(int)
        raw_tfs: Dict[str, Dict[str, int]]  = {}
        doc_lens: Dict[str, int]            = {}

        for did in self.doc_ids:
            toks = corpus[did]
            tf: Dict[str, int] = defaultdict(int)
            for t in toks: tf[t] += 1
            raw_tfs[did]  = dict(tf)
            doc_lens[did] = len(toks)
            for t in tf: df[t] += 1

        self.vocab = {t for t, c in df.items()
                      if self.min_df <= c <= max_df_abs}
        self.idf   = {t: math.log((N + 1) / (df[t] + 1)) + 1.0
                      for t in self.vocab}

        avgdl = sum(doc_lens.values()) / N if N else 1.0

        for did in self.doc_ids:
            tf      = raw_tfs[did]
            max_tf  = max(tf.values()) if tf else 1
            dl      = doc_lens[did]
            vec: Dict[str, float] = {}
            for t, cnt in tf.items():
                if t not in self.vocab: continue
                vec[t] = self._tf(cnt, max_tf, dl, avgdl) * self.idf[t]
            # L2 normalise
            norm = math.sqrt(sum(v*v for v in vec.values()))
            if norm > 0:
                vec = {t: v/norm for t, v in vec.items()}
            self.vectors[did] = vec

    def _query_vec(self, qtoks: List[str],
                   avgdl: float = 1.0) -> Dict[str, float]:
        tf: Dict[str, int] = defaultdict(int)
        for t in qtoks:
            if t in self.vocab: tf[t] += 1
        max_tf = max(tf.values()) if tf else 1
        dl     = len(qtoks)
        vec    = {t: self._tf(cnt, max_tf, dl, avgdl) * self.idf.get(t, 0.0)
                  for t, cnt in tf.items()}
        norm = math.sqrt(sum(v*v for v in vec.values()))
        if norm > 0:
            vec = {t: v/norm for t, v in vec.items()}
        return vec

    def retrieve(self, qtoks: List[str],
                 top_k: int = 1000) -> List[Tuple[str, float]]:
        qvec = self._query_vec(qtoks)
        if not qvec:
            return []
        scores = [(did, cosine_sim_sparse(qvec, self.vectors[did]))
                  for did in self.doc_ids]
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
    all_results = []

    # Token cache
    tok_cache_c: Dict[Tuple, Dict] = {}
    tok_cache_q: Dict[Tuple, Dict] = {}

    for (tf_scheme, vocab, ngram, min_df, max_df, extra) in CONFIGS:
        name = (f"AdvTFIDF_{tf_scheme}_{vocab}_ng={ngram}_"
                f"mindf={min_df}_maxdf={max_df}_p={extra}")
        print(f"\n{'─'*64}\n  {name}\n{'─'*64}")

        key = (vocab, ngram)
        if key not in tok_cache_c:
            print(f"  Tokenising ({vocab}, ng={ngram}) ...")
            tok_cache_c[key] = {
                cid: tokenise(text, vocab, ngram)
                for cid, text in candidates.items()
            }
            tok_cache_q[key] = {
                qid: tokenise(text, vocab, ngram)
                for qid, text in queries.items()
            }

        idx = AdvancedTFIDF(tf_scheme, min_df, max_df, extra)
        idx.fit(tok_cache_c[key])

        results: Dict[str, List[str]] = {}
        for qid in relevance:
            if qid not in tok_cache_q[key]: continue
            ranked   = idx.retrieve(tok_cache_q[key][qid], top_k=args.top_k)
            results[qid] = [d for d, _ in ranked]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MicroF1@10",
                        title="Advanced TF-IDF — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
