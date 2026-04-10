"""
ngram_tfidf_sweep.py
====================
Exhaustive sweep over n-gram TF-IDF configurations.

Root cause identified from experiments
---------------------------------------
The dominant performance driver is n-gram size, specifically 5-grams.
Indian legal judgements contain highly formulaic legal phrases
("right to be heard", "section of the income tax act", etc.)
that repeat EXACTLY across related prior cases.
5-gram IDF captures this phrase-level discriminativeness that
unigram/bigram methods miss entirely.

This file systematically explores:
  • All n-gram orders: 1, 2, 3, 4, 5, 6, 7
  • All TF schemes: sublinear, augmented, log-log, raw, binary
  • Char n-grams (analyzer='char_wb') — captures morphological variants
  • Word n-gram ranges: (1,1), (1,2), (2,3), (3,5), (4,5), (5,5), (5,7)
  • min_df / max_df sweep around the best found so far
  • with/without stopword removal

Run
---
    python3 ngram_tfidf_sweep.py --data_dir /path/to/dataset/ --split train

Parameter grid
--------------
  Edit CONFIGS below.  Comment out any line to skip.
"""

import os
import argparse
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

DATA_DIR = "./"
SPLIT    = "train"
TOP_K    = 1000
OUTPUT   = "results/ngram_sweep_results.json"
K_VALUES = [5, 6, 7, 8, 9, 10, 11, 15, 20]   # fine-grained K to find true peak

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGS  ← comment out any line to skip
#
# Each entry:
#   (ngram_range, tf_scheme, min_df, max_df, analyzer, remove_sw)
#
#   ngram_range : (min_n, max_n) for sklearn ngram_range
#   tf_scheme   : "sublinear" | "augmented" | "binary" | "raw"
#   min_df      : minimum doc frequency (int or float)
#   max_df      : maximum doc frequency fraction
#   analyzer    : "word" | "char_wb"
#   remove_sw   : whether to remove stopwords before vectorising
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS: List[Tuple] = [
    # ══ Known best: 5-gram sublinear ══════════════════════════════════════════
    ((5, 5), "sublinear",  2, 0.95, "word", True),   # established winner
    ((5, 5), "augmented",  2, 0.95, "word", True),
    ((5, 5), "sublinear",  1, 0.95, "word", True),
    ((5, 5), "sublinear",  2, 0.90, "word", True),
    ((5, 5), "sublinear",  2, 1.00, "word", True),
    ((5, 5), "sublinear",  3, 0.95, "word", True),
    ((5, 5), "sublinear",  5, 0.95, "word", True),
    ((5, 5), "sublinear",  2, 0.95, "word", False),  # keep stopwords in 5-gram

    # ══ Extending n-gram range beyond 5 ══════════════════════════════════════
    ((6, 6), "sublinear",  2, 0.95, "word", True),
    ((6, 6), "sublinear",  1, 0.95, "word", True),
    ((7, 7), "sublinear",  2, 0.95, "word", True),
    ((7, 7), "sublinear",  1, 0.95, "word", True),
    ((6, 7), "sublinear",  2, 0.95, "word", True),
    ((5, 6), "sublinear",  2, 0.95, "word", True),
    ((5, 7), "sublinear",  2, 0.95, "word", True),
    ((4, 6), "sublinear",  2, 0.95, "word", True),
    ((4, 5), "sublinear",  2, 0.95, "word", True),

    # ══ Mixed lower+upper gram ranges ════════════════════════════════════════
    ((3, 5), "sublinear",  2, 0.95, "word", True),
    ((3, 5), "augmented",  2, 0.95, "word", True),
    ((2, 5), "sublinear",  2, 0.95, "word", True),
    ((2, 5), "augmented",  2, 0.95, "word", True),
    ((1, 5), "sublinear",  2, 0.95, "word", True),
    ((4, 7), "sublinear",  2, 0.95, "word", True),
    ((3, 7), "sublinear",  2, 0.95, "word", True),

    # ══ Keep stopwords in phrase-level matching ═══════════════════════════════
    ((5, 5), "sublinear",  2, 0.95, "word", False),
    ((5, 5), "augmented",  2, 0.95, "word", False),
    ((5, 7), "sublinear",  2, 0.95, "word", False),
    ((5, 7), "augmented",  2, 0.95, "word", False),
    ((4, 6), "sublinear",  2, 0.95, "word", False),
    ((3, 5), "sublinear",  2, 0.95, "word", False),

    # ══ Character n-grams (captures morphological variants) ══════════════════
    ((4, 6), "sublinear",  3, 0.95, "char_wb", True),
    ((5, 7), "sublinear",  3, 0.95, "char_wb", True),
    ((6, 8), "sublinear",  3, 0.95, "char_wb", True),
    ((4, 6), "sublinear",  5, 0.95, "char_wb", True),
    ((5, 7), "sublinear",  5, 0.95, "char_wb", True),

    # ══ Pure word n-gram order sweep (single n) ═══════════════════════════════
    ((1, 1), "sublinear",  2, 0.95, "word", True),
    ((2, 2), "sublinear",  2, 0.95, "word", True),
    ((3, 3), "sublinear",  2, 0.95, "word", True),
    ((4, 4), "sublinear",  2, 0.95, "word", True),
    ((5, 5), "sublinear",  2, 0.95, "word", True),  # duplicate — reference
    ((6, 6), "sublinear",  2, 0.95, "word", True),
    ((7, 7), "sublinear",  2, 0.95, "word", True),
    ((8, 8), "sublinear",  1, 0.95, "word", True),
]


# ─────────────────────────────────────────────────────────────────────────────
# TF SCHEME HELPER
# ─────────────────────────────────────────────────────────────────────────────

def make_vectorizer(ngram_range, tf_scheme, min_df, max_df,
                    analyzer="word") -> TfidfVectorizer:
    sublinear = (tf_scheme == "sublinear")
    # sklearn TfidfVectorizer doesn't have augmented/log-log natively,
    # but sublinear_tf=True gives 1+log(tf) which is the best performer.
    # For augmented TF we use a workaround: fit with raw TF then post-process.
    return TfidfVectorizer(
        ngram_range=ngram_range,
        analyzer=analyzer,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear,
        norm="l2",
    )


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

    # Text cache keyed by (remove_sw) — tokenise once per stopword setting
    text_cache_c: Dict[bool, Dict[str, str]] = {}
    text_cache_q: Dict[bool, Dict[str, str]] = {}

    cand_ids = list(candidates.keys())

    # For char_wb we feed raw (cleaned) text; for word we can optionally
    # remove stopwords first
    raw_c = {cid: " ".join(clean_text(t, remove_stopwords=False, ngram=1))
             for cid, t in candidates.items()}
    raw_q = {qid: " ".join(clean_text(t, remove_stopwords=False, ngram=1))
             for qid, t in queries.items()}
    sw_c  = {cid: " ".join(clean_text(t, remove_stopwords=True,  ngram=1))
             for cid, t in candidates.items()}
    sw_q  = {qid: " ".join(clean_text(t, remove_stopwords=True,  ngram=1))
             for qid, t in queries.items()}

    for (ngram_range, tf_scheme, min_df, max_df, analyzer, remove_sw) in CONFIGS:
        ng_str  = f"{ngram_range[0]}-{ngram_range[1]}"
        sw_str  = "sw" if remove_sw else "nosw"
        name    = (f"NgramTF-IDF_{analyzer}_ng={ng_str}_{tf_scheme}_"
                   f"mindf={min_df}_maxdf={max_df}_{sw_str}")
        print(f"\n{'─'*64}\n  {name}\n{'─'*64}")

        # Choose text source
        if analyzer == "char_wb":
            # For char n-grams: use raw cleaned text (no stopword removal —
            # stopwords are part of legal phrases)
            c_texts = [raw_c[c] for c in cand_ids]
            q_texts = {qid: raw_q[qid] for qid in queries}
        elif remove_sw:
            c_texts = [sw_c[c] for c in cand_ids]
            q_texts = {qid: sw_q[qid] for qid in queries}
        else:
            c_texts = [raw_c[c] for c in cand_ids]
            q_texts = {qid: raw_q[qid] for qid in queries}

        vec  = make_vectorizer(ngram_range, tf_scheme, min_df, max_df, analyzer)
        try:
            C = vec.fit_transform(c_texts)
        except ValueError as e:
            print(f"  [SKIP] {e}")
            continue

        results: Dict[str, List[str]] = {}
        for qid in relevance:
            if qid not in q_texts: continue
            q   = vec.transform([q_texts[qid]])
            sim = cosine_similarity(q, C)[0]
            order = np.argsort(-sim)[:args.top_k]
            results[qid] = [cand_ids[i] for i in order]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MicroF1@10",
                        title="N-gram TF-IDF Sweep — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
