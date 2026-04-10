"""
ngram_rrf_fusion.py
====================
Reciprocal Rank Fusion of multiple 5-gram TF-IDF variants.

Motivation from experiments
----------------------------
  • 5-gram TF-IDF (sublinear, mindf=2, maxdf=0.95) = MicroF1@10 = 0.360
  • The pivot slope does not help (all slopes give identical results)
  • Multi-field (nouns + full) with ng5 gets to 0.3127@10

The next logical step: fuse MULTIPLE 5-gram TF-IDF systems that differ
in their vocabulary / text preprocessing, using Reciprocal Rank Fusion.

RRF score: RRF(d) = Σ_r  1 / (k + rank_r(d))   k=60

Systems fused (all on full 2000-candidate pool):
  A. TF-IDF 5-gram, sublinear TF,   stopwords removed
  B. TF-IDF 5-gram, augmented TF,   stopwords removed
  C. TF-IDF 5-gram, sublinear TF,   stopwords KEPT (catches legal phrases
                                     that contain "of", "the", "in", etc.)
  D. TF-IDF 6-gram, sublinear TF,   stopwords removed
  E. TF-IDF (4,6)-gram, sublinear,  stopwords removed
  F. TF-IDF 5-gram, sublinear,      nouns only (Method-1 vocabulary)
  G. TF-IDF 5-gram, sublinear,      nouns+verbs (Method-2 vocabulary)
  H. BM25+ 5-gram,                   stopwords removed  (if BM25 results available)

Run
---
    python3 ngram_rrf_fusion.py --data_dir /path/to/dataset/ --split train

Parameter grid
--------------
  Edit FUSION_GROUPS below.  Comment out any group to skip.
"""

import os
import re
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
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
OUTPUT   = "results/ngram_rrf_results.json"
K_VALUES = [5, 6, 7, 8, 9, 10, 11, 15, 20]

RRF_K_VALUES = [10, 30, 60, 120]   # test multiple RRF k constants

# ─────────────────────────────────────────────────────────────────────────────
# FUSION_GROUPS  ← comment out any group to skip
#
# Each entry: (group_name, [list of system_keys])
# Available system keys: A, B, C, D, E, F, G
#   (see SYSTEM_DEFS below for what each letter means)
# ─────────────────────────────────────────────────────────────────────────────

FUSION_GROUPS: List[Tuple] = [
    # ── 2-way fusions ─────────────────────────────────────────────────────────
    ("RRF_A+B",          ["A", "B"]),    # sub + augmented
    ("RRF_A+C",          ["A", "C"]),    # sw-removed + sw-kept
    ("RRF_A+D",          ["A", "D"]),    # 5g + 6g
    ("RRF_A+E",          ["A", "E"]),    # 5g + (4,6)g
    ("RRF_A+F",          ["A", "F"]),    # full 5g + nouns 5g
    ("RRF_A+G",          ["A", "G"]),    # full 5g + nouns_verbs 5g
    ("RRF_C+F",          ["C", "F"]),    # sw-kept 5g + nouns 5g
    ("RRF_D+E",          ["D", "E"]),    # 6g + (4,6)g
    ("RRF_F+G",          ["F", "G"]),    # nouns 5g + nv 5g

    # ── 3-way fusions ─────────────────────────────────────────────────────────
    ("RRF_A+B+C",        ["A", "B", "C"]),
    ("RRF_A+C+F",        ["A", "C", "F"]),
    ("RRF_A+C+G",        ["A", "C", "G"]),
    ("RRF_A+D+F",        ["A", "D", "F"]),
    ("RRF_A+E+F",        ["A", "E", "F"]),
    ("RRF_A+B+F",        ["A", "B", "F"]),
    ("RRF_A+F+G",        ["A", "F", "G"]),
    ("RRF_C+F+G",        ["C", "F", "G"]),
    ("RRF_A+D+E",        ["A", "D", "E"]),

    # ── 4-way fusions ─────────────────────────────────────────────────────────
    ("RRF_A+B+C+F",      ["A", "B", "C", "F"]),
    ("RRF_A+B+C+G",      ["A", "B", "C", "G"]),
    ("RRF_A+C+F+G",      ["A", "C", "F", "G"]),
    ("RRF_A+D+E+F",      ["A", "D", "E", "F"]),
    ("RRF_A+B+F+G",      ["A", "B", "F", "G"]),

    # ── 5-way and 7-way fusions ────────────────────────────────────────────────
    ("RRF_A+B+C+F+G",    ["A", "B", "C", "F", "G"]),
    ("RRF_A+B+C+D+F+G",  ["A", "B", "C", "D", "F", "G"]),
    ("RRF_ALL",           ["A", "B", "C", "D", "E", "F", "G"]),
]

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

# These are built at runtime
# A: TF-IDF (5,5) sublinear  stopwords removed
# B: TF-IDF (5,5) augmented  stopwords removed
# C: TF-IDF (5,5) sublinear  stopwords KEPT
# D: TF-IDF (6,6) sublinear  stopwords removed
# E: TF-IDF (4,6) sublinear  stopwords removed
# F: TF-IDF (5,5) sublinear  NOUNS ONLY
# G: TF-IDF (5,5) sublinear  NOUNS + VERBS

SYSTEM_DEFS = {
    "A": dict(ngram=(5,5), sublinear=True,  sw=True,  vocab="full"),
    "B": dict(ngram=(5,5), sublinear=False, sw=True,  vocab="full"),  # augmented
    "C": dict(ngram=(5,5), sublinear=True,  sw=False, vocab="full"),
    "D": dict(ngram=(6,6), sublinear=True,  sw=True,  vocab="full"),
    "E": dict(ngram=(4,6), sublinear=True,  sw=True,  vocab="full"),
    "F": dict(ngram=(5,5), sublinear=True,  sw=True,  vocab="nouns"),
    "G": dict(ngram=(5,5), sublinear=True,  sw=True,  vocab="nouns_verbs"),
}

# ─────────────────────────────────────────────────────────────────────────────
# POS HELPERS (for F and G)
# ─────────────────────────────────────────────────────────────────────────────

NOUN_TAGS = {"NN", "NNS", "NNP"}
VERB_TAGS = {"VB", "VBZ", "VBN", "VBD"}
_lem  = WordNetLemmatizer()
_CITE = re.compile(r'\[?\?CITATION\?\]?|<CITATION_\d+>', re.IGNORECASE)
_PUN  = re.compile(r'[",\-\'_]')

def _prep(t): return _PUN.sub(" ", _CITE.sub(" ", t))

def _get_tokens(text, vocab, sw):
    if vocab == "nouns":
        toks = word_tokenize(_prep(text))
        base = [_lem.lemmatize(w.lower(),'n') for w,t in pos_tag(toks)
                if t in NOUN_TAGS]
    elif vocab == "nouns_verbs":
        toks = word_tokenize(_prep(text)); base=[]
        for w,t in pos_tag(toks):
            if t in NOUN_TAGS: base.append(_lem.lemmatize(w.lower(),'n'))
            elif t in VERB_TAGS: base.append(_lem.lemmatize(w.lower(),'v'))
    else:
        base = clean_text(text, remove_stopwords=sw, ngram=1)
    return " ".join(base)


# ─────────────────────────────────────────────────────────────────────────────
# RRF
# ─────────────────────────────────────────────────────────────────────────────

def rrf_fuse(ranked_lists: List[Dict[str, List[str]]],
             qids: List[str],
             k: int = 60,
             top_k: int = 1000) -> Dict[str, List[str]]:
    out = {}
    for qid in qids:
        scores: Dict[str, float] = defaultdict(float)
        for rl in ranked_lists:
            for rank, cid in enumerate(rl.get(qid, [])):
                scores[cid] += 1.0 / (k + rank + 1)
        out[qid] = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    return out


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
    cand_ids  = list(candidates.keys())
    qids      = [qid for qid in queries if qid in relevance]
    all_results = []

    # ── Build all base ranked lists ───────────────────────────────────────────
    print("\nBuilding base retrieval systems ...")
    base_ranked: Dict[str, Dict[str, List[str]]] = {}
    text_cache:  Dict[Tuple, Dict[str, str]] = {}   # (vocab, sw) → {id: str}

    for sys_key, cfg in SYSTEM_DEFS.items():
        vocab  = cfg["vocab"]
        sw     = cfg["sw"]
        ngram  = cfg["ngram"]
        sub    = cfg["sublinear"]
        print(f"  System {sys_key}: ng={ngram}, sub={sub}, vocab={vocab}, sw={sw}")

        # Prepare text (cache by vocab+sw since POS is expensive)
        cache_key = (vocab, sw)
        if cache_key not in text_cache:
            print(f"    Preparing text (vocab={vocab}, sw={sw}) ...")
            text_cache[cache_key] = {
                did: _get_tokens(text, vocab, sw)
                for did, text in {**candidates, **queries}.items()
            }

        tc = text_cache[cache_key]
        c_texts = [tc[c] for c in cand_ids]
        q_texts = {qid: tc[qid] for qid in queries}

        # Fit TF-IDF
        vec = TfidfVectorizer(
            ngram_range=ngram,
            analyzer="word",
            min_df=2,
            max_df=0.95,
            sublinear_tf=sub,
            norm="l2",
        )
        try:
            C = vec.fit_transform(c_texts)
        except ValueError as e:
            print(f"    [SKIP] {e}")
            continue

        ranked: Dict[str, List[str]] = {}
        for qid in qids:
            q   = vec.transform([q_texts[qid]])
            sim = cosine_similarity(q, C)[0]
            order = np.argsort(-sim)[:args.top_k]
            ranked[qid] = [cand_ids[i] for i in order]

        base_ranked[sys_key] = ranked

        # Evaluate individual system
        m = evaluate_all(ranked, relevance, k_values=K_VALUES,
                         label=f"Base_{sys_key}", verbose=False)
        m["model"] = f"Base_{sys_key}_{vocab}_ng{ngram[0]}-{ngram[1]}_sub{sub}_sw{sw}"
        all_results.append(m)
        print(f"    MAP={m['MAP']:.4f}  MicroF1@10={m.get('MicroF1@10',0):.4f}")

    # ── RRF fusions ───────────────────────────────────────────────────────────
    for (group_name, sys_keys) in FUSION_GROUPS:
        valid = [base_ranked[k] for k in sys_keys if k in base_ranked]
        if not valid:
            print(f"\n  [SKIP] {group_name} — no valid systems")
            continue

        print(f"\n  {group_name}  ({len(valid)} systems)")

        for rrf_k in RRF_K_VALUES:
            name  = f"{group_name}_k={rrf_k}"
            fused = rrf_fuse(valid, qids, k=rrf_k, top_k=args.top_k)
            m     = evaluate_all(fused, relevance, k_values=K_VALUES,
                                 label=name, verbose=False)
            m["model"]  = name
            m["rrf_k"]  = rrf_k
            m["systems"] = "+".join(sys_keys)
            all_results.append(m)
            print(f"    k={rrf_k}: MAP={m['MAP']:.4f}  "
                  f"MicroF1@7={m.get('MicroF1@7',0):.4f}  "
                  f"MicroF1@10={m.get('MicroF1@10',0):.4f}")

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MicroF1@10",
                        title="N-gram RRF Fusion — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
