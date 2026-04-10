"""
citation_network_retrieval.py
==============================
Citation-network-based Prior Case Retrieval.
Extracts <CITATION_XXXXXXX> tags from the raw legal text.

Run
---
    python3 citation_network_retrieval.py --data_dir /path/to/dataset/ --split train

Pure citation measures
----------------------
  BC         — Bibliographic Coupling   |A∩B| / √(|A|·|B|)
  Jaccard    — |A∩B| / |A∪B|
  Dice       — 2|A∩B| / (|A|+|B|)
  IDFCosine  — IDF-weighted citation cosine
  Cocitation — Co-citation count (docs citing both A and B)

Hybrid
------
  Citation measure + BM25, α-weighted Z-normalised combination.

Parameter grid
--------------
  Edit the PURE_CONFIGS and HYBRID_CONFIGS lists.
  Comment out any line to skip that config.
"""

import os
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

from utils import (
    load_split, clean_text, extract_citations, evaluate_all,
    save_results, print_results_table, save_results_csv,
    compute_idf, z_norm,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "/home/raghavgrover/Desktop/Sem6/INLP/Project_Final/Experiments"
SPLIT    = "test"
TOP_K    = 1000
OUTPUT   = "results/citation_results.json"
K_VALUES = [5, 6, 7, 8, 9, 10, 11, 15, 20]

# ─────────────────────────────────────────────────────────────────────────────
# PURE CITATION CONFIGS  ← comment out any line to skip
# Each entry: (method,)
#   method : "BC" | "Jaccard" | "Dice" | "IDFCosine" | "Cocitation"
# ─────────────────────────────────────────────────────────────────────────────

PURE_CONFIGS: List[Tuple] = [
    ("BC",),
    ("Jaccard",),
    ("Dice",),
    ("IDFCosine",),
    ("Cocitation",),
]

# ─────────────────────────────────────────────────────────────────────────────
# HYBRID CONFIGS  ← comment out any line to skip
# Each entry: (cite_method, bm25_ngram, alpha)
#   cite_method : "BC" | "Jaccard" | "Dice" | "IDFCosine"
#   bm25_ngram  : n-gram for BM25 lexical component
#   alpha       : weight of citation score (1-alpha = BM25 weight)
# ─────────────────────────────────────────────────────────────────────────────

HYBRID_CONFIGS: List[Tuple] = [
    # ── BC + BM25 ─────────────────────────────────────────────────────────────
    ("BC", 1, 0.1),
    ("BC", 1, 0.3),
    ("BC", 1, 0.5),
    ("BC", 1, 0.7),
    ("BC", 1, 0.9),
    ("BC", 5, 0.3),
    ("BC", 5, 0.5),
    ("BC", 5, 0.7),

    # ── Jaccard + BM25 ────────────────────────────────────────────────────────
    ("Jaccard", 1, 0.3),
    ("Jaccard", 1, 0.5),
    ("Jaccard", 1, 0.7),
    ("Jaccard", 5, 0.5),

    # ── Dice + BM25 ───────────────────────────────────────────────────────────
    ("Dice", 1, 0.3),
    ("Dice", 1, 0.5),
    ("Dice", 5, 0.5),

    # ── IDFCosine + BM25 ──────────────────────────────────────────────────────
    ("IDFCosine", 1, 0.3),
    ("IDFCosine", 1, 0.5),
    ("IDFCosine", 1, 0.7),
    ("IDFCosine", 5, 0.3),
    ("IDFCosine", 5, 0.5),
    ("IDFCosine", 5, 0.7),
]


# ─────────────────────────────────────────────────────────────────────────────
# SIMILARITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def bc(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))

def jaccard(a: Set[str], b: Set[str]) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0

def dice(a: Set[str], b: Set[str]) -> float:
    d = len(a) + len(b)
    return 2 * len(a & b) / d if d else 0.0

def idf_cosine(a: Set[str], b: Set[str], idf: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot  = sum(idf.get(c, 1.0) ** 2 for c in a & b)
    na   = math.sqrt(sum(idf.get(c, 1.0) ** 2 for c in a))
    nb   = math.sqrt(sum(idf.get(c, 1.0) ** 2 for c in b))
    return dot / (na * nb) if na * nb else 0.0

def cocitation_score(a: Set[str], b: Set[str],
                     cite_to_docs: Dict[str, Set[str]]) -> float:
    """Number of docs that cite at least one ref from both a and b."""
    docs_a = set().union(*(cite_to_docs.get(c, set()) for c in a))
    docs_b = set().union(*(cite_to_docs.get(c, set()) for c in b))
    return float(len(docs_a & docs_b))

_SIM_FNS = {
    "BC":        bc,
    "Jaccard":   jaccard,
    "Dice":      dice,
    "IDFCosine": None,   # needs idf extra arg
    "Cocitation": None,  # needs cite_to_docs extra arg
}


# ─────────────────────────────────────────────────────────────────────────────
# BM25 (minimal inline)
# ─────────────────────────────────────────────────────────────────────────────

class _MiniBM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1; self.b = b

    def fit(self, corpus: Dict[str, List[str]]):
        self.ids  = list(corpus.keys()); self.N = len(self.ids)
        self.df   = defaultdict(int)
        self.tfs  = []
        self.lens = []
        for did in self.ids:
            tf = defaultdict(int)
            for t in corpus[did]: tf[t] += 1
            self.tfs.append(dict(tf)); self.lens.append(len(corpus[did]))
            for t in tf: self.df[t] += 1
        self.avgdl = sum(self.lens) / self.N if self.N else 1.0

    def _idf(self, t):
        d = self.df.get(t, 0)
        return math.log((self.N - d + 0.5) / (d + 0.5) + 1)

    def scores_dict(self, qtoks: List[str]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for i, did in enumerate(self.ids):
            sc = 0.0
            for t in set(qtoks):
                f = self.tfs[i].get(t, 0)
                if f == 0: continue
                sc += self._idf(t) * f * (self.k1+1) / (
                    f + self.k1*(1-self.b+self.b*self.lens[i]/self.avgdl))
            out[did] = sc
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
    cand_ids = list(candidates.keys())

    # ── Extract citations ────────────────────────────────────────────────────
    print("\nExtracting citations ...")
    q_cites: Dict[str, Set[str]] = {
        qid: extract_citations(t) for qid, t in queries.items()
    }
    c_cites: Dict[str, Set[str]] = {
        cid: extract_citations(t) for cid, t in candidates.items()
    }

    n_q = sum(1 for c in q_cites.values() if c)
    n_c = sum(1 for c in c_cites.values() if c)
    print(f"  Queries with citations   : {n_q}/{len(q_cites)}")
    print(f"  Candidates with citations: {n_c}/{len(c_cites)}")

    # IDF over citation IDs
    cite_df: Dict[str, int] = defaultdict(int)
    for cset in c_cites.values():
        for c in cset: cite_df[c] += 1
    cite_idf: Dict[str, float] = {
        c: math.log((len(c_cites)+1) / (df+1)) + 1.0
        for c, df in cite_df.items()
    }

    # Co-citation inverted index
    cite_to_docs: Dict[str, Set[str]] = defaultdict(set)
    for cid, cset in c_cites.items():
        for c in cset: cite_to_docs[c].add(cid)

    # ── Pre-compute BM25 models for each unique ngram in HYBRID_CONFIGS ──────
    bm25_ngrams = set(ng for (_, ng, _) in HYBRID_CONFIGS)
    bm25_models: Dict[int, _MiniBM25] = {}
    bm25_tok_q:  Dict[int, Dict]      = {}
    bm25_tok_c:  Dict[int, Dict]      = {}

    for ng in bm25_ngrams:
        print(f"  Building BM25 index (ng={ng}) ...")
        bm25_tok_c[ng] = {cid: clean_text(t, True, ng) for cid, t in candidates.items()}
        bm25_tok_q[ng] = {qid: clean_text(t, True, ng) for qid, t in queries.items()}
        m = _MiniBM25(k1=1.5, b=0.75)
        m.fit(bm25_tok_c[ng])
        bm25_models[ng] = m

    all_results = []

    # ── PURE CITATION MEASURES ───────────────────────────────────────────────
    for (method,) in PURE_CONFIGS:
        name = f"Citation_{method}"
        print(f"\n{'─'*60}\n  {name}\n{'─'*60}")

        results: Dict[str, List[str]] = {}
        for qid in relevance:
            qc = q_cites.get(qid, set())
            sc: Dict[str, float] = {}
            for cid in cand_ids:
                cc = c_cites.get(cid, set())
                if method == "IDFCosine":
                    sc[cid] = idf_cosine(qc, cc, cite_idf)
                elif method == "Cocitation":
                    sc[cid] = cocitation_score(qc, cc, cite_to_docs)
                else:
                    sc[cid] = _SIM_FNS[method](qc, cc)
            ranked = sorted(cand_ids, key=lambda c: sc.get(c, 0.0), reverse=True)
            results[qid] = ranked[: args.top_k]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    # ── HYBRID: CITATION + BM25 ──────────────────────────────────────────────
    for (cite_method, bm25_ng, alpha) in HYBRID_CONFIGS:
        name = f"Hybrid_{cite_method}_BM25ng={bm25_ng}_a={alpha}"
        print(f"\n{'─'*60}\n  {name}\n{'─'*60}")

        bm25_m = bm25_models[bm25_ng]
        bm25_q = bm25_tok_q[bm25_ng]

        results: Dict[str, List[str]] = {}
        for qid in relevance:
            qc    = q_cites.get(qid, set())
            qtoks = bm25_q.get(qid, [])

            cite_raw: Dict[str, float] = {}
            for cid in cand_ids:
                cc = c_cites.get(cid, set())
                if cite_method == "IDFCosine":
                    cite_raw[cid] = idf_cosine(qc, cc, cite_idf)
                else:
                    cite_raw[cid] = _SIM_FNS[cite_method](qc, cc)

            bm25_raw = bm25_m.scores_dict(qtoks)
            cite_z   = z_norm(cite_raw)
            bm25_z   = z_norm(bm25_raw)

            combined = {
                cid: alpha * cite_z.get(cid, 0.0) +
                     (1 - alpha) * bm25_z.get(cid, 0.0)
                for cid in cand_ids
            }
            ranked = sorted(cand_ids, key=lambda c: combined[c], reverse=True)
            results[qid] = ranked[: args.top_k]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MAP",
                        title="Citation Network — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
