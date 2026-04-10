"""
tfidf_ngram_analysis.py
=======================
TF-IDF Cosine Similarity Prior Case Retrieval — Higher Order N-Grams
Testing n-grams (3, 4, 5, 6, 7, 8) with 4 TF weighting schemes, each tested on 3 document-frequency configurations.

Run
---
    python3 tfidf_ngram_analysis.py --data_dir /path/to/dataset/ --split train
"""

import os
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from utils import (
    load_split, clean_text, evaluate_all,
    save_results, print_results_table, save_results_csv
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "/home/raghavgrover/Desktop/Sem6/INLP/Project_Final/Experiments"
SPLIT    = "test"
TOP_K    = 1000
OUTPUT   = "results/tfidf_ngram_analysis_results.json"
K_VALUES = [5, 6, 7, 8, 9, 10, 11, 15, 20]

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGS
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS: List[Tuple] = []

for n in [3, 4, 5, 6, 7, 8]:
    for scheme in ["raw", "log", "binary", "augmented"]:
        CONFIGS.append((scheme, n, 1, 1.00))
        CONFIGS.append((scheme, n, 2, 1.00))
        CONFIGS.append((scheme, n, 2, 0.95))


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


class TFIDFIndexFast:
    def __init__(self, tf_scheme="log", min_df=1, max_df_frac=1.0):
        self.tf_fn       = _TF_FNS[tf_scheme]
        self.min_df      = min_df
        self.max_df_frac = max_df_frac
        self.vocab: Dict[str, int] = {}
        self.idf_arr: np.ndarray   = np.array([])
        self.cand_matrix = None
        self.doc_ids: List[str]    = []

    def fit(self, doc_ids: List[str], raw_tfs: Dict[str, Dict[str, int]], df: Dict[str, int]):
        self.doc_ids = doc_ids
        N = len(doc_ids)
        max_df_abs = int(self.max_df_frac * N)

        # 1. Build Vocab & IDF
        valid_vocab = [t for t, cnt in df.items() if self.min_df <= cnt <= max_df_abs]
        self.vocab = {t: i for i, t in enumerate(valid_vocab)}
        
        self.idf_arr = np.zeros(len(self.vocab), dtype=np.float32)
        for t, i in self.vocab.items():
            self.idf_arr[i] = math.log((N + 1) / (df[t] + 1)) + 1.0

        # 2. Build Candidate Matrix
        rows, cols, data = [], [], []
        for row_idx, did in enumerate(doc_ids):
            tf = raw_tfs[did]
            if not tf:
                continue
                
            maxt = max(tf.values())
            vec_sq_norm = 0.0
            row_cols = []
            row_data = []
            
            for t, cnt in tf.items():
                col_idx = self.vocab.get(t)
                if col_idx is not None:
                    val = self.tf_fn(cnt, maxt) * self.idf_arr[col_idx]
                    row_cols.append(col_idx)
                    row_data.append(val)
                    vec_sq_norm += val * val
                    
            norm = math.sqrt(vec_sq_norm)
            if norm > 0:
                for c, v in zip(row_cols, row_data):
                    rows.append(row_idx)
                    cols.append(c)
                    data.append(v / norm)

        self.cand_matrix = csr_matrix((data, (rows, cols)), shape=(len(doc_ids), len(self.vocab)), dtype=np.float32)

    def retrieve_batch(self, query_dict: Dict[str, List[str]], top_k: int = 1000) -> Dict[str, List[str]]:
        qids = list(query_dict.keys())
        rows, cols, data = [], [], []
        
        for r, qid in enumerate(qids):
            tokens = query_dict[qid]
            tf = defaultdict(int)
            for t in tokens:
                if t in self.vocab:
                    tf[t] += 1
            if not tf:
                continue
                
            maxt = max(tf.values())
            vec_sq_norm = 0.0
            row_cols = []
            row_data = []
            
            for t, cnt in tf.items():
                col_idx = self.vocab[t]
                val = self.tf_fn(cnt, maxt) * self.idf_arr[col_idx]
                row_cols.append(col_idx)
                row_data.append(val)
                vec_sq_norm += val * val
                
            norm = math.sqrt(vec_sq_norm)
            if norm > 0:
                for c, v in zip(row_cols, row_data):
                    rows.append(r)
                    cols.append(c)
                    data.append(v / norm)
        
        q_matrix = csr_matrix((data, (rows, cols)), shape=(len(qids), len(self.vocab)), dtype=np.float32)
        
        # Batch Cosine Similarity (Both matrices are L2 normalized, so dot product = cosine sim)
        sims = q_matrix.dot(self.cand_matrix.T)
        
        results = {}
        # Convert doc_ids to a numpy array for fast indexing
        doc_ids_arr = np.array(self.doc_ids)
        
        # Iterating over the sparse matrix rows natively is slower if dense is small, 
        # but 1700 candidates is small enough to densify row by row
        for r, qid in enumerate(qids):
            row_sims = sims[r].toarray().flatten()
            k = min(top_k, len(self.doc_ids))
            
            # Fast top-K selection
            if k < len(self.doc_ids):
                top_indices = np.argpartition(row_sims, -k)[-k:]
                # Sort within the top K
                top_indices = top_indices[np.argsort(-row_sims[top_indices])]
            else:
                top_indices = np.argsort(-row_sims)
                
            results[qid] = doc_ids_arr[top_indices].tolist()
            
        return results


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
    
    # Precomputed bases per n-gram
    c_raw_tfs:    Dict[int, Dict[str, Dict[str, int]]] = {}
    c_df:         Dict[int, Dict[str, int]] = {}
    c_doc_ids:    Dict[int, List[str]] = {}

    all_results = []

    for (tf_scheme, ngram, min_df, max_df_frac) in CONFIGS:
        name = (f"TFIDF_{tf_scheme}_ng={ngram}_"
                f"mindf={min_df}_maxdf={max_df_frac}")
        print(f"\n{'─'*60}\n  {name}\n{'─'*60}")

        if ngram not in tok_cache_c:
            print(f"  [Cache] Generating {ngram}-grams and precomputing TFs...")
            tok_cache_c[ngram] = {
                cid: clean_text(t, remove_stopwords=True, ngram=ngram)
                for cid, t in candidates.items()
            }
            tok_cache_q[ngram] = {
                qid: clean_text(t, remove_stopwords=True, ngram=ngram)
                for qid, t in queries.items()
            }
            
            # Precompute document raw TFs and DF once per n-gram
            doc_ids = list(tok_cache_c[ngram].keys())
            c_doc_ids[ngram] = doc_ids
            df = defaultdict(int)
            raw_tfs = {}
            
            for did in doc_ids:
                tf = defaultdict(int)
                for t in tok_cache_c[ngram][did]:
                    tf[t] += 1
                raw_tfs[did] = dict(tf)
                for t in tf:
                    df[t] += 1
            
            c_raw_tfs[ngram] = raw_tfs
            c_df[ngram] = dict(df)

        idx = TFIDFIndexFast(tf_scheme, min_df, max_df_frac)
        idx.fit(c_doc_ids[ngram], c_raw_tfs[ngram], c_df[ngram])

        # Filter queries to only those in the relevance dictionary natively
        valid_queries = {qid: tok_cache_q[ngram][qid] for qid in relevance if qid in tok_cache_q[ngram]}
        
        results = idx.retrieve_batch(valid_queries, top_k=args.top_k)
        
        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MAP",
                        title="TF-IDF N-GRAM ANALYSIS (3 to 8)")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
