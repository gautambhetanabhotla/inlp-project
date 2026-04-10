"""
rr_filtered_tfidf.py
====================
TF-IDF Prior Case Retrieval — Rhetorical Role Filtering Analysis
Testing all 127 possible combinations of the 7 Rhetorical Roles using the
top-3 TF-IDF configurations identified in previous experiments.

Run
---
    python3 rr_filtered_tfidf.py
"""

import os
import math
import glob
import json
import argparse
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from utils import (
    clean_text, evaluate_all,
    save_results, print_results_table, save_results_csv
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "/home/raghavgrover/Desktop/Sem6/INLP/Project_Final/Experiments"
SPLIT    = "test_rr"
TOP_K    = 1000
OUTPUT   = "results/rr_filtered_tfidf_results.json"
K_VALUES = [5, 6, 7, 8, 9, 10, 11]

ALL_ROLES = [
    "Argument",
    "Fact",
    "Precedent",
    "RatioOfTheDecision",
    "RulingByLowerCourt",
    "RulingByPresentCourt",
    "Statute"
]

# Top 3 Configs from regular TF-IDF fast sweep
CONFIGS = [
    ("augmented", 3, 2, 1.00),
    ("augmented", 3, 2, 0.95),
    ("binary",    3, 2, 1.00),
]


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF INDEX (Fast formulation)
# ─────────────────────────────────────────────────────────────────────────────

def _tf_binary(count: int, max_tf: int) -> float:
    return 1.0 if count > 0 else 0.0

def _tf_augmented(count: int, max_tf: int) -> float:
    return 0.5 + 0.5 * count / max_tf if max_tf > 0 else 0.0

_TF_FNS = {
    "binary":    _tf_binary,
    "augmented": _tf_augmented,
}


class TFIDFIndexFast:
    def __init__(self, tf_scheme="augmented", min_df=2, max_df_frac=1.0):
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
            tf = raw_tfs.get(did, {})
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

    def retrieve_batch(self, query_tfs: Dict[str, Dict[str, int]]) -> Dict[str, List[str]]:
        qids = list(query_tfs.keys())
        rows, cols, data = [], [], []
        
        for r, qid in enumerate(qids):
            tf = query_tfs[qid]
            if not tf:
                continue
                
            maxt = max(tf.values())
            vec_sq_norm = 0.0
            row_cols = []
            row_data = []
            
            for t, cnt in tf.items():
                if t in self.vocab:
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
        
        # Batch Cosine Similarity
        sims = q_matrix.dot(self.cand_matrix.T)
        
        results = {}
        doc_ids_arr = np.array(self.doc_ids)
        
        for r, qid in enumerate(qids):
            row_sims = sims[r].toarray().flatten()
            top_indices = np.argsort(-row_sims)
            results[qid] = doc_ids_arr[top_indices].tolist()
            
        return results

# ─────────────────────────────────────────────────────────────────────────────
# RR DATA LOADER & PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

def load_and_tokenize_rr_split(base_dir: str, split: str, ngram: int = 3):
    """
    Parses <Role>\\t<Text> documents and pre-tokenizes them per Rhetorical Role.
    Returns:
    - cand_roles: {doc_id: {role: [filtered_ngrams]} }
    - query_roles: {doc_id: {role: [filtered_ngrams]} }
    - relevance: {query_id: [cand_ids]}
    """
    split_dir = os.path.join(base_dir, f"ik_{split}")
    query_dir = os.path.join(split_dir, "query")
    cand_dir  = os.path.join(split_dir, "candidate")
    
    # Load relevance
    relevance = {}
    json_path = os.path.join(split_dir, f"test.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            raw_rel = json.load(f)
            if "Query Set" in raw_rel:
                for item in raw_rel["Query Set"]:
                    qid = item["id"].replace(".txt", "")
                    cands = [c.replace(".txt", "") for c in item.get("relevant candidates", [])]
                    relevance[qid] = cands
            else:
                relevance = raw_rel
            
    def _parse_dir(dpath: str) -> Dict[str, Dict[str, List[str]]]:
        docs = {}
        if not os.path.exists(dpath):
            return docs
            
        for fname in tqdm(os.listdir(dpath), desc=f"Loading {dpath}"):
            if not fname.endswith(".txt"): continue
            did = fname.replace(".txt", "")
            with open(os.path.join(dpath, fname), "r", encoding="utf-8", errors="replace") as f:
                role_texts = defaultdict(list)
                for line in f:
                    parts = line.strip().split("\t", 1)
                    if len(parts) == 2:
                        role, text = parts
                        if role in ALL_ROLES:
                            role_texts[role].append(text)
                
                # Pre-tokenize
                docs[did] = {}
                for role, sents in role_texts.items():
                    combined_text = " ".join(sents)
                    tokens = clean_text(combined_text, remove_stopwords=True, ngram=ngram)
                    if len(tokens) > 0:
                        docs[did][role] = tokens
        return docs

    print("Parsing candidates...")
    cand_roles = _parse_dir(cand_dir)
    print("Parsing queries...")
    query_roles = _parse_dir(query_dir)
    
    return query_roles, cand_roles, relevance

def merge_tokens_for_roles(docs_roles: Dict[str, Dict[str, List[str]]], active_roles: Tuple[str]) -> Dict[str, List[str]]:
    """Combines pre-tokenized arrays for the active roles only."""
    merged = {}
    valid_roles_set = set(active_roles)
    for did, roles_dict in docs_roles.items():
        combined = []
        for r, toks in roles_dict.items():
            if r in valid_roles_set:
                combined.extend(toks)
        if combined:
            merged[did] = combined
    return merged
    

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

    # Pre-tokenize data once as 3-grams
    q_roles, c_roles, relevance = load_and_tokenize_rr_split(args.data_dir, args.split, ngram=3)
    
    # Generate all 127 role combinations
    all_combos = []
    for r in range(1, len(ALL_ROLES) + 1):
        for combo in combinations(ALL_ROLES, r):
            all_combos.append(combo)
            
    print(f"Generated {len(all_combos)} rhetorical role combinations.")

    all_results = []
    
    for combo in tqdm(all_combos, desc="Evaluating Combinations"):
        # Create dataset for this combination
        combo_name = "-".join([c[:3] for c in combo]) # concise readable name
        
        # Merge tokens
        c_toks = merge_tokens_for_roles(c_roles, combo)
        q_toks = merge_tokens_for_roles(q_roles, combo)
        
        # Build document DF and raw TFs
        c_doc_ids = list(c_toks.keys())
        df = defaultdict(int)
        c_raw_tfs = {}
        for did in c_doc_ids:
            tf = defaultdict(int)
            for t in c_toks[did]:
                tf[t] += 1
            c_raw_tfs[did] = dict(tf)
            for t in tf.keys():
                df[t] += 1
                
        # Build query raw TFs
        q_raw_tfs = {}
        for qid in q_toks:
            if qid not in relevance: continue
            tf = defaultdict(int)
            for t in q_toks[qid]:
                tf[t] += 1
            q_raw_tfs[qid] = dict(tf)
            
        for (tf_scheme, ngram, min_df, max_df_frac) in CONFIGS:
            idx = TFIDFIndexFast(tf_scheme, min_df, max_df_frac)
            idx.fit(c_doc_ids, c_raw_tfs, dict(df))
            
            results = idx.retrieve_batch(q_raw_tfs)
            
            # Label strictly with the combo
            full_combo_str = "[" + "-".join(combo) + "]"
            model_label = f"TFIDF_{tf_scheme}_ng=3_mindf={min_df}_maxdf={max_df_frac}_{full_combo_str}"
            
            m = evaluate_all(results, relevance, k_values=K_VALUES, label=model_label)
            all_results.append(m)

    save_results(all_results, args.output)
    
    # Group and save top 5 results to a TXT file
    import sys
    
    top_5_path = args.output.replace(".json", "_top5.txt")
    
    # Still print to terminal for user visibility
    print("\n\n=== Top Role Combinations (Augmented, maxdf=1.0) ===")
    subset_results1 = [r for r in all_results if "augmented" in r['model'] and "maxdf=1.0_" in r['model']]
    subset_results1.sort(key=lambda x: x.get("MAP", 0), reverse=True)
    print_results_table(subset_results1[:5], sort_by="MAP", title="Top Augmented 1.0 Configurations")

    print("\n\n=== Top Role Combinations (Augmented, maxdf=0.95) ===")
    subset_results2 = [r for r in all_results if "augmented" in r['model'] and "maxdf=0.95_" in r['model']]
    subset_results2.sort(key=lambda x: x.get("MAP", 0), reverse=True)
    print_results_table(subset_results2[:5], sort_by="MAP", title="Top Augmented 0.95 Configurations")

    print("\n\n=== Top Role Combinations (Binary, maxdf=1.0) ===")
    subset_results3 = [r for r in all_results if "binary" in r['model'] and "maxdf=1.0_" in r['model']]
    subset_results3.sort(key=lambda x: x.get("MAP", 0), reverse=True)
    print_results_table(subset_results3[:5], sort_by="MAP", title="Top Binary 1.0 Configurations")
    
    # Save to file
    with open(top_5_path, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        try:
            print("=== Top Role Combinations (Augmented, maxdf=1.0) ===")
            print_results_table(subset_results1[:5], sort_by="MAP", title="Top Augmented 1.0 Configurations")
            print("\n\n=== Top Role Combinations (Augmented, maxdf=0.95) ===")
            print_results_table(subset_results2[:5], sort_by="MAP", title="Top Augmented 0.95 Configurations")
            print("\n\n=== Top Role Combinations (Binary, maxdf=1.0) ===")
            print_results_table(subset_results3[:5], sort_by="MAP", title="Top Binary 1.0 Configurations")
        finally:
            sys.stdout = original_stdout

    save_results_csv(all_results, args.output.replace(".json", ".csv"))
    print(f"\nAll results saved to {args.output}")
    print(f"Top 5 text exported to {top_5_path}")

if __name__ == "__main__":
    main()
