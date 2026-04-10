"""
rr_late_fusion_sweep.py
=======================
Exhaustive Late Fusion Sweep over Top 3 TF-IDF Configs and targeted weight grids.

Run
---
    python3 rr_late_fusion_sweep.py
"""

import os
import math
import json
import argparse
from collections import defaultdict
from typing import Dict, List

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from utils import (
    clean_text, evaluate_all,
    save_results, print_results_table, save_results_csv
)

# ── GLOBAL SETTINGS ──────────────────────────────────────────────────────────

DATA_DIR = "/home/raghavgrover/Desktop/Sem6/INLP/Project_Final/Experiments"
SPLIT    = "test_rr"
OUTPUT   = "results/rr_late_fusion_sweep_results.json"
K_VALUES = [5, 6, 7, 8, 9, 10]

ALL_ROLES = [
    "Argument", "Fact", "Precedent", "RatioOfTheDecision",
    "RulingByLowerCourt", "RulingByPresentCourt", "Statute"
]

# Top 3 TF-IDF Configs
CONFIGS = [
    {"name": "Aug_1.0",  "scheme": "augmented", "ngram": 3, "min_df": 2, "max_df_frac": 1.00},
    {"name": "Aug_0.95", "scheme": "augmented", "ngram": 3, "min_df": 2, "max_df_frac": 0.95},
    {"name": "Bin_1.0",  "scheme": "binary",    "ngram": 3, "min_df": 2, "max_df_frac": 1.00},
]

# Generate Focused Weight Grids based on previous results
# We know Fact, Ratio, and Precedent are king.
WEIGHT_GRIDS = []

# Base combinations of the top 3 roles
for f_wt in [1.0, 2.0, 3.0, 4.0]:
    for r_wt in [2.0, 3.0, 4.0]:
        for p_wt in [1.0, 2.0, 3.0]:
            # Strict subset (only top 3)
            WEIGHT_GRIDS.append({
                "Fact": f_wt, "RatioOfTheDecision": r_wt, "Precedent": p_wt,
                "Argument": 0.0, "Statute": 0.0, "RulingByPresentCourt": 0.0, "RulingByLowerCourt": 0.0
            })
            # Top 3 + slight context from Argument/Statute
            WEIGHT_GRIDS.append({
                "Fact": f_wt, "RatioOfTheDecision": r_wt, "Precedent": p_wt,
                "Argument": 0.5, "Statute": 0.5, "RulingByPresentCourt": 0.0, "RulingByLowerCourt": 0.0
            })

# ── TF-IDF INDEX ─────────────────────────────────────────────────────────────

def _tf_binary(count: int, max_tf: int) -> float:
    return 1.0 if count > 0 else 0.0

def _tf_augmented(count: int, max_tf: int) -> float:
    return 0.5 + 0.5 * count / max_tf if max_tf > 0 else 0.0

_TF_FNS = {"binary": _tf_binary, "augmented": _tf_augmented}

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
        if N == 0: return
            
        max_df_abs = int(self.max_df_frac * N)
        valid_vocab = [t for t, cnt in df.items() if self.min_df <= cnt <= max_df_abs]
        self.vocab = {t: i for i, t in enumerate(valid_vocab)}
        
        self.idf_arr = np.zeros(len(self.vocab), dtype=np.float32)
        for t, i in self.vocab.items():
            self.idf_arr[i] = math.log((N + 1) / (df[t] + 1)) + 1.0

        rows, cols, data = [], [], []
        for row_idx, did in enumerate(doc_ids):
            tf = raw_tfs.get(did, {})
            if not tf: continue
                
            maxt = max(tf.values())
            vec_sq_norm = 0.0
            row_cols, row_data = [], []
            
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

    def get_score_matrix(self, qids: List[str], query_tfs: Dict[str, Dict[str, int]]) -> np.ndarray:
        if self.cand_matrix is None or len(self.vocab) == 0:
            return np.zeros((len(qids), len(self.doc_ids)), dtype=np.float32)

        rows, cols, data = [], [], []
        for r, qid in enumerate(qids):
            tf = query_tfs.get(qid, {})
            if not tf: continue
                
            maxt = max(tf.values())
            vec_sq_norm = 0.0
            row_cols, row_data = [], []
            
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
        return q_matrix.dot(self.cand_matrix.T).toarray()

# ── LATE FUSION RETRIEVER ────────────────────────────────────────────────────

class LateFusionRetriever:
    def __init__(self, roles: List[str], config: dict):
        self.roles = roles
        self.indexes = {
            r: TFIDFIndexFast(config["scheme"], config["min_df"], config["max_df_frac"]) 
            for r in roles
        }
        self.cand_doc_ids = []

    def fit(self, cand_roles_dict: Dict[str, Dict[str, List[str]]]):
        self.cand_doc_ids = sorted(list(cand_roles_dict.keys()))
        for role in self.roles:
            df = defaultdict(int)
            raw_tfs = {}
            for did in self.cand_doc_ids:
                toks = cand_roles_dict[did].get(role, [])
                tf = defaultdict(int)
                for t in toks: tf[t] += 1
                if tf:
                    raw_tfs[did] = dict(tf)
                    for t in tf.keys(): df[t] += 1
            self.indexes[role].fit(self.cand_doc_ids, raw_tfs, dict(df))

    def retrieve_with_weights(self, query_roles_dict: Dict[str, Dict[str, List[str]]], relevance: dict, weights: Dict[str, float]) -> dict:
        qids = sorted(list(relevance.keys()))
        final_scores = np.zeros((len(qids), len(self.cand_doc_ids)), dtype=np.float32)
        
        for role in self.roles:
            if weights.get(role, 0.0) == 0.0: continue
            q_raw_tfs = {}
            for qid in qids:
                toks = query_roles_dict[qid].get(role, [])
                tf = defaultdict(int)
                for t in toks: tf[t] += 1
                q_raw_tfs[qid] = dict(tf)
            final_scores += weights[role] * self.indexes[role].get_score_matrix(qids, q_raw_tfs)
            
        results = {}
        doc_ids_arr = np.array(self.cand_doc_ids)
        for r, qid in enumerate(qids):
            top_indices = np.argsort(-final_scores[r])
            results[qid] = doc_ids_arr[top_indices].tolist()
        return results

# ── DATA LOADER ──────────────────────────────────────────────────────────────

def load_and_tokenize_rr_split(base_dir: str, split: str, ngram: int = 3):
    split_dir = os.path.join(base_dir, f"ik_{split}")
    query_dir = os.path.join(split_dir, "query")
    cand_dir  = os.path.join(split_dir, "candidate")
    
    relevance = {}
    with open(os.path.join(split_dir, f"test.json")) as f:
        raw_rel = json.load(f)
        for item in raw_rel.get("Query Set", []):
            relevance[item["id"].replace(".txt", "")] = [c.replace(".txt", "") for c in item.get("relevant candidates", [])]
            
    def _parse_dir(dpath: str) -> Dict[str, Dict[str, List[str]]]:
        docs = {}
        for fname in tqdm(os.listdir(dpath), desc=f"Loading {dpath}"):
            if not fname.endswith(".txt"): continue
            did = fname.replace(".txt", "")
            with open(os.path.join(dpath, fname), "r", encoding="utf-8", errors="replace") as f:
                role_texts = defaultdict(list)
                for line in f:
                    parts = line.strip().split("\t", 1)
                    if len(parts) == 2 and parts[0] in ALL_ROLES:
                        role_texts[parts[0]].append(parts[1])
                docs[did] = {r: clean_text(" ".join(sents), remove_stopwords=True, ngram=ngram) for r, sents in role_texts.items() if sents}
        return docs

    return _parse_dir(query_dir), _parse_dir(cand_dir), relevance

# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--split",    default=SPLIT)
    parser.add_argument("--output",   default=OUTPUT)
    args = parser.parse_args()

    q_roles, c_roles, relevance = load_and_tokenize_rr_split(args.data_dir, args.split, ngram=3)
    all_results = []
    
    print(f"\nStarting Grid Sweep. Total Grids per config: {len(WEIGHT_GRIDS)}")

    for config in CONFIGS:
        print(f"\n[{config['name']}] Fitting Base Indexes...")
        retriever = LateFusionRetriever(ALL_ROLES, config)
        retriever.fit(c_roles)
        
        for i, weights in enumerate(tqdm(WEIGHT_GRIDS, desc=f"Sweeping Grids for {config['name']}")):
            w_str = f"F{int(weights['Fact'])}R{int(weights['RatioOfTheDecision'])}P{int(weights['Precedent'])}"
            if weights["Argument"] > 0: w_str += "A0.5"
            model_label = f"LF_{config['name']}_{w_str}"
            
            results = retriever.retrieve_with_weights(q_roles, relevance, weights)
            metrics = evaluate_all(results, relevance, k_values=K_VALUES, label=model_label)
            all_results.append(metrics)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_results(all_results, args.output)
    
    print("\n\n=== Top 10 Absolute Best Configurations (Sorted by MAP) ===")
    all_results.sort(key=lambda x: x.get("MAP", 0), reverse=True)
    print_results_table(all_results[:10], sort_by="MAP", title="Ultimate Late Fusion Tuning Results")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))

if __name__ == "__main__":
    main()