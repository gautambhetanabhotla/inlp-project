"""
Hybrid BM25 + SentenceBERT Re-ranking Pipeline for Prior Case Retrieval
========================================================================
Stage 1 : BM25 retrieves top-N candidates per query (fast, lexical)
Stage 2 : SentenceBERT re-ranks those top-N candidates using cosine similarity
          on mean-pooled sentence embeddings — much better for semantic similarity
          than vanilla CLS-token BERT.

Compared to hybrid_retrieval.py (CLS-token BERT), SBERT produces embeddings
specifically optimised for cosine similarity, giving better semantic ranking.

Usage:
    python3 hybrid_retrieval_sbert.py path/to/config.json

Config file keys:
    path_prior_cases   : folder containing candidate .txt files
    path_current_cases : folder containing query .txt files
    true_labels_json   : path to label json (train.json / test.json)
    checkpoint         : SentenceTransformer model id, e.g.:
                           "sentence-transformers/all-MiniLM-L6-v2"   (fast, good quality)
                           "sentence-transformers/all-mpnet-base-v2"   (best quality)
                           "sentence-transformers/msmarco-bert-base-dot-v5" (retrieval-tuned)
                           "law-ai/InLegalBERT"  (legal domain, used via SBERT wrapper)
    bm25_top_n         : how many candidates BM25 shortlists per query (default 100)
    bm25_n_gram        : n-gram value for BM25 (default 1)
    alpha              : score fusion weight — 0=pure SBERT, 1=pure BM25 (default 0.3)
    batch_size         : SBERT encoding batch size (default 32)
"""

import os, sys, re, json, time, pickle as pkl
import numpy as np
import pandas as pd
from tqdm import tqdm
import evaluate_at_K

# ── Argument & config ──────────────────────────────────────────────────────────
assert len(sys.argv) == 2, "Usage: python3 hybrid_retrieval_sbert.py path/to/config.json"

current_time = "_".join(time.ctime().split()) + "_" + str(os.getpid())
save_folder  = f"./exp_results/HYBRID_SBERT_{current_time}"
os.makedirs(save_folder)

with open(sys.argv[1], "r") as f:
    cfg = json.load(f)
with open(f"{save_folder}/config_file.json", "w") as f:
    json.dump(cfg, f, indent=4)

PATH_CANDIDATES = cfg["path_prior_cases"]
PATH_QUERIES    = cfg["path_current_cases"]
LABELS_JSON     = cfg["true_labels_json"]
CHECKPOINT      = cfg["checkpoint"]
BM25_TOP_N      = int(cfg.get("bm25_top_n", 100))
BM25_N_GRAM     = int(cfg.get("bm25_n_gram", 1))
ALPHA           = float(cfg.get("alpha", 0.3))   # score = alpha*bm25 + (1-alpha)*sbert
BATCH_SIZE      = int(cfg.get("batch_size", 32))

assert os.path.isdir(PATH_CANDIDATES), f"Candidate dir not found: {PATH_CANDIDATES}"
assert os.path.isdir(PATH_QUERIES),    f"Query dir not found: {PATH_QUERIES}"
assert os.path.isfile(LABELS_JSON),    f"Labels file not found: {LABELS_JSON}"

print(f"Save folder : {save_folder}")
print(f"Checkpoint  : {CHECKPOINT}")
print(f"BM25 top-N  : {BM25_TOP_N}  |  n-gram : {BM25_N_GRAM}  |  alpha : {ALPHA}")

# ── Imports ────────────────────────────────────────────────────────────────────
import torch, gc
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from sentence_transformers import SentenceTransformer

gc.collect()
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD CORPUS
# ══════════════════════════════════════════════════════════════════════════════
def load_txt_folder(folder: str) -> dict:
    """Returns {int_case_id: text} for every .txt file in folder."""
    docs = {}
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".txt"):
            continue
        nums = re.findall(r"\d+", fname)
        if not nums:
            continue
        doc_id = int(nums[0])
        with open(os.path.join(folder, fname), "r", encoding="utf-8", errors="ignore") as fh:
            docs[doc_id] = " ".join(fh.read().split())
    return docs

print("Loading corpus …")
candidate_docs = load_txt_folder(PATH_CANDIDATES)
query_docs     = load_txt_folder(PATH_QUERIES)
print(f"  Queries: {len(query_docs)}  |  Candidates: {len(candidate_docs)}")

candidate_ids   = sorted(candidate_docs.keys())
query_ids       = sorted(query_docs.keys())
candidate_texts = [candidate_docs[i] for i in candidate_ids]
query_texts     = [query_docs[i]     for i in query_ids]

# ══════════════════════════════════════════════════════════════════════════════
# 2. BM25 FIRST-STAGE RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════
class BM25:
    def __init__(self, b=0.7, k1=1.6, n_gram=1):
        self.vectorizer = TfidfVectorizer(
            max_df=0.65, min_df=1, use_idf=True,
            ngram_range=(n_gram, n_gram)
        )
        self.b, self.k1 = b, k1

    def fit(self, X):
        self.vectorizer.fit(X)
        # Pre-compute and cache corpus matrix once — avoids re-transforming 4320
        # candidates on every single query call.
        self._X_raw = super(TfidfVectorizer, self.vectorizer).transform(X)
        self._len_X = self._X_raw.sum(1).A1
        self._X_csc = self._X_raw.tocsc()          # CSC needed for column slicing
        self.avdl   = self._X_raw.sum(1).mean()
        # Pre-compute the doc-length normalisation term (constant across queries)
        self._dl_norm = self.k1 * (1 - self.b + self.b * self._len_X / self.avdl)

    def transform(self, q):
        """Score all corpus docs against a single query string."""
        k1 = self.k1
        q_vec, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q_vec)
        X_csc  = self._X_csc[:, q_vec.indices]
        denom  = X_csc + self._dl_norm[:, None]
        idf    = self.vectorizer._tfidf.idf_[None, q_vec.indices] - 1.0
        numer  = X_csc.multiply(np.broadcast_to(idf, X_csc.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1

print("\nFitting BM25 …")
bm25 = BM25(n_gram=BM25_N_GRAM)
bm25.fit(candidate_texts)

print(f"Running BM25 first-stage retrieval (top-{BM25_TOP_N}) …")
bm25_scores_all: dict[int, dict[int, float]] = {}
bm25_shortlist:  dict[int, list[int]]        = {}

for q_id, q_text in tqdm(zip(query_ids, query_texts), total=len(query_ids)):
    scores  = bm25.transform(q_text)
    s_min, s_max = scores.min(), scores.max()
    norm    = (scores - s_min) / (s_max - s_min + 1e-12)
    bm25_scores_all[q_id] = {c_id: float(norm[i]) for i, c_id in enumerate(candidate_ids)}
    top_n   = np.argsort(scores)[::-1][:BM25_TOP_N]
    bm25_shortlist[q_id]  = [candidate_ids[i] for i in top_n]

print("BM25 stage done.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. SBERT EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading SentenceTransformer: {CHECKPOINT} …")
sbert = SentenceTransformer(CHECKPOINT, device=device)

# SBERT truncates internally; for very long legal docs we truncate to max seq len
MAX_TOKENS = sbert.get_max_seq_length()
print(f"  Model max seq length: {MAX_TOKENS} tokens")

def encode(texts: list[str]) -> np.ndarray:
    """Encode a list of texts, returning (N, H) normalised embeddings."""
    return sbert.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,   # L2-normalised → dot product == cosine sim
        convert_to_numpy=True,
    )

# Pre-compute all query embeddings in one batched pass
print("Encoding all query documents …")
query_emb_matrix = encode(query_texts)   # (Q, H)
query_emb_dict   = {q_id: query_emb_matrix[i] for i, q_id in enumerate(query_ids)}

# For each query: encode only its BM25 shortlist
print("Encoding shortlisted candidates & computing SBERT scores …")
transformer_scores: dict[int, dict[int, float]] = {}

for q_id in tqdm(query_ids):
    shortlist   = bm25_shortlist[q_id]
    short_texts = [candidate_docs[c] for c in shortlist]
    short_embs  = encode(short_texts)              # (BM25_TOP_N, H)

    q_emb       = query_emb_dict[q_id]             # (H,) already normalised
    # dot product of normalised vectors == cosine similarity
    sims        = short_embs @ q_emb               # (BM25_TOP_N,)

    # normalise to [0, 1]
    mn, mx = sims.min(), sims.max()
    sims_norm = (sims - mn) / (mx - mn + 1e-12)

    transformer_scores[q_id] = {c_id: float(sims_norm[i])
                                 for i, c_id in enumerate(shortlist)}

del sbert, query_emb_matrix
gc.collect()
torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════════════════════
# 4. SCORE FUSION  (alpha * BM25 + (1-alpha) * SBERT)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFusing scores …")
# Build fused score matrix as numpy array (fast, avoids iloc bottleneck)
q_idx = {q_id: i for i, q_id in enumerate(query_ids)}
c_idx = {c_id: i for i, c_id in enumerate(candidate_ids)}

fused_matrix = np.zeros((len(query_ids), len(candidate_ids)), dtype=np.float32)

for q_id in query_ids:
    qi = q_idx[q_id]
    bm25_row  = bm25_scores_all[q_id]      # {c_id: score} for ALL candidates
    sbert_row = transformer_scores[q_id]   # {c_id: score} for shortlist only
    for ci, c_id in enumerate(candidate_ids):
        b = bm25_row.get(c_id, 0.0)
        s = sbert_row.get(c_id, 0.0)
        fused_matrix[qi, ci] = ALPHA * b + (1.0 - ALPHA) * s

# ══════════════════════════════════════════════════════════════════════════════
# 5. BUILD SIMILARITY MATRIX & EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
with open(LABELS_JSON, "r") as f:
    true_labels = json.load(f)

def obtain_sim_df_from_labels(labels):
    query_numbers     = [int(re.findall(r"\d+", i["id"])[0]) for i in labels["Query Set"]]
    relevant_cases    = [i["relevant candidates"]            for i in labels["Query Set"]]
    relevant_cases    = [[int(re.findall(r"\d+", j)[0]) for j in rc] for rc in relevant_cases]
    relevant_cases    = dict(zip(query_numbers, relevant_cases))
    candidate_numbers = sorted([int(re.findall(r"\d+", i["id"])[0]) for i in labels["Candidate Set"]])

    rows = {}
    for qn in sorted(relevant_cases):
        row = {}
        for cn in candidate_numbers:
            if cn == qn:                       row[cn] = -1
            elif cn in relevant_cases[qn]:     row[cn] = 1
            else:                              row[cn] = 0
        rows[qn] = row

    df = pd.DataFrame(rows).T
    df.insert(0, "query_case_id", list(rows.keys()))
    return df.reset_index(drop=True)

gold_df = obtain_sim_df_from_labels(true_labels)
sim_df  = obtain_sim_df_from_labels(true_labels)

# ── Vectorized fill (no slow iloc loop) ───────────────────────────────────────
print("Filling similarity matrix (vectorized) …")
col_name       = "query_case_id"
col_candidates = [c for c in sim_df.columns if c != col_name]

# Map label-file candidate order to fused_matrix column order
label_c_ids = [int(c) for c in col_candidates]
col_remap   = [c_idx[c] for c in label_c_ids]   # index into fused_matrix columns

# Map label-file query order to fused_matrix row order
label_q_ids = list(sim_df[col_name].values)
row_remap   = [q_idx[q] for q in label_q_ids]

# Fill all scores at once
score_block            = fused_matrix[np.ix_(row_remap, col_remap)]  # (Q, C)
sim_df[col_candidates] = score_block.astype(float)

filled_csv_path = f"{save_folder}/filled_similarity_matrix.csv"
sim_df.to_csv(filled_csv_path, index=False)

print("Computing evaluation metrics …")
output_numbers = evaluate_at_K.get_f1_vs_K(gold_df, sim_df)

with open(f"{save_folder}/output.json", "w") as f:
    json.dump(output_numbers, f, indent=4)

# ── Pretty print ───────────────────────────────────────────────────────────────
best_k  = int(np.argmax(output_numbers["f1_vs_K"]))
best_f1 = output_numbers["f1_vs_K"][best_k]
best_p  = output_numbers["precision_vs_K"][best_k]
best_r  = output_numbers["recall_vs_K"][best_k]

print("\n" + "="*58)
print(f"  Hybrid BM25 + SBERT Results")
print(f"  Checkpoint : {CHECKPOINT}")
print(f"  alpha      : {ALPHA}  |  BM25 top-N : {BM25_TOP_N}")
print("="*58)
print(f"  Best K     : {best_k + 1}")
print(f"  Precision  : {best_p*100:.2f}%")
print(f"  Recall     : {best_r*100:.2f}%")
print(f"  F1         : {best_f1*100:.2f}%")
print("="*58)
print(f"\nResults saved to: {save_folder}/")
