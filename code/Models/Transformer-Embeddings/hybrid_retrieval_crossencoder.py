"""
Hybrid BM25 + Cross-Encoder Re-ranking Pipeline for Prior Case Retrieval
=========================================================================
Stage 1 : BM25 retrieves top-N candidates per query (fast, lexical)
Stage 2 : Cross-encoder scores every (query, shortlisted-candidate) pair jointly,
           giving much higher accuracy than bi-encoders because both texts are
           seen together by the model.
Score fusion: final = alpha * norm_bm25 + (1 - alpha) * norm_cross
"""

import os, sys, json, time, re, codecs
import numpy as np
import pandas as pd
import pickle as pkl
import tqdm
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers.cross_encoder import CrossEncoder

assert len(sys.argv) == 2, "Usage: python hybrid_retrieval_crossencoder.py <config.json>"

# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFIG
# ══════════════════════════════════════════════════════════════════════════════
with open(sys.argv[1]) as f:
    cfg = json.load(f)

PATH_CANDIDATES  = cfg["path_prior_cases"]
PATH_QUERIES     = cfg["path_current_cases"]
PATH_LABELS      = cfg["true_labels_json"]
CHECKPOINT       = cfg.get("checkpoint", "cross-encoder/ms-marco-MiniLM-L-6-v2")
BM25_TOP_N       = int(cfg.get("bm25_top_n", 100))
BM25_N_GRAM      = int(cfg.get("bm25_n_gram", 1))
ALPHA            = float(cfg.get("alpha", 0.5))   # 0=pure cross-enc, 1=pure BM25
BATCH_SIZE       = int(cfg.get("batch_size", 32))
MAX_LENGTH       = int(cfg.get("max_length", 512))

current_time = "_".join(time.ctime().split()) + "_" + str(os.getpid())
save_folder  = f"./exp_results/HYBRID_CE_{current_time}"
os.makedirs(save_folder)

with open(f"{save_folder}/config_file.json", "w") as f:
    json.dump(cfg, f, indent=4)

print(f"\nSave folder : {save_folder}")
print(f"Checkpoint  : {CHECKPOINT}")
print(f"BM25 top-N  : {BM25_TOP_N}  |  n-gram : {BM25_N_GRAM}  |  alpha : {ALPHA}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD CORPUS
# ══════════════════════════════════════════════════════════════════════════════
import evaluate_at_K

def load_txt_folder(folder):
    docs = {}
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".txt"):
            continue
        doc_id = int(re.findall(r"\d+", fname)[0])
        fpath = os.path.join(folder, fname)
        with codecs.open(fpath, "r", "utf-8", errors="ignore") as fh:
            docs[doc_id] = " ".join(fh.read().split())
    return docs

print("\nLoading corpus …")
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
        self._X_raw  = super(TfidfVectorizer, self.vectorizer).transform(X)
        self._len_X  = self._X_raw.sum(1).A1
        self._X_csc  = self._X_raw.tocsc()
        self.avdl    = self._X_raw.sum(1).mean()
        self._dl_norm = self.k1 * (1 - self.b + self.b * self._len_X / self.avdl)

    def transform(self, q):
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
bm25_scores_all: dict = {}   # query_id -> {cand_id: norm_bm25_score}
bm25_shortlist:  dict = {}   # query_id -> [top-N cand_ids]

for q_id, q_text in tqdm.tqdm(zip(query_ids, query_texts), total=len(query_ids)):
    scores = bm25.transform(q_text)
    s_min, s_max = scores.min(), scores.max()
    norm = (scores - s_min) / (s_max - s_min + 1e-12)
    bm25_scores_all[q_id] = {c_id: float(norm[i]) for i, c_id in enumerate(candidate_ids)}
    top_n_idx = np.argsort(scores)[::-1][:BM25_TOP_N]
    bm25_shortlist[q_id] = [candidate_ids[i] for i in top_n_idx]

print("BM25 stage done.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. CROSS-ENCODER RE-RANKING
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading CrossEncoder: {CHECKPOINT} …")
model = CrossEncoder(CHECKPOINT, max_length=MAX_LENGTH)

print(f"Cross-encoder scoring (query, candidate) pairs …")
ce_scores_all: dict = {}   # query_id -> {cand_id: ce_score}

for q_id, q_text in tqdm.tqdm(zip(query_ids, query_texts), total=len(query_ids)):
    shortlisted = bm25_shortlist[q_id]
    # Build (query_text, candidate_text) pairs for this query
    pairs = [(q_text, candidate_docs[c_id]) for c_id in shortlisted]
    # Score all pairs in batches
    scores = model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)
    # Normalise to [0, 1]
    s_min, s_max = scores.min(), scores.max()
    norm = (scores - s_min) / (s_max - s_min + 1e-12)
    ce_scores_all[q_id] = {c_id: float(norm[i]) for i, c_id in enumerate(shortlisted)}

print("Cross-encoder stage done.")

# ══════════════════════════════════════════════════════════════════════════════
# 4. SCORE FUSION & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with open(PATH_LABELS) as f:
    true_labels = json.load(f)

def obtain_sim_df(labels):
    query_numbers    = [int(re.findall(r"\d+", i["id"])[0]) for i in labels["Query Set"]]
    relevant_cases   = [i["relevant candidates"]            for i in labels["Query Set"]]
    relevant_cases   = [[int(re.findall(r"\d+", j)[0]) for j in i] for i in relevant_cases]
    relevant_cases   = {i: j for i, j in zip(query_numbers, relevant_cases)}
    candidate_numbers = sorted([int(re.findall(r"\d+", i["id"])[0]) for i in labels["Candidate Set"]])

    rows = {}
    for qn in sorted(relevant_cases):
        rows[qn] = {c: (-1 if c == qn else (1 if c in relevant_cases[qn] else 0))
                    for c in candidate_numbers}
    df = pd.DataFrame(rows).T
    df.insert(0, "query_case_id", list(rows.keys()))
    return df.reset_index(drop=True)

gold_df = obtain_sim_df(true_labels)
sim_df  = obtain_sim_df(true_labels)
column_name       = "query_case_id"
column_candidates = list(sim_df.columns)[1:]

print("\nFusing scores …")
score_rows = []
for q_id in sim_df[column_name].values:
    row = []
    for c_id in column_candidates:
        bm25_s = bm25_scores_all[q_id].get(int(c_id), 0.0)
        ce_s   = ce_scores_all[q_id].get(int(c_id), 0.0)   # 0 for non-shortlisted
        row.append(ALPHA * bm25_s + (1 - ALPHA) * ce_s)
    score_rows.append(row)
sim_df[column_candidates] = np.array(score_rows, dtype=float)

sim_df.to_csv(f"{save_folder}/filled_similarity_matrix.csv")

output = evaluate_at_K.get_f1_vs_K(gold_df, sim_df)
with open(f"{save_folder}/output.json", "w") as f:
    json.dump(output, f, indent=4)

# Pretty-print best K
f1s = output["f1_vs_K"]
best_k = int(np.argmax(f1s)) + 1
best_f1 = f1s[best_k - 1]
best_p  = output["precision_vs_K"][best_k - 1]
best_r  = output["recall_vs_K"][best_k - 1]
print("\n" + "=" * 58)
print(f"  Best K     : {best_k}")
print(f"  Precision  : {best_p*100:.2f}%")
print(f"  Recall     : {best_r*100:.2f}%")
print(f"  F1         : {best_f1*100:.2f}%")
print("=" * 58)
print(f"\nResults saved to: {save_folder}/")
