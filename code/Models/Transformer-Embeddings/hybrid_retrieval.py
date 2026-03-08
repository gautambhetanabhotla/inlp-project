"""
Hybrid BM25 + Transformer Re-ranking Pipeline for Prior Case Retrieval
=======================================================================
Stage 1 : BM25 retrieves top-N candidates per query (fast, lexical)
Stage 2 : A legal-domain transformer (InLegalBERT / InCaseLawBERT / bert-base)
          re-ranks those top-N candidates using dense cosine similarity.

This is significantly faster than all-pairs transformer scoring while
being more accurate than BM25 alone.

Usage:
    python3 hybrid_retrieval.py path/to/config.json

Config file keys:
    path_prior_cases   : folder containing candidate .txt files
    path_current_cases : folder containing query .txt files
    true_labels_json   : path to label json (train.json / test.json)
    checkpoint         : HuggingFace model id or local path
                         e.g. "law-ai/InLegalBERT", "law-ai/InCaseLawBERT",
                              "bert-base-uncased", "distilbert-base-uncased"
    top512             : "True"  -> truncate doc to first 512 tokens
                         "False" -> segment doc, embed each segment, max-pool
    bm25_top_n         : how many candidates BM25 shortlists per query (e.g. 100)
    bm25_n_gram        : n-gram value for BM25 (1 = unigram recommended)
    alpha              : interpolation weight for score fusion (0=pure transformer,
                         1=pure BM25, 0.3 recommended for re-ranking)
"""

import os, sys, re, json, time, codecs, pickle as pkl
import numpy as np
import pandas as pd
from tqdm import tqdm
import evaluate_at_K

# ── Argument & config ──────────────────────────────────────────────────────────
assert len(sys.argv) == 2, "Usage: python3 hybrid_retrieval.py path/to/config.json"

current_time = "_".join(time.ctime().split()) + "_" + str(os.getpid())
save_folder  = f"./exp_results/HYBRID_{current_time}"
os.makedirs(save_folder)

with open(sys.argv[1], "r") as f:
    cfg = json.load(f)
with open(f"{save_folder}/config_file.json", "w") as f:
    json.dump(cfg, f, indent=4)

PATH_CANDIDATES  = cfg["path_prior_cases"]
PATH_QUERIES     = cfg["path_current_cases"]
LABELS_JSON      = cfg["true_labels_json"]
CHECKPOINT       = cfg["checkpoint"]
TOP512           = cfg.get("top512", "True")
BM25_TOP_N       = int(cfg.get("bm25_top_n", 100))
BM25_N_GRAM      = int(cfg.get("bm25_n_gram", 1))
ALPHA            = float(cfg.get("alpha", 0.3))   # score = alpha*bm25 + (1-alpha)*transformer

assert os.path.isdir(PATH_CANDIDATES),  f"Candidate dir not found: {PATH_CANDIDATES}"
assert os.path.isdir(PATH_QUERIES),     f"Query dir not found: {PATH_QUERIES}"
assert os.path.isfile(LABELS_JSON),     f"Labels file not found: {LABELS_JSON}"
assert TOP512 in ("True", "False"),     "top512 must be 'True' or 'False'"

print(f"Save folder : {save_folder}")
print(f"Checkpoint  : {CHECKPOINT}")
print(f"BM25 top-N  : {BM25_TOP_N}  |  n-gram : {BM25_N_GRAM}  |  alpha : {ALPHA}")

# ── Imports (heavy ones after config validation) ───────────────────────────────
import torch, gc
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from transformers import AutoTokenizer, AutoModel

gc.collect()
torch.cuda.empty_cache()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
    """Scikit-learn based BM25 (same implementation as the original run_script.py)."""
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
        b, k1 = self.b, self.k1
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

# For each query: get BM25 scores over ALL candidates, keep top-N shortlist
print(f"Running BM25 first-stage retrieval (top-{BM25_TOP_N}) …")
bm25_scores_all: dict[int, dict[int, float]] = {}   # query_id -> {cand_id: bm25_score}
bm25_shortlist:  dict[int, list[int]]        = {}   # query_id -> [top-N cand_ids]

for q_id, q_text in tqdm(zip(query_ids, query_texts), total=len(query_ids)):
    scores = bm25.transform(q_text)
    # normalise to [0, 1]
    s_min, s_max = scores.min(), scores.max()
    norm_scores  = (scores - s_min) / (s_max - s_min + 1e-12)

    bm25_scores_all[q_id] = {c_id: float(norm_scores[i]) for i, c_id in enumerate(candidate_ids)}

    top_n_idx = np.argsort(scores)[::-1][:BM25_TOP_N]
    bm25_shortlist[q_id] = [candidate_ids[i] for i in top_n_idx]

print(f"BM25 stage done. Average shortlist size: {BM25_TOP_N}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. TRANSFORMER EMBEDDINGS (only on shortlisted candidates)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading transformer: {CHECKPOINT} …")

if "distilbert" in CHECKPOINT.lower():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
elif "bert" in CHECKPOINT.lower():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
else:
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    if CHECKPOINT == "law-ai/InCaseLawBERT":
        tokenizer.model_max_length = 510

model = AutoModel.from_pretrained(CHECKPOINT).to(device)
model.eval()

EMBED_BATCH = 16   # reduce to 8 if OOM on GPU; fine on CPU too

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Returns (N, hidden_size) numpy array.
    If TOP512=="True"  -> truncate each doc to 512 tokens, embed once.
    If TOP512=="False" -> split into 512-token chunks, embed each, max-pool.
    """
    all_vecs = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        with torch.no_grad():
            if TOP512 == "True":
                enc = tokenizer(batch, padding=True, truncation=True,
                                max_length=512, return_tensors="pt").to(device)
                out = model(**enc, output_hidden_states=True)
                # CLS token of last hidden layer
                vec = out.hidden_states[-1][:, 0, :].cpu().numpy()
                all_vecs.append(vec)
            else:
                # chunk-based: embed per-document
                for doc in batch:
                    tokens = tokenizer(doc, return_tensors="pt",
                                       truncation=False)["input_ids"][0]
                    chunk_vecs = []
                    CHUNK = 510
                    for start in range(0, len(tokens), CHUNK):
                        chunk_ids = tokens[start : start + CHUNK].unsqueeze(0).to(device)
                        out = model(chunk_ids, output_hidden_states=True)
                        chunk_vecs.append(out.hidden_states[-1][0, 0, :].cpu().numpy())
                    # max-pool across chunks
                    all_vecs.append(np.max(np.stack(chunk_vecs), axis=0, keepdims=True))
    return np.vstack(all_vecs)  # (N, H)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


# Pre-compute query embeddings
print("Embedding query documents …")
query_embeddings = {}
for q_id, q_text in tqdm(zip(query_ids, query_texts), total=len(query_ids)):
    query_embeddings[q_id] = embed_texts([q_text])[0]   # (H,)

# For each query, embed only its BM25 shortlisted candidates
print("Embedding shortlisted candidate documents & scoring …")
transformer_scores: dict[int, dict[int, float]] = {}

for q_id in tqdm(query_ids):
    shortlist   = bm25_shortlist[q_id]
    short_texts = [candidate_docs[c] for c in shortlist]
    short_embs  = embed_texts(short_texts)              # (BM25_TOP_N, H)
    q_emb       = query_embeddings[q_id]                # (H,)

    t_scores = {}
    for c_id, c_emb in zip(shortlist, short_embs):
        t_scores[c_id] = cosine_sim(q_emb, c_emb)

    # normalise transformer scores to [0, 1]
    vals = list(t_scores.values())
    mn, mx = min(vals), max(vals)
    transformer_scores[q_id] = {c: (s - mn) / (mx - mn + 1e-12) for c, s in t_scores.items()}

del model, tokenizer
gc.collect()
torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════════════════════
# 4. SCORE FUSION  (alpha * BM25 + (1-alpha) * Transformer)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFusing scores …")
fused_scores: dict[int, dict[int, float]] = {}

for q_id in query_ids:
    fused = {}
    for c_id in candidate_ids:
        b_score = bm25_scores_all[q_id].get(c_id, 0.0)
        t_score = transformer_scores[q_id].get(c_id, 0.0)   # 0 if not in shortlist
        fused[c_id] = ALPHA * b_score + (1.0 - ALPHA) * t_score
    fused_scores[q_id] = fused

with open(f"{save_folder}/fused_scores.json", "w") as f:
    json.dump({str(k): {str(kk): vv for kk, vv in v.items()}
               for k, v in fused_scores.items()}, f)

# ══════════════════════════════════════════════════════════════════════════════
# 5. BUILD SIMILARITY MATRIX & EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
with open(LABELS_JSON, "r") as f:
    true_labels = json.load(f)

def obtain_sim_df_from_labels(labels):
    query_numbers    = [int(re.findall(r"\d+", i["id"])[0]) for i in labels["Query Set"]]
    relevant_cases   = [i["relevant candidates"]            for i in labels["Query Set"]]
    relevant_cases   = [[int(re.findall(r"\d+", j)[0]) for j in rc] for rc in relevant_cases]
    relevant_cases   = dict(zip(query_numbers, relevant_cases))
    candidate_numbers = sorted([int(re.findall(r"\d+", i["id"])[0]) for i in labels["Candidate Set"]])

    rows = {}
    for qn in sorted(relevant_cases):
        row = {}
        for cn in candidate_numbers:
            if cn == qn:               row[cn] = -1
            elif cn in relevant_cases[qn]: row[cn] = 1
            else:                      row[cn] = 0
        rows[qn] = row

    df = pd.DataFrame(rows).T
    df.insert(0, "query_case_id", list(rows.keys()))
    return df.reset_index(drop=True)

gold_df = obtain_sim_df_from_labels(true_labels)
sim_df  = obtain_sim_df_from_labels(true_labels)

# ── Vectorized fill (replaces slow row-by-row iloc loop) ──────────────────────
print("Filling similarity matrix (vectorized) …")
col_name       = "query_case_id"
col_candidates = [c for c in sim_df.columns if c != col_name]

q_idx = {q_id: i for i, q_id in enumerate(query_ids)}
c_idx = {c_id: i for i, c_id in enumerate(candidate_ids)}

# Build full fused matrix as numpy array
fused_matrix = np.zeros((len(query_ids), len(candidate_ids)), dtype=np.float32)
for q_id in query_ids:
    qi = q_idx[q_id]
    for ci, c_id in enumerate(candidate_ids):
        fused_matrix[qi, ci] = fused_scores[q_id].get(c_id, 0.0)

label_c_ids = [int(c) for c in col_candidates]
label_q_ids = list(sim_df[col_name].values)
col_remap   = [c_idx[c] for c in label_c_ids]
row_remap   = [q_idx[q] for q in label_q_ids]

score_block            = fused_matrix[np.ix_(row_remap, col_remap)]
sim_df[col_candidates] = score_block.astype(float)

filled_csv_path = f"{save_folder}/filled_similarity_matrix.csv"
sim_df.to_csv(filled_csv_path, index=False)

print("Computing evaluation metrics …")
output_numbers = evaluate_at_K.get_f1_vs_K(gold_df, sim_df)

with open(f"{save_folder}/output.json", "w") as f:
    json.dump(output_numbers, f, indent=4)

# ── Pretty print ───────────────────────────────────────────────────────────────
best_k   = int(np.argmax(output_numbers["f1_vs_K"]))
best_f1  = output_numbers["f1_vs_K"][best_k]
best_p   = output_numbers["precision_vs_K"][best_k]
best_r   = output_numbers["recall_vs_K"][best_k]

print("\n" + "="*55)
print(f"  Hybrid BM25 + Transformer Results")
print(f"  Checkpoint : {CHECKPOINT}")
print(f"  alpha      : {ALPHA}  |  BM25 top-N : {BM25_TOP_N}")
print("="*55)
print(f"  Best K     : {best_k + 1}")
print(f"  Precision  : {best_p*100:.2f}%")
print(f"  Recall     : {best_r*100:.2f}%")
print(f"  F1         : {best_f1*100:.2f}%")
print("="*55)
print(f"\nResults saved to: {save_folder}/")
