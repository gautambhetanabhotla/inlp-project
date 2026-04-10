"""
Hybrid TF-IDF + Chunk-Aggregate SBERT Pipeline for Prior Case Retrieval
=======================================================================
Why chunking?  Legal docs have median ~5000 tokens; 99% exceed 512 tokens.
Standard BERT/SBERT truncates at 512, seeing only the first ~6% of the text.

Approach:
  Stage 1 : TF-IDF top-N shortlist (cosine similarity, full text)
  Stage 2 : Each document split into overlapping token chunks.
            All candidate chunks are pre-encoded upfront.
            Query is also chunked.
            Doc score = max cosine similarity over all (query_chunk, cand_chunk) pairs.
  Fusion  : alpha * norm_tfidf + (1-alpha) * chunk_max_score
"""

import os, sys, json, time, re, codecs
import numpy as np
import pandas as pd
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from tfidf_utils import clean_text
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

assert len(sys.argv) == 2, "Usage: python hybrid_retrieval_chunk_sbert.py <config.json>"

# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFIG
# ══════════════════════════════════════════════════════════════════════════════
with open(sys.argv[1]) as f:
    cfg = json.load(f)

PATH_CANDIDATES = cfg["path_prior_cases"]
PATH_QUERIES    = cfg["path_current_cases"]
PATH_LABELS     = cfg["true_labels_json"]
CHECKPOINT      = cfg.get("checkpoint", "sentence-transformers/all-MiniLM-L6-v2")
TFIDF_TOP_N     = int(cfg.get("tfidf_top_n", 100))
TFIDF_N_GRAM    = int(cfg.get("tfidf_n_gram", 1))
TFIDF_MIN_DF    = int(cfg.get("tfidf_min_df", 2))
TFIDF_MAX_DF    = float(cfg.get("tfidf_max_df", 0.95))
TFIDF_TF_SCHEME = cfg.get("tfidf_tf_scheme", "log")   # "log" → sublinear_tf=True
ALPHA           = float(cfg.get("alpha", 0.3))
CHUNK_TOKENS    = int(cfg.get("chunk_tokens", 256))   # tokens per chunk (excl. special)
CHUNK_STRIDE    = int(cfg.get("chunk_stride", 128))   # overlap between consecutive chunks
ENCODE_BATCH    = int(cfg.get("encode_batch", 128))   # SBERT encode batch size

current_time = "_".join(time.ctime().split()) + "_" + str(os.getpid())
save_folder  = f"./exp_results/HYBRID_CHUNK_{current_time}"
os.makedirs(save_folder)

with open(f"{save_folder}/config_file.json", "w") as f:
    json.dump(cfg, f, indent=4)

print(f"\nSave folder : {save_folder}")
print(f"Checkpoint  : {CHECKPOINT}")
print(f"TF-IDF top-N : {TFIDF_TOP_N}  |  n-gram : {TFIDF_N_GRAM}  |  min_df : {TFIDF_MIN_DF}  |  max_df : {TFIDF_MAX_DF}  |  tf_scheme : {TFIDF_TF_SCHEME}  |  alpha : {ALPHA}")
print(f"Chunk size  : {CHUNK_TOKENS} tokens  |  stride : {CHUNK_STRIDE} tokens")

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD CORPUS
# ══════════════════════════════════════════════════════════════════════════════
def load_txt_folder(folder):
    docs = {}
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".txt"):
            continue
        doc_id = int(re.findall(r"\d+", fname)[0])
        with codecs.open(os.path.join(folder, fname), "r", "utf-8", errors="ignore") as fh:
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
# 2. TF-IDF FIRST-STAGE RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════
print("\nFitting TF-IDF …")
# Use clean_text as analyzer: strips <CITATION_*> tags, removes stopwords, builds n-grams
vectorizer = TfidfVectorizer(
    analyzer=lambda text: clean_text(text, remove_stopwords=True, ngram=TFIDF_N_GRAM),
    min_df=TFIDF_MIN_DF,
    max_df=TFIDF_MAX_DF,
    sublinear_tf=(TFIDF_TF_SCHEME == "log"),   # log(1+tf) when True
    use_idf=True,
    norm="l2",
)
tfidf_matrix = vectorizer.fit_transform(candidate_texts)   # (n_cands, vocab), l2-normalised

print(f"Running TF-IDF first-stage retrieval (top-{TFIDF_TOP_N}) …")
tfidf_scores_all = {}   # query_id -> {cand_id: norm_tfidf_score}
tfidf_shortlist  = {}   # query_id -> [top-N cand_ids]

for q_id, q_text in tqdm.tqdm(zip(query_ids, query_texts), total=len(query_ids)):
    q_vec  = vectorizer.transform([q_text])                      # (1, vocab)
    scores = (tfidf_matrix @ q_vec.T).toarray().flatten()        # cosine (both l2-normed)
    s_min, s_max = scores.min(), scores.max()
    norm = (scores - s_min) / (s_max - s_min + 1e-12)
    tfidf_scores_all[q_id] = {c_id: float(norm[i]) for i, c_id in enumerate(candidate_ids)}
    top_n_idx = np.argsort(scores)[::-1][:TFIDF_TOP_N]
    tfidf_shortlist[q_id] = [candidate_ids[i] for i in top_n_idx]

print("TF-IDF stage done.")

# ══════════════════════════════════════════════════════════════════════════════
# 3. CHUNKING UTILITY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading tokenizer for chunking …")
# Use SBERT's own tokenizer for accurate chunk boundaries
sbert_model = SentenceTransformer(CHECKPOINT)
tokenizer   = sbert_model.tokenizer

def text_to_chunks(text: str, chunk_tokens: int, stride: int) -> list[str]:
    """Tokenize text and split into overlapping chunks, returning decoded strings."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) == 0:
        return [text[:100] or "[EMPTY]"]
    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + chunk_tokens, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        if end == len(token_ids):
            break
        start += stride
    return chunks if chunks else [text[:100]]

# ══════════════════════════════════════════════════════════════════════════════
# 4. PRE-ENCODE ALL CANDIDATE CHUNKS (shortlisted candidates only)
# ══════════════════════════════════════════════════════════════════════════════
# Collect the union of all shortlisted candidate IDs across queries
shortlisted_cand_ids = set()
for sl in tfidf_shortlist.values():
    shortlisted_cand_ids.update(sl)
shortlisted_cand_ids = sorted(shortlisted_cand_ids)

print(f"\nEncoding {len(shortlisted_cand_ids)} shortlisted candidates (chunked) …")
cand_chunks_list  = []   # flat list of all chunk strings
cand_chunk_to_doc = []   # parallel: doc_id for each chunk

for c_id in tqdm.tqdm(shortlisted_cand_ids):
    chunks = text_to_chunks(candidate_docs[c_id], CHUNK_TOKENS, CHUNK_STRIDE)
    cand_chunks_list.extend(chunks)
    cand_chunk_to_doc.extend([c_id] * len(chunks))

print(f"  Total candidate chunks: {len(cand_chunks_list)}")
print(f"  Avg chunks/doc: {len(cand_chunks_list)/len(shortlisted_cand_ids):.1f}")

# Encode all candidate chunks in one batched pass
cand_chunk_embs = sbert_model.encode(
    cand_chunks_list,
    batch_size=ENCODE_BATCH,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True,
)   # shape: (total_chunks, dim)

# Build a dict: doc_id -> row indices in cand_chunk_embs
cand_chunk_to_doc = np.array(cand_chunk_to_doc)
cand_idx_map = {}   # doc_id -> np.array of indices
for c_id in shortlisted_cand_ids:
    cand_idx_map[c_id] = np.where(cand_chunk_to_doc == c_id)[0]

print("Candidate chunk encoding done.")

# ══════════════════════════════════════════════════════════════════════════════
# 5. QUERY CHUNK ENCODING + SCORE AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nEncoding queries and computing chunk-max SBERT scores …")

chunk_scores_all = {}   # query_id -> {cand_id: max_chunk_score}

for q_id, q_text in tqdm.tqdm(zip(query_ids, query_texts), total=len(query_ids)):
    q_chunks     = text_to_chunks(q_text, CHUNK_TOKENS, CHUNK_STRIDE)
    q_chunk_embs = sbert_model.encode(
        q_chunks,
        batch_size=ENCODE_BATCH,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )   # shape: (n_q_chunks, dim)

    shortlist    = tfidf_shortlist[q_id]
    chunk_scores_all[q_id] = {}

    for c_id in shortlist:
        c_embs = cand_chunk_embs[cand_idx_map[c_id]]   # (n_c_chunks, dim)
        # similarity matrix: (n_q_chunks, n_c_chunks), take global max
        sim_matrix = q_chunk_embs @ c_embs.T            # dot = cosine (both normalised)
        chunk_scores_all[q_id][c_id] = float(sim_matrix.max())

print("Chunk-SBERT scoring done.")

# ══════════════════════════════════════════════════════════════════════════════
# 6. SCORE FUSION & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with open(PATH_LABELS) as f:
    true_labels = json.load(f)

def obtain_sim_df(labels):
    query_numbers     = [int(re.findall(r"\d+", i["id"])[0]) for i in labels["Query Set"]]
    relevant_cases    = [i["relevant candidates"]            for i in labels["Query Set"]]
    relevant_cases    = [[int(re.findall(r"\d+", j)[0]) for j in i] for i in relevant_cases]
    relevant_cases    = {i: j for i, j in zip(query_numbers, relevant_cases)}
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
column_candidates = list(sim_df.columns)[1:]

print("\nFusing scores and building similarity matrix …")

# Normalise chunk scores per query to [0,1] before fusion
score_rows = []
for q_id in sim_df["query_case_id"].values:
    cs = chunk_scores_all.get(q_id, {})
    # global min-max over shortlisted candidates for this query
    vals = list(cs.values())
    vmin, vmax = (min(vals), max(vals)) if vals else (0.0, 1.0)
    row = []
    for c_id in column_candidates:
        tfidf_s     = tfidf_scores_all[q_id].get(int(c_id), 0.0)
        chunk_s_raw = cs.get(int(c_id), 0.0)
        chunk_s = (chunk_s_raw - vmin) / (vmax - vmin + 1e-12)
        row.append(ALPHA * tfidf_s + (1 - ALPHA) * chunk_s)
    score_rows.append(row)

sim_df[column_candidates] = np.array(score_rows, dtype=float)
sim_df.to_csv(f"{save_folder}/filled_similarity_matrix.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 6b. EVALUATION  (micro Precision, Recall, F1 vs K=1..20)
# ══════════════════════════════════════════════════════════════════════════════
SKIP_IDS = {1864396, 1508893}

gold_indexed = gold_df.set_index("query_case_id")
sim_indexed  = sim_df.set_index("query_case_id")
cand_col_ids = np.array([int(c) for c in column_candidates])

query_data = []
for q_id in sim_indexed.index.values:
    if q_id in SKIP_IDS:
        continue
    if q_id not in gold_indexed.index:
        continue
    gold_row = gold_indexed.loc[q_id]
    if isinstance(gold_row, pd.DataFrame):
        gold_row = gold_row.iloc[0]
    gold = gold_row.values
    actual = {str(c) for c, g in zip(cand_col_ids, gold) if g == 1 or g == -2}

    sim_row = sim_indexed.loc[q_id]
    if isinstance(sim_row, pd.DataFrame):
        sim_row = sim_row.iloc[0]
    sim_scores = sim_row.values
    sorted_cands = [str(c) for _, c in
                    sorted(zip(sim_scores, cand_col_ids), key=lambda x: x[0], reverse=True)]
    if str(q_id) in sorted_cands:
        sorted_cands.remove(str(q_id))
    query_data.append((actual, sorted_cands))

precision_vs_K, recall_vs_K, f1_vs_K = [], [], []
for k in range(1, 21):
    tp = fp = fn = 0
    for actual, sorted_cands in query_data:
        top_k = set(sorted_cands[:k])
        tp += len(top_k & actual)
        fp += len(top_k - actual)
        fn += len(actual - top_k)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    precision_vs_K.append(prec)
    recall_vs_K.append(rec)
    f1_vs_K.append(f1)

output = {
    "precision_vs_K": precision_vs_K,
    "recall_vs_K":    recall_vs_K,
    "f1_vs_K":        f1_vs_K,
}
with open(f"{save_folder}/output.json", "w") as f:
    json.dump(output, f, indent=4)

best_k = int(np.argmax(f1_vs_K)) + 1
print("\n" + "=" * 58)
print(f"  Best K     : {best_k}")
print(f"  Precision  : {precision_vs_K[best_k-1]*100:.2f}%")
print(f"  Recall     : {recall_vs_K[best_k-1]*100:.2f}%")
print(f"  F1         : {f1_vs_K[best_k-1]*100:.2f}%")
print("=" * 58)
print(f"\nResults saved to: {save_folder}/")
