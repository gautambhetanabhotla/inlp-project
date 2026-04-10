"""
bge_standalone.py
=================
Standalone dense retrieval using BGE (no TF-IDF pool).
Encodes all 1727 candidates and all 237 queries, then ranks by cosine similarity.

Optionally adds mean-over-chunks strategy (--chunks) to handle documents
longer than 512 tokens by splitting into overlapping 512-token windows
and averaging their embeddings.

Usage
-----
    # Truncate only (matches current bge_rerank.py behaviour):
    python bge_standalone.py

    # With chunking (better for long legal docs):
    python bge_standalone.py --chunks --stride 256
"""

import os, json, argparse, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tfidf_utils import evaluate_all

K_VALUES = [5, 10, 15, 20, 50, 100, 200]

parser = argparse.ArgumentParser()
parser.add_argument("--cand_dir",   default="dataset/ik_test 4/candidate")
parser.add_argument("--query_dir",  default="dataset/ik_test 4/query")
parser.add_argument("--labels",     default="dataset/ik_test 4/test.json")
parser.add_argument("--model",      default="BAAI/bge-base-en-v1.5")
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--device",     default=None)
parser.add_argument("--cache_dir",  default="Cache")
parser.add_argument("--query_prefix", default="Represent this sentence for searching relevant passages: ")
parser.add_argument("--pooling",    default="cls", choices=["cls","mean"])
parser.add_argument("--chunks",     action="store_true",
                    help="Enable chunk-then-mean-pool for long docs (slower but handles >512 tok docs)")
parser.add_argument("--stride",     type=int, default=256,
                    help="Stride between chunks when --chunks is enabled (default: 256)")
parser.add_argument("--out",        default=None)
args = parser.parse_args()

# ── device ────────────────────────────────────────────────────────────────────
if args.device:
    device = args.device
elif torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Device: {device}  |  chunking={'on, stride='+str(args.stride) if args.chunks else 'OFF (truncate@512)'}")

# ── load docs ─────────────────────────────────────────────────────────────────
def load_folder(folder, ids=None):
    docs = {}
    for fn in sorted(os.listdir(folder)):
        if fn.endswith(".txt") and (ids is None or fn in ids):
            with open(os.path.join(folder, fn), errors="ignore") as f:
                docs[fn] = f.read()
    return docs

print("Loading documents …")
candidate_docs = load_folder(args.cand_dir)
query_docs     = load_folder(args.query_dir)
cand_ids  = sorted(candidate_docs.keys())
query_ids = sorted(query_docs.keys())
print(f"  {len(query_ids)} queries  |  {len(cand_ids)} candidates (full corpus, no TF-IDF pool)")

with open(args.labels) as f:
    true_labels = json.load(f)
gold_indexed = {
    item["id"]: set(item.get("relevant candidates", []))
    for item in true_labels["Query Set"]
}

# ── model ─────────────────────────────────────────────────────────────────────
print(f"\nLoading model: {args.model} …")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
model     = AutoModel.from_pretrained(args.model, cache_dir=args.cache_dir)
model.eval().to(device)
print(f"  loaded in {time.time()-t0:.1f}s")

# ── helpers ───────────────────────────────────────────────────────────────────
def _pool_hidden(last_hidden_state, attention_mask):
    if args.pooling == "cls":
        return last_hidden_state[:, 0]
    mask_exp = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)


def _encode_batch(texts):
    """Encode a small batch of strings → (N, dim) normed float32 numpy."""
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=args.max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
    embs = _pool_hidden(out.last_hidden_state, enc["attention_mask"])
    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
    if device == "mps":
        torch.mps.empty_cache()
    return embs.cpu().float().numpy()


def encode_chunked(texts, desc=""):
    """
    For each text: tokenise fully, split into overlapping windows of
    max_length tokens with stride, encode each window, then mean-pool
    the chunk embeddings → single vector per document.
    """
    all_embs = []
    for i, text in enumerate(texts):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        stride    = args.stride
        win       = args.max_length - 2          # leave room for [CLS]/[SEP]
        chunks    = []
        start = 0
        while start < len(token_ids):
            chunk = token_ids[start : start + win]
            chunks.append(tokenizer.decode(chunk))
            if start + win >= len(token_ids):
                break
            start += stride
        if not chunks:
            chunks = [text]
        # encode in sub-batches
        chunk_embs = []
        for b in range(0, len(chunks), args.batch_size):
            chunk_embs.append(_encode_batch(chunks[b : b + args.batch_size]))
        chunk_embs = np.vstack(chunk_embs)        # (n_chunks, dim)
        doc_emb    = chunk_embs.mean(axis=0)
        doc_emb   /= (np.linalg.norm(doc_emb) + 1e-9)
        all_embs.append(doc_emb)
        if desc and (i % 50 == 0):
            print(f"  {desc}: {i}/{len(texts)}", flush=True)
    return np.vstack(all_embs)


def encode_truncate(texts, desc="", is_query=False):
    """Simple truncation encode — same as bge_rerank.py."""
    if is_query and args.query_prefix:
        texts = [args.query_prefix + t for t in texts]
    all_embs = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i : i + args.batch_size]
        all_embs.append(_encode_batch(batch))
        if desc and ((i // args.batch_size) % 20 == 0):
            print(f"  {desc}: {min(i+args.batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.vstack(all_embs)


# ── encode ────────────────────────────────────────────────────────────────────
encode_fn = encode_chunked if args.chunks else encode_truncate

print("\nEncoding candidates …")
cand_texts = [candidate_docs[d] for d in cand_ids]
cand_embs  = encode_fn(cand_texts, desc="candidates")

print("Encoding queries …")
q_texts = [query_docs[d] for d in query_ids]
if args.chunks:
    q_embs = encode_chunked(q_texts, desc="queries")
else:
    q_embs = encode_truncate(q_texts, desc="queries", is_query=True)

# ── rank ──────────────────────────────────────────────────────────────────────
print("\nRanking …")
scores_matrix = q_embs @ cand_embs.T        # (n_queries, n_cands)
ranked_results = {}
for qi, q_id in enumerate(query_ids):
    order = np.argsort(scores_matrix[qi])[::-1]
    ranked_results[q_id] = [cand_ids[i] for i in order if cand_ids[i] != q_id]

# ── evaluate ──────────────────────────────────────────────────────────────────
model_tag = args.model.split("/")[-1]
chunk_tag = f"-chunks{args.stride}" if args.chunks else "-trunc512"
label     = f"BGE-standalone ({model_tag}{chunk_tag})"

relevance = {q: list(v) for q, v in gold_indexed.items()}
m = evaluate_all(ranked_results, relevance, k_values=K_VALUES, label=label, verbose=True)

# best micro-F1
query_data = []
for q_id, ranked in ranked_results.items():
    actual = gold_indexed.get(q_id)
    if actual:
        query_data.append((actual, ranked))

best_f1, best_k = 0.0, 0
for k in range(1, 201):
    preds_at_k = [r[:k] for _, r in query_data]
    f1s = []
    for (actual, _), preds in zip(query_data, preds_at_k):
        tp = len(set(preds) & actual)
        p  = tp / len(preds) if preds else 0
        r  = tp / len(actual) if actual else 0
        f1s.append(2*p*r/(p+r) if p+r else 0)
    avg = sum(f1s) / len(f1s)
    if avg > best_f1:
        best_f1, best_k = avg, k

print(f"\n  BestMicroF1 = {best_f1*100:.2f}% @ K={best_k}")

# ── save ──────────────────────────────────────────────────────────────────────
out = args.out or f"retrieval-predictions-bge-standalone-{model_tag}{chunk_tag}.json"
with open(out, "w") as f:
    json.dump(ranked_results, f)
print(f"  Predictions saved → {out}")
