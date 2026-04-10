"""
bge_rerank.py
=============
Stage 2 of the TF-IDF → BGE pipeline.
Takes a candidate pool JSON (e.g. TF-IDF top-200 predictions) and reranks
each query's pool using cosine similarity of BGE-style dense embeddings.

Model choices (--model flag)
----------------------------
  Fast / CPU-friendly (fits on MacBook Air):
    BAAI/bge-base-en-v1.5       110M params, 512-token window  [DEFAULT]
    BAAI/bge-large-en-v1.5      335M params, 512-token window  (may OOM on MPS)

  Long-context (32k tokens, require more RAM  ~6–14 GB):
    dunzhang/stella_en_1.5B_v5  1.5B params – best from paper on long docs
    Salesforce/SFR-Embedding-2_R  Mistral-7B based – best MAP on IL-PCR in paper

Usage
-----
    # Stage 1 (if not done yet):
    python tfidf_retrieve.py

    # Stage 2 – rerank TF-IDF top-200 with BGE-large (default):
    python bge_rerank.py --pool retrieval-predictions-tfidf-n10-log-top200.json

    # Use a longer-context model:
    python bge_rerank.py --pool retrieval-predictions-tfidf-n10-log-top200.json \\
                         --model dunzhang/stella_en_1.5B_v5 --max_length 8192

    # Adjust batch size for memory:
    python bge_rerank.py --pool ... --batch_size 4
"""

import os, json, argparse, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tfidf_utils import evaluate_all

K_VALUES = [5, 10, 15, 20, 50, 100, 200]

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--pool",       required=True,
                    help="JSON file {query_id: [cand_id,...]} — pool to rerank")
parser.add_argument("--cand_dir",   default="dataset/ik_test 4/candidate")
parser.add_argument("--query_dir",  default="dataset/ik_test 4/query")
parser.add_argument("--labels",     default="dataset/ik_test 4/test.json")
parser.add_argument("--model",      default="BAAI/bge-base-en-v1.5",
                    help="HuggingFace embedding model name (default: BAAI/bge-base-en-v1.5)")
parser.add_argument("--max_length", type=int, default=512,
                    help="Token truncation limit (default: 512; use 8192+ for long-context models)")
parser.add_argument("--batch_size", type=int, default=4,
                    help="Encoding batch size — reduce if OOM (default: 4)")
parser.add_argument("--device",     default=None,
                    help="Force device: cpu, mps, cuda (default: auto-detect)")
parser.add_argument("--cache_dir",  default="Cache",
                    help="HuggingFace model cache directory (default: Cache/)")
parser.add_argument("--out",        default=None,
                    help="Output predictions JSON path. Auto-named if omitted.")
parser.add_argument("--query_prefix", default="Represent this sentence for searching relevant passages: ",
                    help="Instruction prefix prepended to queries (BGE-style). Set to '' to disable.")
parser.add_argument("--pooling",    default="cls", choices=["cls", "mean"],
                    help="Pooling strategy: cls (BGE default) or mean (default: cls)")
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
print(f"Device: {device}")

# ── load pool ─────────────────────────────────────────────────────────────────
with open(args.pool) as f:
    pool_predictions = json.load(f)
print(f"Pool: {len(pool_predictions)} queries")

pool_cand_ids = set()
for cands in pool_predictions.values():
    pool_cand_ids.update(cands)
print(f"Unique pool candidates: {len(pool_cand_ids)}")

# ── load documents (only pool candidates, not full 1727) ──────────────────────
def load_folder(folder, ids=None):
    docs = {}
    for fn in sorted(os.listdir(folder)):
        if fn.endswith(".txt") and (ids is None or fn in ids):
            with open(os.path.join(folder, fn), errors="ignore") as f:
                docs[fn] = f.read()
    return docs

print("Loading documents …")
candidate_docs      = load_folder(args.cand_dir, ids=pool_cand_ids)
query_docs          = load_folder(args.query_dir)
query_ids           = sorted(query_docs.keys())
pool_cand_ids_list  = sorted(pool_cand_ids)
print(f"  {len(query_ids)} queries  |  {len(pool_cand_ids_list)} pool candidates")

# ── ground truth ──────────────────────────────────────────────────────────────
with open(args.labels) as f:
    true_labels = json.load(f)
gold_indexed = {
    item["id"]: set(item.get("relevant candidates", []))
    for item in true_labels["Query Set"]
}

# ── load model ────────────────────────────────────────────────────────────────
print(f"\nLoading model: {args.model} …")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
model     = AutoModel.from_pretrained(args.model, cache_dir=args.cache_dir)
model.eval().to(device)
print(f"  loaded in {time.time()-t0:.1f}s")

# ── embedding helpers ─────────────────────────────────────────────────────────
def mean_pool(last_hidden_state, attention_mask):
    """Masked mean pooling over token dimension."""
    mask_exp = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)


def pool(last_hidden_state, attention_mask):
    """Pool according to --pooling flag: cls token or masked mean."""
    if args.pooling == "cls":
        return last_hidden_state[:, 0]   # [CLS] is first token for BERT-style models
    return mean_pool(last_hidden_state, attention_mask)


def encode(texts, desc="", is_query=False):
    """Encode a list of strings, returning a (N, dim) float32 numpy array."""
    if is_query and args.query_prefix:
        texts = [args.query_prefix + t for t in texts]
    all_embs = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i : i + args.batch_size]
        enc   = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
        embs = pool(out.last_hidden_state, enc["attention_mask"])
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu().float().numpy())
        del enc, out, embs
        if device == "mps":
            torch.mps.empty_cache()
        if desc and ((i // args.batch_size) % 20 == 0):
            print(f"  {desc}: {min(i + args.batch_size, len(texts))}/{len(texts)}", flush=True)
    return np.vstack(all_embs)

# ── encode ────────────────────────────────────────────────────────────────────
print("\nEncoding pool candidates …")
cand_texts = [candidate_docs[d] for d in pool_cand_ids_list]
cand_embs  = encode(cand_texts, desc="candidates")   # (n_pool, dim)

print("Encoding queries …")
q_texts = [query_docs[d] for d in query_ids]
q_embs  = encode(q_texts, desc="queries", is_query=True)   # (n_queries, dim)

cand_idx_map = {c: i for i, c in enumerate(pool_cand_ids_list)}

# ── rerank ────────────────────────────────────────────────────────────────────
print("\nReranking …")
ranked_results = {}
for qi, q_id in enumerate(query_ids):
    pool = pool_predictions.get(q_id, [])
    if not pool:
        ranked_results[q_id] = []
        continue
    pool_idxs  = [cand_idx_map[c] for c in pool if c in cand_idx_map]
    pool_embs  = cand_embs[pool_idxs]              # (pool_size, dim)
    scores     = pool_embs @ q_embs[qi]            # cosine (both L2-normed)
    order      = np.argsort(scores)[::-1]
    ranked_results[q_id] = [pool[i] for i in order]

# ── evaluate ──────────────────────────────────────────────────────────────────
model_tag = args.model.split("/")[-1]
relevance = {q: list(v) for q, v in gold_indexed.items()}
m = evaluate_all(
    ranked_results, relevance,
    k_values=K_VALUES,
    label=f"BGE-rerank ({model_tag}) on {os.path.basename(args.pool)}",
    verbose=True,
)

# best micro-F1 sweep
query_data = []
for q_id, ranked in ranked_results.items():
    actual = gold_indexed.get(q_id)
    if actual:
        query_data.append((actual, [c for c in ranked if c != q_id]))

best_f1, best_k = 0.0, 1
for k in range(1, max(K_VALUES) + 1):
    tp = fp = fn = 0
    for actual, ranked in query_data:
        top_k = set(ranked[:k])
        tp += len(top_k & actual)
        fp += len(top_k - actual)
        fn += len(actual - top_k)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    if f > best_f1:
        best_f1, best_k = f, k

print(f"\n  ── Summary ──────────────────────────────────────────────────────")
print(f"  Model      = {args.model}")
print(f"  Pool       = {args.pool}")
print(f"  MAP        = {m['MAP']*100:.2f}%")
print(f"  MRR        = {m['MRR']*100:.2f}%")
print(f"  MicroF1@10 = {m.get('MicroF1@10', 0)*100:.2f}%")
print(f"  BestMicroF1= {best_f1*100:.2f}% @ K={best_k}")
print(f"  ─────────────────────────────────────────────────────────────────")

# ── save predictions ──────────────────────────────────────────────────────────
pool_base = os.path.splitext(os.path.basename(args.pool))[0]
out_path  = args.out or f"retrieval-predictions-bge-{model_tag}-rerank-{pool_base}.json"
with open(out_path, "w") as f:
    json.dump(ranked_results, f)
print(f"\n  Predictions saved → {out_path}")
