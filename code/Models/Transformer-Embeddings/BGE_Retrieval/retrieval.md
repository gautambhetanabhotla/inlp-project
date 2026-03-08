# BGE Retrieval: Long-Document Chunking

## Summary

`BAAI/bge-base-en` has a hard 512-token context window. Legal cases in the ILPCR dataset have a **median length of 4,778 tokens** (98% of documents exceed 512 tokens), so the model was silently truncating every document to its first ~1–2 paragraphs.

This update adds overlapping token-level chunking with mean pooling, so the full text of each document is encoded rather than just the opening section.

---

## Approach

Each document is split into overlapping 510-token chunks. Every chunk is embedded independently, then the chunk embeddings are **mean-pooled and re-normalised** into a single document vector.

\`\`\`
Document (e.g. 4,778 tokens)
│
├── Chunk 1 : tokens   0 – 509   → embedding e1
├── Chunk 2 : tokens 446 – 955   → embedding e2
├── Chunk 3 : tokens 892 – 1401  → embedding e3
│   ...
└── Chunk N : tokens ...         → embedding eN

Final embedding = normalise( mean(e1, e2, ..., eN) )
\`\`\`

Overlap of 64 tokens between consecutive chunks preserves context across boundaries.

---

## Corpus Statistics

| Statistic | Tokens |
|-----------|-------:|
| Model max context | 512 |
| Median doc length | 4,778 |
| 90th percentile | 17,449 |
| Max doc length | 127,766 |
| % docs > 512 tokens | 98.0% |

---

## Results

| Metric | Before chunking | After chunking | Δ |
|--------|:--------------:|:--------------:|:---:|
| Precision@5 | 0.1629 | **0.3114** | +91% |
| Recall@10 | 0.2671 | **0.4131** | +55% |
| MAP | 0.1837 | **0.3291** | +79% |
| NDCG@10 | 0.2539 | **0.4222** | +66% |
| Queries evaluated | 237 | 237 | — |

---

## Run Command

\`\`\`bash
python3 bge_retrieval.py \
    --corpus_dir  ../BM25/data/corpus/ik_test \
    --output_path ./exp_results/bge_scores.json \
    --labels_json ../BM25/data/corpus/ik_test/test.json \
    --model_name  BAAI/bge-base-en \
    --top_k       100 \
    --batch_size  32 \
    --chunk_size  510 \
    --chunk_overlap 64
\`\`\`

---

## Full Code

\`\`\`python
"""
BGE Embedding-based Prior Case Retrieval (PCR)
================================================
Uses BAAI/bge-base-en via sentence-transformers to embed legal case documents,
compute cosine similarity, retrieve the top-K most relevant candidate cases
for each query, and evaluate with Precision@K, Recall@K, MAP, and NDCG@K.

Usage:
    python bge_retrieval.py \
        --corpus_dir  ./data/corpus/ik_test \
        --output_path ./exp_results/bge_scores.json \
        --labels_json ./data/corpus/ik_test/test.json \
        --model_name  BAAI/bge-base-en \
        --top_k       100 \
        --batch_size  32 \
        --chunk_size  510 \
        --chunk_overlap 64

Output JSON format:
    {
        "<query_id>": {
            "<candidate_id>": <cosine_similarity_score>,
            ...
        },
        ...
    }

Metrics printed and saved alongside the scores JSON:
    Precision@5, Recall@10, MAP, NDCG@10
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_documents(dir_path: str, id_padding: int = 10) -> dict[str, str]:
    docs: dict[str, str] = {}
    dir_path = Path(dir_path)

    for fname in sorted(dir_path.iterdir()):
        if fname.suffix != ".txt":
            continue
        numbers = re.findall(r"\d+", fname.stem)
        if not numbers:
            continue
        doc_id = str(int(numbers[0])).zfill(id_padding)
        with open(fname, "r", encoding="utf-8") as fh:
            text = fh.read()
        text = " ".join(text.split())
        docs[doc_id] = text

    return docs


def detect_padding_length(corpus_dir: str) -> int:
    lower = corpus_dir.lower()
    if "coliee" in lower:
        return 6
    return 10


def chunk_text_by_tokens(
    text: str,
    tokenizer,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    stride = chunk_size - overlap
    chunks = []
    for start in range(0, max(1, len(token_ids)), stride):
        chunk_ids = token_ids[start : start + chunk_size]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        if start + chunk_size >= len(token_ids):
            break
    return chunks if chunks else [text]


def encode_documents(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    desc: str,
    query_prefix: str = "",
    device: str = "cpu",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> np.ndarray:
    tokenizer = model.tokenizer
    max_seq = model.get_max_seq_length() or chunk_size
    all_embeddings: list[np.ndarray] = []

    for text in tqdm(texts, desc=desc, unit="doc"):
        if query_prefix:
            text = query_prefix + text

        token_ids = tokenizer.encode(text, add_special_tokens=True)
        if len(token_ids) <= max_seq:
            chunks = [text]
        else:
            chunks = chunk_text_by_tokens(text, tokenizer, chunk_size=max_seq - 2, overlap=chunk_overlap)

        chunk_embs = model.encode(
            chunks,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
            show_progress_bar=False,
        )

        doc_emb = chunk_embs.mean(axis=0)
        norm = np.linalg.norm(doc_emb)
        if norm > 0:
            doc_emb = doc_emb / norm

        all_embeddings.append(doc_emb)

    return np.vstack(all_embeddings).astype(np.float32)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def load_ground_truth(labels_json_path: str) -> dict[str, list[str]]:
    with open(labels_json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    ground_truth: dict[str, list[str]] = {}
    for entry in data["Query Set"]:
        qid = entry["id"].replace(".txt", "")
        relevant = [c.replace(".txt", "") for c in entry.get("relevant candidates", [])]
        ground_truth[qid] = relevant
    return ground_truth


def precision_at_k(relevant: set, retrieved: list, k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / k


def recall_at_k(relevant: set, retrieved: list, k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def average_precision(relevant: set, retrieved: list) -> float:
    if not relevant:
        return 0.0
    hits, running_sum = 0, 0.0
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hits += 1
            running_sum += hits / rank
    return running_sum / len(relevant)


def ndcg_at_k(relevant: set, retrieved: list, k: int) -> float:
    top_k = retrieved[:k]
    dcg = sum(1.0 / np.log2(r + 1) for r, d in enumerate(top_k, 1) if d in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(
    results: dict[str, dict[str, float]],
    ground_truth: dict[str, list[str]],
    p_k: int = 5,
    r_k: int = 10,
    ndcg_k: int = 10,
) -> dict[str, float]:
    prec_scores, rec_scores, ap_scores, ndcg_scores = [], [], [], []
    skipped = 0
    for qid, relevant_list in ground_truth.items():
        if qid not in results:
            skipped += 1
            continue
        relevant = set(relevant_list)
        retrieved = list(results[qid].keys())
        prec_scores.append(precision_at_k(relevant, retrieved, k=p_k))
        rec_scores.append(recall_at_k(relevant, retrieved, k=r_k))
        ap_scores.append(average_precision(relevant, retrieved))
        ndcg_scores.append(ndcg_at_k(relevant, retrieved, k=ndcg_k))
    if skipped:
        log.warning("%d queries skipped.", skipped)
    return {
        f"precision@{p_k}":  float(np.mean(prec_scores))  if prec_scores  else 0.0,
        f"recall@{r_k}":     float(np.mean(rec_scores))   if rec_scores   else 0.0,
        "MAP":               float(np.mean(ap_scores))    if ap_scores    else 0.0,
        f"ndcg@{ndcg_k}":   float(np.mean(ndcg_scores))  if ndcg_scores  else 0.0,
        "num_queries_evaluated": len(prec_scores),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    log.info("Device: %s", device)

    corpus_dir = Path(args.corpus_dir)
    candidate_dir = corpus_dir / "candidate"
    query_dir = corpus_dir / "query"

    assert candidate_dir.is_dir(), f"Candidate dir not found: {candidate_dir}"
    assert query_dir.is_dir(), f"Query dir not found: {query_dir}"

    id_pad = detect_padding_length(str(corpus_dir))
    candidate_docs = load_documents(str(candidate_dir), id_pad)
    query_docs     = load_documents(str(query_dir), id_pad)

    candidate_ids   = sorted(candidate_docs.keys())
    candidate_texts = [candidate_docs[cid] for cid in candidate_ids]
    query_ids       = sorted(query_docs.keys())
    query_texts     = [query_docs[qid] for qid in query_ids]

    log.info("Loading model: %s", args.model_name)
    model = SentenceTransformer(args.model_name, device=device)

    query_prefix = ""
    if "bge" in args.model_name.lower():
        query_prefix = "Represent this sentence for searching relevant passages: "

    candidate_embs = encode_documents(
        model, candidate_texts, args.batch_size, "Candidates",
        query_prefix="", device=device,
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
    )
    query_embs = encode_documents(
        model, query_texts, args.batch_size, "Queries",
        query_prefix=query_prefix, device=device,
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
    )

    sim_matrix = np.dot(query_embs, candidate_embs.T)
    top_k = min(args.top_k, len(candidate_ids))
    results: dict[str, dict[str, float]] = {}

    for q_idx, qid in enumerate(tqdm(query_ids, desc="Top-K retrieval")):
        sims = sim_matrix[q_idx]
        if len(candidate_ids) > 5000:
            top_indices = np.argpartition(sims, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
        else:
            top_indices = np.argsort(sims)[::-1][:top_k]
        results[qid] = {
            candidate_ids[c_idx]: float(round(float(sims[c_idx]), 6))
            for c_idx in top_indices
            if candidate_ids[c_idx] != qid
        }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    log.info("Results saved → %s", output_path)

    labels_path = args.labels_json
    if labels_path is None:
        for stem in ("test", "val", "train"):
            p = corpus_dir / f"{stem}.json"
            if p.is_file():
                labels_path = str(p)
                break

    if labels_path and Path(labels_path).is_file():
        ground_truth = load_ground_truth(labels_path)
        metrics = compute_metrics(results, ground_truth)

        log.info("─" * 45)
        for metric, value in metrics.items():
            if metric == "num_queries_evaluated":
                log.info("  %-22s %d", metric, int(value))
            else:
                log.info("  %-22s %.4f", metric, value)
        log.info("─" * 45)

        metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump({"model": args.model_name, "metrics": metrics}, fh, indent=2)
        log.info("Metrics saved → %s", metrics_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir",     type=str, default="./data/corpus/ik_test")
    parser.add_argument("--output_path",    type=str, default="./exp_results/bge_scores.json")
    parser.add_argument("--model_name",     type=str, default="BAAI/bge-base-en")
    parser.add_argument("--labels_json",    type=str, default=None)
    parser.add_argument("--top_k",          type=int, default=100)
    parser.add_argument("--batch_size",     type=int, default=32)
    parser.add_argument("--chunk_size",     type=int, default=510)
    parser.add_argument("--chunk_overlap",  type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
\`\`\`

---

## Output Metrics (`bge_scores_metrics.json`)

\`\`\`json
{
  "model": "BAAI/bge-base-en",
  "metrics": {
    "precision@5": 0.3113924050632912,
    "recall@10": 0.4130534453648184,
    "MAP": 0.3290502256481126,
    "ndcg@10": 0.4221554059253057,
    "num_queries_evaluated": 237
  }
}
\`\`\`