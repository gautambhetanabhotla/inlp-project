# IL-PCR Experiment Report
**Dataset**: IK (Indian Kanoon) — Training Split  
**Task**: Precedent Case Retrieval (PCR)  
**Metric**: F1@K (reported at best K, K ∈ {1…20})  
**Date**: March 7, 2026

---

## Completed Experiments

### 1. BM25 Baseline (Unigram)

| Parameter | Value |
|-----------|-------|
| Model | BM25 |
| n-gram | 1 (unigram) |
| Corpus | ik_train (full text) |

**Results:**

| K | Recall | Precision | F1 |
|---|--------|-----------|-----|
| 1 | 3.73% | 23.94% | 6.45% |
| 2 | 6.33% | 20.31% | 9.65% |
| 3 | 8.08% | 17.29% | 11.02% |
| 4 | 9.57% | 15.36% | 11.79% |
| **5** | **10.59%** | **13.59%** | **11.90%** ← best |
| 6 | 11.45% | 12.25% | 11.84% |
| 7 | 12.34% | 11.31% | 11.80% |
| 10 | 14.51% | 9.31% | 11.34% |
| 20 | 18.46% | 5.93% | 8.97% |

**Best F1: 11.90% @ K=5**

---

### 2. Hybrid BM25 + InLegalBERT (Top-512 Truncation)

| Parameter | Value |
|-----------|-------|
| Model | `law-ai/InLegalBERT` |
| Encoding strategy | Truncate to first 512 tokens |
| BM25 top-N candidates | 100 |
| BM25 n-gram | 1 |
| Fusion weight α | 0.3 (BM25) / 0.7 (BERT) |
| Corpus | ik_train (full text) |

**Results:**

| K | Recall | Precision | F1 |
|---|--------|-----------|-----|
| 1 | 1.04% | 6.65% | 1.79% |
| 5 | 3.65% | 4.69% | 4.11% |
| 10 | 5.92% | 3.80% | 4.71% |
| **20** | **10.25%** | **3.29%** | **4.98%** ← best |

**Best F1: 4.98% @ K=20**  
⚠️ Significantly worse than BM25 alone — top-512 truncation loses critical legal content.

---

### 3. Hybrid BM25 + SBERT MiniLM (Full Document)

| Parameter | Value |
|-----------|-------|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Encoding strategy | Full document (truncated to 512 tokens by SBERT) |
| BM25 top-N candidates | 100 |
| BM25 n-gram | 1 |
| Fusion weight α | 0.3 (BM25) / 0.7 (SBERT) |
| Batch size | 32 |
| Corpus | ik_train (full text) |

**Results:**

| K | Recall | Precision | F1 |
|---|--------|-----------|-----|
| 1 | 2.79% | 17.90% | 4.82% |
| 5 | 8.87% | 11.39% | 9.98% |
| 6 | 9.91% | 10.60% | 10.24% |
| **7** | **10.80%** | **9.90%** | **10.33%** ← best |
| 10 | 13.09% | 8.40% | 10.24% |
| 20 | 17.77% | 5.70% | 8.63% |

**Best F1: 10.33% @ K=7**  
⚠️ Slightly worse than BM25 alone — full-doc SBERT loses information for long legal documents.

---

### 4. Hybrid BM25 + SBERT MiniLM with Chunking ⭐ Best Result

| Parameter | Value |
|-----------|-------|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Encoding strategy | Chunked: 256 token windows, 128 token stride |
| BM25 top-N candidates | 100 |
| BM25 n-gram | 1 |
| Fusion weight α | 0.3 (BM25) / 0.7 (SBERT) |
| Encode batch size | 128 |
| Corpus | ik_train (full text) |

**Results:**

| K | Recall | Precision | F1 |
|---|--------|-----------|-----|
| 1 | 5.60% | 35.91% | 9.68% |
| 2 | 9.68% | 31.08% | 14.77% |
| 3 | 12.87% | 27.53% | 17.54% |
| 4 | 15.26% | 24.49% | 18.80% |
| 5 | 17.35% | 22.27% | 19.51% |
| **6** | **18.99%** | **20.31%** | **19.63%** ← best |
| 7 | 19.97% | 18.31% | 19.10% |
| 10 | 22.66% | 14.55% | 17.72% |
| 20 | 27.02% | 8.67% | 13.13% |

**Best F1: 19.63% @ K=6** ← **Current state-of-the-art on this setup**

---

## Summary Comparison

| # | Method | Encoding | Best F1 | Best K | vs BM25 |
|---|--------|----------|---------|--------|---------|
| 1 | BM25 (unigram) | — | 11.90% | 5 | baseline |
| 2 | BM25 + SBERT (full doc) | 512-token truncation | 10.33% | 7 | −1.57pp |
| 3 | BM25 + InLegalBERT | top-512 truncation | 4.98% | 20 | −6.92pp |
| **4** | **BM25 + SBERT (chunked)** | **256-tok / 128-stride** | **19.63%** | **6** | **+7.73pp** |

---

## Key Findings

1. **Chunking is essential for long legal documents.** SBERT with 256-token chunks + 128-token stride (max-pooled over chunks) gives a **+7.73pp** gain over BM25, vs. a **−1.57pp** loss with full-doc encoding.

2. **InLegalBERT with top-512 truncation severely degrades performance** (4.98%), likely because legal judgments bury relevant content deep in the document — the first 512 tokens are mostly preamble/header.

3. **BM25 alone is a strong baseline** (11.90%). Full-doc SBERT actually hurts, confirming that without chunking, semantic models cannot handle the length of Indian legal judgments.

4. **Low fusion weight for BM25 (α=0.3)** used throughout — all hybrid models rely more heavily on SBERT similarity. This works well for the chunked variant but is suboptimal when SBERT encoding is poor.

---


