# IL-PCR Hybrid Retrieval — Experiment Report
**Date:** April 10, 2026  
**Task:** Legal case retrieval (IL-PCR dataset, 237 queries, 1727 candidates)  
**Primary metric:** MicroF1 at optimal K  
**Secondary metrics:** MAP, MRR, NDCG@10  

---

## Methodology

### 1. Task and Dataset

The IL-PCR (Indian Legal – Prior Case Retrieval) task requires: given a query judgment from the Indian Supreme Court, retrieve all prior cases cited in it from a corpus of 1,727 candidate judgments. The evaluation set contains 237 query documents. Two document IDs (1864396, 1508893) are excluded from evaluation due to annotation issues in the gold labels.

**Corpus characteristics:**
- Documents are Indian Supreme Court full-text judgments (~5,000–50,000 words each).
- Average tokenised length: ~14,000 sub-word tokens (well beyond any transformer's 512-token context window).
- Relevance signal: verbatim citation strings (e.g., `AIR 1973 SC 1461`, `(1999) 4 SCC 260`) that appear in both the query and relevant candidates.
- Average relevant candidates per query: ~3–5.

---

### 2. Evaluation Metrics

**MicroF1@K** (primary): For a fixed retrieval depth K, precision and recall are computed for each query (retrieved set = top-K), then micro-averaged across all queries. The reported MicroF1 is the maximum over K ∈ {5, 6, 7, 8, 9, 10, 11, 15, 20}.

$$\text{MicroF1} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}, \quad \text{TP/FP/FN summed across all queries}$$

**MAP** (Mean Average Precision): Average of per-query Average Precision at all relevant document ranks.

**MRR** (Mean Reciprocal Rank): $\frac{1}{|Q|}\sum_{q} \frac{1}{\text{rank of first relevant doc}}$. Measures how quickly the top-1 relevant document is found.

**NDCG@10** (Normalised Discounted Cumulative Gain at 10): Graded relevance measure; rewards systems that place relevant documents higher in the top-10.

---

### 3. Pipeline Architecture

The pipeline (`hybrid_transformer_rerank.py`) is a two-stage retrieval system:

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1 — TF-IDF First-Stage Retrieval                 │
│  Input: 1727 candidates + 1 query                       │
│  Output: Shortlist of top-N (default 200) candidates    │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  Stage 2 — Transformer Re-ranking (optional)            │
│  Input: top-N shortlist                                 │
│  Output: model score per candidate                      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  Score Fusion                                           │
│  final = α·tfidf + β·citation + (1−α−β)·model          │
└────────────────────────┬────────────────────────────────┘
                         │
                   Ranked list → Evaluation
```

---

### 4. Stage 1: Improved TF-IDF

The original hybrid pipeline (`hybrid_retrieval_chunk_sbert.py`) had a bug: the augmented TF scheme was only applied to the candidate matrix via sklearn's `sublinear_tf` flag, but **not** to queries. This made augmented TF asymmetric and effectively unused.

**Fix applied:** A custom `_augment_tf()` function applies the formula $0.5 + 0.5 \cdot \frac{tf}{\max\_tf\_per\_doc}$ in-place on both the candidate CSR matrix and the query matrix before multiplying by the IDF vector.

**TF-IDF construction:**

| Parameter | Description | Values tested |
|---|---|---|
| `scheme` | TF weighting | `log` (sublinear), `augmented` (0.5+0.5·tf/max), `binary` |
| `ngram` | Max n-gram order | 4, 6, 7, 9 |
| `min_df` | Min document frequency | 2 (fixed) |
| `max_df` | Max document frequency | 0.95 (fixed) |
| `top_n` | Stage-1 shortlist size | 50, 100, 150, 200, 300 |

Scores are min-max normalised per query before fusion.

---

### 5. Stage 2: Bi-Encoder with Sliding-Window Chunking

Since legal documents greatly exceed transformer context windows, a **sliding-window chunking** strategy is used:

1. The full document text is tokenised (sub-word tokenisation by the model's tokenizer).
2. The token sequence is split into overlapping chunks of `chunk_tokens` tokens with a step of `chunk_stride` tokens (50% overlap by default).
3. All candidate chunks are encoded once and cached. At query time, each query is also chunked and encoded.
4. Similarity scores between all (query-chunk, candidate-chunk) pairs are aggregated:

| Aggregation | Formula | Properties |
|---|---|---|
| `max_chunk` | $\max_{i,j} \cos(q_i, c_j)$ | Fast; captures single best window |
| `mean_chunk` | $\text{mean}_i \max_j \cos(q_i, c_j)$ | Softer; averages query-chunk bests |
| `sum_top3/5` | $\sum_{k=1}^{K} \text{top-K}(\max_i \cos(q_i, c_j))$ | Accumulates evidence from multiple candidate windows |

**Key parameters:**

| Parameter | Default | Tested values |
|---|---|---|
| `chunk_tokens` | 256 | 128, 256, 512 |
| `chunk_stride` | 128 | 64, 128, 256 |
| `encode_batch` | 128 | fixed |

Average chunks per document: ~70 (at chunk=256, stride=128). Total chunks across 1,727 candidates: ~121,000.

---

### 6. Citation Overlap Score

A third retrieval signal is computed without any model:

$$\text{citation\_recall}(q, c) = \frac{|C_q \cap C_c|}{|C_q|}$$

where $C_q$ and $C_c$ are the sets of Indian SC citation strings extracted from query $q$ and candidate $c$ respectively.

**Citation patterns matched** (regular expressions):
- `AIR YYYY S.C. NNN` — All India Reporter, Supreme Court
- `(YYYY) N SCC NNN` — Supreme Court Cases (variant 1)
- `YYYY (N) SCC NNN` — Supreme Court Cases (variant 2)
- `[YYYY] N SCR NNN` — Supreme Court Reporter
- `MANU/SC/NNNN/YYYY` — Manupatra neutral citation

This signal is:
- **Exact-match** (immune to embedding blur: `AIR 1973 SC 1461` ≠ `AIR 1974 SC 1461`)
- **Not IDF-diluted** (common citations still contribute equally)
- **Near-zero cost** (~2s precompute for all 1,964 documents)

---

### 7. Score Fusion

The three signals are combined linearly:

$$\text{final}(q, c) = \alpha \cdot s_{\text{tfidf}}(q,c) + \beta \cdot s_{\text{citation}}(q,c) + (1-\alpha-\beta) \cdot s_{\text{model}}(q,c)$$

where $\alpha$ is the TF-IDF weight, $\beta$ is the citation weight, and $1-\alpha-\beta$ is the transformer weight. Setting $\beta=0$ reduces to the standard two-way hybrid. Setting $\alpha=0, \beta=0$ is pure-transformer mode.

TF-IDF scores are min-max normalised per query. Transformer bi-encoder scores are also min-max normalised per query to $[0,1]$ before fusion. Citation scores are naturally in $[0,1]$.

---

### 8. Models Evaluated

**Bi-encoders (sentence transformers):**

| Model | Parameters | Domain | Training |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 22M | General | MS MARCO + NLI |
| `all-mpnet-base-v2` | 110M | General | MS MARCO + NLI |
| `all-roberta-large-v1` | 355M | General | MS MARCO + NLI |
| `multi-qa-MiniLM-L6-cos-v1` | 22M | General | QA datasets |
| `paraphrase-MiniLM-L6-v2` | 22M | General | Paraphrase detection |
| `bert-base-uncased` | 110M | General | MLM only (no retrieval fine-tuning) |
| `law-ai/InLegalBERT` | 110M | Indian Legal | Legal corpus MLM |
| `law-ai/InCaseLawBERT` | 110M | Indian Legal | Legal corpus MLM |
| `nlpaueb/legal-bert-base-uncased` | 110M | Legal (EU/US) | Legal corpus MLM |

**Note:** InLegalBERT, InCaseLawBERT, and LegalBERT are BERT-style masked language models without retrieval fine-tuning. They are used as bi-encoders with mean-pooling (sentence-transformers wraps them with a mean pooling layer and random projection head).

**Cross-encoders:**

| Model | Parameters | Training |
|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | MS MARCO passage ranking |

Cross-encoders jointly encode (query, candidate) pairs for more accurate scoring but require $O(N)$ forward passes per query. Texts were pre-truncated to `max_length × 4` characters (~2048 chars) before tokenisation to prevent OOM on long documents.

---

### 9. Experiment Groups

| Group | # Exp. | Purpose |
|---|---|---|
| R | 7 | Reference: pure TF-IDF baselines (scheme × n-gram ablation) |
| I | 5 | MiniLM + improved TF-IDF; alpha and scheme exploration |
| A | 4 | Alpha sweep from 0.50 to 0.95 (MiniLM, fixed scheme/n-gram) |
| G | 1 | Chunk aggregation: mean_chunk vs max_chunk |
| C | 3 | Chunk size ablation: 128, 256, 512 tokens |
| M | 11 | Full model sweep: 9 models × alpha ∈ {0.8, 0.9} |
| CE | 7 | Cross-encoder: top-N ∈ {50, 100, 200}, multiple models *(skipped due to runtime)* |
| TN | 4 | Top-N shortlist size: 50, 100, 150, 300 |
| S | 3 | sum_topK aggregation: sum_top3, sum_top5 |
| CO | 6 | Citation overlap: beta ∈ {0.05, 0.10, 0.20}, 3-way hybrid variants |
| T | 4 | Pure transformer (skip_tfidf=True): 4 models, full corpus shortlist |

**Total experiments run: 48** (CE group skipped). Each experiment saves results as both JSON (per-K curves) and a row in `all_results.csv`.

---

## Baselines

| Label | Scheme | n-gram | MicroF1 | MAP | MRR | NDCG@10 |
|---|---|---|---|---|---|---|
| R_log_n4 | log | 4 | 0.4409 | 0.5906 | 0.7960 | 0.6836 |
| R_log_n7 | log | 7 | 0.4490 | 0.5961 | 0.8015 | 0.6879 |
| R_log_n9 | log | 9 | 0.4496 | 0.5970 | 0.8014 | 0.6877 |
| R_aug_n6 | augmented | 6 | 0.4496 | 0.5977 | 0.7957 | 0.6862 |
| **R_aug_n7** | **augmented** | **7** | **0.4507** | 0.5943 | 0.7935 | 0.6824 |
| R_aug_n9 | augmented | 9 | 0.4501 | 0.5899 | 0.7851 | 0.6793 |
| R_bin_n7 | binary | 7 | 0.4490 | 0.5910 | 0.7928 | 0.6811 |

**Best pure TF-IDF: 0.4507 (R_aug_n7, augmented scheme, n-gram=7)**

Key observations:
- Augmented TF scheme narrowly beats log (0.4507 vs 0.4496 at n=7).
- n-gram order peaks at 6–7; going to 9 slightly regresses on MicroF1.
- Binary TF-IDF is competitive (0.4490) but consistently below augmented.
- For MAP/MRR/NDCG, log n=9 performs best — the metrics have different sensitivities to TF scheme.

---

## Group I — MiniLM + Improved TF-IDF (Alpha Study)

All experiments: MiniLM-L6-v2, augmented scheme (except I_log_n4_a06), chunk=256/128, top_n=200.

| Label | Scheme | n | α | MicroF1 | Δ vs best TF-IDF |
|---|---|---|---|---|---|
| I_log_n4_a06 | log | 4 | 0.6 | 0.4046 | −4.61 |
| I_aug_n7_a06 | augmented | 4 | 0.6 | 0.3996 | −5.11 |
| I_aug_n7_a08 | augmented | 4 | 0.8 | 0.4291 | −2.16 |
| I_aug_n7_a09 | augmented | 4 | 0.9 | 0.4385 | −1.22 |
| I_aug_n9_a08 | augmented | 9 | 0.8 | 0.4321 | −1.86 |

**Findings:**
- The original hybrid failure (α=0.6) is confirmed and explained: giving 40% weight to MiniLM embeddings actively hurts retrieval by 4–5 points.
- Increasing α improves F1 monotonically; the transformer's contribution is net-negative at any α tested here in Group I.
- n-gram=9 at α=0.8 (0.4321) is slightly better than n-gram=4 (0.4291), but the optimal n-gram for pure TF-IDF (n=7) does not appear to transfer cleanly to the hybrid setting.

---

## Group A — Alpha Sweep (0.50 → 0.95)

MiniLM-L6-v2, augmented, n=4, chunk=256/128, top_n=200.

| α | MicroF1 | MAP | MRR | NDCG@10 |
|---|---|---|---|---|
| 0.50 | 0.3887 | 0.4995 | 0.7251 | 0.5851 |
| 0.70 | 0.4179 | 0.5392 | 0.7581 | 0.6251 |
| 0.80 | 0.4291 | 0.5663 | 0.7752 | 0.6507 |
| 0.85 | 0.4327 | 0.5785 | 0.7897 | 0.6653 |
| 0.90 | 0.4385 | 0.5873 | 0.7907 | 0.6711 |
| **0.95** | **0.4468** | 0.5948 | 0.7949 | 0.6836 |

**Findings:**
- F1 increases monotonically as transformer weight → 0. The transformer signal is pure noise for this legal retrieval task.
- At α=0.95 (only 5% transformer weight) the hybrid reaches 0.4468 — just 0.39 points below pure TF-IDF (0.4507).
- The conclusion is clear: for IL-PCR, general-purpose sentence transformers add no retrieval value; they dilute a strong TF-IDF signal.

---

## Group G — Chunk Aggregation (max vs mean)

MiniLM-L6-v2, augmented, n=4, α=0.8, chunk=256/128.

| Aggregation | MicroF1 | MAP |
|---|---|---|
| max_chunk | 0.4291 | 0.5663 |
| mean_chunk | 0.4273 | 0.5655 |

**Finding:** `max_chunk` marginally outperforms `mean_chunk`. Intuitively, the single most-similar passage to the query is a better relevance proxy than averaging across all windows (which dilutes the signal with off-topic paragraphs).

---

## Group C — Chunk Size Ablation

MiniLM-L6-v2, augmented, n=4, α=0.8, top_n=200.

| Chunk / Stride | MicroF1 | MAP | Elapsed (s) |
|---|---|---|---|
| 128 / 64 | 0.4095 | 0.5379 | 256 |
| **256 / 128** | **0.4291** | **0.5663** | **264** |
| 512 / 128 | 0.4257 | 0.5581 | 294 |
| 512 / 256 | 0.4256 | 0.5552 | 170 |

**Finding:** Chunk=256 with stride=128 (50% overlap) is the sweet spot. Smaller chunks (128) lose context; larger chunks (512) pack too much irrelevant text per window and give the model a noisier representation despite faster inference (fewer chunks to encode).

---

## Group M — Model Sweep

All at augmented, n=4, top_n=200, chunk=256/128, max_chunk.

### α = 0.8

| Model | MicroF1 | MAP | MRR | NDCG@10 | Elapsed (s) |
|---|---|---|---|---|---|
| paraphrase-MiniLM-L6-v2 | 0.3970 | 0.5249 | 0.7665 | 0.6128 | 187 |
| bert-base-uncased | 0.4173 | 0.5338 | 0.7559 | 0.6276 | 1163 |
| multi-qa-MiniLM-L6-cos-v1 | 0.4190 | 0.5529 | 0.7667 | 0.6385 | 323 |
| all-mpnet-base-v2 | 0.4244 | 0.5738 | 0.7949 | 0.6613 | 1174 |
| InLegalBERT | 0.4246 | 0.5592 | 0.7875 | 0.6475 | 1180 |
| LegalBERT | 0.4247 | 0.5611 | 0.7764 | 0.6525 | 1281 |
| RoBERTa-large | 0.4323 | 0.5742 | 0.7875 | 0.6654 | 3683 |
| InCaseLawBERT | 0.4155 | 0.5620 | 0.7887 | 0.6476 | 1316 |

### α = 0.9

| Model | MicroF1 | MAP | MRR | NDCG@10 |
|---|---|---|---|---|
| InLegalBERT | 0.4418 | 0.5824 | 0.7919 | 0.6728 |
| InCaseLawBERT | 0.4362 | 0.5828 | 0.7912 | 0.6729 |
| **RoBERTa-large** | **0.4435** | 0.5904 | 0.7921 | 0.6801 |

**Best hybrid at α=0.9:** RoBERTa-large 0.4435 — still 0.72 points below pure TF-IDF.

**Findings:**
- No model at α=0.8 beats MiniLM at α=0.95. The choice of model matters less than the alpha.
- Domain-specific models (InLegalBERT, InCaseLawBERT, LegalBERT) do **not** outperform general purpose models at α=0.8 — they score 0.4246, 0.4155, 0.4247 vs RoBERTa-large's 0.4323.
- At α=0.9, InLegalBERT (0.4418) does close the gap vs RoBERTa (0.4435), justifying its usage when speed matters (RoBERTa-large is 3× slower).
- paraphrase-MiniLM is worst: it's tuned for semantic similarity tasks (paraphrase detection), not retrieval ranking.
- InCaseLawBERT loads with UNEXPECTED MLM / NSP keys — it's a pre-trained LM, not a fine-tuned bi-encoder — which explains underperformance despite legal domain pretraining.

---

## Group TN — Top-N Shortlist Size

MiniLM-L6-v2, augmented, n=4, α=0.8, chunk=256/128.

| top_n | Candidates re-ranked | MicroF1 | MAP |
|---|---|---|---|
| 50 | 50 | 0.4273 | 0.5646 |
| 100 | 100 | 0.4279 | 0.5667 |
| 150 | 150 | 0.4279 | 0.5670 |
| **200** | **200** | **0.4291** | **0.5663** |
| 300 | 300 | 0.4309 | 0.5682 |

**Finding:** F1 improves weakly as top_n increases. The difference from 50 to 300 is only 0.36 points, confirming TF-IDF Stage 1 recall is already saturated — nearly all true positives fall in the top 50. The default top_n=200 is reasonable; going to 300 gives negligible gain at no extra cost (retrieval time is dominated by encoding, not scoring).

---

## Group S — Sum-TopK Aggregation

Replaces `max_chunk` (single best window) with `sum_top3` / `sum_top5` (sum of K best windows per candidate).

| Label | Model | Agg | MicroF1 | MAP | MRR |
|---|---|---|---|---|---|
| I_aug_n7_a08 (ref) | MiniLM | max_chunk | 0.4291 | 0.5663 | 0.7752 |
| S_sum_top3 | MiniLM | sum_top3 | 0.4285 | 0.5731 | 0.7891 |
| S_sum_top5 | MiniLM | sum_top5 | 0.4262 | 0.5704 | 0.7909 |
| M_inlegalbert (ref) | InLegalBERT | max_chunk | 0.4246 | 0.5592 | 0.7875 |
| S_inlegal_sum_top3 | InLegalBERT | sum_top3 | 0.4309 | 0.5736 | 0.7914 |

**Findings:**
- `sum_top3` slightly **hurts** MiniLM MicroF1 (42.85 vs 42.91) but **improves** MAP and MRR — indicating it finds relevant document better overall but not at the critical K cut-off.
- For InLegalBERT, sum_top3 helps F1 by 0.63 points (43.09 vs 42.46).
- The hypothesis that distributed citation evidence across paragraphs would be better captured by sum aggregation is partially supported — MAP/MRR improve — but the F1 gain is marginal.

---

## Group CO — Citation Overlap Signal

Adds a third score component: exact Indian SC citation recall (`citation_beta` × |Q_cites ∩ C_cites| / |Q_cites|).

### TF-IDF only + Citation (no model)

| Label | α (TF-IDF) | β (citation) | model weight | MicroF1 | MAP |
|---|---|---|---|---|---|
| CO_tfidf_b05 | 0.95 | 0.05 | 0.00 | 0.4458 | 0.5981 |
| CO_tfidf_b10 | 0.90 | 0.10 | 0.00 | 0.4446 | 0.5983 |
| CO_tfidf_b20 | 0.80 | 0.20 | 0.00 | 0.4440 | 0.5964 |

Pure TF-IDF (augmented n=4) augmented with citation signal.  
Note: augmented n=4 alone ≈ 0.4409 (comparable to R_log_n4).  
**CO_tfidf_b05 at 0.4458** is a meaningful gain of ~+0.49pp over augmented n=4, confirming that exact citation matching adds signal. However it remains below the best TF-IDF (R_aug_n7 = 0.4507) which simply uses a higher n-gram order.

### 3-Way Hybrid (TF-IDF + Citation + Model)

| Label | Model | α | β | model weight | MicroF1 | MAP |
|---|---|---|---|---|---|---|
| CO_minilm_b10 | MiniLM | 0.75 | 0.10 | 0.15 | 0.4321 | 0.5752 |
| CO_inlegal_b10 | InLegalBERT | 0.75 | 0.10 | 0.15 | 0.4301 | 0.5693 |
| CO_ce_b10 | CE-MiniLM | 0.55 | 0.10 | 0.35 | **0.2179** | 0.2834 |

**CO_ce_b10 (0.2179)** is catastrophically bad. Root cause: the cross-encoder was applied with `char_limit=2048` (a fix applied during the run to prevent OOM). For Indian SC judgments that can exceed 50,000 characters, only the first ~500 tokens of each document reached the model. The intro/preamble of judgments contains no substantive legal reasoning — the model scored all documents near-identically, giving effectively random scores. Combined with a high model weight (0.35) at alpha=0.55, this destroyed the TF-IDF signal.

---

## Group T — Pure Transformer (No TF-IDF)

`skip_tfidf=True`: all 1727 candidates scored, no TF-IDF shortlisting.

| Model | MicroF1 | MAP | MRR | Elapsed (s) |
|---|---|---|---|---|
| T_inlegalbert | 0.2890 | 0.3325 | 0.6043 | 1179 |
| T_incaselaw | 0.2927 | 0.3528 | 0.6233 | 1267 |
| T_mpnet | 0.3277 | 0.3928 | 0.6393 | 1220 |
| **T_minilm** | **0.3457** | 0.4048 | 0.6403 | 254 |

**Findings:**
- All pure-transformer systems fail catastrophically vs TF-IDF (−10 to −16 points).
- Domain-specific legal models (InLegalBERT, InCaseLawBERT) perform **worst** — they were trained on Indian SC text but as masked language models, not retrieval bi-encoders. Their embeddings do not encode citation/precedent proximity.
- General-purpose MiniLM marginally beats all legal models — highlighting that fine-tuning for semantic similarity (SNLI/NLI data) transfers better than domain LM pretraining for retrieval.
- MRR scores (60–64%) suggest models often rank a relevant document in the top 5, but MicroF1 suffers because they fail to rank the full relevant set comprehensively.

---

## Overall Results Summary

All experiments sorted by MicroF1:

| Rank | Label | Group | MicroF1 | MAP | MRR | NDCG@10 | Model |
|---|---|---|---|---|---|---|---|
| 1 | **R_aug_n7** | R | **0.4507** | 0.5943 | 0.7935 | 0.6824 | — |
| 2 | R_aug_n6 | R | 0.4496 | 0.5977 | 0.7957 | 0.6862 | — |
| 3 | R_log_n9 | R | 0.4496 | 0.5970 | 0.8014 | 0.6877 | — |
| 4 | R_aug_n9 | R | 0.4501 | 0.5899 | 0.7851 | 0.6793 | — |
| 5 | R_log_n7 | R | 0.4490 | 0.5961 | 0.8015 | 0.6879 | — |
| 6 | R_bin_n7 | R | 0.4490 | 0.5910 | 0.7928 | 0.6811 | — |
| 7 | A_aug_n7_a095 | A | 0.4468 | 0.5948 | 0.7949 | 0.6836 | MiniLM |
| 8 | CO_tfidf_b05 | CO | 0.4458 | 0.5981 | 0.7938 | 0.6839 | — |
| 9 | CO_tfidf_b10 | CO | 0.4446 | 0.5983 | 0.7964 | 0.6834 | — |
| 10 | R_log_n4 | R | 0.4409 | 0.5906 | 0.7960 | 0.6836 | — |
| 11 | M_roberta_a09 | M | 0.4435 | 0.5904 | 0.7921 | 0.6801 | RoBERTa-large |
| 12 | M_inlegalbert_a09 | M | 0.4418 | 0.5824 | 0.7919 | 0.6728 | InLegalBERT |
| 13 | I_aug_n7_a09 | I | 0.4385 | 0.5873 | 0.7907 | 0.6711 | MiniLM |
| 14 | M_incaselaw_a09 | M | 0.4362 | 0.5828 | 0.7912 | 0.6729 | InCaseLawBERT |
| 15 | A_aug_n7_a085 | A | 0.4327 | 0.5785 | 0.7897 | 0.6653 | MiniLM |
| 16 | M_roberta | M | 0.4323 | 0.5742 | 0.7875 | 0.6654 | RoBERTa-large |
| 17 | CO_tfidf_b20 | CO | 0.4440 | 0.5964 | 0.7930 | 0.6811 | — |
| 18 | I_aug_n9_a08 | I | 0.4321 | 0.5685 | 0.7744 | 0.6559 | MiniLM |
| 19 | CO_minilm_b10 | CO | 0.4321 | 0.5752 | 0.7854 | 0.6613 | MiniLM |
| 20 | S_inlegal_sum_top3 | S | 0.4309 | 0.5736 | 0.7914 | 0.6663 | InLegalBERT |
| 21 | TN_top300 | TN | 0.4309 | 0.5682 | 0.7784 | 0.6552 | MiniLM |
| 22 | CO_inlegal_b10 | CO | 0.4301 | 0.5693 | 0.7873 | 0.6580 | InLegalBERT |
| 23 | I_aug_n7_a08 | I | 0.4291 | 0.5663 | 0.7752 | 0.6507 | MiniLM |
| 24 | S_sum_top3 | S | 0.4285 | 0.5731 | 0.7891 | 0.6636 | MiniLM |
| 25 | G_aug_n7_a08_meanchunk | G | 0.4273 | 0.5655 | 0.7760 | 0.6543 | MiniLM |
| 26 | TN_top50 | TN | 0.4273 | 0.5646 | 0.7750 | 0.6502 | MiniLM |
| 27 | TN_top100 | TN | 0.4279 | 0.5667 | 0.7756 | 0.6512 | MiniLM |
| 28 | TN_top150 | TN | 0.4279 | 0.5670 | 0.7757 | 0.6504 | MiniLM |
| 29 | S_sum_top5 | S | 0.4262 | 0.5704 | 0.7909 | 0.6594 | MiniLM |
| 30 | M_legalbert | M | 0.4247 | 0.5611 | 0.7764 | 0.6525 | LegalBERT |
| 31 | M_inlegalbert | M | 0.4246 | 0.5592 | 0.7875 | 0.6475 | InLegalBERT |
| 32 | M_mpnet | M | 0.4244 | 0.5738 | 0.7949 | 0.6613 | MPNet |
| 33 | C_chunk512_str128 | C | 0.4257 | 0.5581 | 0.7655 | 0.6437 | MiniLM |
| 34 | C_chunk512_str256 | C | 0.4256 | 0.5552 | 0.7722 | 0.6424 | MiniLM |
| 35 | A_aug_n7_a070 | A | 0.4179 | 0.5392 | 0.7581 | 0.6251 | MiniLM |
| 36 | M_bert_base | M | 0.4173 | 0.5338 | 0.7559 | 0.6276 | BERT-base |
| 37 | M_multiqa | M | 0.4190 | 0.5529 | 0.7667 | 0.6385 | multi-qa-MiniLM |
| 38 | M_incaselaw | M | 0.4155 | 0.5620 | 0.7887 | 0.6476 | InCaseLawBERT |
| 39 | I_log_n4_a06 | I | 0.4046 | 0.5279 | 0.7451 | 0.6119 | MiniLM |
| 40 | C_chunk128_str64 | C | 0.4095 | 0.5379 | 0.7561 | 0.6242 | MiniLM |
| 41 | I_aug_n7_a06 | I | 0.3996 | 0.5174 | 0.7382 | 0.6006 | MiniLM |
| 42 | M_sbert_para | M | 0.3970 | 0.5249 | 0.7665 | 0.6128 | paraphrase-MiniLM |
| 43 | A_aug_n7_a050 | A | 0.3887 | 0.4995 | 0.7251 | 0.5851 | MiniLM |
| 44 | T_minilm | T | 0.3457 | 0.4048 | 0.6403 | 0.4935 | MiniLM |
| 45 | T_mpnet | T | 0.3277 | 0.3928 | 0.6393 | 0.4834 | MPNet |
| 46 | T_incaselaw | T | 0.2927 | 0.3528 | 0.6233 | 0.4399 | InCaseLawBERT |
| 47 | T_inlegalbert | T | 0.2890 | 0.3325 | 0.6043 | 0.4254 | InLegalBERT |
| 48 | CO_ce_b10 | CO | 0.2179 | 0.2834 | 0.5344 | 0.3381 | CE-MiniLM |

---

## Key Findings

### 1. No hybrid method beats pure TF-IDF on MicroF1

The best result in the entire sweep is **R_aug_n7 = 0.4507** — a pure TF-IDF baseline with augmented TF scheme and n-gram=7. Every transformer-augmented method performs below this ceiling.

The best hybrid MicroF1 is **A_aug_n7_a095 = 0.4468** (α=0.95, only 5% transformer weight) — 0.39 points below.

### 2. Alpha is the dominant hyperparameter

Across all hybrid experiments, increasing α (TF-IDF weight) monotonically improves MicroF1. This confirms the transformer signal is net-negative noise for this task. The optimal TF-IDF weight for any transformer model is effectively 1.0.

### 3. General transformers outperform legal-domain models

Despite InLegalBERT and InCaseLawBERT being pre-trained on Indian SC judgments, they do not outperform general-purpose models (RoBERTa-large, MiniLM) when used as bi-encoders. The reason: these models are BERT-style masked language models without retrieval fine-tuning. Their embeddings are not optimised for similarity ranking.

### 4. Pure transformer retrieval fails badly (−10 to −16pp)

Group T confirms the pre-experiment prediction: without TF-IDF shortlisting, transformer bi-encoders retrieve largely irrelevant documents. On this legal task:
- Legal terminology (citation strings, section references) is critical.
- A single law citation like "AIR 1973 SC 1461" carries more retrieval signal than any semantic proximity that words like "held", "Bench", "appellant" carry.
- Transformers cannot distinguish "AIR 1973 SC 1461" from "AIR 1974 SC 1461" in embedding space; TF-IDF treats them as entirely different n-grams.

### 5. Citation overlap adds marginal signal

`CO_tfidf_b05` (0.4458) improves over augmented n=4 baseline (~0.4409) by +0.49pp. However it remains below augmented n=7 (0.4507). The citation overlap signal is partially redundant with high-n-gram TF-IDF: a shared citation string (e.g. "AIR 1973 SC 1461") already contributes strongly as a 5+ gram TF-IDF match.



---

## Recommendations

| Priority | Action | Expected gain |
|---|---|---|
| 1 | **Use R_aug_n7 as final TF-IDF model** (augmented, n=7) | Proven 0.4507 |
| 2 | **Increase n-gram to 7–9 in all hybrid experiments** (currently n=4 default) | May push CO_tfidf_b05 to ~0.4550 |
| 3 | **Try α=0.95 with RoBERTa-large** (best hybrid model at α=0.9) | Estimated 0.4470–0.4490 |
| 4 | **Re-run CE experiments with proper chunked cross-encoder** (score per chunk, take max) | Unknown; could recoup CE |
| 5 | **InLegalBERT at α=0.95** is the best speed-accuracy tradeoff vs RoBERTa-large | ~0.4450 estimate, 3× faster |

---

## Configuration Recommendations (For Final System)

```
TF-IDF: scheme=augmented, ngram=7, min_df=2, max_df=0.95, top_n=200
Hybrid:  alpha=0.95, model=MiniLM-L6-v2 (or skip transformer entirely)
Chunks:  chunk_tokens=256, chunk_stride=128, agg=max_chunk
Citation: beta=0.05, alpha=0.90 (if citation overlap is included)
```

**Best reproducible result: pure TF-IDF, R_aug_n7 = 0.4507 MicroF1**
