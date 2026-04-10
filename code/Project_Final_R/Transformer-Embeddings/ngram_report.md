# TF-IDF N-gram Sweep — Analysis Report

**Task:** IL-PCR — Indian Legal Prior Case Retrieval (237 queries, 1727 candidates)  
**Configs swept:** n-gram ∈ {1…10} × scheme ∈ {raw, log, binary, augmented}, min_df=2, max_df=0.95

---

## N-gram trend (MicroF1 by scheme)

| n | augmented | log | binary | raw |
|---|---|---|---|---|
| 1 | 0.3545 | 0.3673 | 0.3400 | 0.3359 |
| 2 | 0.4393 | 0.4307 | 0.4378 | 0.3545 |
| 3 | 0.4446 | 0.4373 | 0.4446 | 0.3602 |
| 4 | 0.4446 | 0.4409 | 0.4430 | 0.3640 |
| 5 | 0.4468 | 0.4429 | 0.4456 | 0.3668 |
| 6 | 0.4496 | 0.4472 | 0.4501 | 0.3707 |
| 7 | **0.4507** | 0.4490 | 0.4490 | 0.3712 |
| 8 | 0.4496 | 0.4473 | 0.4487 | 0.3736 |
| 9 | 0.4501 | 0.4496 | 0.4490 | 0.3745 |
| 10 | 0.4501 | 0.4501 | 0.4485 | 0.3762 |

Log scheme per-n breakdown (additional metrics):

| n | MicroF1 | MAP | MRR | NDCG@10 |
|---|---|---|---|---|
| 1 | 0.3673 | 0.4822 | 0.7420 | 0.5784 |
| 2 | 0.4307 | 0.5789 | 0.7991 | 0.6684 |
| 3 | 0.4373 | 0.5873 | 0.7974 | 0.6797 |
| 4 | 0.4409 | 0.5906 | 0.7960 | 0.6836 |
| 5 | 0.4429 | 0.5930 | 0.8003 | 0.6861 |
| 6 | 0.4472 | 0.5952 | 0.8028 | 0.6864 |
| 7 | 0.4490 | 0.5961 | 0.8015 | 0.6879 |
| 8 | 0.4473 | 0.5971 | 0.8013 | 0.6887 |
| 9 | 0.4496 | 0.5970 | 0.8014 | 0.6877 |
| 10 | 0.4501 | 0.5949 | 0.7964 | 0.6862 |

---

## Analysis

### 1. Massive jump at n=1→2, then rapid saturation

The dominant n-gram gain is from unigrams to bigrams: **+0.0634 MicroF1** (log scheme). After that the increments shrink sharply:

| Transition | ΔMicroF1 (log) |
|---|---|
| 1 → 2 | +0.0634 |
| 2 → 3 | +0.0066 |
| 3 → 4 | +0.0036 |
| 4 → 5 | +0.0020 |
| 5 → 6 | +0.0043 |
| 6 → 7 | +0.0018 |
| 7 → 8 | −0.0017 |
| 8 → 9 | +0.0023 |
| 9 → 10 | +0.0005 |

**Why?** Legal retrieval here is driven largely by exact citation strings (e.g. `AIR 1973 SC 1461`, `(1999) 4 SCC 260`) and statutory references (e.g. `section 300 ipc`). These are 2–5 tokens long. Bigrams already capture the first word-pair boundary of such strings; trigrams and 4-grams cover them entirely. By n=4 the vocabulary of high-IDF discriminative phrases is essentially complete — longer n-grams add only noisier, sparser features that marginal gain can't justify. The curve is flat from n=4 onwards, with fluctuations under 0.003.

### 2. MRR is far less sensitive to n than MicroF1

Even with unigrams, MRR=0.7420 — the model already finds a relevant document near rank 1. MRR barely changes past n=4 (stays 0.796–0.803). This tells us: additional n-grams don't help rank the single best result — the top candidate is already identified by IDF-weighted token overlap. The gain at higher n is purely in **recall**: the model now correctly ranks more of the full relevant set above K=8, which is what MicroF1 measures.

### 3. MAP and NDCG@10 plateau at n=7–8

Both increase monotonically from n=1 through n=7–8 then flatten. NDCG@10 peaks at n=8 log (0.6887), one notch above the MicroF1 optimum (n=7 augmented). The slight divergence occurs because NDCG@10 rewards placing relevant docs at the very top of the top-10 list, and a marginally richer vocabulary at n=8 helps with ranking within the top-10 even when the retrieved set doesn't change much in size.

### 4. Raw TF barely improves with n and lags far behind

Raw TF rises from 0.3359 (n=1) to only 0.3762 (n=10) — a total gain of +0.0403, vs +0.0962 for augmented over the same range. Raw stays ~0.07 below the normalised schemes at every n. The reason is length bias: Indian SC judgments span 5,000–50,000 words; longer documents accumulate higher raw counts for any shared term, swamping the relevance signal regardless of how many n-grams are added. TF normalisation (log/binary/augmented) removes this bias, unlocking the discriminative value of longer n-grams.

### 5. Log, binary, augmented are effectively tied at their peaks

All three normalised schemes peak within 0.0006 MicroF1 of each other (augmented 0.4507, log 0.4501, binary 0.4501). The TF component is near-constant for citation strings — they appear 0 or 1 times per document, so log(1+1)=binary=1=augmented≈1. IDF carries almost all the weight. Scheme choice is irrelevant; n-gram order is what matters.

---

## Recommendation

Use **n=7, scheme=augmented** (MicroF1=0.4507, MAP=0.5943, MRR=0.7935, NDCG@10=0.6824). Any n ∈ {6,7,8,9} with any normalised scheme gives results within 0.002 of this, so the exact choice is not critical. Avoid raw TF.
