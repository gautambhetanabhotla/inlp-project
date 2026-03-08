# Experiment Analysis Results

## Experiment: `pure_bm25_raw.py`

### Methodology

**Objective:** Evaluate baseline Information Retrieval algorithms (BM25 and TF-IDF) on raw text documents without applying advanced preprocessing like regex stripping or lemmatization (i.e., preserving numbers and punctuation).
**Input:** Raw text formats of Prior Case Retrieval (PCR) queries and a pool of candidate documents.
**Pipeline:**

1. **BM25 Pipeline:** Texts are tokenized and processed using a standard `TfidfVectorizer` to extract term frequencies and document lengths. The relevance score for each query against all candidates is then computed using the classical BM25 probabilistic ranking formula (with $k_1$ and $b$ parameters).
2. **TF-IDF Pipeline:** Candidate texts are converted into TF-IDF vector representations utilizing both unigrams and bigrams. Query vectors are generated the same way.
**Comparison/Evaluation:** For each query, the algorithms return a ranked list of candidate documents based on either BM25 score or Cosine Similarity (TF-IDF). The top-K retrieved documents are then evaluated against a ground truth dataset across standard IR metrics like Precision, Recall, F1-scores, MAP, MRR, and NDCG at $K \in \{1, 5, 10, 20\}$.

---

- **Candidate Set Size:** 1727 documents
- **Query Set Size:** 237 documents

### 1. Pure Raw-Text BM25

- **Description:** A fast BM25 implementation identical to the original paper's logic, running directly on raw text.
- **Parameters:** `k1=1.6`, `b=0.7`

**Results:**

| K | Micro-F1 | Micro-P | Micro-R | Macro-F1 | MAP | MRR | NDCG |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.0733 | 0.3038 | 0.0417 | 0.0917 | 0.3077 | 0.3077 | 0.3077 |
| 5 | 0.1380 | 0.1696 | 0.1164 | 0.1379 | 0.1766 | 0.3769 | 0.2383 |
| 10 | 0.1333 | 0.1152 | 0.1581 | 0.1268 | 0.1503 | 0.3884 | 0.2294 |
| 20 | 0.1141 | 0.0778 | 0.2137 | 0.1074 | 0.1504 | 0.3946 | 0.2465 |

### 2. Pure Raw TF-IDF Cosine

- **Description:** TF-IDF vectorization with Cosine Similarity on raw text.
- **Parameters:** `ngram_range=(1,2)` (Unigram + Bigram)

**Results:**

| K | Micro-F1 | Micro-P | Micro-R | Macro-F1 | MAP | MRR | NDCG |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.1283 | 0.5316 | 0.0730 | 0.1938 | 0.5385 | 0.5385 | 0.5385 |
| 5 | 0.3269 | 0.4017 | 0.2756 | 0.3292 | 0.4381 | 0.6600 | 0.5306 |
| 10 | 0.3446 | 0.2979 | 0.4088 | 0.3212 | 0.4067 | 0.6676 | 0.5299 |
| 20 | 0.2808 | 0.1916 | 0.5258 | 0.2563 | 0.4089 | 0.6696 | 0.5495 |

---

## Experiment: `run_experiments.py` (Hyperparameter Tuning & Advanced Models)

This experiment extends the baseline models by adding hyperparameter tuning for BM25, trying Doc2Vec and Latent Semantic Analysis (LSA), as well as evaluating standard token preprocessing via SpaCy (lemmatization and stop-word removal).

### Methodology (Preprocessing & Tuning)

**Objective:** Evaluate classical IR techniques utilizing NLP pre-processing on queries and candidate documents to measure impact of hyperparameter tuning and dense representation.
**Input:** Tokenized formats of Prior Case Retrieval (PCR) queries and a pool of candidate documents.
**Pipeline:**

1. **Preprocessing:** Text is lowercased, non-alphabet characters are stripped, and SpaCy is used to extract lemmas while discarding stop words and short tokens.
2. **Models Evaluated:**
    - **BM25 (Standard, High K1, Low B):** Tests the impact of term saturation ($k_1$) and length normalization ($b$).
    - **TF-IDF (Unigram vs. Uni+Bigram):** Tests whether phrase extraction improves precision.
    - **LSA (100 vs. 200 components):** Tests dimensionality reduction over TF-IDF vectors using SVD.
**Comparison/Evaluation:** Same ground-truth evaluation metrics (Micro-F1, Precision, Recall, MAP, MRR, NDCG) across $K \in \{1, 5, 10, 20\}$ to establish the state-of-the-art among baseline unsupervised models.

### Results

#### BM25 Variants

##### BM25 Standard (`k1=1.5`, `b=0.75`)

| K | Micro-F1 | Micro-P | Micro-R | Macro-F1 | MAP | MRR | NDCG |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.0713 | 0.2954 | 0.0405 | 0.0903 | 0.2991 | 0.2991 | 0.2991 |
| 5 | 0.1298 | 0.1595 | 0.1094 | 0.1313 | 0.1728 | 0.3730 | 0.2333 |
| 10 | 0.1259 | 0.1089 | 0.1494 | 0.1201 | 0.1462 | 0.3840 | 0.2213 |
| 20 | 0.1082 | 0.0738 | 0.2027 | 0.1023 | 0.1474 | 0.3891 | 0.2388 |

##### BM25 High K1 (`k1=2.0`, `b=0.75`)

| K | Micro-F1 | Micro-P | Micro-R | Macro-F1 | MAP | MRR | NDCG |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.0815 | 0.3376 | 0.0463 | 0.1095 | 0.3419 | 0.3419 | 0.3419 |
| 5 | 0.1470 | 0.1806 | 0.1239 | 0.1456 | 0.1971 | 0.4117 | 0.2617 |
| 10 | 0.1455 | 0.1257 | 0.1726 | 0.1386 | 0.1695 | 0.4233 | 0.2523 |
| 20 | 0.1225 | 0.0835 | 0.2293 | 0.1154 | 0.1686 | 0.4278 | 0.2680 |

##### BM25 Low B (`k1=1.5`, `b=0.5`)

| K | Micro-F1 | Micro-P | Micro-R | Macro-F1 | MAP | MRR | NDCG |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.0458 | 0.1899 | 0.0261 | 0.0654 | 0.1923 | 0.1923 | 0.1923 |
| 5 | 0.0804 | 0.0987 | 0.0677 | 0.0815 | 0.1070 | 0.2378 | 0.1447 |
| 10 | 0.0805 | 0.0696 | 0.0955 | 0.0760 | 0.0919 | 0.2482 | 0.1403 |
| 20 | 0.0687 | 0.0468 | 0.1285 | 0.0647 | 0.0923 | 0.2542 | 0.1520 |

#### TF-IDF Variants

##### TF-IDF Unigram

| K | Micro-F1 | Micro-P | Micro-R | Macro-F1 | MAP | MRR | NDCG |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.1293 | 0.5359 | 0.0735 | 0.1904 | 0.5427 | 0.5427 | 0.5427 |
| 5 | 0.3043 | 0.3738 | 0.2565 | 0.3052 | 0.4123 | 0.6536 | 0.5032 |
| 10 | 0.3163 | 0.2734 | 0.3752 | 0.2953 | 0.3760 | 0.6622 | 0.4986 |
| 20 | 0.2653 | 0.1810 | 0.4968 | 0.2428 | 0.3789 | 0.6637 | 0.5203 |

##### TF-IDF Unigram + Bigram

| K | Micro-F1 | Micro-P | Micro-R | Macro-F1 | MAP | MRR | NDCG |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.1293 | 0.5359 | 0.0735 | 0.1947 | 0.5427 | 0.5427 | 0.5427 |
| 5 | 0.3152 | 0.3873 | 0.2658 | 0.3198 | 0.4315 | 0.6559 | 0.5219 |
| 10 | 0.3407 | 0.2945 | 0.4042 | 0.3183 | 0.4061 | 0.6647 | 0.5265 |
| 20 | 0.2814 | 0.1920 | 0.5269 | 0.2569 | 0.4077 | 0.6667 | 0.5479 |

#### LSA (Latent Semantic Analysis)

##### LSA Cosine 100 Components

| K | Micro-F1 | Micro-P | Micro-R | Macro-F1 | MAP | MRR | NDCG |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.1253 | 0.5190 | 0.0712 | 0.1788 | 0.5256 | 0.5256 | 0.5256 |
| 5 | 0.2981 | 0.3662 | 0.2513 | 0.3013 | 0.3987 | 0.6464 | 0.4907 |
| 10 | 0.3134 | 0.2709 | 0.3717 | 0.2921 | 0.3700 | 0.6545 | 0.4923 |
| 20 | 0.2666 | 0.1819 | 0.4991 | 0.2429 | 0.3741 | 0.6559 | 0.5165 |

##### LSA Cosine 200 Components

| K | Micro-F1 | Micro-P | Micro-R | Macro-F1 | MAP | MRR | NDCG |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.1385 | 0.5738 | 0.0787 | 0.2117 | 0.5812 | 0.5812 | 0.5812 |
| 5 | 0.3166 | 0.3890 | 0.2669 | 0.3222 | 0.4349 | 0.6843 | 0.5277 |
| 10 | 0.3295 | 0.2848 | 0.3909 | 0.3077 | 0.4016 | 0.6918 | 0.5252 |
| 20 | 0.2746 | 0.1873 | 0.5142 | 0.2512 | 0.4045 | 0.6928 | 0.5466 |

---

## Experiment: `nlp_sentence_filter_bm25.py` (Sentence-level Event Filtering + BM25)

This experiment evaluates a hybrid pipeline that combines linguistic NLP parsing (via SpaCy) with classical probabilistic retrieval (BM25). Instead of treating the candidate document as a single bag of words, we filter documents based on whether their individual *sentences* overlap meaningfully with the *concepts* extracted from the query.

### Experiment 3 Methodology

**Objective:** To improve precision by dynamically filtering out irrelevant noise from candidate documents before applying BM25 scoring. If a sentence in a candidate document does not share core conceptual overlap with the query, it is discarded.
**Input:** Raw text formats of Prior Case Retrieval (PCR) queries and a pool of candidate documents.
**Pipeline:**

1. **Concept Extraction (Query):** The query is segmented into sentences. SpaCy extracts key "Event Concepts" consisting purely of Nouns, Proper Nouns, and Verbs (lemmatized, excluding stopwords).
2. **Concept Extraction (Candidates):** Every candidate document is segmented into sentences, and the same "Event Concepts" are extracted for each sentence.
3. **Dynamic Filtering:** For each query, we iterate through the candidate pool. We retain a candidate's sentence *only if* its extracted concepts intersect with the query's total concepts by a predefined `overlap` threshold (e.g., $\ge 2$ matching concepts).
4. **Scoring:** A separate BM25 model is built for the dynamically filtered candidate corpus specific to each query, and the query is scored against this tailored corpus.
**Comparison/Evaluation:** Same standard metrics (Micro-F1, Precision, Recall, MAP, MRR, NDCG) across $K \in \{1, 5, 10, 20\}$ computed against the ground truth. We evaluate variations in the BM25 parameters ($k_1$, $b$) and the strictness of the overlap filtering threshold.

### Experiment 3 Results

#### NLP Filter BM25 Standard (`overlap>=2`, `k1=1.6`, `b=0.75`)

| K | Micro-F1 | Micro-P | Micro-R | Macro-F1 | MAP | MRR | NDCG |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.0774 | 0.3207 | 0.0440 | 0.1004 | 0.3248 | 0.3248 | 0.3248 |
| 5 | 0.1525 | 0.1873 | 0.1285 | 0.1538 | 0.2000 | 0.4046 | 0.2665 |
| 10 | 0.1464 | 0.1266 | 0.1737 | 0.1410 | 0.1710 | 0.4166 | 0.2561 |
| 20 | 0.1209 | 0.0825 | 0.2264 | 0.1136 | 0.1700 | 0.4210 | 0.2685 |

#### NLP Filter BM25 Strict (`overlap>=3`, `k1=1.6`, `b=0.75`)

| K | Micro-F1 | Micro-P | Micro-R | Macro-F1 | MAP | MRR | NDCG |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.0764 | 0.3165 | 0.0434 | 0.0997 | 0.3205 | 0.3205 | 0.3205 |
| 5 | 0.1573 | 0.1932 | 0.1326 | 0.1591 | 0.2030 | 0.4090 | 0.2714 |
| 10 | 0.1523 | 0.1316 | 0.1807 | 0.1467 | 0.1748 | 0.4188 | 0.2616 |
| 20 | 0.1249 | 0.0852 | 0.2339 | 0.1179 | 0.1740 | 0.4232 | 0.2754 |

#### NLP Filter BM25 Low B (`overlap>=2`, `k1=1.6`, `b=0.5`)

| K | Micro-F1 | Micro-P | Micro-R | Macro-F1 | MAP | MRR | NDCG |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.0489 | 0.2025 | 0.0278 | 0.0685 | 0.2051 | 0.2051 | 0.2051 |
| 5 | 0.0968 | 0.1190 | 0.0816 | 0.0962 | 0.1268 | 0.2660 | 0.1692 |
| 10 | 0.0879 | 0.0759 | 0.1042 | 0.0833 | 0.1046 | 0.2755 | 0.1570 |
| 20 | 0.0813 | 0.0555 | 0.1523 | 0.0763 | 0.1069 | 0.2828 | 0.1750 |

#### NLP Filter BM25 High K1 (`overlap>=2`, `k1=2.0`, `b=0.75`)

| K | Micro-F1 | Micro-P | Micro-R | Macro-F1 | MAP | MRR | NDCG |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.0804 | 0.3333 | 0.0457 | 0.1032 | 0.3376 | 0.3376 | 0.3376 |
| 5 | 0.1690 | 0.2076 | 0.1424 | 0.1718 | 0.2203 | 0.4374 | 0.2938 |
| 10 | 0.1645 | 0.1422 | 0.1951 | 0.1559 | 0.1901 | 0.4450 | 0.2809 |
| 20 | 0.1327 | 0.0905 | 0.2484 | 0.1247 | 0.1877 | 0.4506 | 0.2931 |
