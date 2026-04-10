# Legal Statute Identification (LSI) — Model Evaluation Report

---

## 1. Introduction

This report documents the evaluation of a **pretrained Legal Statute Identification (LSI) model** on sample Indian legal case texts. The goal of the LSI task is to automatically identify the relevant sections of the Indian Penal Code (IPC) that apply to a given case description — a practically important problem in legal AI that reduces the manual effort required to tag case documents with applicable statutes.

All predictions in this report are produced by a **pretrained model**. No fine-tuning or retraining was performed as part of this evaluation. The model weights (`pytorch_model.bin`) were loaded directly and used in inference mode.

---

## 2. Task Description

Given the text of an Indian legal judgment or case summary, the LSI task requires predicting a **subset of IPC sections** from a fixed vocabulary of 100 labels (e.g., *Section 302 — Murder*, *Section 376 — Rape*, *Section 420 — Cheating*). It is a **multi-label classification** problem: a single case may involve multiple offences and therefore multiple applicable sections.

The label vocabulary (`label_vocab.json`) contains 100 IPC sections ranging from procedural provisions (Section 34 — Common Intention) to substantive offences (Section 302 — Murder, Section 420 — Cheating, etc.).

---

## 3. Model Architecture

The pretrained model is a **Hierarchical BERT (HierBERT)** architecture designed specifically for long legal documents that exceed the standard 512-token limit of transformer models.

### 3.1 Base Encoder — InLegalBERT

The sentence-level encoder is **`law-ai/InLegalBERT`**, a BERT model pretrained on a large corpus of Indian legal text. Using a domain-specific pretrained model (rather than general-purpose BERT) is critical for legal NLP, as legal language contains highly specialised vocabulary, Latin phrases, and citation patterns not well-represented in general corpora.

### 3.2 Hierarchical Document Encoding

Legal judgments are long documents (often thousands of words) that cannot be processed in a single BERT pass. HierBERT addresses this with a two-level encoding strategy:

**Level 1 — Sentence Encoding:** Each sentence in the document is independently encoded by InLegalBERT. The `[CLS]` token representation is extracted as the sentence embedding. To manage GPU memory, sentences are processed in fragments of 32 at a time.

**Level 2 — Document Encoding:** The sequence of sentence embeddings is passed through a **Bidirectional LSTM with Attention (LstmAttn)**. This layer captures inter-sentence dependencies and produces a single document-level representation via a weighted attention sum over sentence outputs.

### 3.3 Classification Head

The document-level hidden state is passed through a **linear classification layer** projecting to the 100-dimensional label space. A **sigmoid activation** (not softmax) is applied to produce independent probabilities for each label, appropriate for multi-label classification. A **BCE with Logits Loss** weighted by inverse label frequency was used during training to handle class imbalance.

### 3.4 Architecture Summary

```
Input Document (sentences)
        │
        ▼
┌─────────────────────────┐
│  InLegalBERT (per sent) │  ← Sentence-level encoder
│  → [CLS] embedding      │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Bi-LSTM + Attention    │  ← Document-level encoder
│  → document vector      │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Linear(768 → 100)      │  ← Classification head
│  → Sigmoid              │
└─────────────────────────┘
        │
        ▼
  Multi-label predictions
  (threshold = 0.5)
```

---

## 4. Inference Setup

| Parameter | Value |
|---|---|
| Base model | `law-ai/InLegalBERT` |
| Pretrained weights | `pytorch_model.bin` |
| Device | CPU |
| Prediction threshold | 0.5 |
| Max sentences per doc | 128 |
| Max tokens per sentence | 128 |
| Fragment size (BERT batch) | 32 |

Predictions were generated using `test.py`, a standalone inference script that replicates the model architecture from the original `lsi.py` training code and loads the pretrained weights in evaluation mode (`model.eval()`), with gradients disabled.

---

## 5. Test Samples

Four synthetic case summaries were constructed in the judicial register of Indian court judgments (13 sentences each), covering distinct categories of IPC offences:

| Sample ID | Case Type | Expected Key Sections |
|---|---|---|
| `sample_murder_dowry` | Dowry death / murder | 302, 304B, 498A, 34 |
| `sample_robbery_hurt` | Robbery with grievous hurt | 392, 394, 397, 120B |
| `sample_cheating_forgery` | Bank fraud and forgery | 420, 467, 468, 471, 415 |
| `sample_kidnapping_assault` | Kidnapping and rape of minor | 363, 366, 366A, 376, 376(2) |

---

## 6. Results

### 6.1 Sample 1 — Murder and Dowry Death (`sample_murder_dowry`)

**Predicted statutes (confidence ≥ 0.5):**

| Section | Description | Confidence |
|---|---|---|
| Section 304B | Dowry death | 0.926 |
| Section 498A | Cruelty by husband / relatives | 0.824 |
| Section 302 | Murder | 0.508 |

**Top-5 predictions:**

| Rank | Section | Confidence |
|---|---|---|
| 1 | Section 304B | 0.926 |
| 2 | Section 498A | 0.824 |
| 3 | Section 302 | 0.508 |
| 4 | Section 34 | 0.347 |
| 5 | Section 304 | 0.232 |

**Analysis:** The model correctly identifies the core offences. Section 304B (dowry death) and 498A (cruelty) are predicted with very high confidence, which is appropriate given the explicit mention of dowry harassment and suspicious death within three years of marriage. Section 302 is predicted at the threshold boundary (0.508), reflecting the realistic ambiguity between murder and dowry death in such cases. Section 34 (common intention) appears in the top-5 at 0.347 — just below threshold — consistent with the multi-accused scenario described. The omission of Section 34 as a confirmed prediction is the only minor gap.

---

### 6.2 Sample 2 — Robbery with Hurt (`sample_robbery_hurt`)

**Predicted statutes (confidence ≥ 0.5):**

| Section | Description | Confidence |
|---|---|---|
| Section 394 | Hurt during robbery | 0.988 |
| Section 397 | Robbery with deadly weapon | 0.943 |
| Section 392 | Robbery | 0.806 |

**Top-5 predictions:**

| Rank | Section | Confidence |
|---|---|---|
| 1 | Section 394 | 0.988 |
| 2 | Section 397 | 0.943 |
| 3 | Section 392 | 0.806 |
| 4 | Section 34 | 0.450 |
| 5 | Section 302 | 0.324 |

**Analysis:** This is the strongest result across all samples. All three primary robbery-related sections are predicted with very high confidence. The hierarchical specificity is correctly captured — 392 (robbery) < 394 (robbery + hurt) < 397 (robbery + deadly weapon) — all three being simultaneously applicable as charged. Section 34 (common intention) at 0.450 narrowly misses the threshold despite the multi-accused framing, and Section 120B (criminal conspiracy) was not predicted, likely because conspiracy charges are less frequent in training data for robbery cases.

---

### 6.3 Sample 3 — Cheating and Forgery (`sample_cheating_forgery`)

**Predicted statutes (confidence ≥ 0.5):**

| Section | Description | Confidence |
|---|---|---|
| Section 420 | Cheating | 0.890 |
| Section 468 | Forgery for cheating | 0.853 |
| Section 467 | Forgery of valuable security | 0.810 |
| Section 471 | Using forged document as genuine | 0.805 |
| Section 415 | Cheating (definition) | 0.622 |

**Top-5 predictions:**

| Rank | Section | Confidence |
|---|---|---|
| 1 | Section 420 | 0.890 |
| 2 | Section 468 | 0.853 |
| 3 | Section 467 | 0.810 |
| 4 | Section 471 | 0.805 |
| 5 | Section 415 | 0.622 |

**Analysis:** A perfect prediction — all five expected sections are identified with high confidence. The model cleanly distinguishes the cheating cluster (415, 420) from the forgery cluster (467, 468, 471), which are conceptually related but legally distinct. This suggests InLegalBERT has learned strong representations for financial fraud terminology (bank documents, forged instruments, misappropriation). Section 415 (the definitional provision for cheating) being predicted alongside Section 420 (the punishing provision) is particularly noteworthy.

---

### 6.4 Sample 4 — Kidnapping and Sexual Assault (`sample_kidnapping_assault`)

**Predicted statutes (confidence ≥ 0.5):**

| Section | Description | Confidence |
|---|---|---|
| Section 366A | Procuring minor girl | 0.930 |
| Section 363 | Kidnapping | 0.880 |
| Section 366 | Kidnapping for marriage/illicit intercourse | 0.713 |
| Section 376 | Rape | 0.676 |

**Top-5 predictions:**

| Rank | Section | Confidence |
|---|---|---|
| 1 | Section 366A | 0.930 |
| 2 | Section 363 | 0.880 |
| 3 | Section 366 | 0.713 |
| 4 | Section 376 | 0.676 |
| 5 | Section 376(2) | 0.490 |

**Analysis:** The model correctly predicts 4 of the 5 expected sections. Section 376(2) (aggravated rape of a minor) narrowly misses the threshold at 0.490, appearing just below it in 5th place. The distinction between Section 376 (rape) and Section 376(2) (aggravated rape) is subtle and depends on age-related context that can be ambiguous at the sentence level. Section 506 (criminal intimidation) was not predicted, which may reflect that threatening language in the context of a kidnapping case is less frequently tagged with 506 in training data.

---

## 7. Overall Summary

| Sample | Expected Sections | Correctly Predicted | Missed | Extra |
|---|---|---|---|---|
| `sample_murder_dowry` | 302, 304B, 498A, 34 | 302, 304B, 498A | 34 | — |
| `sample_robbery_hurt` | 392, 394, 397, 120B | 392, 394, 397 | 120B, 34 | — |
| `sample_cheating_forgery` | 420, 467, 468, 471, 415 | 420, 467, 468, 471, 415 | — | — |
| `sample_kidnapping_assault` | 363, 366, 366A, 376, 376(2) | 363, 366, 366A, 376 | 376(2), 506 | — |

The pretrained LSI model performs strongly across all four test cases with **no false positives** and only a small number of missed sections, most of which appear just below the 0.5 threshold in the top-5 rankings. The model demonstrates particularly robust performance on the cheating/forgery case (5/5 correct) and the robbery case (3/3 core sections correct at very high confidence).

---

## 8. Observations and Limitations

**Strengths of the pretrained model:**
- Domain-adapted BERT encoder (InLegalBERT) effectively captures Indian legal language.
- Hierarchical architecture handles multi-sentence case descriptions without truncation.
- No false positives observed across all four test samples.
- Confidence scores are well-calibrated: missed sections tend to appear at 0.35–0.49, close to the threshold.

**Limitations:**
- The model is run on **CPU**, which significantly increases inference time. GPU inference is recommended for large-scale evaluation.
- Ancillary procedural sections (Section 34 — common intention, Section 120B — conspiracy) are consistently under-predicted, possibly due to class imbalance in training data.
- The prediction threshold of 0.5 is fixed; a lower threshold (e.g., 0.4) would recover several near-miss predictions at the cost of potential false positives.
- Test samples are synthetic; performance on real case documents from the ILSI benchmark may differ.

---

## 9. Running the Evaluation

### Dependencies
```bash
pip install torch transformers datasets scikit-learn
```

### Inference on custom text
```bash
python test.py
```
Edit the `SAMPLES` dictionary in `test.py` to provide custom case text as a list of sentences per document.

### Benchmark evaluation
```bash
# After generating ilsi-test-pred.json via test.py:
python Eval/lsi-eval.py
```
This computes the **macro-F1 score** against the gold standard (`Eval/ilsi-test-gold.json`).

---

## 10. References

- **Dataset & Task:** ILSI (Indian Legal Statute Identification), introduced as part of the FIRE/AILA shared tasks on legal AI.
- **Base Model:** `law-ai/InLegalBERT` — BERT pretrained on Indian legal corpora, available on HuggingFace.
- **Architecture:** Hierarchical BERT with Bidirectional LSTM Attention, as described in `lsi.py`.
- **Evaluation Metric:** Macro-averaged F1 score across all 100 IPC section labels.
