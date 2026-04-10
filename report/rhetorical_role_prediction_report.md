# Rhetorical Role Prediction in Legal Judgments

## Overview

This report documents an experiment in **automated rhetorical role labeling** of Indian legal judgments using a pretrained deep learning model. The task involves classifying each sentence of a court judgment into one of several functional roles — such as stating facts, citing precedents, or declaring a ruling — to help structure and understand legal texts automatically.

---

## Task Description

Legal judgments are long, complex documents that mix multiple rhetorical functions within a single document. **Rhetorical Role Labeling (RRL)** is the NLP task of assigning a functional label to each sentence, answering the question: *"What role does this sentence play in the judgment?"*

The labels used in this experiment include:

| Label | Meaning |
|---|---|
| **Fact** | Factual background or procedural history of the case |
| **Precedent** | Reference to or discussion of prior case law |
| **RulingByLowerCourt** | A decision or order made by a lower court |

---

## System Architecture

The model used is a **Multi-Task Learning (MTL) Classifier** combining BERT embeddings with stacked BiLSTM layers and a CRF decoder. The pipeline has three stages:

### Stage 1 — Sentence Embedding (BERT)
Each sentence in the judgment is encoded using `bert-base-uncased`. The `[CLS]` token embedding (768-dimensional) serves as the sentence representation. For the binary/shift module, a **context-enriched embedding** is formed by concatenating the previous, current, and next sentence embeddings, yielding a 2304-dimensional vector per sentence.

### Stage 2 — Sequential Modeling (BiLSTM × 2)
Two bidirectional LSTM modules process the sequences of sentence embeddings:

- **`LSTM_Emitter_Binary`** — Takes the 2304-dim context embeddings and learns a coarse binary/shift signal across the document sequence. It outputs both tag emissions and hidden states.
- **`LSTM_Emitter`** — Takes the 768-dim BERT embeddings *combined* with the hidden states from the binary module. This cross-module fusion allows the main classifier to benefit from the contextual shift signals learned by the first LSTM.

### Stage 3 — CRF Decoding
Two **Conditional Random Field (CRF)** decoders are applied — one for the main rhetorical role sequence, one for the binary sequence. The CRF models label transitions across sentences (e.g., a Fact is more likely to be followed by another Fact than to jump directly to a ruling), producing globally coherent label sequences via **Viterbi decoding**.

### Architecture Diagram

```
Input Sentences
      │
      ▼
[BERT Encoder] ─────────────────────────────────────────────────┐
      │                                                          │
  768-dim CLS embeddings                           2304-dim context embeddings
      │                                                          │
      │                                               ┌──────────────────┐
      │                                               │ LSTM_Emitter_    │
      │                                               │ Binary (BiLSTM)  │
      │                                               └────────┬─────────┘
      │                                                        │
      │                              hidden states ◄───────────┘
      │                                   │
      ▼                                   ▼
┌──────────────────────────────────────────────┐
│         LSTM_Emitter (BiLSTM)                │
│  concat(BERT_emb, hidden_binary) → 1536-dim  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
               [CRF Decoder]
                       │
                       ▼
            Predicted Role Labels
```

---

## Inference Pipeline

The inference script processes a full judgment end-to-end:

1. **Sentence Tokenization** — The raw judgment text is split into sentences using NLTK's `sent_tokenize`, filtering out very short fragments (< 15 characters).
2. **BERT Encoding** — Each sentence is tokenized and passed through `bert-base-uncased` to extract its `[CLS]` embedding.
3. **Context Embedding** — For each sentence `i`, a 2304-dim vector is formed by concatenating embeddings of sentences `i-1`, `i`, and `i+1` (with zero-padding at boundaries).
4. **Model Forward Pass** — Both embedding sequences are batched and passed through the MTL classifier.
5. **Label Decoding** — The predicted integer indices are mapped back to human-readable role labels via `idx2tag`.

---

## Results

The model was applied to a 1963 Indian Supreme Court judgment (*Petitioner vs. Respondent & Others*, Civil Appeal Nos. 30 and 31 of 1963) concerning an election dispute under the Representation of the People Act, 1951. A total of **185 sentences** were processed and labeled.

### Sample Predictions

The table below shows representative examples of each predicted rhetorical role, drawn from the full output:

| # | Sentence | Predicted Role |
|---|----------|----------------|
| 3 | *"The validity of the election of the appellant to [ORG] at the third general elections held in the month of February, 1962, was challenged by two of the electors of the constituency..."* | **Fact** |
| 64 | *"The foundation of the argument is that there has been a non-compliance with the provisions of s. 82."* | **Precedent** |
| 11 | *"Appeals by special leave from the judgment and order dated August 31, 1962, of [ORG] in D. [NAME] Civil Writ Petitions Nos. 376 and 377 of 1962."* | **RulingByLowerCourt** |

### Label Distribution Observations

Across the 185 predicted sentences, three dominant patterns emerge:

- **Fact** labels appear primarily in the early sections of the judgment (sentences 3–60), covering procedural history, statutory provisions cited, and the grounds of objection raised by the appellant.
- **Precedent** labels dominate the analytical middle section (sentences 64–130), where the court reasons through each preliminary objection, discusses prior decisions, and applies legal standards.
- **RulingByLowerCourt** labels appear throughout, especially in the preamble (sentences 1–30) and conclusion (sentences 138–185), capturing procedural context and the lower court's orders.

---

## Key Design Decisions

**Why MTL (Multi-Task Learning)?**
Training a secondary binary/shift task alongside the main labeling task acts as a regularizer. The binary module learns document-level structure (e.g., detecting major topic shifts), which in turn improves the main classifier's ability to track rhetorical flow within a judgment.

**Why CRF over a simple softmax?**
A softmax output layer predicts each sentence independently. A CRF models the entire label sequence jointly, penalizing unlikely transitions (e.g., a `RulingByLowerCourt` in the middle of a fact-narration block) and producing more coherent outputs across the document.

**Why context-enriched embeddings for the binary module?**
Rhetorical shifts are inherently relational — a sentence signals a shift only in relation to its neighbors. Feeding `[prev, curr, next]` embeddings to the shift module gives it the information needed to detect these boundaries without requiring the main model to do so implicitly.

---

## Conclusion

The pretrained MTL model successfully assigns rhetorical roles to sentences across a full Indian Supreme Court judgment. The sequential CRF decoder, combined with BERT's contextual sentence representations and the cross-module fusion of binary shift signals, produces label sequences that are both locally accurate and globally coherent. This kind of automated annotation can support downstream legal AI applications such as judgment summarization, argument mining, and case retrieval.
