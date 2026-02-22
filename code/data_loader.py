"""
Data loading and preprocessing utilities for Indian legal documents.
Handles the IL-PCR and IL-TUR datasets plus rhetorical role annotations.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset

from config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentence tokeniser helpers
# ---------------------------------------------------------------------------

def sentence_tokenize(text: str) -> list[str]:
    """Split a legal document into sentences.

    Uses a regex-based approach tuned for Indian legal text (handles
    abbreviations like 'S.', 'Sec.', 'v.', 'Hon\'ble', etc.).
    """
    # Protect common legal abbreviations
    protected = text
    abbreviations = [
        r"(?<!\w)S\.", r"(?<!\w)Sec\.", r"(?<!\w)Art\.",
        r"(?<!\w)No\.", r"(?<!\w)Sr\.", r"(?<!\w)Jr\.",
        r"(?<!\w)Dr\.", r"(?<!\w)Mr\.", r"(?<!\w)Mrs\.",
        r"(?<!\w)Ms\.", r"(?<!\w)Hon\.", r"(?<!\w)vs\.",
        r"(?<!\w)v\.", r"(?<!\w)i\.e\.", r"(?<!\w)e\.g\.",
    ]
    for abbr in abbreviations:
        protected = re.sub(abbr, lambda m: m.group().replace(".", "<DOT>"), protected)

    # Split on sentence-ending punctuation followed by space + uppercase
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9("])', protected)

    sentences = [s.replace("<DOT>", ".").strip() for s in raw_sentences if s.strip()]
    return sentences


# ---------------------------------------------------------------------------
# Legal document representation
# ---------------------------------------------------------------------------

class LegalDocument:
    """Represents a single legal case document."""

    def __init__(
        self,
        doc_id: str,
        text: str,
        sentences: Optional[list[str]] = None,
        rhetorical_roles: Optional[list[str]] = None,
        cited_cases: Optional[list[str]] = None,
        cited_statutes: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ):
        self.doc_id = doc_id
        self.text = text
        self.sentences = sentences or sentence_tokenize(text)
        self.rhetorical_roles = rhetorical_roles  # per-sentence labels
        self.cited_cases = cited_cases or []
        self.cited_statutes = cited_statutes or []
        self.metadata = metadata or {}

    # ---------- segment helpers ----------

    def get_segments(self) -> dict[str, str]:
        """Return text grouped by rhetorical role."""
        if self.rhetorical_roles is None:
            return {"Full": self.text}
        segments: dict[str, list[str]] = {}
        for sent, role in zip(self.sentences, self.rhetorical_roles):
            segments.setdefault(role, []).append(sent)
        return {role: " ".join(sents) for role, sents in segments.items()}

    # ---------- serialisation ----------

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "sentences": self.sentences,
            "rhetorical_roles": self.rhetorical_roles,
            "cited_cases": self.cited_cases,
            "cited_statutes": self.cited_statutes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LegalDocument":
        return cls(**d)


# ---------------------------------------------------------------------------
# Citation & statute extraction  (regex-based)
# ---------------------------------------------------------------------------

# Indian case citation patterns
_CASE_CITATION_PATTERNS = [
    # AIR citations: AIR 2020 SC 123
    re.compile(r"AIR\s+\d{4}\s+\w+\s+\d+", re.IGNORECASE),
    # SCC citations: (2020) 5 SCC 123
    re.compile(r"\(\d{4}\)\s+\d+\s+SCC\s+\d+", re.IGNORECASE),
    # SCR citations: [2020] 3 SCR 456
    re.compile(r"\[\d{4}\]\s+\d+\s+SCR\s+\d+", re.IGNORECASE),
    # Generic "X v. Y" or "X vs. Y" patterns
    re.compile(
        r"[A-Z][a-zA-Z\s&.]+?\s+(?:v\.|vs\.?)\s+[A-Z][a-zA-Z\s&.]+?(?=,|\.|;|\)|\])",
        re.MULTILINE,
    ),
]

# Indian statute patterns
_STATUTE_PATTERNS = [
    # "Section 302 of the Indian Penal Code"
    re.compile(
        r"(?:Section|Sec\.?|S\.?)\s+\d+[A-Za-z]?\s+(?:of\s+(?:the\s+)?)?([A-Z][\w\s,()]+?(?:Act|Code|Ordinance|Rules|Regulation)(?:,?\s*\d{4})?)",
        re.IGNORECASE,
    ),
    # "Article 21 of the Constitution"
    re.compile(
        r"(?:Article|Art\.?)\s+\d+[A-Za-z]?\s+(?:of\s+(?:the\s+)?)?([A-Z][\w\s,()]+?(?:Constitution|Act))",
        re.IGNORECASE,
    ),
    # Direct act references: Indian Penal Code, 1860
    re.compile(
        r"(?:the\s+)?([A-Z][\w\s]+?(?:Act|Code|Ordinance|Rules|Regulation),?\s*\d{4})",
        re.IGNORECASE,
    ),
]


def extract_case_citations(text: str) -> list[str]:
    """Extract case citations from legal text."""
    citations = []
    for pattern in _CASE_CITATION_PATTERNS:
        for m in pattern.finditer(text):
            citation = m.group().strip().rstrip(",;.")
            if len(citation) > 5:
                citations.append(citation)
    return list(set(citations))


def extract_statute_references(text: str) -> list[str]:
    """Extract statute / act references from legal text."""
    statutes: list[str] = []
    for pattern in _STATUTE_PATTERNS:
        for m in pattern.finditer(text):
            statute = m.group().strip().rstrip(",;.")
            statutes.append(statute)
    # Normalise whitespace
    statutes = [re.sub(r"\s+", " ", s) for s in statutes]
    return list(set(statutes))


# ---------------------------------------------------------------------------
# Dataset loaders  — HuggingFace IL-TUR
# ---------------------------------------------------------------------------

ILTUR_DATASET = "Exploration-Lab/IL-TUR"
ILTUR_REVISION = "script"


def load_iltur_rr(
    split: str = "CL_train",
) -> list[dict]:
    """Load the Rhetorical Role (RR) task from HuggingFace IL-TUR.

    Available splits:
        CL_train, CL_dev, CL_test   (Competition Law)
        IT_train, IT_dev, IT_test    (Income Tax)

    Each example has:
        id:     str              — Case ID
        text:   list[str]        — sentences
        labels: list[int]        — per-sentence RR label (0-12)

    Returns a list of dicts, one per document.
    """
    ds = load_dataset(ILTUR_DATASET, "RR", split=split, revision=ILTUR_REVISION)
    data = []
    for row in ds:
        data.append({
            "id": row["id"],
            "sentences": row["text"],      # already a list of sentences
            "labels": row["labels"],       # list of ints (0-12)
        })
    logger.info("Loaded IL-TUR RR split '%s': %d documents", split, len(data))
    return data


def load_iltur_pcr(
    query_split: str = "train_queries",
    candidate_split: str = "train_candidates",
) -> tuple[list[LegalDocument], list[LegalDocument], dict]:
    """Load the Prior Case Retrieval (PCR) task from HuggingFace IL-TUR.

    Query splits:      train_queries, dev_queries, test_queries
    Candidate splits:  train_candidates, dev_candidates, test_candidates

    Returns (queries, candidates, qrels) where:
        queries     — list of LegalDocument (query cases)
        candidates  — list of LegalDocument (candidate corpus)
        qrels       — {query_id: [relevant_candidate_ids]}
    """
    q_ds = load_dataset(ILTUR_DATASET, "PCR", split=query_split, revision=ILTUR_REVISION)
    c_ds = load_dataset(ILTUR_DATASET, "PCR", split=candidate_split, revision=ILTUR_REVISION)

    queries: list[LegalDocument] = []
    qrels: dict[str, list[str]] = {}
    for row in q_ds:
        doc = _hf_pcr_to_document(row)
        queries.append(doc)
        rel = row.get("relevant_candidates")
        if rel:
            qrels[doc.doc_id] = [str(r) for r in rel]

    candidates: list[LegalDocument] = []
    for row in c_ds:
        candidates.append(_hf_pcr_to_document(row))

    logger.info(
        "Loaded IL-TUR PCR: %d queries (%s), %d candidates (%s), %d qrels",
        len(queries), query_split, len(candidates), candidate_split, len(qrels),
    )
    return queries, candidates, qrels


def _hf_pcr_to_document(row: dict) -> LegalDocument:
    """Convert a HuggingFace PCR row into a LegalDocument."""
    doc_id = str(row["id"])
    sentences = row["text"]  # list of strings
    text = " ".join(sentences)

    doc = LegalDocument(
        doc_id=doc_id,
        text=text,
        sentences=sentences,
    )
    # Auto-extract citations and statutes from text
    doc.cited_cases = extract_case_citations(text)
    doc.cited_statutes = extract_statute_references(text)
    return doc


# ---------------------------------------------------------------------------
# Torch dataset wrappers
# ---------------------------------------------------------------------------

class SentenceClassificationDataset(TorchDataset):
    """Dataset for training the RR sentence classifier."""

    def __init__(self, sentences: list[str], labels: list[int], tokenizer, max_length: int = 512):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sentences[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": self.labels[idx],
        }


class ContextualSentenceDataset(TorchDataset):
    """Dataset that includes surrounding sentence context for RR classification."""

    def __init__(
        self,
        documents: list[list[str]],
        labels: list[list[int]],
        tokenizer,
        max_length: int = 512,
        context_window: int = 2,
    ):
        self.samples: list[tuple[str, int]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for doc_sents, doc_labels in zip(documents, labels):
            for i, (sent, label) in enumerate(zip(doc_sents, doc_labels)):
                # Build context: preceding + current + following sentences
                start = max(0, i - context_window)
                end = min(len(doc_sents), i + context_window + 1)
                context = " [SEP] ".join(doc_sents[start:end])
                self.samples.append((context, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": label,
        }
