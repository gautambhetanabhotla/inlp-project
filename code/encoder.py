"""
Segment encoder module.

Produces dense vector representations of legal document segments
(Facts, Arguments, Ratio Decidendi, etc.) using a transformer encoder
with various pooling strategies.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from config import Config, EncoderConfig
from data_loader import LegalDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------

def _mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling over non-padding tokens."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)


def _max_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Max pooling over non-padding tokens."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    token_embeddings[mask_expanded == 0] = -1e9
    return token_embeddings.max(1).values


# ---------------------------------------------------------------------------
# Text dataset for batch encoding
# ---------------------------------------------------------------------------

class _TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


# ---------------------------------------------------------------------------
# Segment encoder
# ---------------------------------------------------------------------------

class SegmentEncoder:
    """Encodes text segments into dense vectors using a transformer model."""

    def __init__(self, cfg: EncoderConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModel.from_pretrained(cfg.model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Encode a list of texts into dense vectors.

        Args:
            texts: list of text strings to encode.
            batch_size: override default batch size.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.empty((0, self.cfg.embedding_dim))

        bs = batch_size or self.cfg.batch_size
        all_embeddings: list[np.ndarray] = []

        for start in range(0, len(texts), bs):
            batch_texts = texts[start : start + bs]
            encoded = self.tokenizer(
                batch_texts,
                max_length=self.cfg.max_seq_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**encoded)
            token_embs = outputs.last_hidden_state

            if self.cfg.pooling_strategy == "cls":
                embeddings = token_embs[:, 0, :]
            elif self.cfg.pooling_strategy == "max":
                embeddings = _max_pooling(token_embs, encoded["attention_mask"])
            else:  # "mean" (default)
                embeddings = _mean_pooling(token_embs, encoded["attention_mask"])

            # L2-normalise
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string."""
        return self.encode([text])[0]

    # ---------- Document-level helpers ----------

    def encode_document_segments(self, doc: LegalDocument) -> dict[str, np.ndarray]:
        """Encode each rhetorical-role segment of a document.

        Returns a dict mapping role name → embedding vector.
        """
        segments = doc.get_segments()
        result: dict[str, np.ndarray] = {}
        for role, text in segments.items():
            if text.strip():
                result[role] = self.encode_single(text)
        return result

    def encode_document_full(self, doc: LegalDocument) -> np.ndarray:
        """Encode the full document text (truncated to max_seq_length).

        For long documents, splits into chunks and averages.
        """
        text = doc.text
        # Split into chunks if necessary
        tokens = self.tokenizer.tokenize(text)
        max_tokens = self.cfg.max_seq_length - 2  # account for [CLS] and [SEP]

        if len(tokens) <= max_tokens:
            return self.encode_single(text)

        # Sliding window chunking with 50% overlap
        stride = max_tokens // 2
        chunks: list[str] = []
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i : i + max_tokens]
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)
            if i + max_tokens >= len(tokens):
                break

        chunk_embeddings = self.encode(chunks)
        # Weighted average: later chunks get slightly less weight
        weights = np.array([1.0 / (1 + 0.1 * i) for i in range(len(chunks))])
        weights /= weights.sum()
        avg_embedding = (chunk_embeddings * weights[:, None]).sum(axis=0)
        avg_embedding /= np.linalg.norm(avg_embedding) + 1e-9
        return avg_embedding


# ---------------------------------------------------------------------------
# Pre-computed embedding index for fast retrieval
# ---------------------------------------------------------------------------

class EmbeddingIndex:
    """Stores pre-computed embeddings for a corpus and supports similarity search."""

    def __init__(self):
        self.doc_ids: list[str] = []
        self.segment_embeddings: dict[str, dict[str, np.ndarray]] = {}  # doc_id → {role → emb}
        self.full_embeddings: dict[str, np.ndarray] = {}                # doc_id → emb

    def add(self, doc_id: str, segment_embs: dict[str, np.ndarray], full_emb: np.ndarray):
        self.doc_ids.append(doc_id)
        self.segment_embeddings[doc_id] = segment_embs
        self.full_embeddings[doc_id] = full_emb

    def segment_similarity(
        self,
        query_segments: dict[str, np.ndarray],
        candidate_id: str,
        weights: dict[str, float],
    ) -> float:
        """Compute weighted segment-aligned similarity between query and candidate."""
        candidate_segments = self.segment_embeddings.get(candidate_id, {})
        total_sim = 0.0
        total_weight = 0.0

        for role, weight in weights.items():
            q_emb = query_segments.get(role)
            c_emb = candidate_segments.get(role)
            if q_emb is not None and c_emb is not None:
                sim = float(np.dot(q_emb, c_emb))
                total_sim += weight * sim
                total_weight += weight

        return total_sim / total_weight if total_weight > 0 else 0.0

    def full_text_similarity(self, query_emb: np.ndarray, candidate_id: str) -> float:
        """Cosine similarity using full-document embeddings."""
        c_emb = self.full_embeddings.get(candidate_id)
        if c_emb is None:
            return 0.0
        return float(np.dot(query_emb, c_emb))

    def save(self, path: Path):
        """Persist the index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "doc_ids": self.doc_ids,
            "segment_embeddings": {
                doc_id: {role: emb.tolist() for role, emb in roles.items()}
                for doc_id, roles in self.segment_embeddings.items()
            },
            "full_embeddings": {
                doc_id: emb.tolist() for doc_id, emb in self.full_embeddings.items()
            },
        }
        import json
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info("Saved embedding index (%d documents) to %s", len(self.doc_ids), path)

    def load(self, path: Path):
        """Load a saved index."""
        import json
        with open(path) as f:
            data = json.load(f)
        self.doc_ids = data["doc_ids"]
        self.segment_embeddings = {
            doc_id: {role: np.array(emb) for role, emb in roles.items()}
            for doc_id, roles in data["segment_embeddings"].items()
        }
        self.full_embeddings = {
            doc_id: np.array(emb) for doc_id, emb in data["full_embeddings"].items()
        }
        logger.info("Loaded embedding index with %d documents", len(self.doc_ids))
