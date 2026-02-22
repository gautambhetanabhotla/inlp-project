"""
Retrieval pipeline.

Combines segment-level semantic similarity with Knowledge-Graph-based
similarity to rank candidate prior cases for a given query case.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import Config, RetrievalConfig
from data_loader import LegalDocument
from encoder import EmbeddingIndex, SegmentEncoder
from knowledge_graph import LegalKnowledgeGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """A single scored candidate case."""
    doc_id: str
    score: float
    semantic_score: float
    graph_score: float
    segment_scores: dict  # role → similarity


# ---------------------------------------------------------------------------
# BM25 baseline
# ---------------------------------------------------------------------------

class BM25Retriever:
    """BM25 baseline retriever using rank_bm25."""

    def __init__(self):
        self._index = None
        self._doc_ids: list[str] = []

    def build_index(self, documents: list[LegalDocument]):
        from rank_bm25 import BM25Okapi

        self._doc_ids = [doc.doc_id for doc in documents]
        tokenized = [doc.text.lower().split() for doc in documents]
        self._index = BM25Okapi(tokenized)
        logger.info("BM25 index built with %d documents", len(documents))

    def retrieve(self, query_text: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Return top-k (doc_id, score) pairs."""
        if self._index is None:
            raise RuntimeError("BM25 index not built. Call build_index() first.")
        tokenized_query = query_text.lower().split()
        scores = self._index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self._doc_ids[i], float(scores[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# Main retrieval system
# ---------------------------------------------------------------------------

class LegalCaseRetriever:
    """
    Hybrid retrieval system that combines:
      1. Segment-aligned semantic similarity (Facts↔Facts, Ratio↔Ratio, …)
      2. Knowledge-graph-based similarity (shared statutes, citations)
      3. Optional BM25 first-stage retrieval for candidate generation
    """

    def __init__(
        self,
        cfg: Config,
        encoder: SegmentEncoder,
        embedding_index: EmbeddingIndex,
        knowledge_graph: LegalKnowledgeGraph,
        bm25: Optional[BM25Retriever] = None,
    ):
        self.cfg = cfg
        self.ret_cfg = cfg.retrieval
        self.encoder = encoder
        self.index = embedding_index
        self.kg = knowledge_graph
        self.bm25 = bm25

    # ---------- Candidate generation ----------

    def _first_stage_candidates(self, query: LegalDocument, top_k: int) -> list[str]:
        """Generate initial candidate set (BM25 or full corpus)."""
        if self.bm25 is not None:
            results = self.bm25.retrieve(query.text, top_k=top_k)
            return [doc_id for doc_id, _ in results]
        else:
            # Return all documents in the index
            return [did for did in self.index.doc_ids if did != query.doc_id]

    # ---------- Scoring ----------

    def _compute_semantic_score(
        self,
        query_segments: dict[str, np.ndarray],
        query_full_emb: np.ndarray,
        candidate_id: str,
    ) -> tuple[float, dict]:
        """Compute segment-aligned + full-text semantic score."""
        # Segment-aligned similarity
        seg_score = self.index.segment_similarity(
            query_segments, candidate_id, self.ret_cfg.segment_weights
        )
        # Full-text similarity as fallback
        full_score = self.index.full_text_similarity(query_full_emb, candidate_id)

        # Blend: prefer segment-level if segments exist, else fallback to full
        has_segments = bool(query_segments) and bool(
            self.index.segment_embeddings.get(candidate_id)
        )
        if has_segments:
            semantic = 0.7 * seg_score + 0.3 * full_score
        else:
            semantic = full_score

        # Per-role breakdown for interpretability
        role_scores = {}
        cand_segs = self.index.segment_embeddings.get(candidate_id, {})
        for role in self.ret_cfg.segment_weights:
            q = query_segments.get(role)
            c = cand_segs.get(role)
            if q is not None and c is not None:
                role_scores[role] = float(np.dot(q, c))

        return semantic, role_scores

    def _compute_graph_score(self, query_id: str, candidate_id: str) -> float:
        """Compute graph-based similarity score."""
        # Structural similarity (shared statutes + citations)
        structural = self.kg.graph_similarity(query_id, candidate_id)

        # Node2Vec embedding similarity
        emb_sim = self.kg.embedding_similarity(query_id, candidate_id)

        return 0.6 * structural + 0.4 * emb_sim

    # ---------- Main retrieval ----------

    def retrieve(self, query: LegalDocument, top_k: Optional[int] = None) -> list[RetrievalResult]:
        """Retrieve and rank candidate prior cases for a query case.

        Args:
            query: The query LegalDocument (should already have rhetorical roles).
            top_k: Number of results to return (default from config).

        Returns:
            Ranked list of RetrievalResult objects.
        """
        top_k = top_k or self.ret_cfg.top_k

        # Step 1: Encode the query
        query_segments = self.encoder.encode_document_segments(query)
        query_full_emb = self.encoder.encode_document_full(query)

        # Step 2: Generate candidates
        candidate_pool_size = self.ret_cfg.rerank_top_k if self.ret_cfg.rerank else top_k
        candidates = self._first_stage_candidates(query, candidate_pool_size)

        # Step 3: Score each candidate
        results: list[RetrievalResult] = []
        for cand_id in candidates:
            if cand_id == query.doc_id:
                continue

            semantic, role_scores = self._compute_semantic_score(
                query_segments, query_full_emb, cand_id
            )
            graph = self._compute_graph_score(query.doc_id, cand_id)

            combined = (
                self.ret_cfg.semantic_weight * semantic
                + self.ret_cfg.graph_weight * graph
            )

            results.append(RetrievalResult(
                doc_id=cand_id,
                score=combined,
                semantic_score=semantic,
                graph_score=graph,
                segment_scores=role_scores,
            ))

        # Step 4: Sort and return top-k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ---------- Batch retrieval ----------

    def retrieve_batch(
        self,
        queries: list[LegalDocument],
        top_k: Optional[int] = None,
    ) -> dict[str, list[RetrievalResult]]:
        """Retrieve for multiple queries.

        Returns {query_doc_id: [RetrievalResult, ...]}.
        """
        results = {}
        for i, query in enumerate(queries):
            logger.info("Retrieving for query %d/%d: %s", i + 1, len(queries), query.doc_id)
            results[query.doc_id] = self.retrieve(query, top_k)
        return results
