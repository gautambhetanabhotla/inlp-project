"""
Knowledge Graph construction and graph embedding module.

Builds a heterogeneous graph of Cases and Statutes from the legal corpus,
then generates node embeddings via Node2Vec for use in retrieval.
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

from config import Config, KGConfig
from data_loader import LegalDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Statute normalisation
# ---------------------------------------------------------------------------

_NORMALISE_MAP = {
    "ipc": "Indian Penal Code, 1860",
    "indian penal code": "Indian Penal Code, 1860",
    "crpc": "Code of Criminal Procedure, 1973",
    "code of criminal procedure": "Code of Criminal Procedure, 1973",
    "cpc": "Code of Civil Procedure, 1908",
    "code of civil procedure": "Code of Civil Procedure, 1908",
    "evidence act": "Indian Evidence Act, 1872",
    "indian evidence act": "Indian Evidence Act, 1872",
    "constitution of india": "Constitution of India",
    "constitution": "Constitution of India",
}


def normalise_statute(name: str) -> str:
    """Normalise statute names to canonical forms."""
    cleaned = re.sub(r"\s+", " ", name.strip().lower())
    # Remove leading "the"
    cleaned = re.sub(r"^the\s+", "", cleaned)
    # Check known aliases
    for alias, canonical in _NORMALISE_MAP.items():
        if alias in cleaned:
            return canonical
    # Fallback: title-case & strip trailing punctuation
    return name.strip().rstrip(",;.").title()


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class LegalKnowledgeGraph:
    """Heterogeneous graph of cases and statutes."""

    NODE_TYPE_CASE = "case"
    NODE_TYPE_STATUTE = "statute"

    EDGE_TYPE_CITES = "cites"          # case → case
    EDGE_TYPE_REFERS = "refers_to"      # case → statute

    def __init__(self, cfg: KGConfig):
        self.cfg = cfg
        self.graph = nx.DiGraph()
        self._case_nodes: set[str] = set()
        self._statute_nodes: set[str] = set()
        self._embeddings: Optional[dict[str, np.ndarray]] = None

    # ---------- Graph construction ----------

    def add_document(self, doc: LegalDocument):
        """Add a single document (and its citations) to the graph."""
        case_id = doc.doc_id

        # Add case node
        if case_id not in self._case_nodes:
            self.graph.add_node(case_id, node_type=self.NODE_TYPE_CASE)
            self._case_nodes.add(case_id)

        # Case → Case citation edges
        for cited in doc.cited_cases:
            cited_id = cited.strip()
            if cited_id not in self._case_nodes:
                self.graph.add_node(cited_id, node_type=self.NODE_TYPE_CASE)
                self._case_nodes.add(cited_id)
            self.graph.add_edge(case_id, cited_id, edge_type=self.EDGE_TYPE_CITES)

        # Case → Statute edges
        for statute in doc.cited_statutes:
            statute_norm = normalise_statute(statute)
            if statute_norm not in self._statute_nodes:
                self.graph.add_node(statute_norm, node_type=self.NODE_TYPE_STATUTE)
                self._statute_nodes.add(statute_norm)
            self.graph.add_edge(case_id, statute_norm, edge_type=self.EDGE_TYPE_REFERS)

    def build_from_corpus(self, documents: list[LegalDocument]):
        """Build the full graph from a list of documents."""
        for doc in documents:
            self.add_document(doc)
        logger.info(
            "Knowledge Graph built: %d nodes (%d cases, %d statutes), %d edges",
            self.graph.number_of_nodes(),
            len(self._case_nodes),
            len(self._statute_nodes),
            self.graph.number_of_edges(),
        )

    # ---------- Graph queries ----------

    def get_shared_statutes(self, case_a: str, case_b: str) -> list[str]:
        """Return statutes cited by both cases."""
        statutes_a = {
            n for n in self.graph.successors(case_a)
            if self.graph.nodes[n].get("node_type") == self.NODE_TYPE_STATUTE
        } if self.graph.has_node(case_a) else set()

        statutes_b = {
            n for n in self.graph.successors(case_b)
            if self.graph.nodes[n].get("node_type") == self.NODE_TYPE_STATUTE
        } if self.graph.has_node(case_b) else set()

        return list(statutes_a & statutes_b)

    def get_citation_overlap(self, case_a: str, case_b: str) -> int:
        """Return the number of cases cited by both case_a and case_b."""
        cited_a = {
            n for n in self.graph.successors(case_a)
            if self.graph.nodes[n].get("node_type") == self.NODE_TYPE_CASE
        } if self.graph.has_node(case_a) else set()

        cited_b = {
            n for n in self.graph.successors(case_b)
            if self.graph.nodes[n].get("node_type") == self.NODE_TYPE_CASE
        } if self.graph.has_node(case_b) else set()

        return len(cited_a & cited_b)

    def graph_similarity(self, case_a: str, case_b: str) -> float:
        """Compute a graph-based similarity score between two cases.

        Combines Jaccard similarity over shared statutes and shared citations.
        """
        if not self.graph.has_node(case_a) or not self.graph.has_node(case_b):
            return 0.0

        # Statute Jaccard
        stat_a = {
            n for n in self.graph.successors(case_a)
            if self.graph.nodes[n].get("node_type") == self.NODE_TYPE_STATUTE
        }
        stat_b = {
            n for n in self.graph.successors(case_b)
            if self.graph.nodes[n].get("node_type") == self.NODE_TYPE_STATUTE
        }
        stat_jaccard = (
            len(stat_a & stat_b) / len(stat_a | stat_b) if stat_a | stat_b else 0.0
        )

        # Citation Jaccard
        cite_a = {
            n for n in self.graph.successors(case_a)
            if self.graph.nodes[n].get("node_type") == self.NODE_TYPE_CASE
        }
        cite_b = {
            n for n in self.graph.successors(case_b)
            if self.graph.nodes[n].get("node_type") == self.NODE_TYPE_CASE
        }
        cite_jaccard = (
            len(cite_a & cite_b) / len(cite_a | cite_b) if cite_a | cite_b else 0.0
        )

        # Direct citation bonus
        direct = 0.0
        if self.graph.has_edge(case_a, case_b) or self.graph.has_edge(case_b, case_a):
            direct = 0.3

        return 0.4 * stat_jaccard + 0.3 * cite_jaccard + direct

    # ---------- Node2Vec graph embeddings ----------

    def compute_embeddings(self):
        """Compute Node2Vec embeddings for all nodes in the graph."""
        from node2vec import Node2Vec

        # Node2Vec works on undirected graphs
        undirected = self.graph.to_undirected()

        n2v = Node2Vec(
            undirected,
            dimensions=self.cfg.embedding_dim,
            walk_length=self.cfg.walk_length,
            num_walks=self.cfg.num_walks,
            p=self.cfg.p,
            q=self.cfg.q,
            workers=self.cfg.workers,
            quiet=True,
        )

        model = n2v.fit(
            window=self.cfg.window_size,
            min_count=1,
            batch_words=4,
            epochs=self.cfg.epochs,
        )

        self._embeddings = {}
        for node in self.graph.nodes():
            if node in model.wv:
                self._embeddings[node] = model.wv[node]

        logger.info("Computed Node2Vec embeddings for %d nodes", len(self._embeddings))

    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """Retrieve the embedding vector for a graph node."""
        if self._embeddings is None:
            return None
        return self._embeddings.get(node_id)

    def embedding_similarity(self, case_a: str, case_b: str) -> float:
        """Cosine similarity between Node2Vec embeddings of two cases."""
        if self._embeddings is None:
            return 0.0
        emb_a = self._embeddings.get(case_a)
        emb_b = self._embeddings.get(case_b)
        if emb_a is None or emb_b is None:
            return 0.0
        cos = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8)
        return float(cos)

    # ---------- Persistence ----------

    def save(self, directory: Path):
        """Save graph and embeddings to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save graph as edge list + node attributes
        nx.write_gml(self.graph, str(directory / "legal_kg.gml"))

        # Save embeddings
        if self._embeddings is not None:
            emb_data = {k: v.tolist() for k, v in self._embeddings.items()}
            with open(directory / "node_embeddings.json", "w") as f:
                json.dump(emb_data, f)

        logger.info("Saved KG to %s", directory)

    def load(self, directory: Path):
        """Load graph and embeddings from disk."""
        directory = Path(directory)

        gml_path = directory / "legal_kg.gml"
        if gml_path.exists():
            self.graph = nx.read_gml(str(gml_path))
            self._case_nodes = {
                n for n, d in self.graph.nodes(data=True)
                if d.get("node_type") == self.NODE_TYPE_CASE
            }
            self._statute_nodes = {
                n for n, d in self.graph.nodes(data=True)
                if d.get("node_type") == self.NODE_TYPE_STATUTE
            }
            logger.info(
                "Loaded KG: %d nodes, %d edges",
                self.graph.number_of_nodes(),
                self.graph.number_of_edges(),
            )

        emb_path = directory / "node_embeddings.json"
        if emb_path.exists():
            with open(emb_path) as f:
                raw = json.load(f)
            self._embeddings = {k: np.array(v) for k, v in raw.items()}
            logger.info("Loaded %d node embeddings", len(self._embeddings))

    # ---------- Analytics ----------

    def get_statistics(self) -> dict:
        """Return basic graph statistics."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_cases": len(self._case_nodes),
            "num_statutes": len(self._statute_nodes),
            "avg_degree": (
                sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1)
            ),
            "num_connected_components": nx.number_weakly_connected_components(self.graph),
        }
