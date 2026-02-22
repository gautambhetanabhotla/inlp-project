"""
Similar Case Matching in Indian Legal Documents
================================================

Main orchestration script that ties together:
  1. Data loading
  2. Rhetorical Role segmentation
  3. Knowledge Graph construction
  4. Segment encoding + indexing
  5. Hybrid retrieval
  6. Evaluation

Usage:
    python main.py --mode full          # Run the full pipeline
    python main.py --mode train_rr      # Train only the RR classifier
    python main.py --mode build_kg      # Build only the Knowledge Graph
    python main.py --mode index         # Build the embedding index
    python main.py --mode retrieve      # Run retrieval (requires index + KG)
    python main.py --mode evaluate      # Evaluate retrieval results
    python main.py --mode demo          # Run a demo on synthetic data
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

from config import Config
from data_loader import (
    LegalDocument,
    extract_case_citations,
    extract_statute_references,
    load_iltur_pcr,
    load_iltur_rr,
    sentence_tokenize,
)
from encoder import EmbeddingIndex, SegmentEncoder
from evaluation import Evaluator, compare_systems
from knowledge_graph import LegalKnowledgeGraph
from retrieval import BM25Retriever, LegalCaseRetriever
from rhetorical_roles import RRTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seed everything
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_load_data(
    cfg: Config,
    query_split: str = "test_queries",
    candidate_split: str = "test_candidates",
) -> tuple[list[LegalDocument], list[LegalDocument], dict]:
    """Load the IL-PCR corpus and relevance judgments from HuggingFace."""
    logger.info("Loading IL-TUR PCR dataset from HuggingFace …")
    queries, candidates, qrels = load_iltur_pcr(query_split, candidate_split)
    logger.info(
        "Loaded %d queries, %d candidates, %d qrels",
        len(queries), len(candidates), len(qrels),
    )
    return queries, candidates, qrels


def step_train_rr(cfg: Config):
    """Train the Rhetorical Role classifier on the IL-TUR RR dataset.

    Downloads the dataset directly from HuggingFace.
    Uses Competition Law + Income Tax splits.
    """
    logger.info("Training Rhetorical Role classifier …")

    # Load both domains from HuggingFace
    cl_train = load_iltur_rr("CL_train")
    it_train = load_iltur_rr("IT_train")
    train_data = cl_train + it_train

    cl_dev = load_iltur_rr("CL_dev")
    it_dev = load_iltur_rr("IT_dev")
    dev_data = cl_dev + it_dev

    logger.info("RR training data: %d docs, dev: %d docs", len(train_data), len(dev_data))

    trainer = RRTrainer(cfg)

    # Labels are already ints (0–12) from the dataset
    train_docs = [item["sentences"] for item in train_data]
    train_labels = [item["labels"] for item in train_data]
    train_loader = trainer.prepare_dataset_from_ids(train_docs, train_labels)

    dev_docs = [item["sentences"] for item in dev_data]
    dev_labels = [item["labels"] for item in dev_data]
    val_loader = trainer.prepare_dataset_from_ids(dev_docs, dev_labels)

    trainer.train(train_loader, val_loader)
    trainer.save(cfg.paths.models_dir / "rr_final.pt")

    return trainer


def step_segment_documents(
    cfg: Config,
    documents: list[LegalDocument],
    trainer: RRTrainer = None,
) -> list[LegalDocument]:
    """Apply RR segmentation to all documents."""
    logger.info("Segmenting %d documents …", len(documents))

    if trainer is None:
        # Try loading a saved model
        model_path = cfg.paths.models_dir / "rr_best.pt"
        if not model_path.exists():
            model_path = cfg.paths.models_dir / "rr_final.pt"
        if model_path.exists():
            trainer = RRTrainer(cfg)
            trainer.load(model_path)
        else:
            logger.warning("No RR model found. Documents will not be segmented.")
            return documents

    for doc in documents:
        trainer.segment_document(doc)

    logger.info("Segmentation complete.")
    return documents


def step_build_knowledge_graph(
    cfg: Config,
    documents: list[LegalDocument],
) -> LegalKnowledgeGraph:
    """Build the Legal Knowledge Graph from the corpus."""
    logger.info("Building Knowledge Graph …")
    kg = LegalKnowledgeGraph(cfg.kg)
    kg.build_from_corpus(documents)

    stats = kg.get_statistics()
    logger.info("KG stats: %s", stats)

    # Compute Node2Vec embeddings
    if kg.graph.number_of_nodes() > 1:
        try:
            kg.compute_embeddings()
        except Exception as e:
            logger.warning("Node2Vec embedding failed (install node2vec?): %s", e)

    kg.save(cfg.paths.kg_dir)
    return kg


def step_build_index(
    cfg: Config,
    documents: list[LegalDocument],
) -> tuple[SegmentEncoder, EmbeddingIndex]:
    """Encode all documents and build the embedding index."""
    logger.info("Building embedding index for %d documents …", len(documents))
    encoder = SegmentEncoder(cfg.encoder, device=cfg.device)
    index = EmbeddingIndex()

    for i, doc in enumerate(documents):
        segment_embs = encoder.encode_document_segments(doc)
        full_emb = encoder.encode_document_full(doc)
        index.add(doc.doc_id, segment_embs, full_emb)

        if (i + 1) % 50 == 0:
            logger.info("  Encoded %d / %d documents", i + 1, len(documents))

    index.save(cfg.paths.processed_data_dir / "embedding_index.json")
    return encoder, index


def step_retrieve(
    cfg: Config,
    queries: list[LegalDocument],
    corpus: list[LegalDocument],
    encoder: SegmentEncoder,
    index: EmbeddingIndex,
    kg: LegalKnowledgeGraph,
) -> dict[str, list]:
    """Run retrieval for all query documents."""
    logger.info("Running retrieval for %d queries …", len(queries))

    # Build BM25 index for first-stage retrieval
    bm25 = BM25Retriever()
    bm25.build_index(corpus)

    retriever = LegalCaseRetriever(
        cfg=cfg,
        encoder=encoder,
        embedding_index=index,
        knowledge_graph=kg,
        bm25=bm25,
    )

    all_results = retriever.retrieve_batch(queries)
    return all_results


def step_evaluate(cfg: Config, all_results: dict, qrels: dict):
    """Evaluate retrieval results."""
    logger.info("Evaluating …")
    evaluator = Evaluator(cfg.eval)
    eval_output = evaluator.evaluate_all(all_results, qrels)
    evaluator.print_results(eval_output)

    # Save results
    results_path = cfg.paths.results_dir / "eval_results.json"
    serialisable = {
        "aggregated": eval_output["aggregated"],
        "num_queries": eval_output["num_queries"],
        "per_query": {
            qid: {k: float(v) for k, v in metrics.items()}
            for qid, metrics in eval_output["per_query"].items()
        },
    }
    with open(results_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info("Results saved to %s", results_path)

    return eval_output


# ---------------------------------------------------------------------------
# Demo with synthetic data
# ---------------------------------------------------------------------------

def run_demo(cfg: Config):
    """Run a self-contained demo with synthetic legal documents."""
    logger.info("=" * 60)
    logger.info("  DEMO MODE — using synthetic data")
    logger.info("=" * 60)

    # Create synthetic documents
    documents = _create_synthetic_corpus()
    logger.info("Created %d synthetic documents", len(documents))

    # Designate the first 2 as queries, rest as corpus
    queries = documents[:2]
    corpus = documents[2:]

    # Build relevance judgments (for demo)
    qrels = {
        queries[0].doc_id: [corpus[0].doc_id, corpus[2].doc_id],
        queries[1].doc_id: [corpus[1].doc_id, corpus[3].doc_id],
    }

    # Step 1: Build Knowledge Graph
    logger.info("\n--- Step 1: Knowledge Graph ---")
    kg = LegalKnowledgeGraph(cfg.kg)
    kg.build_from_corpus(documents)
    stats = kg.get_statistics()
    logger.info("KG stats: %s", stats)

    # Step 2: Encode documents
    logger.info("\n--- Step 2: Encoding documents ---")
    encoder = SegmentEncoder(cfg.encoder, device=cfg.device)
    index = EmbeddingIndex()

    for doc in documents:
        seg_embs = encoder.encode_document_segments(doc)
        full_emb = encoder.encode_document_full(doc)
        index.add(doc.doc_id, seg_embs, full_emb)

    # Step 3: Retrieve
    logger.info("\n--- Step 3: Retrieval ---")
    bm25 = BM25Retriever()
    bm25.build_index(corpus)

    retriever = LegalCaseRetriever(
        cfg=cfg,
        encoder=encoder,
        embedding_index=index,
        knowledge_graph=kg,
        bm25=bm25,
    )

    all_results = {}
    for query in queries:
        results = retriever.retrieve(query, top_k=5)
        all_results[query.doc_id] = results
        logger.info("\nQuery: %s", query.doc_id)
        for rank, r in enumerate(results, 1):
            logger.info(
                "  Rank %d: %s  (score=%.4f, semantic=%.4f, graph=%.4f)",
                rank, r.doc_id, r.score, r.semantic_score, r.graph_score,
            )

    # Step 4: Evaluate
    logger.info("\n--- Step 4: Evaluation ---")
    evaluator = Evaluator(cfg.eval)
    eval_output = evaluator.evaluate_all(all_results, qrels)
    evaluator.print_results(eval_output)

    # Ablation: compare with BM25-only baseline
    logger.info("\n--- Ablation: BM25 baseline ---")
    from retrieval import RetrievalResult
    bm25_results = {}
    for query in queries:
        bm25_ranked = bm25.retrieve(query.text, top_k=5)
        bm25_results[query.doc_id] = [
            RetrievalResult(doc_id=did, score=s, semantic_score=0, graph_score=0, segment_scores={})
            for did, s in bm25_ranked
        ]

    comparison = compare_systems(
        {"BM25": bm25_results, "Hybrid (ours)": all_results},
        qrels,
        cfg.eval,
    )

    logger.info("Demo complete!")


def _create_synthetic_corpus() -> list[LegalDocument]:
    """Generate synthetic Indian legal documents for demonstration."""
    docs = [
        # Query 1: Murder case
        LegalDocument(
            doc_id="query_001",
            text=(
                "The appellant was convicted under Section 302 of the Indian Penal Code, 1860 "
                "for the murder of the deceased. The prosecution alleged that on the night of "
                "15th January 2018, the appellant struck the deceased with a sharp weapon. "
                "Three eyewitnesses testified to seeing the appellant at the scene. "
                "The trial court relied on circumstantial evidence and the testimony of witnesses. "
                "The defence argued that the identification was flawed and the witnesses were unreliable. "
                "The court held that the chain of circumstances was complete and upheld the conviction. "
                "Reference was made to Sharad Birdhichand Sarda v. State of Maharashtra."
            ),
            rhetorical_roles=[
                "Facts", "Facts", "Facts",
                "Arguments", "Arguments",
                "Ratio Decidendi", "Ratio Decidendi",
                "Precedents",
            ],
            cited_cases=["Sharad Birdhichand Sarda v. State of Maharashtra"],
            cited_statutes=["Indian Penal Code, 1860"],
        ),
        # Query 2: Constitutional rights
        LegalDocument(
            doc_id="query_002",
            text=(
                "The petitioner challenged the detention order under Article 21 of the Constitution "
                "of India, claiming violation of fundamental rights. The State argued the detention "
                "was necessary for public safety under the National Security Act, 1980. "
                "The petitioner had been held without trial for over six months. "
                "The court examined whether the detention satisfied the procedural safeguards "
                "laid down in Maneka Gandhi v. Union of India. "
                "It was held that the right to life includes the right to live with dignity. "
                "The detention order was quashed as it violated due process."
            ),
            rhetorical_roles=[
                "Facts", "Arguments", "Arguments",
                "Facts",
                "Ratio Decidendi", "Ratio Decidendi",
                "Ratio Decidendi",
                "Ruling by Present Court",
            ],
            cited_cases=["Maneka Gandhi v. Union of India"],
            cited_statutes=["Constitution of India", "National Security Act, 1980"],
        ),
        # Corpus doc 1: Similar murder case
        LegalDocument(
            doc_id="case_001",
            text=(
                "The accused was charged under Section 302 and Section 34 of the Indian Penal Code "
                "for the murder of two persons. Eyewitness testimony placed the accused at the scene. "
                "The medical evidence confirmed death by sharp weapon injuries. "
                "The prosecution established motive through prior enmity. "
                "The defence contested the credibility of witnesses. "
                "The court applied the test from Sharad Birdhichand Sarda v. State of Maharashtra "
                "and found the circumstantial evidence convincing."
            ),
            rhetorical_roles=[
                "Facts", "Facts", "Facts",
                "Arguments", "Arguments",
                "Ratio Decidendi", "Ratio Decidendi",
            ],
            cited_cases=["Sharad Birdhichand Sarda v. State of Maharashtra"],
            cited_statutes=["Indian Penal Code, 1860"],
        ),
        # Corpus doc 2: Similar constitutional case
        LegalDocument(
            doc_id="case_002",
            text=(
                "The petitioner filed a habeas corpus petition under Article 226 read with Article 21 "
                "challenging preventive detention under the National Security Act. "
                "The detenu had been held for four months without being informed of the grounds. "
                "The State produced the detention order citing threats to public order. "
                "Relying on Maneka Gandhi v. Union of India, the court examined whether "
                "procedural fairness was maintained. "
                "The court quashed the detention for non-compliance with statutory safeguards."
            ),
            rhetorical_roles=[
                "Facts", "Facts",
                "Facts",
                "Arguments",
                "Ratio Decidendi", "Ratio Decidendi",
                "Ruling by Present Court",
            ],
            cited_cases=["Maneka Gandhi v. Union of India"],
            cited_statutes=["Constitution of India", "National Security Act, 1980"],
        ),
        # Corpus doc 3: Another IPC case
        LegalDocument(
            doc_id="case_003",
            text=(
                "The accused was tried under Section 302 of the Indian Penal Code for causing death "
                "by administering poison. The forensic report confirmed the presence of toxic substances. "
                "The motive was established through financial disputes. "
                "The court noted the principle from Hanumant Govind v. State of Madhya Pradesh "
                "regarding circumstantial evidence. "
                "Conviction was upheld on appeal."
            ),
            rhetorical_roles=[
                "Facts", "Facts",
                "Facts",
                "Ratio Decidendi", "Ratio Decidendi",
                "Ruling by Present Court",
            ],
            cited_cases=["Hanumant Govind v. State of Madhya Pradesh"],
            cited_statutes=["Indian Penal Code, 1860"],
        ),
        # Corpus doc 4: Fundamental rights case
        LegalDocument(
            doc_id="case_004",
            text=(
                "The petitioner sought enforcement of fundamental rights under Article 14 and "
                "Article 21 of the Constitution of India. The State had imposed restrictions "
                "on movement without legal authority. "
                "The petitioner relied on Maneka Gandhi v. Union of India and K.S. Puttaswamy v. "
                "Union of India to argue that personal liberty cannot be curtailed without due process. "
                "The court held that any restriction must satisfy the test of reasonableness "
                "and proportionality. The impugned order was struck down."
            ),
            rhetorical_roles=[
                "Facts", "Facts", "Facts",
                "Arguments", "Arguments",
                "Ratio Decidendi", "Ratio Decidendi",
                "Ruling by Present Court",
            ],
            cited_cases=[
                "Maneka Gandhi v. Union of India",
                "K.S. Puttaswamy v. Union of India",
            ],
            cited_statutes=["Constitution of India"],
        ),
        # Corpus doc 5: Unrelated contract case (negative example)
        LegalDocument(
            doc_id="case_005",
            text=(
                "The plaintiff filed a suit for specific performance of a contract of sale "
                "of immovable property. The defendant denied the existence of a valid contract. "
                "The trial court examined the documents and found the agreement was duly executed. "
                "Under Section 10 of the Indian Contract Act, 1872, all essentials of a valid "
                "contract were satisfied. The court decreed specific performance with costs."
            ),
            rhetorical_roles=[
                "Facts", "Facts",
                "Facts",
                "Ratio Decidendi", "Ratio Decidendi",
                "Ruling by Present Court",
            ],
            cited_cases=[],
            cited_statutes=["Indian Contract Act, 1872"],
        ),
    ]

    return docs


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(cfg: Config):
    """Execute the complete pipeline."""
    logger.info("=" * 60)
    logger.info("  FULL PIPELINE — Similar Case Matching")
    logger.info("=" * 60)

    # 1. Load PCR data from HuggingFace
    queries, corpus, qrels = step_load_data(cfg)
    all_documents = queries + corpus
    logger.info("Queries: %d, Corpus: %d", len(queries), len(corpus))

    # 2. Train RR classifier (downloads RR data from HuggingFace)
    trainer = step_train_rr(cfg)

    # 3. Segment documents
    all_documents = step_segment_documents(cfg, all_documents, trainer)

    # 4. Build Knowledge Graph
    kg = step_build_knowledge_graph(cfg, all_documents)

    # 5. Build embedding index
    encoder, index = step_build_index(cfg, all_documents)

    # 6. Retrieve
    all_results = step_retrieve(cfg, queries, corpus, encoder, index, kg)

    # 7. Evaluate
    step_evaluate(cfg, all_results, qrels)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Similar Case Matching in Indian Legal Documents"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "train_rr", "build_kg", "index", "retrieve", "evaluate", "demo"],
        default="demo",
        help="Pipeline mode (default: demo)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(seed=args.seed)
    if args.device:
        cfg.device = args.device

    set_seed(cfg.seed)
    logger.info("Device: %s", cfg.device)
    logger.info("Mode: %s", args.mode)

    if args.mode == "demo":
        run_demo(cfg)
    elif args.mode == "full":
        run_full_pipeline(cfg)
    elif args.mode == "train_rr":
        step_train_rr(cfg)
    elif args.mode == "build_kg":
        queries, corpus, _ = step_load_data(cfg)
        step_build_knowledge_graph(cfg, queries + corpus)
    elif args.mode == "index":
        queries, corpus, _ = step_load_data(cfg)
        step_build_index(cfg, queries + corpus)
    elif args.mode == "retrieve":
        queries, corpus, qrels = step_load_data(cfg)
        encoder = SegmentEncoder(cfg.encoder, device=cfg.device)
        index = EmbeddingIndex()
        index.load(cfg.paths.processed_data_dir / "embedding_index.json")
        kg = LegalKnowledgeGraph(cfg.kg)
        kg.load(cfg.paths.kg_dir)
        all_results = step_retrieve(cfg, queries, corpus, encoder, index, kg)
        step_evaluate(cfg, all_results, qrels)
    elif args.mode == "evaluate":
        logger.info("Load saved results and evaluate (not yet implemented — use 'retrieve' mode).")


if __name__ == "__main__":
    main()
