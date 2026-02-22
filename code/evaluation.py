"""
Evaluation module.

Computes standard Information Retrieval metrics:
  - Precision@K
  - Recall@K
  - Mean Average Precision (MAP)
  - NDCG@K
"""

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

from config import EvalConfig
from retrieval import RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Precision@K: fraction of top-k results that are relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Recall@K: fraction of relevant docs retrieved in top-k."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return len(set(top_k) & relevant) / len(relevant)


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """Average Precision for a single query."""
    if not relevant:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            hits += 1
            sum_precisions += hits / (i + 1)
    return sum_precisions / len(relevant)


def ndcg_at_k(
    retrieved: list[str],
    relevant: set[str],
    k: int,
    relevance_scores: Optional[dict[str, float]] = None,
) -> float:
    """Normalised Discounted Cumulative Gain at K.

    If relevance_scores is provided, uses graded relevance; otherwise binary.
    """
    def _dcg(doc_list: list[str], limit: int) -> float:
        dcg = 0.0
        for i, doc_id in enumerate(doc_list[:limit]):
            if relevance_scores:
                rel = relevance_scores.get(doc_id, 0.0)
            else:
                rel = 1.0 if doc_id in relevant else 0.0
            dcg += (2 ** rel - 1) / np.log2(i + 2)  # i+2 because log2(1)=0
        return dcg

    dcg = _dcg(retrieved, k)

    # Ideal ordering
    if relevance_scores:
        ideal_order = sorted(relevant, key=lambda d: relevance_scores.get(d, 0.0), reverse=True)
    else:
        ideal_order = list(relevant)
    idcg = _dcg(ideal_order, k)

    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal Rank: 1 / rank of the first relevant document."""
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

class Evaluator:
    """Evaluates retrieval results against relevance judgments."""

    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg

    def evaluate_query(
        self,
        results: list[RetrievalResult],
        relevant_ids: set[str],
        relevance_scores: Optional[dict[str, float]] = None,
    ) -> dict:
        """Compute all metrics for a single query.

        Args:
            results: ranked RetrievalResult list from the retriever.
            relevant_ids: set of ground-truth relevant document IDs.
            relevance_scores: optional graded relevance {doc_id: float}.

        Returns:
            dict of metric_name → value.
        """
        retrieved = [r.doc_id for r in results]
        metrics: dict = {}

        for k in self.cfg.k_values:
            metrics[f"P@{k}"] = precision_at_k(retrieved, relevant_ids, k)
            metrics[f"R@{k}"] = recall_at_k(retrieved, relevant_ids, k)
            metrics[f"NDCG@{k}"] = ndcg_at_k(retrieved, relevant_ids, k, relevance_scores)

        metrics["MAP"] = average_precision(retrieved, relevant_ids)
        metrics["MRR"] = reciprocal_rank(retrieved, relevant_ids)

        return metrics

    def evaluate_all(
        self,
        all_results: dict[str, list[RetrievalResult]],
        qrels: dict[str, list[str]],
        graded_qrels: Optional[dict[str, dict[str, float]]] = None,
    ) -> dict:
        """Compute aggregated metrics over all queries.

        Args:
            all_results: {query_id: [RetrievalResult, ...]}.
            qrels: {query_id: [relevant_doc_ids]}.
            graded_qrels: optional {query_id: {doc_id: relevance_score}}.

        Returns:
            dict with per-query and aggregated (mean) metrics.
        """
        per_query: dict[str, dict] = {}
        aggregated: dict[str, list[float]] = defaultdict(list)

        for query_id, results in all_results.items():
            if query_id not in qrels:
                logger.warning("No qrels for query %s — skipping", query_id)
                continue

            relevant = set(qrels[query_id])
            graded = graded_qrels.get(query_id) if graded_qrels else None
            query_metrics = self.evaluate_query(results, relevant, graded)
            per_query[query_id] = query_metrics

            for metric_name, value in query_metrics.items():
                aggregated[metric_name].append(value)

        # Compute means
        mean_metrics = {
            f"mean_{name}": float(np.mean(values))
            for name, values in aggregated.items()
        }

        return {
            "per_query": per_query,
            "aggregated": mean_metrics,
            "num_queries": len(per_query),
        }

    def print_results(self, eval_output: dict):
        """Pretty-print evaluation results."""
        print("\n" + "=" * 60)
        print("  EVALUATION RESULTS")
        print("=" * 60)
        print(f"  Number of queries evaluated: {eval_output['num_queries']}")
        print("-" * 60)

        agg = eval_output["aggregated"]
        # Sort metrics for clean display
        for name in sorted(agg.keys()):
            print(f"  {name:20s}  {agg[name]:.4f}")

        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Comparative evaluation (for ablation studies)
# ---------------------------------------------------------------------------

def compare_systems(
    system_results: dict[str, dict[str, list[RetrievalResult]]],
    qrels: dict[str, list[str]],
    cfg: EvalConfig,
) -> dict[str, dict]:
    """Evaluate multiple retrieval systems side by side.

    Args:
        system_results: {system_name: {query_id: [RetrievalResult]}}.
        qrels: relevance judgments.
        cfg: evaluation config.

    Returns:
        {system_name: aggregated_metrics}.
    """
    evaluator = Evaluator(cfg)
    comparison = {}

    for sys_name, results in system_results.items():
        eval_out = evaluator.evaluate_all(results, qrels)
        comparison[sys_name] = eval_out["aggregated"]
        logger.info("System '%s': %s", sys_name, eval_out["aggregated"])

    # Print comparison table
    metric_names = sorted(next(iter(comparison.values())).keys())
    header = f"{'System':25s}" + "".join(f"{m:>15s}" for m in metric_names)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for sys_name, metrics in comparison.items():
        row = f"{sys_name:25s}" + "".join(f"{metrics[m]:>15.4f}" for m in metric_names)
        print(row)
    print("=" * len(header) + "\n")

    return comparison
