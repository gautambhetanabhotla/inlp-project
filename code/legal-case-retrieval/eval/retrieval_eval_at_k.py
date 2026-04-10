#!/usr/bin/env python3
"""
Evaluate ranked case retrieval predictions against the ik_test 4 gold file.

Gold format:
{
  "Query Set": [
    {"id": "0001104022.txt", "relevant candidates": ["..."]},
    ...
  ]
}

Prediction format options per query id:
1) ["case1.txt", "case2.txt", ...]                 # ranked retrieval list
2) {"case1.txt": 0.9, "case2.txt": 0.7, ...}        # score map
3) [{"id": "case1.txt", "score": 0.9}, ...]       # ranked objects

Metrics reported for each k:
- microF1@k
- precision@k
- recall@k
- mrr@k
- map@k
- ndcg@k
"""

import argparse
import json
import math
from typing import Dict, Iterable, List, Sequence


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_gold(gold_obj) -> Dict[str, List[str]]:
    if isinstance(gold_obj, dict) and "Query Set" in gold_obj:
        items = gold_obj["Query Set"]
    elif isinstance(gold_obj, list):
        items = gold_obj
    else:
        raise ValueError("Unsupported gold format")

    gold = {}
    for item in items:
        query_id = str(item.get("id") or item.get("query_name") or item.get("query id"))
        relevant = item.get("relevant candidates") or item.get("relevant_candidates") or item.get("cited cases") or []
        gold[query_id] = [str(x) for x in relevant]
    return gold


def _sorted_unique(labels: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for label in labels:
        if label not in seen:
            out.append(label)
            seen.add(label)
    return out


def parse_ranked_predictions(raw_case_pred) -> List[str]:
    if isinstance(raw_case_pred, list):
        if not raw_case_pred:
            return []
        if all(isinstance(x, str) for x in raw_case_pred):
            return _sorted_unique([str(x) for x in raw_case_pred])
        if all(isinstance(x, dict) for x in raw_case_pred):
            scored = []
            for item in raw_case_pred:
                label = item.get("id") or item.get("case_id") or item.get("label") or item.get("name")
                score = item.get("score", item.get("prob", item.get("confidence", 0.0)))
                if label is not None:
                    try:
                        score = float(score)
                    except (TypeError, ValueError):
                        score = 0.0
                    scored.append((str(label), score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return _sorted_unique([label for label, _ in scored])

    if isinstance(raw_case_pred, dict):
        scored = []
        for label, score in raw_case_pred.items():
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0
            scored.append((str(label), score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [label for label, _ in scored]

    return []


def precision_at_k(topk: Sequence[str], relevant: set, k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for label in topk if label in relevant)
    return hits / float(k)


def recall_at_k(topk: Sequence[str], relevant: set) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for label in topk if label in relevant)
    return hits / float(len(relevant))


def reciprocal_rank_at_k(topk: Sequence[str], relevant: set) -> float:
    for idx, label in enumerate(topk, start=1):
        if label in relevant:
            return 1.0 / float(idx)
    return 0.0


def average_precision_at_k(topk: Sequence[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    ap_sum = 0.0
    for rank, label in enumerate(topk, start=1):
        if label in relevant:
            hits += 1
            ap_sum += hits / float(rank)
    denom = float(min(len(relevant), k))
    return ap_sum / denom if denom > 0 else 0.0


def ndcg_at_k(topk: Sequence[str], relevant: set, k: int) -> float:
    dcg = 0.0
    for rank, label in enumerate(topk, start=1):
        if label in relevant:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_at_k(gold: Dict[str, List[str]], pred: Dict[str, object], ks: Sequence[int]):
    case_ids = list(gold.keys())
    ranked_predictions = {case_id: parse_ranked_predictions(pred.get(case_id, [])) for case_id in case_ids}

    results = {}
    for k in ks:
        p_vals = []
        r_vals = []
        rr_vals = []
        ap_vals = []
        ndcg_vals = []

        tp = 0
        fp = 0
        fn = 0

        for case_id in case_ids:
            rel = set(gold.get(case_id, []))
            topk = ranked_predictions[case_id][:k]
            hits = sum(1 for label in topk if label in rel)
            tp += hits
            fp += len(topk) - hits
            fn += len(rel - set(topk))

            p_vals.append(precision_at_k(topk, rel, k))
            r_vals.append(recall_at_k(topk, rel))
            rr_vals.append(reciprocal_rank_at_k(topk, rel))
            ap_vals.append(average_precision_at_k(topk, rel, k))
            ndcg_vals.append(ndcg_at_k(topk, rel, k))

        micro_prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        micro_rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        micro_f1 = (2.0 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0

        results[k] = {
            "microf1@k": micro_f1,
            "precision@k": sum(p_vals) / float(len(p_vals)) if p_vals else 0.0,
            "recall@k": sum(r_vals) / float(len(r_vals)) if r_vals else 0.0,
            "mrr@k": sum(rr_vals) / float(len(rr_vals)) if rr_vals else 0.0,
            "map@k": sum(ap_vals) / float(len(ap_vals)) if ap_vals else 0.0,
            "ndcg@k": sum(ndcg_vals) / float(len(ndcg_vals)) if ndcg_vals else 0.0,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ranked case retrieval predictions for ik_test 4.")
    parser.add_argument("--gold", default="ilsi-test-gold.json", help="Gold JSON file, usually dataset/ik_test 4/test.json copied here.")
    parser.add_argument("--pred", default="../retrieval-predictions.json", help="Ranked predictions JSON file.")
    parser.add_argument("--ks", default="1,5,10,20", help="Comma-separated k values.")
    parser.add_argument("--out", default="retrieval_eval_at_k_results.json", help="Output JSON path.")
    args = parser.parse_args()

    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]
    gold = normalize_gold(load_json(args.gold))
    pred = load_json(args.pred)

    results = evaluate_at_k(gold, pred, ks)

    print("Metrics at k")
    print("k\tmicroF1\tprecision\trecall\tMRR\tMAP\tnDCG")
    for k in ks:
        r = results[k]
        print(
            f"{k}\t{r['microf1@k']:.6f}\t{r['precision@k']:.6f}\t{r['recall@k']:.6f}\t"
            f"{r['mrr@k']:.6f}\t{r['map@k']:.6f}\t{r['ndcg@k']:.6f}"
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({str(k): results[k] for k in ks}, f, indent=2)
    print(f"\nSaved metrics to {args.out}")


if __name__ == "__main__":
    main()
