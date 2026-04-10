#!/usr/bin/env python3
"""
Evaluate ranked retrieval predictions against ik_test test.csv.

CSV columns expected:
- Query Case
- Cited Cases   (stringified Python list)

Metrics at k:
- microF1@k, precision@k, recall@k, mrr@k, map@k, ndcg@k
"""

import argparse
import ast
import csv
import json
import math
from typing import Dict, List, Sequence


def load_predictions(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gold_from_csv(path: str) -> Dict[str, List[str]]:
    gold = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = str(row["Query Case"]).strip()
            cited_raw = row["Cited Cases"].strip()
            if cited_raw.startswith('"""') and cited_raw.endswith('"""'):
                cited_raw = cited_raw[3:-3]
            try:
                cited = ast.literal_eval(cited_raw)
            except Exception:
                cited = []
            if isinstance(cited, str):
                # Some rows parse to a quoted list string (e.g. "['a.txt', 'b.txt']").
                # Parse one more time to get the actual list.
                try:
                    cited = ast.literal_eval(cited)
                except Exception:
                    cited = []
            if not isinstance(cited, list):
                cited = []
            gold[query_id] = [str(x) for x in cited]
    return gold


def precision_at_k(topk: Sequence[str], relevant: set, k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for x in topk if x in relevant)
    return hits / float(k)


def recall_at_k(topk: Sequence[str], relevant: set) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for x in topk if x in relevant)
    return hits / float(len(relevant))


def reciprocal_rank_at_k(topk: Sequence[str], relevant: set) -> float:
    for idx, x in enumerate(topk, start=1):
        if x in relevant:
            return 1.0 / float(idx)
    return 0.0


def average_precision_at_k(topk: Sequence[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    ap_sum = 0.0
    for rank, x in enumerate(topk, start=1):
        if x in relevant:
            hits += 1
            ap_sum += hits / float(rank)
    denom = float(min(len(relevant), k))
    return ap_sum / denom if denom > 0 else 0.0


def ndcg_at_k(topk: Sequence[str], relevant: set, k: int) -> float:
    dcg = 0.0
    for rank, x in enumerate(topk, start=1):
        if x in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = min(len(relevant), k)
    if ideal == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(gold: Dict[str, List[str]], pred: Dict[str, List[str]], ks: Sequence[int]):
    query_ids = list(gold.keys())
    results = {}

    for k in ks:
        tp = 0
        fp = 0
        fn = 0

        p_vals = []
        r_vals = []
        rr_vals = []
        ap_vals = []
        ndcg_vals = []

        for qid in query_ids:
            rel = set(gold.get(qid, []))
            topk = pred.get(qid, [])[:k]

            hits = sum(1 for x in topk if x in rel)
            tp += hits
            fp += len(topk) - hits
            fn += len(rel - set(topk))

            p_vals.append(precision_at_k(topk, rel, k))
            r_vals.append(recall_at_k(topk, rel))
            rr_vals.append(reciprocal_rank_at_k(topk, rel))
            ap_vals.append(average_precision_at_k(topk, rel, k))
            ndcg_vals.append(ndcg_at_k(topk, rel, k))

        micro_p = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        micro_r = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        micro_f1 = (2.0 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

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
    parser = argparse.ArgumentParser(description="Evaluate retrieval predictions using test.csv.")
    parser.add_argument("--gold-csv", default="../dataset/ik_test 4/test.csv", help="Gold CSV path")
    parser.add_argument("--pred", default="../retrieval-predictions-dual.json", help="Predictions JSON path")
    parser.add_argument("--ks", default="1,5,10,20", help="Comma-separated k values")
    parser.add_argument("--out", default="retrieval_eval_from_csv_at_k_results.json", help="Output JSON path")
    args = parser.parse_args()

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    gold = load_gold_from_csv(args.gold_csv)
    pred = load_predictions(args.pred)
    results = evaluate(gold, pred, ks)

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
