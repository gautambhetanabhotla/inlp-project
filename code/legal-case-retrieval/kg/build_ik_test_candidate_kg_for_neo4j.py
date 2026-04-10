#!/usr/bin/env python3
"""
Build a Neo4j-ready knowledge graph from ik_test candidate predicted labels.

Input:
- ik_test_candidate_pred.json: {case_id: [label1, label2, ...]}

Output files (CSV):
- <prefix>_case_nodes.csv
- <prefix>_label_nodes.csv
- <prefix>_case_label_edges.csv
- <prefix>_case_case_edges.csv

Graph model:
- (:Case {case_id, label_count})
- (:Label {name})
- (:Case)-[:HAS_LABEL]->(:Label)
- (:Case)-[:SIMILAR_LABELS {shared_count, shared_labels}]-(:Case)
"""

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_case_labels(case_to_labels):
    norm = {}
    for case_id, labels in case_to_labels.items():
        unique_labels = sorted({str(x).strip() for x in labels if str(x).strip()})
        norm[str(case_id)] = unique_labels
    return norm


def filter_cases(case_to_labels, candidate_dir: str):
    if not candidate_dir:
        return case_to_labels

    cand_path = Path(candidate_dir)
    allowed = {p.name for p in cand_path.glob("*.txt")}
    return {cid: labels for cid, labels in case_to_labels.items() if cid in allowed}


def build_graph(case_to_labels):
    case_ids = sorted(case_to_labels.keys())
    all_labels = sorted({lab for labs in case_to_labels.values() for lab in labs})

    label_to_cases = defaultdict(list)
    for cid in case_ids:
        for lab in case_to_labels[cid]:
            label_to_cases[lab].append(cid)

    pair_to_labels = defaultdict(set)
    for lab, cases in label_to_cases.items():
        ucases = sorted(set(cases))
        for left, right in itertools.combinations(ucases, 2):
            pair_to_labels[(left, right)].add(lab)

    return case_ids, all_labels, pair_to_labels


def write_case_nodes(path, case_ids, case_to_labels):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "label_count", "labels"])
        w.writeheader()
        for cid in case_ids:
            labels = case_to_labels[cid]
            w.writerow({
                "case_id": cid,
                "label_count": len(labels),
                "labels": "|".join(labels),
            })


def write_label_nodes(path, labels):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["label_name"])
        w.writeheader()
        for lab in labels:
            w.writerow({"label_name": lab})


def write_case_label_edges(path, case_ids, case_to_labels):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "label_name"])
        w.writeheader()
        for cid in case_ids:
            for lab in case_to_labels[cid]:
                w.writerow({"case_id": cid, "label_name": lab})


def write_case_case_edges(path, pair_to_labels, min_shared):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["source_case_id", "target_case_id", "shared_count", "shared_labels"],
        )
        w.writeheader()
        for (left, right), labels in sorted(pair_to_labels.items(), key=lambda x: (-len(x[1]), x[0][0], x[0][1])):
            if len(labels) < min_shared:
                continue
            slabels = sorted(labels)
            w.writerow(
                {
                    "source_case_id": left,
                    "target_case_id": right,
                    "shared_count": len(slabels),
                    "shared_labels": "|".join(slabels),
                }
            )


def main():
    ap = argparse.ArgumentParser(description="Build Neo4j KG for ik_test candidate cases from predicted labels.")
    ap.add_argument("--labels-json", default="../ik_test_candidate_pred.json", help="Path to case->labels JSON.")
    ap.add_argument("--candidate-dir", default="../dataset/ik_test 4/candidate", help="Optional candidate directory filter.")
    ap.add_argument("--min-shared", type=int, default=1, help="Minimum shared labels for Case-Case edge.")
    ap.add_argument("--out-prefix", default="ik_test_candidate_kg", help="Output prefix (no extension).")
    args = ap.parse_args()

    raw = load_json(args.labels_json)
    case_to_labels = normalize_case_labels(raw)
    case_to_labels = filter_cases(case_to_labels, args.candidate_dir)

    case_ids, all_labels, pair_to_labels = build_graph(case_to_labels)

    case_nodes = f"{args.out_prefix}_case_nodes.csv"
    label_nodes = f"{args.out_prefix}_label_nodes.csv"
    case_label_edges = f"{args.out_prefix}_case_label_edges.csv"
    case_case_edges = f"{args.out_prefix}_case_case_edges.csv"
    graph_json = f"{args.out_prefix}.json"

    write_case_nodes(case_nodes, case_ids, case_to_labels)
    write_label_nodes(label_nodes, all_labels)
    write_case_label_edges(case_label_edges, case_ids, case_to_labels)
    write_case_case_edges(case_case_edges, pair_to_labels, args.min_shared)

    with open(graph_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cases": len(case_ids),
                "labels": len(all_labels),
                "pair_edges_all": len(pair_to_labels),
                "min_shared": args.min_shared,
                "case_nodes_csv": case_nodes,
                "label_nodes_csv": label_nodes,
                "case_label_edges_csv": case_label_edges,
                "case_case_edges_csv": case_case_edges,
            },
            f,
            indent=2,
        )

    kept_edges = sum(1 for labs in pair_to_labels.values() if len(labs) >= args.min_shared)
    print("Neo4j KG export created")
    print(f"Cases: {len(case_ids)}")
    print(f"Labels: {len(all_labels)}")
    print(f"Case-Case edges (>= {args.min_shared} shared): {kept_edges}")
    print(f"Case-Label edges: {sum(len(v) for v in case_to_labels.values())}")
    print(f"Prefix: {args.out_prefix}")


if __name__ == "__main__":
    main()
