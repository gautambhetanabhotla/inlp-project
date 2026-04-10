#!/usr/bin/env python3
"""
Run KG analyses on the ik_test candidate graph in Neo4j.
All results are compact (limited rows) and saved to kg_analyses_results.json.

Usage:
  python run_kg_analyses.py --uri bolt://127.0.0.1:7687 --user neo4j --password 'Bibek2005*'
"""
from __future__ import annotations
import argparse, json
from neo4j import GraphDatabase


def run(driver, q, **params):
    with driver.session(database="neo4j") as s:
        return [dict(r) for r in s.run(q, **params)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri",      default="bolt://127.0.0.1:7687")
    ap.add_argument("--user",     default="neo4j")
    ap.add_argument("--password", required=True)
    ap.add_argument("--out",      default="kg_analyses_results.json")
    args = ap.parse_args()

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    results = {}

    # ── 1. Graph summary ──────────────────────────────────────────────────────
    results["1_graph_summary"] = run(driver, """
        MATCH (c:Case)  WITH count(c) AS cases
        MATCH (l:Label) WITH cases, count(l) AS labels
        MATCH ()-[r:HAS_LABEL]->()   WITH cases, labels, count(r) AS hl
        MATCH ()-[r:SIMILAR_LABELS]->() RETURN cases, labels, hl AS has_label_edges,
              count(r) AS similar_labels_edges
    """)

    # ── 2. Top-20 labels by case frequency ───────────────────────────────────
    results["2_top20_labels_by_freq"] = run(driver, """
        MATCH (c:Case)-[:HAS_LABEL]->(l:Label)
        RETURN l.name AS label, count(c) AS case_count
        ORDER BY case_count DESC LIMIT 20
    """)

    # ── 3. Label count distribution (how many cases have N labels) ───────────
    results["3_label_count_distribution"] = run(driver, """
        MATCH (c:Case)
        RETURN c.label_count AS num_labels, count(c) AS num_cases
        ORDER BY num_labels
    """)

    # ── 4. Top-20 hub cases (highest SIMILAR_LABELS degree) ──────────────────
    results["4_top20_hub_cases"] = run(driver, """
        MATCH (c:Case)-[r:SIMILAR_LABELS]-()
        RETURN c.case_id AS case_id, c.label_count AS label_count,
               count(r) AS degree
        ORDER BY degree DESC LIMIT 20
    """)

    # ── 5. Top-20 label co-occurrence pairs ───────────────────────────────────
    results["5_top20_label_cooccurrence"] = run(driver, """
        MATCH (l1:Label)<-[:HAS_LABEL]-(c:Case)-[:HAS_LABEL]->(l2:Label)
        WHERE l1.name < l2.name
        RETURN l1.name AS label_a, l2.name AS label_b, count(c) AS co_cases
        ORDER BY co_cases DESC LIMIT 20
    """)

    # ── 6. Shared-count histogram (buckets 1,2,3,4,5,6-10,11+) ──────────────
    results["6_shared_count_histogram"] = run(driver, """
        MATCH ()-[r:SIMILAR_LABELS]->()
        WITH r.shared_count AS sc,
             CASE
               WHEN r.shared_count >= 11 THEN '11+'
               WHEN r.shared_count >= 6  THEN '6-10'
               ELSE toString(r.shared_count)
             END AS bucket
        RETURN bucket, count(sc) AS edge_count
        ORDER BY bucket
    """)

    # ── 7. Cases with zero labels ─────────────────────────────────────────────
    results["7_cases_with_zero_labels"] = run(driver, """
        MATCH (c:Case) WHERE c.label_count = 0
        RETURN count(c) AS zero_label_cases
    """)

    # ── 8. Avg/max/min shared_count on SIMILAR_LABELS edges ──────────────────
    results["8_edge_weight_stats"] = run(driver, """
        MATCH ()-[r:SIMILAR_LABELS]->()
        RETURN round(avg(r.shared_count),2) AS avg_shared,
               max(r.shared_count)          AS max_shared,
               min(r.shared_count)          AS min_shared
    """)

    # ── 9. Top-20 strongly connected case pairs (high shared_count) ───────────
    results["9_top20_strongest_pairs"] = run(driver, """
        MATCH (a:Case)-[r:SIMILAR_LABELS]->(b:Case)
        RETURN a.case_id AS case_a, b.case_id AS case_b,
               r.shared_count AS shared, r.shared_labels AS labels
        ORDER BY r.shared_count DESC LIMIT 20
    """)

    # ── 10. Dense clusters: cases connected to ≥50 others with shared≥3 ───────
    results["10_dense_cluster_sizes"] = run(driver, """
        MATCH (c:Case)-[r:SIMILAR_LABELS]->(n:Case)
        WHERE r.shared_count >= 3
        WITH c, count(n) AS strong_neighbours
        WHERE strong_neighbours >= 50
        RETURN strong_neighbours AS min_3_shared_degree, count(c) AS num_cases
        ORDER BY min_3_shared_degree DESC LIMIT 20
    """)

    driver.close()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Saved to {args.out}")
    print(f"\n=== Summary ===")
    print(f"Graph:        {results['1_graph_summary']}")
    print(f"Top label:    {results['2_top20_labels_by_freq'][0] if results['2_top20_labels_by_freq'] else 'n/a'}")
    print(f"Top hub case: {results['4_top20_hub_cases'][0] if results['4_top20_hub_cases'] else 'n/a'}")
    print(f"Max shared:   {results['8_edge_weight_stats']}")


if __name__ == "__main__":
    main()
