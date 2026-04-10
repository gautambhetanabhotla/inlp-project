"""
Retrieve relevant cases for a given query case using a trained CaseGNN model.

Usage:
  python retrieve.py --query_file /path/to/case.txt --top_k 10
  python retrieve.py --query_id 0001104022 --top_k 10
  python retrieve.py --model_path output/experiments/.../best_model.pt
"""
import argparse
import json
import os
import sys

import torch
import dgl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import CaseGNN

from config import (
    GRAPH_DIR, EMBEDDING_DIR, LABEL_DIR, IK_TEST_DIR, IK_TRAIN_DIR,
    IN_DIM, H_DIM, OUT_DIM, DROPOUT, NUM_HEAD, ID_PAD_LEN
)


def load_casegnn_embeddings(split="test"):
    """Load precomputed CaseGNN embeddings and case name list."""
    emb_path = os.path.join(EMBEDDING_DIR, f"ik_{split}_casegnn_embedding.pt")
    list_path = os.path.join(EMBEDDING_DIR, f"ik_{split}_casegnn_embedding_case_list.json")

    embeddings = torch.load(emb_path, map_location="cpu", weights_only=False)
    with open(list_path) as f:
        case_list = json.load(f)
    return embeddings, case_list


def retrieve(query_embedding, pool_embeddings, pool_case_list, top_k=10, exclude_id=None):
    """
    Compute cosine similarity and return top-k most similar cases.
    """
    q_norm = query_embedding / query_embedding.norm()
    p_norm = pool_embeddings / pool_embeddings.norm(dim=1, keepdim=True)
    sim = torch.mv(p_norm, q_norm)

    if exclude_id and exclude_id in pool_case_list:
        idx = pool_case_list.index(exclude_id)
        sim[idx] = float("-inf")

    top_vals, top_idxs = torch.topk(sim, min(top_k, len(pool_case_list)))
    results = []
    for val, idx in zip(top_vals.tolist(), top_idxs.tolist()):
        results.append((pool_case_list[idx], val))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query_id", type=str, help="Case ID (10-digit, e.g. 0001104022)")
    p.add_argument("--query_file", type=str, help="Path to a raw case text file")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"],
                   help="Which pool to search in")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--model_path", type=str, default=None,
                   help="Path to trained model checkpoint (optional; uses precomputed embeddings by default)")
    args = p.parse_args()

    if not args.query_id and not args.query_file:
        p.error("Provide --query_id or --query_file")

    # Load precomputed embeddings
    pool_emb, pool_cases = load_casegnn_embeddings(args.split)
    print(f"Loaded {len(pool_cases)} case embeddings from {args.split} pool")

    if args.query_id:
        qid = args.query_id.zfill(ID_PAD_LEN)
        if qid not in pool_cases:
            print(f"Error: case {qid} not found in the {args.split} embedding pool")
            sys.exit(1)
        q_idx = pool_cases.index(qid)
        q_emb = pool_emb[q_idx]
        exclude = qid
    else:
        print("Note: --query_file requires re-encoding through the model.")
        print("For now, please use --query_id with a case already in the pool.")
        sys.exit(1)

    results = retrieve(q_emb, pool_emb, pool_cases, args.top_k, exclude_id=exclude)

    print(f"\nTop-{args.top_k} relevant cases for query {qid}:")
    print("-" * 50)
    for rank_i, (case_id, score) in enumerate(results, 1):
        print(f"  {rank_i:2d}. {case_id}.txt  (score: {score:.4f})")

    # Check against ground truth if available
    labels_path = os.path.join(LABEL_DIR, f"{args.split}_labels.json")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            labels = json.load(f)
        qname = qid + ".txt"
        if qname in labels:
            gold = set(labels[qname])
            retrieved = set(c + ".txt" for c, _ in results)
            hits = retrieved & gold
            print(f"\nGround truth: {len(gold)} relevant cases")
            print(f"Hits in top-{args.top_k}: {len(hits)}")
            if hits:
                print(f"  Matched: {sorted(hits)}")


if __name__ == "__main__":
    main()
