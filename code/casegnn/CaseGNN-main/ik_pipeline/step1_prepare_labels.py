"""
Step 1 – Prepare label files and BM25 hard negatives for the ik dataset.

Produces:
  output/labels/train_labels.json        – {query.txt: [relevant_candidate.txt, ...]}
  output/labels/test_labels.json
  output/labels/hard_neg_top50_train.json – JSONL, one JSON object per line
  output/labels/test_candidate_list.json  – {query.txt: [all_candidate.txt, ...]}
  output/labels/train_case_list.json      – sorted list of all train case IDs
  output/labels/test_case_list.json       – sorted list of all test case IDs
"""
import json
import os
import sys
from collections import defaultdict
from tqdm import tqdm

# BM25
from rank_bm25 import BM25Okapi

from config import (
    IK_TRAIN_DIR, IK_TEST_DIR, LABEL_DIR, ensure_dirs
)


def load_ik_json(path):
    """Read ik-format JSON and return {query_name: [relevant_candidate_names]}."""
    with open(path, "r") as f:
        data = json.load(f)
    label_dict = {}
    for entry in data["Query Set"]:
        qname = entry["query_name"]
        cands = entry["relevant candidates"]
        label_dict[qname] = cands
    return label_dict


def gather_case_ids(ik_dir):
    """Return sorted list of ALL unique case file names (query + candidate)."""
    ids = set()
    for subdir in ["query", "candidate"]:
        d = os.path.join(ik_dir, subdir)
        if os.path.isdir(d):
            for fname in os.listdir(d):
                if fname.endswith(".txt"):
                    ids.add(fname)
    return sorted(ids)


def read_case_text(ik_dir, case_name):
    """Read case text; look in query/ first, then candidate/."""
    for subdir in ["query", "candidate"]:
        p = os.path.join(ik_dir, subdir, case_name)
        if os.path.isfile(p):
            with open(p, "r", errors="replace") as f:
                return f.read()
    return ""


def tokenize_simple(text):
    """Whitespace tokenisation (good enough for BM25)."""
    return text.lower().split()


def compute_bm25_hard_negatives(label_dict, ik_dir, all_case_names, top_k=50):
    """
    For each query, compute BM25 scores against all cases and pick the top-k
    *non-relevant* cases as hard negatives.
    Returns JSONL-style list of dicts: [{query: [neg1, neg2, ...]}, ...]
    """
    print("Building BM25 index …")
    corpus_tokens = []
    for cname in tqdm(all_case_names, desc="Tokenising"):
        text = read_case_text(ik_dir, cname)
        corpus_tokens.append(tokenize_simple(text))

    bm25 = BM25Okapi(corpus_tokens)
    name2idx = {n: i for i, n in enumerate(all_case_names)}

    hard_neg_list = []
    for qname, rel_list in tqdm(label_dict.items(), desc="BM25 scoring"):
        query_text = read_case_text(ik_dir, qname)
        scores = bm25.get_scores(tokenize_simple(query_text))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        rel_set = set(rel_list) | {qname}
        negatives = []
        for idx in ranked:
            cand = all_case_names[idx]
            if cand not in rel_set:
                negatives.append(cand)
            if len(negatives) >= top_k:
                break
        hard_neg_list.append({qname: negatives})

    return hard_neg_list


def main():
    ensure_dirs()

    # ── Load labels ──────────────────────────────────────────────────────
    train_labels = load_ik_json(os.path.join(IK_TRAIN_DIR, "train.json"))
    test_labels  = load_ik_json(os.path.join(IK_TEST_DIR, "test.json"))

    # ── Gather all case IDs per split ────────────────────────────────────
    train_cases = gather_case_ids(IK_TRAIN_DIR)
    test_cases  = gather_case_ids(IK_TEST_DIR)

    print(f"Train: {len(train_labels)} queries, {len(train_cases)} total cases")
    print(f"Test : {len(test_labels)} queries, {len(test_cases)} total cases")

    # Filter labels to only include candidates that exist in the pool
    train_cases_set = set(train_cases)
    test_cases_set = set(test_cases)

    for qname in list(train_labels.keys()):
        train_labels[qname] = [c for c in train_labels[qname] if c in train_cases_set]
        if not train_labels[qname]:
            del train_labels[qname]
            print(f"  Warning: removed train query {qname} (no valid candidates in pool)")

    for qname in list(test_labels.keys()):
        test_labels[qname] = [c for c in test_labels[qname] if c in test_cases_set]
        if not test_labels[qname]:
            del test_labels[qname]
            print(f"  Warning: removed test query {qname} (no valid candidates in pool)")

    # ── Save labels ──────────────────────────────────────────────────────
    with open(os.path.join(LABEL_DIR, "train_labels.json"), "w") as f:
        json.dump(train_labels, f, indent=2)
    with open(os.path.join(LABEL_DIR, "test_labels.json"), "w") as f:
        json.dump(test_labels, f, indent=2)

    # ── Save full case lists ─────────────────────────────────────────────
    with open(os.path.join(LABEL_DIR, "train_case_list.json"), "w") as f:
        json.dump(train_cases, f)
    with open(os.path.join(LABEL_DIR, "test_case_list.json"), "w") as f:
        json.dump(test_cases, f)

    # ── Test candidate list (all test cases for each test query) ─────────
    test_candidate_dict = {}
    for qname in test_labels:
        test_candidate_dict[qname] = [c for c in test_cases if c != qname]
    with open(os.path.join(LABEL_DIR, "test_candidate_list.json"), "w") as f:
        json.dump(test_candidate_dict, f)

    # ── BM25 hard negatives for training ─────────────────────────────────
    hn_path = os.path.join(LABEL_DIR, "hard_neg_top50_train.json")
    if os.path.exists(hn_path):
        print(f"BM25 hard negatives already cached at {hn_path}, skipping.")
    else:
        hard_negs = compute_bm25_hard_negatives(train_labels, IK_TRAIN_DIR, train_cases)
        with open(hn_path, "w") as f:
            for entry in hard_negs:
                f.write(json.dumps(entry) + "\n")

    print("Done. Label files saved to:", LABEL_DIR)


if __name__ == "__main__":
    main()
