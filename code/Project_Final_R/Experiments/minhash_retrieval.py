"""
minhash_retrieval.py
=====================
MinHash / Jaccard-based verbatim passage detection for Prior Case Retrieval.

Why this is fundamentally different from TF-IDF
-------------------------------------------------
TF-IDF cosine:  weighted bag-of-words, order-insensitive, IDF dampens common phrases
MinHash Jaccard: unweighted SET similarity over SHINGLES (contiguous chunks)
                 → identical phrases contribute equally regardless of frequency
                 → directly measures how much text is literally shared

The hypothesis
--------------
Indian legal judgements frequently COPY verbatim passages from prior cases:
  • The holding of a prior case (quoted directly)
  • A statutory provision (word for word)
  • A test or principle established in an earlier case

TF-IDF down-weights these because they appear in many documents.
MinHash gives them FULL credit because set Jaccard ignores IDF.

A case that copies 20% of another case verbatim has Jaccard ≈ 0.2.
That's a strong and reliable signal TF-IDF cannot capture.

MinHash algorithm
-----------------
1. Shingle document: extract all contiguous chunks of k tokens
2. Apply b × r hash functions → MinHash signature (b bands × r rows)
3. Two documents collide in a band iff they share ≥ (1/b)^(1/r) Jaccard
4. At query time: compute exact Jaccard between query shingles and
   all candidates (or use LSH to find candidates first)

Shingle types tested
--------------------
  word-k  : k consecutive words  (k=3,5,7,10)
  char-k  : k consecutive characters (k=20,30,50)  ← robust to OCR/formatting

Combination with TF-IDF
------------------------
  final = α · tfidf_score + (1-α) · minhash_jaccard

Run
---
    python3 minhash_retrieval.py --data_dir /path/to/dataset/ --split train

Parameter grid
--------------
  Edit CONFIGS below.  Comment out any line to skip.
"""

import os
import re
import math
import argparse
import hashlib
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import (
    load_split, clean_text, evaluate_all,
    save_results, print_results_table, save_results_csv, z_norm,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS  (edit here)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "./"
SPLIT    = "train"
TOP_K    = 1000
OUTPUT   = "results/minhash_results.json"
K_VALUES = [5, 6, 7, 8, 9, 10, 11, 15, 20]

N_HASH   = 128   # MinHash signature size (larger = more accurate Jaccard est.)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGS  ← comment out any line to skip
# (shingle_type, shingle_size, combination, alpha)
#
#   shingle_type : "word" | "char"
#   shingle_size : tokens or characters per shingle
#   combination  : "minhash_only" | "tfidf_5gram" | "tfidf_1gram"
#   alpha        : weight of TF-IDF score (1-alpha = MinHash weight)
#                  only used when combination != "minhash_only"
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS: List[Tuple] = [
    # ── Pure MinHash (standalone) ─────────────────────────────────────────────
    ("word",  3,  "minhash_only", 0.0),
    ("word",  5,  "minhash_only", 0.0),
    ("word",  7,  "minhash_only", 0.0),
    ("word",  10, "minhash_only", 0.0),
    ("char",  20, "minhash_only", 0.0),
    ("char",  30, "minhash_only", 0.0),
    ("char",  50, "minhash_only", 0.0),
    ("char",  80, "minhash_only", 0.0),

    # ── MinHash + 5-gram TF-IDF (Z-normalised combination) ────────────────────
    ("word",  5,  "tfidf_5gram",  0.7),
    ("word",  5,  "tfidf_5gram",  0.8),
    ("word",  5,  "tfidf_5gram",  0.9),
    ("word",  7,  "tfidf_5gram",  0.7),
    ("word",  7,  "tfidf_5gram",  0.8),
    ("word",  10, "tfidf_5gram",  0.7),
    ("word",  10, "tfidf_5gram",  0.8),
    ("char",  20, "tfidf_5gram",  0.7),
    ("char",  20, "tfidf_5gram",  0.8),
    ("char",  30, "tfidf_5gram",  0.7),
    ("char",  30, "tfidf_5gram",  0.8),
    ("char",  50, "tfidf_5gram",  0.7),
    ("char",  50, "tfidf_5gram",  0.8),
    ("char",  50, "tfidf_5gram",  0.9),

    # ── Equal weight ──────────────────────────────────────────────────────────
    ("word",  5,  "tfidf_5gram",  0.5),
    ("word",  7,  "tfidf_5gram",  0.5),
    ("char",  30, "tfidf_5gram",  0.5),
]


# ─────────────────────────────────────────────────────────────────────────────
# SHINGLING
# ─────────────────────────────────────────────────────────────────────────────

_CITE = re.compile(r'\[?\?CITATION\?\]?|<CITATION_\d+>', re.IGNORECASE)

def _clean_raw(text: str) -> str:
    text = _CITE.sub(" ", text)
    return re.sub(r'\s+', ' ', text.lower()).strip()


def word_shingles(text: str, k: int) -> Set[str]:
    words = _clean_raw(text).split()
    return {" ".join(words[i: i+k]) for i in range(len(words) - k + 1)}


def char_shingles(text: str, k: int) -> Set[str]:
    s = _clean_raw(text)
    return {s[i: i+k] for i in range(len(s) - k + 1)}


def get_shingles(text: str, stype: str, size: int) -> Set[str]:
    if stype == "word":
        return word_shingles(text, size)
    return char_shingles(text, size)


# ─────────────────────────────────────────────────────────────────────────────
# MINHASH
# ─────────────────────────────────────────────────────────────────────────────

# Generate N_HASH random hash function coefficients (a, b) for h(x) = (ax+b) mod p
_PRIME = (1 << 31) - 1   # large Mersenne prime
_RNG   = np.random.RandomState(42)
_A     = _RNG.randint(1, _PRIME, size=N_HASH, dtype=np.int64)
_B     = _RNG.randint(0, _PRIME, size=N_HASH, dtype=np.int64)


def _hash_str(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % _PRIME


def minhash_signature(shingles: Set[str]) -> np.ndarray:
    """Compute MinHash signature of length N_HASH for a set of shingles."""
    sig = np.full(N_HASH, _PRIME, dtype=np.int64)
    for shingle in shingles:
        h = _hash_str(shingle)
        hashes = (_A * h + _B) % _PRIME
        sig = np.minimum(sig, hashes)
    return sig


def jaccard_from_minhash(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Estimate Jaccard similarity from MinHash signatures."""
    return float((sig_a == sig_b).sum()) / N_HASH


def exact_jaccard(a: Set[str], b: Set[str]) -> float:
    """Exact Jaccard — used when sets are small enough."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--split",    default=SPLIT)
    parser.add_argument("--top_k",    type=int, default=TOP_K)
    parser.add_argument("--output",   default=OUTPUT)
    args = parser.parse_args()

    queries, candidates, relevance = load_split(args.data_dir, args.split)
    cand_ids = list(candidates.keys())
    all_results = []

    # ── Build 5-gram TF-IDF index (for hybrid configs) ───────────────────────
    print("\nBuilding 5-gram TF-IDF index ...")
    c_sw = {cid: " ".join(clean_text(t, True, 1)) for cid, t in candidates.items()}
    q_sw = {qid: " ".join(clean_text(t, True, 1)) for qid, t in queries.items()}

    vec5 = TfidfVectorizer(ngram_range=(5,5), min_df=2, max_df=0.95,
                           sublinear_tf=True, norm="l2")
    C5   = vec5.fit_transform([c_sw[c] for c in cand_ids])

    print("Computing 5-gram TF-IDF scores ...")
    tfidf5_scores: Dict[str, Dict[str, float]] = {}
    for qid in relevance:
        if qid not in q_sw: continue
        qvec = vec5.transform([q_sw[qid]])
        sims = cosine_similarity(qvec, C5)[0]
        tfidf5_scores[qid] = {cand_ids[i]: float(sims[i])
                               for i in range(len(cand_ids))}

    # ── Shingle + MinHash cache ────────────────────────────────────────────────
    shingle_cache: Dict[Tuple, Dict[str, Set[str]]]    = {}
    minhash_cache: Dict[Tuple, Dict[str, np.ndarray]]  = {}

    for (stype, ssize, combo, alpha) in CONFIGS:
        name = (f"MinHash_{stype}{ssize}_"
                + ("only" if combo=="minhash_only" else f"{combo}_a={alpha}"))
        print(f"\n{'─'*64}\n  {name}\n{'─'*64}")

        key = (stype, ssize)

        # Compute shingles (cached)
        if key not in shingle_cache:
            print(f"  Shingling corpus ({stype}, k={ssize}) ...")
            shingle_cache[key] = {
                did: get_shingles(text, stype, ssize)
                for did, text in {**candidates, **queries}.items()
            }

        if key not in minhash_cache:
            print(f"  Computing MinHash signatures ...")
            minhash_cache[key] = {
                did: minhash_signature(shingle_cache[key][did])
                for did in shingle_cache[key]
            }

        sc  = shingle_cache[key]
        mhc = minhash_cache[key]
        cand_sigs = np.stack([mhc[c] for c in cand_ids])  # (N_cands, N_HASH)

        results: Dict[str, List[str]] = {}
        for qid in relevance:
            if qid not in mhc: continue

            q_sig  = mhc[qid]                          # (N_HASH,)
            jac_scores = (cand_sigs == q_sig).mean(axis=1)  # (N_cands,)

            if combo == "minhash_only":
                final = jac_scores
            else:
                tf_scores = np.array([
                    tfidf5_scores.get(qid, {}).get(cid, 0.0) for cid in cand_ids
                ])
                # Z-normalise both then combine
                tf_z  = (tf_scores  - tf_scores.mean())  / (tf_scores.std()  + 1e-10)
                jac_z = (jac_scores - jac_scores.mean()) / (jac_scores.std() + 1e-10)
                final = alpha * tf_z + (1 - alpha) * jac_z

            order = np.argsort(-final)[:args.top_k]
            results[qid] = [cand_ids[i] for i in order]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MicroF1@10",
                        title="MinHash Jaccard — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
