"""
citation_advanced.py
=====================
Advanced citation-based retrieval for Indian legal documents.

What the basic citation method did (and why it was limited)
------------------------------------------------------------
  • Extracted citations as a BINARY SET per document
  • Computed Jaccard / BC / IDF-cosine over those binary sets
  • Combined with BM25 (ng=1 or ng=5) as the base retriever

Four novel improvements implemented here
-----------------------------------------

1. FREQUENCY-WEIGHTED CITATION VECTORS (new)
   ─────────────────────────────────────────
   A case cited 4x in a judgment is the central precedent.
   A case cited 1x is a passing reference.
   We represent each document as a TF-IDF vector over its citations,
   where TF = count of times a citation appears, and IDF = inverse
   document frequency of that citation across all candidates.
   Cosine similarity on these vectors is richer than Jaccard.

2. CO-CITATION PARAGRAPH GRAPH (new)
   ───────────────────────────────────
   When two citations appear in the same paragraph, they address the
   same legal point. Build a weighted co-citation graph per document:
   edge(A,B) += 1 whenever A and B co-occur in same paragraph.
   Similarity between two documents = inner product of their co-citation
   edge vectors (normalised). This captures shared legal ARGUMENTS,
   not just shared cited cases.

3. CITATION CONTEXT TF-IDF (new)
   ─────────────────────────────
   Extract the 30-word window around each citation mention.
   Build a TF-IDF representation of this context text per document.
   Two documents citing the same case in similar context = same legal point.
   This is strictly more informative than bare set overlap.

4. BEST BASE RETRIEVER: 5-GRAM TF-IDF (not BM25)
   ─────────────────────────────────────────────────
   The existing hybrid used BM25 (ng=1 or ng=5) as base.
   Our experiments show 5-gram sublinear TF-IDF is the best single method
   (MAP=0.4528 vs BM25ng=5 MAP=0.3804).
   Swapping the base retriever alone should push the hybrid much higher.

5. POSITION-WEIGHTED CITATION SIMILARITY (new)
   ──────────────────────────────────────────────
   First unique citations in a document = main precedents.
   Weight citations by their position: first citation has weight 1.0,
   later citations decay. Overlap among early citations is more diagnostic.

All methods are combined with the 5-gram TF-IDF base using Z-normalised
linear interpolation and multiplicative reranking.

Run
---
    python3 citation_advanced.py --data_dir /path/to/dataset/ --split train

Target: MicroF1@10 > 0.4470 (current best with TF-IDF n-gram)
"""

import os
import re
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from utils import (
    load_split, clean_text, evaluate_all,
    save_results, print_results_table, save_results_csv, z_norm,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "./"
SPLIT    = "train"
TOP_K    = 1000
OUTPUT   = "results/citation_advanced_results.json"
K_VALUES = [5, 6, 7, 8, 9, 10, 11, 15, 20]

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGS  ← comment out any line to skip
#
# Each entry: (method, base_retriever, alpha, rerank_depth)
#
#   method        : citation similarity method (see below)
#   base_retriever: "tfidf5" | "tfidf25" | "none"
#   alpha         : weight of base retriever score (1-alpha = citation weight)
#                   alpha=0.0 → pure citation similarity
#   rerank_depth  : rerank this many top base-retriever results (1000=all)
#
# Citation methods:
#   "freq_idf"     : TF-IDF vector on citations (frequency weighted + IDF)
#   "freq_bc"      : BC with frequency weights
#   "freq_jaccard" : Jaccard with soft/frequency weighting
#   "cocite"       : co-citation paragraph graph similarity
#   "context"      : citation context window TF-IDF similarity
#   "position"     : position-decay weighted citation overlap
#   "idf_cosine"   : standard IDF-cosine (baseline, binary set)
#   "bc"           : standard BC (baseline, binary set)
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS: List[Tuple] = [
    # ── Baselines (binary, best base retriever) ────────────────────────────────
    ("idf_cosine",   "tfidf5",  0.7, 1000),   # baseline replicated with better base
    ("idf_cosine",   "tfidf5",  0.8, 1000),
    ("idf_cosine",   "tfidf5",  0.6, 1000),
    ("idf_cosine",   "tfidf5",  0.5, 1000),
    ("bc",           "tfidf5",  0.7, 1000),
    ("bc",           "tfidf5",  0.8, 1000),

    # ── Frequency-weighted TF-IDF citation vector ──────────────────────────────
    ("freq_idf",     "tfidf5",  0.7, 1000),
    ("freq_idf",     "tfidf5",  0.8, 1000),
    ("freq_idf",     "tfidf5",  0.6, 1000),
    ("freq_idf",     "tfidf5",  0.5, 1000),
    ("freq_idf",     "tfidf5",  0.9, 1000),
    ("freq_idf",     "tfidf5",  0.7,  200),   # rerank only top-200
    ("freq_idf",     "tfidf5",  0.7,  100),
    ("freq_idf",     "tfidf5",  0.8,  100),
    ("freq_idf",     "none",    0.0, 1000),   # standalone

    # ── Frequency-weighted BC ──────────────────────────────────────────────────
    ("freq_bc",      "tfidf5",  0.7, 1000),
    ("freq_bc",      "tfidf5",  0.8, 1000),
    ("freq_bc",      "tfidf5",  0.6, 1000),
    ("freq_bc",      "tfidf5",  0.9, 1000),
    ("freq_bc",      "none",    0.0, 1000),

    # ── Soft Jaccard (frequency) ───────────────────────────────────────────────
    ("freq_jaccard", "tfidf5",  0.7, 1000),
    ("freq_jaccard", "tfidf5",  0.8, 1000),
    ("freq_jaccard", "tfidf5",  0.6, 1000),
    ("freq_jaccard", "none",    0.0, 1000),

    # ── Co-citation paragraph graph ────────────────────────────────────────────
    ("cocite",       "tfidf5",  0.7, 1000),
    ("cocite",       "tfidf5",  0.8, 1000),
    ("cocite",       "tfidf5",  0.6, 1000),
    ("cocite",       "tfidf5",  0.5, 1000),
    ("cocite",       "tfidf5",  0.9, 1000),
    ("cocite",       "none",    0.0, 1000),

    # ── Citation context TF-IDF ────────────────────────────────────────────────
    ("context",      "tfidf5",  0.7, 1000),
    ("context",      "tfidf5",  0.8, 1000),
    ("context",      "tfidf5",  0.6, 1000),
    ("context",      "tfidf5",  0.5, 1000),
    ("context",      "none",    0.0, 1000),

    # ── Position-decay citation overlap ───────────────────────────────────────
    ("position",     "tfidf5",  0.7, 1000),
    ("position",     "tfidf5",  0.8, 1000),
    ("position",     "tfidf5",  0.6, 1000),
    ("position",     "none",    0.0, 1000),

    # ── COMBINATIONS: freq_idf + cocite (blend two citation signals) ───────────
    ("freq_idf+cocite",   "tfidf5", 0.7, 1000),
    ("freq_idf+cocite",   "tfidf5", 0.8, 1000),
    ("freq_idf+cocite",   "tfidf5", 0.6, 1000),
    ("freq_idf+context",  "tfidf5", 0.7, 1000),
    ("freq_idf+context",  "tfidf5", 0.8, 1000),
    ("freq_idf+position", "tfidf5", 0.7, 1000),
    ("freq_idf+position", "tfidf5", 0.8, 1000),
    ("freq_bc+context",   "tfidf5", 0.7, 1000),
    ("cocite+context",    "tfidf5", 0.7, 1000),
    ("cocite+context",    "tfidf5", 0.8, 1000),
    ("all_cite",          "tfidf5", 0.7, 1000),   # freq_idf + cocite + context
    ("all_cite",          "tfidf5", 0.8, 1000),

    # ── Multiplicative boost instead of linear ─────────────────────────────────
    ("boost_freq_idf",   "tfidf5", 0.0, 1000),
    ("boost_cocite",     "tfidf5", 0.0, 1000),
    ("boost_all",        "tfidf5", 0.0, 1000),

    # ── Without base retriever (standalone citation only) ──────────────────────
    ("all_cite",          "none",   0.0, 1000),
]


# ─────────────────────────────────────────────────────────────────────────────
# CITATION EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

CITE_PAT = re.compile(r'<CITATION_(\d+)>')
PARA_SEP = re.compile(r'\n\d+\.\s+')


def extract_citation_freq(text: str) -> Dict[str, int]:
    """Return {citation_id: count_in_doc}."""
    cites = CITE_PAT.findall(text)
    freq: Dict[str, int] = defaultdict(int)
    for c in cites:
        freq[c] += 1
    return dict(freq)


def extract_citation_order(text: str) -> List[str]:
    """Return citations in order of FIRST appearance (unique)."""
    seen = []
    seen_set = set()
    for m in CITE_PAT.finditer(text):
        c = m.group(1)
        if c not in seen_set:
            seen_set.add(c)
            seen.append(c)
    return seen


def extract_cocitation_pairs(text: str) -> Dict[Tuple[str,str], int]:
    """
    Return {(citeA, citeB): count} for pairs appearing in same paragraph.
    Paragraph = numbered section (e.g. "1. ...", "2. ...").
    """
    paras = PARA_SEP.split(text)
    pair_counts: Dict[Tuple[str,str], int] = defaultdict(int)
    for para in paras:
        cites_in_para = list(set(CITE_PAT.findall(para)))
        cites_in_para.sort()
        for i in range(len(cites_in_para)):
            for j in range(i+1, len(cites_in_para)):
                pair = (cites_in_para[i], cites_in_para[j])
                pair_counts[pair] += 1
    return dict(pair_counts)


def extract_context_text(text: str, window: int = 30) -> str:
    """
    Extract the words surrounding each citation mention.
    Returns a concatenated string of all context windows.
    """
    words = text.split()
    ctx_words = []
    # Find citation positions in word list
    word_str = " ".join(words)
    for m in CITE_PAT.finditer(word_str):
        # Approximate word position by counting spaces before match
        char_pos = m.start()
        word_pos = word_str[:char_pos].count(' ')
        start = max(0, word_pos - window)
        end   = min(len(words), word_pos + window + 1)
        chunk = " ".join(w for w in words[start:end]
                         if not CITE_PAT.match(w))
        ctx_words.append(chunk)
    return " ".join(ctx_words)


# ─────────────────────────────────────────────────────────────────────────────
# SIMILARITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def freq_idf_cosine(freq_a: Dict[str,int], freq_b: Dict[str,int],
                    idf: Dict[str,float]) -> float:
    """TF-IDF cosine where TF = citation frequency, IDF = across candidates."""
    common = set(freq_a) & set(freq_b)
    if not common: return 0.0
    dot = sum(freq_a[c] * freq_b[c] * idf.get(c,1.0)**2 for c in common)
    na  = math.sqrt(sum((freq_a[c] * idf.get(c,1.0))**2 for c in freq_a))
    nb  = math.sqrt(sum((freq_b[c] * idf.get(c,1.0))**2 for c in freq_b))
    return dot / (na * nb) if na * nb else 0.0


def freq_bc(freq_a: Dict[str,int], freq_b: Dict[str,int]) -> float:
    """Bibliographic coupling weighted by citation frequency."""
    common = set(freq_a) & set(freq_b)
    if not common: return 0.0
    shared = sum(min(freq_a[c], freq_b[c]) for c in common)
    total  = math.sqrt(sum(freq_a.values()) * sum(freq_b.values()))
    return shared / total if total else 0.0


def freq_jaccard_soft(freq_a: Dict[str,int], freq_b: Dict[str,int]) -> float:
    """Soft Jaccard: min(freq_a[c], freq_b[c]) / max(freq_a[c], freq_b[c])."""
    all_keys = set(freq_a) | set(freq_b)
    if not all_keys: return 0.0
    num = sum(min(freq_a.get(k,0), freq_b.get(k,0)) for k in all_keys)
    den = sum(max(freq_a.get(k,0), freq_b.get(k,0)) for k in all_keys)
    return num / den if den else 0.0


def cocite_similarity(pairs_a: Dict[Tuple,int], pairs_b: Dict[Tuple,int]) -> float:
    """
    Inner product of co-citation edge vectors (L2-normalised).
    Each unique pair is a dimension; value is co-citation count.
    """
    if not pairs_a or not pairs_b: return 0.0
    common = set(pairs_a) & set(pairs_b)
    if not common: return 0.0
    dot = sum(pairs_a[p] * pairs_b[p] for p in common)
    na  = math.sqrt(sum(v**2 for v in pairs_a.values()))
    nb  = math.sqrt(sum(v**2 for v in pairs_b.values()))
    return dot / (na * nb) if na * nb else 0.0


def position_weighted(order_a: List[str], order_b: List[str],
                      decay: float = 0.7) -> float:
    """
    Weight each citation by 1/(1+rank)^decay.
    Overlap score = sum of geometric mean of weights for shared citations.
    """
    if not order_a or not order_b: return 0.0
    wt_a = {c: 1.0/(1+i)**decay for i, c in enumerate(order_a)}
    wt_b = {c: 1.0/(1+i)**decay for i, c in enumerate(order_b)}
    common = set(wt_a) & set(wt_b)
    if not common: return 0.0
    overlap = sum(math.sqrt(wt_a[c] * wt_b[c]) for c in common)
    na = math.sqrt(sum(w**2 for w in wt_a.values()))
    nb = math.sqrt(sum(w**2 for w in wt_b.values()))
    return overlap / (na * nb) if na * nb else 0.0


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
    N = len(cand_ids)

    print("\nExtracting citation features ...")

    # ── Citation frequency dicts ──────────────────────────────────────────────
    q_freq: Dict[str, Dict[str,int]] = {
        qid: extract_citation_freq(t) for qid, t in queries.items()
    }
    c_freq: Dict[str, Dict[str,int]] = {
        cid: extract_citation_freq(t) for cid, t in candidates.items()
    }

    # ── Citation IDF (over candidate corpus) ──────────────────────────────────
    cite_df: Dict[str, int] = defaultdict(int)
    for freq in c_freq.values():
        for c in freq:
            cite_df[c] += 1
    cite_idf: Dict[str, float] = {
        c: math.log((N+1)/(df+1)) + 1.0 for c, df in cite_df.items()
    }
    print(f"  Unique cited cases in candidate corpus: {len(cite_idf)}")
    print(f"  Queries with citations: {sum(1 for f in q_freq.values() if f)}/{len(q_freq)}")
    print(f"  Candidates with citations: {sum(1 for f in c_freq.values() if f)}/{len(c_freq)}")

    # ── Co-citation pairs ──────────────────────────────────────────────────────
    print("  Extracting co-citation pairs ...")
    q_cocite: Dict[str, Dict[Tuple,int]] = {
        qid: extract_cocitation_pairs(t) for qid, t in queries.items()
    }
    c_cocite: Dict[str, Dict[Tuple,int]] = {
        cid: extract_cocitation_pairs(t) for cid, t in candidates.items()
    }
    n_cocite_q = sum(1 for p in q_cocite.values() if p)
    n_cocite_c = sum(1 for p in c_cocite.values() if p)
    print(f"  Queries with co-citation pairs: {n_cocite_q}")
    print(f"  Candidates with co-citation pairs: {n_cocite_c}")

    # ── Citation order (position) ─────────────────────────────────────────────
    q_order: Dict[str, List[str]] = {
        qid: extract_citation_order(t) for qid, t in queries.items()
    }
    c_order: Dict[str, List[str]] = {
        cid: extract_citation_order(t) for cid, t in candidates.items()
    }

    # ── Citation context TF-IDF ────────────────────────────────────────────────
    print("  Building citation context TF-IDF ...")
    c_ctx_texts = [extract_context_text(candidates[c]) for c in cand_ids]
    q_ctx_texts = {qid: extract_context_text(queries[qid]) for qid in queries}

    ctx_vec = TfidfVectorizer(ngram_range=(2,4), min_df=2, max_df=0.95,
                               sublinear_tf=True, norm="l2")
    try:
        C_ctx = ctx_vec.fit_transform(c_ctx_texts)
        ctx_ok = True
    except ValueError:
        ctx_ok = False
        print("  [WARN] context TF-IDF failed (too sparse); skipping context method")

    # ── 5-gram TF-IDF base retriever ─────────────────────────────────────────
    print("\nBuilding 5-gram TF-IDF base retriever ...")
    c_sw5 = [" ".join(clean_text(candidates[c], True, 1)) for c in cand_ids]
    q_sw5 = {qid: " ".join(clean_text(queries[qid], True, 1)) for qid in queries}

    vec5 = TfidfVectorizer(ngram_range=(5,5), min_df=2, max_df=0.95,
                           sublinear_tf=True, norm="l2")
    C5   = vec5.fit_transform(c_sw5)

    tfidf5_scores: Dict[str, np.ndarray]       = {}
    tfidf5_ranked: Dict[str, List[Tuple]]      = {}
    for qid in relevance:
        if qid not in q_sw5: continue
        qvec = vec5.transform([q_sw5[qid]])
        sims = cosine_similarity(qvec, C5)[0]
        tfidf5_scores[qid] = sims
        order = np.argsort(-sims)
        tfidf5_ranked[qid] = [(cand_ids[i], float(sims[i])) for i in order]

    print(f"  Base retriever built: {C5.shape}")

    # ── Pre-compute citation context scores (slow — do once) ─────────────────
    ctx_scores_cache: Dict[str, np.ndarray] = {}
    if ctx_ok:
        print("Computing citation context scores ...")
        for qid in relevance:
            if qid not in q_ctx_texts: continue
            qtxt = q_ctx_texts[qid]
            if not qtxt.strip():
                ctx_scores_cache[qid] = np.zeros(N)
                continue
            try:
                qvec = ctx_vec.transform([qtxt])
                ctx_scores_cache[qid] = cosine_similarity(qvec, C_ctx)[0]
            except Exception:
                ctx_scores_cache[qid] = np.zeros(N)

    # ── Build cand index map ──────────────────────────────────────────────────
    cand_idx = {cid: i for i, cid in enumerate(cand_ids)}

    # ─────────────────────────────────────────────────────────────────────────
    # RUN CONFIGS
    # ─────────────────────────────────────────────────────────────────────────
    for (method, base_retr, alpha, rerank_depth) in CONFIGS:

        boost = method.startswith("boost_")
        m_key = method.replace("boost_", "") if boost else method

        name = f"CitAdv_{method}_base={base_retr}_a={alpha}_d={rerank_depth}"
        print(f"\n{'─'*64}\n  {name}\n{'─'*64}")

        results: Dict[str, List[str]] = {}

        for qid in relevance:
            if qid not in tfidf5_ranked: continue

            qf   = q_freq.get(qid, {})
            qc   = q_cocite.get(qid, {})
            qo   = q_order.get(qid, [])
            tf5  = tfidf5_scores[qid]
            pool = [cid for cid, _ in tfidf5_ranked[qid]]

            if base_retr == "none":
                top_n = cand_ids   # evaluate over all candidates
                rest  = []
            else:
                top_n = pool[:rerank_depth]
                rest  = pool[rerank_depth:]

            # Compute citation score for each candidate in pool
            cite_sc_arr = np.zeros(N)

            sub_methods = m_key.split("+") if "+" in m_key else [m_key]

            # Accumulate all citation sub-method scores
            subscores = {sm: np.zeros(N) for sm in sub_methods}

            for sm in sub_methods:
                for cid in top_n:
                    idx = cand_idx[cid]
                    cf  = c_freq.get(cid, {})
                    cc  = c_cocite.get(cid, {})
                    co  = c_order.get(cid, [])

                    if sm == "freq_idf":
                        subscores[sm][idx] = freq_idf_cosine(qf, cf, cite_idf)
                    elif sm == "freq_bc":
                        subscores[sm][idx] = freq_bc(qf, cf)
                    elif sm == "freq_jaccard":
                        subscores[sm][idx] = freq_jaccard_soft(qf, cf)
                    elif sm == "cocite":
                        subscores[sm][idx] = cocite_similarity(qc, cc)
                    elif sm == "context" and ctx_ok:
                        subscores[sm][idx] = ctx_scores_cache.get(qid, np.zeros(N))[idx]
                    elif sm == "position":
                        subscores[sm][idx] = position_weighted(qo, co)
                    elif sm == "idf_cosine":
                        # Standard IDF-cosine (binary)
                        qa = set(qf); ca = set(cf)
                        common = qa & ca
                        if common:
                            dot = sum(cite_idf.get(c,1.0)**2 for c in common)
                            na_ = math.sqrt(sum(cite_idf.get(c,1.0)**2 for c in qa))
                            nb_ = math.sqrt(sum(cite_idf.get(c,1.0)**2 for c in ca))
                            subscores[sm][idx] = dot/(na_*nb_) if na_*nb_ else 0.0
                    elif sm == "bc":
                        qa = set(qf); ca = set(cf)
                        inter = len(qa & ca)
                        subscores[sm][idx] = (inter/math.sqrt(len(qa)*len(ca))
                                              if qa and ca else 0.0)
                    elif sm in ("all_cite",):
                        # Composite: freq_idf + cocite + context
                        subscores[sm][idx] = (
                            freq_idf_cosine(qf, cf, cite_idf)
                            + cocite_similarity(qc, cc)
                            + (ctx_scores_cache.get(qid, np.zeros(N))[idx]
                               if ctx_ok else 0.0)
                        )

            # Merge sub-method scores (average)
            for sm in sub_methods:
                cite_sc_arr += subscores[sm]
            cite_sc_arr /= len(sub_methods)

            if base_retr == "none":
                # Pure citation ranking
                final = cite_sc_arr
                order = np.argsort(-final)[:args.top_k]
                results[qid] = [cand_ids[i] for i in order]
                continue

            # Combine with base retriever
            top_n_idx = [cand_idx[c] for c in top_n]

            if boost:
                # Multiplicative: tfidf5_score * (1 + cite_sim)
                reranked = sorted(
                    top_n,
                    key=lambda c: float(tf5[cand_idx[c]]) *
                                  (1.0 + cite_sc_arr[cand_idx[c]]),
                    reverse=True
                )
                results[qid] = reranked + rest
            else:
                # Linear Z-normalised combination
                tf_sub  = tf5[top_n_idx]
                cit_sub = cite_sc_arr[top_n_idx]

                tf_mean, tf_std   = tf_sub.mean(),  tf_sub.std()  + 1e-10
                ct_mean, ct_std   = cit_sub.mean(), cit_sub.std() + 1e-10
                tf_z  = (tf_sub  - tf_mean)  / tf_std
                cit_z = (cit_sub - ct_mean)  / ct_std

                combined = alpha * tf_z + (1 - alpha) * cit_z

                order_local = np.argsort(-combined)
                reranked = [top_n[i] for i in order_local]
                results[qid] = reranked + rest

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MicroF1@10",
                        title="Advanced Citation Retrieval — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
