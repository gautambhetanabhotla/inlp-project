"""
rrf_fusion.py
=============
Reciprocal Rank Fusion (RRF) of multiple retrieval systems.

RRF Formula (Cormack et al., 2009):
    RRF(d) = Σ_r  1 / (k + rank_r(d))

    k = 60  (standard default, robust to variations)

Key insight from results analysis
----------------------------------
The analysis showed:
  • TF-IDF (augmented, 5-gram, mindf=2, maxdf=0.95) = MicroF1@10=0.4403
    but only on 234 queries (small candidate pool / train split)
  • Method1 nouns+sublinear TF-IDF = MicroF1@10=0.2617 on 817-query pool
  • No ensemble has been tried on the LARGE candidate pool (817 queries)

RRF is parameter-free (just k), does not require score normalisation,
and consistently beats individual systems and simple score fusion.

Systems fused
--------------
  The file loads pre-computed ranked-list result JSONs and fuses them.
  You can fuse any subset of the 817-query result files.

  Alternatively, this file also COMPUTES fresh ranked lists on the fly
  for the systems you want to fuse, so no external files are needed.

Run
---
    python3 rrf_fusion.py --data_dir /path/to/dataset/ --split train

Parameter grid
--------------
  Edit FUSION_GROUPS below.  Comment out any group to skip.
"""

import os
import re
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from utils import (
    load_split, clean_text, evaluate_all,
    save_results, print_results_table, save_results_csv,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS  (edit here)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "./"
SPLIT    = "train"
TOP_K    = 1000
OUTPUT   = "results/rrf_fusion_results.json"
K_VALUES = [5, 10, 20, 50, 100]

RRF_K = 60   # standard RRF constant (robust, rarely needs tuning)

# ─────────────────────────────────────────────────────────────────────────────
# FUSION_GROUPS  ← comment out any group to skip
# Each entry: (group_name, [list of retriever_keys])
#
# Available retriever keys (built fresh below):
#   "tfidf_full_ng1"    : TF-IDF, full text, unigrams, sublinear
#   "tfidf_full_ng5"    : TF-IDF, full text, 5-gram, sublinear
#   "tfidf_full_ng3"    : TF-IDF, full text, 3-gram, sublinear
#   "tfidf_nouns_ng1"   : TF-IDF, nouns only, unigrams, sublinear (≈Method1)
#   "tfidf_nouns_ng3"   : TF-IDF, nouns only, 3-gram, sublinear
#   "tfidf_nv_ng1"      : TF-IDF, nouns+verbs, unigrams, sublinear (≈Method2)
#   "tfidf_nv_ng3"      : TF-IDF, nouns+verbs, 3-gram, sublinear
#   "bm25_full_ng5"     : BM25 okapi, full text, 5-gram
#   "bm25_nouns_ng5"    : BM25 okapi, nouns only, 5-gram
#   "bm25_nv_ng5"       : BM25 okapi, nouns+verbs, 5-gram
# ─────────────────────────────────────────────────────────────────────────────

FUSION_GROUPS: List[Tuple] = [
    # ── 2-way fusions ─────────────────────────────────────────────────────────
    ("RRF_tfidf_full_ng5+nouns_ng1",
        ["tfidf_full_ng5", "tfidf_nouns_ng1"]),
    ("RRF_tfidf_full_ng5+nv_ng1",
        ["tfidf_full_ng5", "tfidf_nv_ng1"]),
    ("RRF_tfidf_nouns_ng1+nouns_ng3",
        ["tfidf_nouns_ng1", "tfidf_nouns_ng3"]),
    ("RRF_tfidf_nouns_ng1+nv_ng1",
        ["tfidf_nouns_ng1", "tfidf_nv_ng1"]),
    ("RRF_bm25_full_ng5+tfidf_full_ng5",
        ["bm25_full_ng5", "tfidf_full_ng5"]),
    ("RRF_bm25_nouns_ng5+tfidf_nouns_ng1",
        ["bm25_nouns_ng5", "tfidf_nouns_ng1"]),
    ("RRF_bm25_nv_ng5+tfidf_nv_ng1",
        ["bm25_nv_ng5", "tfidf_nv_ng1"]),
    ("RRF_bm25_full_ng5+tfidf_nouns_ng1",
        ["bm25_full_ng5", "tfidf_nouns_ng1"]),

    # ── 3-way fusions ─────────────────────────────────────────────────────────
    ("RRF_tfidf_full_ng5+nouns_ng1+nv_ng1",
        ["tfidf_full_ng5", "tfidf_nouns_ng1", "tfidf_nv_ng1"]),
    ("RRF_tfidf_full_ng3+nouns_ng1+nv_ng1",
        ["tfidf_full_ng3", "tfidf_nouns_ng1", "tfidf_nv_ng1"]),
    ("RRF_bm25_ng5+tfidf_full_ng5+tfidf_nouns_ng1",
        ["bm25_full_ng5", "tfidf_full_ng5", "tfidf_nouns_ng1"]),
    ("RRF_bm25_ng5+tfidf_full_ng5+tfidf_nv_ng1",
        ["bm25_full_ng5", "tfidf_full_ng5", "tfidf_nv_ng1"]),
    ("RRF_bm25_nouns+tfidf_full_ng5+tfidf_nouns_ng1",
        ["bm25_nouns_ng5", "tfidf_full_ng5", "tfidf_nouns_ng1"]),

    # ── 4-way fusions ─────────────────────────────────────────────────────────
    ("RRF_bm25ng5+tfidf_ng5+nouns_ng1+nv_ng1",
        ["bm25_full_ng5","tfidf_full_ng5","tfidf_nouns_ng1","tfidf_nv_ng1"]),
    ("RRF_all10",
        ["tfidf_full_ng1","tfidf_full_ng5","tfidf_full_ng3",
         "tfidf_nouns_ng1","tfidf_nouns_ng3",
         "tfidf_nv_ng1","tfidf_nv_ng3",
         "bm25_full_ng5","bm25_nouns_ng5","bm25_nv_ng5"]),
]

# ─────────────────────────────────────────────────────────────────────────────
# POS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

NOUN_TAGS = {"NN","NNS","NNP"}
VERB_TAGS = {"VB","VBZ","VBN","VBD"}
_lem  = WordNetLemmatizer()
_CITE = re.compile(r'\[?\?CITATION\?\]?|<CITATION_\d+>', re.IGNORECASE)
_PUN  = re.compile(r'[",\-\'_]')

def _pre(text):
    return _PUN.sub(" ", _CITE.sub(" ", text))

def _nouns(text):
    toks = word_tokenize(_pre(text))
    return [_lem.lemmatize(w.lower(),'n') for w,t in pos_tag(toks)
            if t in NOUN_TAGS]

def _nouns_verbs(text):
    toks = word_tokenize(_pre(text)); out=[]
    for w,t in pos_tag(toks):
        if t in NOUN_TAGS: out.append(_lem.lemmatize(w.lower(),'n'))
        elif t in VERB_TAGS: out.append(_lem.lemmatize(w.lower(),'v'))
    return out

def _ngram(tokens, n):
    if n==1: return tokens
    out = list(tokens)
    for k in range(2,n+1):
        for i in range(len(tokens)-k+1):
            out.append("_".join(tokens[i:i+k]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL SYSTEMS (built fresh)
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_ranker(cand_strs, query_strs, cand_ids, query_ids,
                       min_df=2, max_df=0.95, sublinear=True):
    """Returns ranked results dict."""
    vec = TfidfVectorizer(min_df=min_df, max_df=max_df,
                          sublinear_tf=sublinear, norm="l2")
    C = vec.fit_transform(cand_strs)
    results = {}
    for i, qid in enumerate(query_ids):
        q    = vec.transform([query_strs[i]])
        sims = cosine_similarity(q, C)[0]
        order = np.argsort(-sims)
        results[qid] = [cand_ids[j] for j in order]
    return results


def build_bm25_ranker(cand_tok, query_tok, k1=1.2, b=0.75):
    """Returns ranked results dict using Okapi BM25."""
    ids = list(cand_tok.keys()); N = len(ids)
    df = defaultdict(int); tfs = {}; lens = {}
    for cid, toks in cand_tok.items():
        tf = defaultdict(int)
        for t in toks: tf[t]+=1
        tfs[cid]=dict(tf); lens[cid]=len(toks)
        for t in tf: df[t]+=1
    avgdl = sum(lens.values())/N if N else 1.0

    def idf(t):
        d=df.get(t,0)
        return math.log((N-d+0.5)/(d+0.5)+1)

    results = {}
    for qid, qtoks in query_tok.items():
        scores=[]
        for cid in ids:
            sc=sum(idf(t)*tfs[cid].get(t,0)*(k1+1)/(
                   tfs[cid].get(t,0)+k1*(1-b+b*lens[cid]/avgdl))
                   for t in set(qtoks) if tfs[cid].get(t,0))
            scores.append((cid,sc))
        scores.sort(key=lambda x:x[1],reverse=True)
        results[qid]=[c for c,_ in scores]
    return results


# ─────────────────────────────────────────────────────────────────────────────
# RRF CORE
# ─────────────────────────────────────────────────────────────────────────────

def rrf_fuse(ranked_lists: List[Dict[str, List[str]]],
             qids: List[str],
             k: int = 60,
             top_k: int = 1000) -> Dict[str, List[str]]:
    """
    ranked_lists : list of {qid -> [cid_in_rank_order]}
    Returns      : {qid -> RRF-fused ranked list}
    """
    results = {}
    for qid in qids:
        scores: Dict[str, float] = defaultdict(float)
        for rl in ranked_lists:
            for rank, cid in enumerate(rl.get(qid, [])):
                scores[cid] += 1.0 / (k + rank + 1)
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        results[qid] = ranked[:top_k]
    return results


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
    cand_ids  = list(candidates.keys())
    query_ids = list(queries.keys())

    # ── Build all base ranked lists ───────────────────────────────────────────
    print("\nBuilding base retrieval systems ...")

    # --- Tokenise for POS variants ---
    print("  POS-tagging documents ...")
    nouns_c = {cid: _ngram(_nouns(t), 1)  for cid,t in candidates.items()}
    nouns_q = {qid: _ngram(_nouns(t), 1)  for qid,t in queries.items()}
    nv_c    = {cid: _ngram(_nouns_verbs(t), 1)  for cid,t in candidates.items()}
    nv_q    = {qid: _ngram(_nouns_verbs(t), 1)  for qid,t in queries.items()}
    nouns3_c = {cid: _ngram(_nouns(t), 3) for cid,t in candidates.items()}
    nouns3_q = {qid: _ngram(_nouns(t), 3) for qid,t in queries.items()}
    nv3_c   = {cid: _ngram(_nouns_verbs(t), 3) for cid,t in candidates.items()}
    nv3_q   = {qid: _ngram(_nouns_verbs(t), 3) for qid,t in queries.items()}

    def join(d): return {k: " ".join(v) for k,v in d.items()}

    # --- TF-IDF systems ---
    print("  Building TF-IDF rankers ...")
    def tfidf_rank(cstr, qstr):
        return build_tfidf_ranker(
            [cstr[c] for c in cand_ids],
            [qstr[q] for q in query_ids],
            cand_ids, query_ids
        )

    def full_str(corpus, ng):
        return {k: " ".join(clean_text(t, True, ng))
                for k,t in corpus.items()}

    print("    tfidf_full_ng1 ...")
    fs1c = full_str(candidates, 1); fs1q = full_str(queries, 1)
    print("    tfidf_full_ng3 ...")
    fs3c = full_str(candidates, 3); fs3q = full_str(queries, 3)
    print("    tfidf_full_ng5 ...")
    fs5c = full_str(candidates, 5); fs5q = full_str(queries, 5)

    base_systems: Dict[str, Dict[str, List[str]]] = {}

    base_systems["tfidf_full_ng1"]  = tfidf_rank(fs1c,     fs1q)
    base_systems["tfidf_full_ng3"]  = tfidf_rank(fs3c,     fs3q)
    base_systems["tfidf_full_ng5"]  = tfidf_rank(fs5c,     fs5q)
    base_systems["tfidf_nouns_ng1"] = tfidf_rank(join(nouns_c), join(nouns_q))
    base_systems["tfidf_nouns_ng3"] = tfidf_rank(join(nouns3_c),join(nouns3_q))
    base_systems["tfidf_nv_ng1"]    = tfidf_rank(join(nv_c),   join(nv_q))
    base_systems["tfidf_nv_ng3"]    = tfidf_rank(join(nv3_c),  join(nv3_q))

    # --- BM25 systems ---
    print("  Building BM25 rankers ...")
    def full_tok(corpus, ng):
        return {k: clean_text(t, True, ng) for k,t in corpus.items()}

    ft5c = full_tok(candidates, 5); ft5q = full_tok(queries, 5)
    nouns5_c = {cid: _ngram(_nouns(t), 5) for cid,t in candidates.items()}
    nouns5_q = {qid: _ngram(_nouns(t), 5) for qid,t in queries.items()}
    nv5_c    = {cid: _ngram(_nouns_verbs(t), 5) for cid,t in candidates.items()}
    nv5_q    = {qid: _ngram(_nouns_verbs(t), 5) for qid,t in queries.items()}

    base_systems["bm25_full_ng5"]  = build_bm25_ranker(ft5c,     ft5q)
    base_systems["bm25_nouns_ng5"] = build_bm25_ranker(nouns5_c, nouns5_q)
    base_systems["bm25_nv_ng5"]    = build_bm25_ranker(nv5_c,    nv5_q)

    all_results = []
    qids_with_rel = [qid for qid in query_ids if qid in relevance]

    # ── Evaluate individual systems first ─────────────────────────────────────
    for sys_name, sys_results in base_systems.items():
        filtered = {q: v for q, v in sys_results.items() if q in relevance}
        m = evaluate_all(filtered, relevance, k_values=K_VALUES,
                         label=f"Base_{sys_name}", verbose=False)
        m["model"] = f"Base_{sys_name}"
        all_results.append(m)
        print(f"  Base {sys_name}: MAP={m['MAP']:.4f}  "
              f"MicroF1@10={m.get('MicroF1@10',0):.4f}")

    # ── RRF fusion ────────────────────────────────────────────────────────────
    for (group_name, retriever_keys) in FUSION_GROUPS:
        print(f"\n{'─'*64}\n  {group_name}\n{'─'*64}")
        ranked_lists = [base_systems[k] for k in retriever_keys
                        if k in base_systems]
        if not ranked_lists:
            print("  [SKIP] No valid retrievers.")
            continue

        fused = rrf_fuse(ranked_lists, qids_with_rel, k=RRF_K, top_k=args.top_k)

        # Vary RRF k parameter
        for rrf_k in [10, 30, 60, 120]:
            name = f"{group_name}_k={rrf_k}"
            fused_k = rrf_fuse(ranked_lists, qids_with_rel,
                               k=rrf_k, top_k=args.top_k)
            m = evaluate_all(fused_k, relevance, k_values=K_VALUES, label=name)
            all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MicroF1@10",
                        title="RRF Fusion — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
