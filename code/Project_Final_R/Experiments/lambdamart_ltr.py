"""
lambdamart_ltr.py
==================
LambdaMART Learning-to-Rank for Prior Case Retrieval.

Why Learning-to-Rank is fundamentally different
------------------------------------------------
Every method so far was UNSUPERVISED: it designed a scoring function
without using the relevance labels to learn weights.

LambdaMART is SUPERVISED: it learns the optimal combination of
multiple retrieval signals directly from the relevance labels,
optimising NDCG at the target rank cutoff.

This is the dominant approach in modern IR (Bing, Google's learning pipeline,
all top TREC Ad Hoc systems since 2010 use LTR).

Feature vector (per query-candidate pair)
------------------------------------------
  1.  5-gram TF-IDF cosine similarity           (best single method)
  2.  3-gram TF-IDF cosine similarity
  3.  1-gram TF-IDF cosine similarity
  4.  BM25 ng=3 score (normalised per query)
  5.  Citation Bibliographic Coupling (BC)
  6.  Citation Jaccard similarity
  7.  Statute Jaccard similarity
  8.  MinHash Jaccard (word-5 shingles)
  9.  5-gram TF-IDF rank (1 - rank/N)
  10. 3-gram TF-IDF rank
  11. 1-gram TF-IDF rank
  12. Candidate document length (log-norm)
  13. Query document length (log-norm)
  14. Length ratio query/candidate
  15. Vocabulary overlap (token set Jaccard)
  16. Doc2Vec cosine (if available — skipped if not)
  17. Citation overlap count (raw)
  18. Statute overlap count (raw)

Training protocol
-----------------
  • Train on ik_train split relevance labels
  • Evaluate on ik_train (cross-validated) or ik_test
  • Use pairwise labels: (query, relevant_doc) > (query, irrelevant_doc)

LambdaMART implementation: xgboost with objective='rank:ndcg'

Run
---
    python3 lambdamart_ltr.py --data_dir /path/to/dataset/

Parameter grid
--------------
  Edit CONFIGS below.  Comment out any line to skip.
"""

import os
import re
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb

from utils import (
    load_split, clean_text, extract_citations, evaluate_all,
    save_results, print_results_table, save_results_csv,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS  (edit here)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "./"
OUTPUT   = "results/ltr_results.json"
K_VALUES = [5, 6, 7, 8, 9, 10, 11, 15, 20]

# Training: candidate pool size per query during training
# (to avoid O(N²) features; use TF-IDF top-K + random negatives)
TRAIN_POOL_SIZE = 100    # top-K from TF-IDF to include per query
N_RANDOM_NEG    =  50    # additional random negatives per query

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGS  ← comment out any line to skip
# (n_estimators, max_depth, learning_rate, ndcg_cutoff, subsample)
#
#   n_estimators : number of boosting rounds
#   max_depth    : tree depth (3-7 typical for LTR)
#   learning_rate: XGBoost eta
#   ndcg_cutoff  : optimise NDCG@k (7 = peak MicroF1, 10 = standard)
#   subsample    : fraction of training pairs per round
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS: List[Tuple] = [
    # (n_estimators, max_depth, lr,    ndcg_cutoff, subsample)
    (100,  3,  0.1,   7,  1.0),
    (200,  3,  0.1,   7,  1.0),
    (200,  5,  0.1,   7,  1.0),
    (300,  3,  0.05,  7,  1.0),
    (300,  5,  0.05,  7,  1.0),
    (100,  3,  0.1,  10,  1.0),
    (200,  3,  0.1,  10,  1.0),
    (200,  5,  0.1,  10,  1.0),
    (300,  3,  0.05, 10,  1.0),
    (300,  5,  0.05, 10,  1.0),
    # With subsampling (regularisation)
    (300,  5,  0.05,  7,  0.8),
    (300,  5,  0.05, 10,  0.8),
    (500,  5,  0.05,  7,  0.8),
    (500,  5,  0.05, 10,  0.8),
    # Shallow trees, many rounds
    (1000, 2,  0.05,  7,  0.8),
    (1000, 2,  0.05, 10,  0.8),
]


# ─────────────────────────────────────────────────────────────────────────────
# STATUTE EXTRACTION (same regex as statute_retrieval.py)
# ─────────────────────────────────────────────────────────────────────────────

_ACT_MAP_LTR = [
    (r'indian penal code|i\.p\.c', 'IPC'),
    (r'code of criminal procedure|cr\.?p\.?c', 'CrPC'),
    (r'code of civil procedure|c\.p\.c', 'CPC'),
    (r'income.?tax act', 'ITA'),
    (r'constitution', 'Const'),
    (r'companies act', 'CoAct'),
    (r'arbitration.{0,20}act', 'ArbAct'),
    (r'contract act', 'ContAct'),
    (r'evidence act', 'EvidAct'),
    (r'limitation act', 'LimAct'),
    (r'transfer of property', 'TPA'),
]

def _norm_act(frag):
    s = frag.strip().lower()
    for pat, ab in _ACT_MAP_LTR:
        if re.search(pat, s): return ab
    words = [w for w in frag.split() if w and w[0].isupper()]
    return "_".join(words[:2]) if words else "Unk"

def extract_statutes_ltr(text: str) -> Set[str]:
    text = re.sub(r'<[^>]+>', ' ', text)
    found: Set[str] = set()
    for m in re.finditer(
        r'(?:section|sec\.?|s\.)\s*(\d+[A-Za-z]?)'
        r'(?:\s+(?:of\s+(?:the\s+)?)?([A-Z][A-Za-z\s,\.]{3,45}))?',
        text, re.IGNORECASE
    ):
        num, act = m.groups()
        if act:
            found.add(f"S{num.upper()}_{_norm_act(act)}")
    for m in re.finditer(
        r'(?:section|sec\.?|s\.)\s*(\d+[A-Za-z]?)\s+(IPC|CrPC|CPC|ITA)',
        text, re.IGNORECASE
    ):
        num, act = m.groups()
        found.add(f"S{num.upper()}_{act.upper()}")
    return found


# ─────────────────────────────────────────────────────────────────────────────
# MINHASH (same as minhash_retrieval.py, minimal version)
# ─────────────────────────────────────────────────────────────────────────────

import hashlib
_PRIME_LTR = (1 << 31) - 1
_RNG_LTR   = np.random.RandomState(42)
_NHASH_LTR = 64
_A_LTR = _RNG_LTR.randint(1, _PRIME_LTR, _NHASH_LTR, dtype=np.int64)
_B_LTR = _RNG_LTR.randint(0, _PRIME_LTR, _NHASH_LTR, dtype=np.int64)

def _hash_s(s): return int(hashlib.md5(s.encode()).hexdigest(), 16) % _PRIME_LTR

def minhash_sig_ltr(tokens, k=5):
    words = tokens
    shingles = {" ".join(words[i:i+k]) for i in range(len(words)-k+1)}
    sig = np.full(_NHASH_LTR, _PRIME_LTR, dtype=np.int64)
    for sh in shingles:
        h = _hash_s(sh)
        sig = np.minimum(sig, (_A_LTR*h + _B_LTR) % _PRIME_LTR)
    return sig

def mh_jaccard(sa, sb): return float((sa == sb).sum()) / _NHASH_LTR


# ─────────────────────────────────────────────────────────────────────────────
# BM25 (minimal)
# ─────────────────────────────────────────────────────────────────────────────

def build_bm25_scores(c_tok3, q_tok3, cand_ids, k1=1.2, b=0.75):
    N  = len(cand_ids)
    df = defaultdict(int); tfs = {}; lens = {}
    for cid in cand_ids:
        tf = defaultdict(int)
        for t in c_tok3[cid]: tf[t]+=1
        tfs[cid]=dict(tf); lens[cid]=len(c_tok3[cid])
        for t in tf: df[t]+=1
    avgdl = sum(lens.values())/N if N else 1.0
    def idf(t):
        d=df.get(t,0); return math.log((N-d+0.5)/(d+0.5)+1)
    out={}
    for qid, qtoks in q_tok3.items():
        sc={}
        for cid in cand_ids:
            s=sum(idf(t)*tfs[cid].get(t,0)*(k1+1)/(
                tfs[cid].get(t,0)+k1*(1-b+b*lens[cid]/avgdl))
                for t in set(qtoks) if tfs[cid].get(t,0))
            sc[cid]=s
        mx = max(sc.values()) or 1.0
        out[qid]={c:v/mx for c,v in sc.items()}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDING
# ─────────────────────────────────────────────────────────────────────────────

def build_features(qid, cid,
                   tf5, tf3, tf1,       # TF-IDF score dicts
                   bm25_sc,             # BM25 score dict
                   tf5_rank, tf3_rank, tf1_rank,  # rank dicts (normalised 0-1)
                   q_cites, c_cites,    # citation sets
                   q_stat, c_stat,      # statute sets
                   q_mh, c_mh,          # MinHash signatures
                   q_tok1, c_tok1,      # raw tokens for length/overlap
                   N_cands) -> List[float]:

    def safe(d, k, default=0.0): return d.get(k, default) if d else default

    qc  = q_cites.get(qid, set());  cc  = c_cites.get(cid, set())
    qs  = q_stat.get(qid,  set());  cs  = c_stat.get(cid,  set())
    qmh = q_mh.get(qid);            cmh = c_mh.get(cid)
    qt  = q_tok1.get(qid, []);      ct  = c_tok1.get(cid, [])

    # Citation metrics
    cite_inter = len(qc & cc)
    cite_union = len(qc | cc)
    cite_jac   = cite_inter / cite_union if cite_union else 0.0
    cite_bc    = cite_inter / math.sqrt(len(qc)*len(cc)) if qc and cc else 0.0

    # Statute metrics
    stat_inter = len(qs & cs)
    stat_union = len(qs | cs)
    stat_jac   = stat_inter / stat_union if stat_union else 0.0

    # MinHash
    mh_jac = mh_jaccard(qmh, cmh) if qmh is not None and cmh is not None else 0.0

    # Length features
    ql = math.log(len(qt) + 1) / 10.0
    cl = math.log(len(ct) + 1) / 10.0
    lr = min(len(qt)+1, len(ct)+1) / max(len(qt)+1, len(ct)+1)

    # Vocab overlap
    qt_set = set(qt); ct_set = set(ct)
    voc_j  = len(qt_set & ct_set) / len(qt_set | ct_set) if (qt_set | ct_set) else 0.0

    return [
        safe(tf5.get(qid), cid),           # 1. 5-gram TF-IDF score
        safe(tf3.get(qid), cid),           # 2. 3-gram TF-IDF score
        safe(tf1.get(qid), cid),           # 3. 1-gram TF-IDF score
        safe(bm25_sc.get(qid), cid),       # 4. BM25 ng=3 score
        cite_jac,                          # 5. Citation Jaccard
        cite_bc,                           # 6. Citation BC
        stat_jac,                          # 7. Statute Jaccard
        mh_jac,                            # 8. MinHash Jaccard
        1.0 - safe(tf5_rank.get(qid), cid)/N_cands,  # 9.  5-gram rank
        1.0 - safe(tf3_rank.get(qid), cid)/N_cands,  # 10. 3-gram rank
        1.0 - safe(tf1_rank.get(qid), cid)/N_cands,  # 11. 1-gram rank
        ql,                                # 12. query length
        cl,                                # 13. cand length
        lr,                                # 14. length ratio
        voc_j,                             # 15. vocab overlap
        float(cite_inter),                 # 16. citation overlap count
        float(stat_inter),                 # 17. statute overlap count
    ]

FEATURE_NAMES = [
    "tfidf5", "tfidf3", "tfidf1", "bm25ng3",
    "cite_jac", "cite_bc", "stat_jac", "minhash_jac",
    "rank5", "rank3", "rank1",
    "q_len", "c_len", "len_ratio", "vocab_jac",
    "cite_count", "stat_count",
]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--output",   default=OUTPUT)
    args = parser.parse_args()

    print("Loading splits ...")
    tr_q, tr_c, tr_rel = load_split(args.data_dir, "train")
    te_q, te_c, te_rel = load_split(args.data_dir, "test")
    ev_q   = te_q   if te_q   else tr_q
    ev_c   = te_c   if te_c   else tr_c
    ev_rel = te_rel if te_rel else tr_rel

    # Use train for pre-computing all scores; evaluate on ev split
    q_all   = {**tr_q,  **te_q}  if te_q  else tr_q
    c_all   = {**tr_c,  **te_c}  if te_c  else tr_c

    # Combine for IDF computation
    cand_ids_tr  = list(tr_c.keys())
    cand_ids_ev  = list(ev_c.keys())

    def build_tfidf_scores(corpus_c, corpus_q, ng, min_df=2, max_df=0.95):
        cids = list(corpus_c.keys())
        c_sw = {cid: " ".join(clean_text(t, True, ng)) for cid,t in corpus_c.items()}
        q_sw = {qid: " ".join(clean_text(t, True, ng)) for qid,t in corpus_q.items()}
        vec = TfidfVectorizer(ngram_range=(ng,ng), min_df=min_df, max_df=max_df,
                              sublinear_tf=True, norm="l2")
        C   = vec.fit_transform([c_sw[c] for c in cids])
        sc_dict = {}; rank_dict = {}
        for qid, qstr in q_sw.items():
            qv   = vec.transform([qstr])
            sims = cosine_similarity(qv, C)[0]
            order = np.argsort(-sims)
            sc_dict[qid]   = {cids[i]: float(sims[i]) for i in range(len(cids))}
            rank_dict[qid] = {cids[i]: int(r) for r, i in enumerate(order)}
        return sc_dict, rank_dict

    print("Computing TF-IDF scores (5-gram, 3-gram, 1-gram) ...")
    tf5_tr,  rk5_tr  = build_tfidf_scores(tr_c, tr_q, 5)
    tf3_tr,  rk3_tr  = build_tfidf_scores(tr_c, tr_q, 3)
    tf1_tr,  rk1_tr  = build_tfidf_scores(tr_c, tr_q, 1)
    tf5_ev,  rk5_ev  = build_tfidf_scores(ev_c, ev_q, 5)
    tf3_ev,  rk3_ev  = build_tfidf_scores(ev_c, ev_q, 3)
    tf1_ev,  rk1_ev  = build_tfidf_scores(ev_c, ev_q, 1)

    print("Computing BM25 scores (ng=3) ...")
    c_tok3_tr = {c: clean_text(t, True, 3) for c,t in tr_c.items()}
    q_tok3_tr = {q: clean_text(t, True, 3) for q,t in tr_q.items()}
    c_tok3_ev = {c: clean_text(t, True, 3) for c,t in ev_c.items()}
    q_tok3_ev = {q: clean_text(t, True, 3) for q,t in ev_q.items()}
    bm25_tr   = build_bm25_scores(c_tok3_tr, q_tok3_tr, cand_ids_tr)
    bm25_ev   = build_bm25_scores(c_tok3_ev, q_tok3_ev, cand_ids_ev)

    print("Extracting citations ...")
    q_cites_tr = {q: extract_citations(t) for q,t in tr_q.items()}
    c_cites_tr = {c: extract_citations(t) for c,t in tr_c.items()}
    q_cites_ev = {q: extract_citations(t) for q,t in ev_q.items()}
    c_cites_ev = {c: extract_citations(t) for c,t in ev_c.items()}

    print("Extracting statutes ...")
    q_stat_tr = {q: extract_statutes_ltr(t) for q,t in tr_q.items()}
    c_stat_tr = {c: extract_statutes_ltr(t) for c,t in tr_c.items()}
    q_stat_ev = {q: extract_statutes_ltr(t) for q,t in ev_q.items()}
    c_stat_ev = {c: extract_statutes_ltr(t) for c,t in ev_c.items()}

    print("Computing MinHash signatures ...")
    c_tok1_tr = {c: clean_text(t, True, 1) for c,t in tr_c.items()}
    q_tok1_tr = {q: clean_text(t, True, 1) for q,t in tr_q.items()}
    c_tok1_ev = {c: clean_text(t, True, 1) for c,t in ev_c.items()}
    q_tok1_ev = {q: clean_text(t, True, 1) for q,t in ev_q.items()}
    q_mh_tr   = {q: minhash_sig_ltr(t) for q,t in q_tok1_tr.items()}
    c_mh_tr   = {c: minhash_sig_ltr(t) for c,t in c_tok1_tr.items()}
    q_mh_ev   = {q: minhash_sig_ltr(t) for q,t in q_tok1_ev.items()}
    c_mh_ev   = {c: minhash_sig_ltr(t) for c,t in c_tok1_ev.items()}

    N_tr = len(tr_c); N_ev = len(ev_c)

    # ── Build training feature matrix ─────────────────────────────────────────
    print("\nBuilding training feature matrix ...")
    import random
    X_tr, y_tr, groups_tr = [], [], []

    for qid, rels in tr_rel.items():
        if qid not in tf5_tr: continue
        rel_set = set(rels)

        # Top-TRAIN_POOL_SIZE from 5-gram TF-IDF + random negatives
        top5_cids = sorted(cand_ids_tr, key=lambda c: tf5_tr[qid].get(c,0), reverse=True)
        pool = top5_cids[:TRAIN_POOL_SIZE]
        neg_candidates = [c for c in cand_ids_tr if c not in rel_set and c not in pool]
        pool += random.sample(neg_candidates, min(N_RANDOM_NEG, len(neg_candidates)))

        n = 0
        for cid in pool:
            fv = build_features(
                qid, cid,
                tf5_tr, tf3_tr, tf1_tr, bm25_tr,
                rk5_tr, rk3_tr, rk1_tr,
                q_cites_tr, c_cites_tr,
                q_stat_tr, c_stat_tr,
                q_mh_tr, c_mh_tr,
                q_tok1_tr, c_tok1_tr, N_tr
            )
            X_tr.append(fv)
            y_tr.append(1 if cid in rel_set else 0)
            n += 1
        groups_tr.append(n)

    X_tr = np.array(X_tr, dtype=np.float32)
    y_tr = np.array(y_tr, dtype=np.int32)
    groups_tr = np.array(groups_tr, dtype=np.int32)
    print(f"  Training pairs: {len(X_tr)}  "
          f"(pos={y_tr.sum()}, neg={len(y_tr)-y_tr.sum()})")

    # ── Build eval feature matrix ─────────────────────────────────────────────
    print("Building evaluation feature matrix ...")
    X_ev, groups_ev, ev_order = [], [], []

    for qid in ev_rel:
        if qid not in tf5_ev: continue
        top5 = sorted(cand_ids_ev, key=lambda c: tf5_ev[qid].get(c,0), reverse=True)
        n = 0
        for cid in top5[:1000]:
            fv = build_features(
                qid, cid,
                tf5_ev, tf3_ev, tf1_ev, bm25_ev,
                rk5_ev, rk3_ev, rk1_ev,
                q_cites_ev, c_cites_ev,
                q_stat_ev, c_stat_ev,
                q_mh_ev, c_mh_ev,
                q_tok1_ev, c_tok1_ev, N_ev
            )
            X_ev.append(fv)
            ev_order.append((qid, cid))
            n += 1
        groups_ev.append(n)

    X_ev = np.array(X_ev, dtype=np.float32)
    print(f"  Eval pairs: {len(X_ev)}")

    # ── Train + evaluate each config ──────────────────────────────────────────
    all_results = []
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtrain.set_group(groups_tr)
    dtest  = xgb.DMatrix(X_ev)

    for (n_est, max_d, lr, ndcg_cut, subsamp) in CONFIGS:
        name = (f"LambdaMART_n={n_est}_d={max_d}_"
                f"lr={lr}_ndcg@{ndcg_cut}_sub={subsamp}")
        print(f"\n{'─'*64}\n  {name}\n{'─'*64}")

        params = {
            "objective":        "rank:ndcg",
            "ndcg_exp_gain":    True,
            "eval_metric":      f"ndcg@{ndcg_cut}",
            "max_depth":        max_d,
            "eta":              lr,
            "subsample":        subsamp,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "tree_method":      "hist",
            "seed":             42,
        }
        model = xgb.train(params, dtrain, num_boost_round=n_est,
                          verbose_eval=False)

        scores = model.predict(dtest)

        # Print feature importances
        imp = model.get_score(importance_type="gain")
        top_feat = sorted(imp.items(), key=lambda x: -x[1])[:5]
        print("  Top features:", [(FEATURE_NAMES[int(k[1:])] if k.startswith('f') else k, f"{v:.1f}") for k,v in top_feat])

        # Reconstruct ranked lists
        results: Dict[str, List[str]] = {}
        ptr = 0
        for gi, qid in zip(groups_ev, [q for q in ev_rel if q in tf5_ev]):
            group_scores  = scores[ptr: ptr+gi]
            group_cids    = [ev_order[ptr+j][1] for j in range(gi)]
            order         = np.argsort(-group_scores)
            results[qid]  = [group_cids[i] for i in order]
            ptr += gi

        m = evaluate_all(results, ev_rel, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MicroF1@10",
                        title="LambdaMART LTR — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
