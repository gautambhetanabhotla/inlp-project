"""
analyze_rr_full.py
==================
Comprehensive analysis of RR role combinations for IL-PCR retrieval.

Experiments
-----------
  A. Role subset combinations (which roles to include)
  B. Role-weighted boosting (repeat high-value role sentences N times)
  C. Label-as-signal (prepend role label word to each sentence repeatedly)
  D. Role-specific TF-IDF (separate vectorizers per role, fuse scores)
  E. Preprocessing variants (keep vs strip stopwords, n-gram sweep)

Metrics: MAP, MRR, P/R/F1@K, NDCG@K, MicroF1@K  (K=[5,6,7,8,9,10,11,15,20])
Primary sort: BestMicroF1 (consistent with hybrid pipeline)

Usage
-----
    cd Models/RR
    python analyze_rr_full.py
    python analyze_rr_full.py --ngram 3 --min_df 2   # adjust global params
"""

import os, re, json, sys, argparse
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../Transformer-Embeddings'))
from tfidf_utils import clean_text, evaluate_all

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--cand_dir",  default="ik_test_rr/candidate")
parser.add_argument("--query_dir", default="ik_test_rr/query")
parser.add_argument("--labels",    default="ik_test_rr/test.json")
parser.add_argument("--orig_cand", default="../BM25/data/corpus/ik_test/candidate",
                    help="Original (non-RR) candidates for baseline comparison")
parser.add_argument("--orig_query", default="../BM25/data/corpus/ik_test/query")
parser.add_argument("--ngram",    type=int,   default=4)
parser.add_argument("--min_df",   type=int,   default=2)
parser.add_argument("--max_df",   type=float, default=0.95)
args = parser.parse_args()

SKIP_IDS = {1864396, 1508893}
K_VALUES = [5, 6, 7, 8, 9, 10, 11, 15, 20]
ALL_ROLES = {'Argument', 'Fact', 'Precedent', 'RatioOfTheDecision',
             'RulingByLowerCourt', 'RulingByPresentCourt', 'Statute'}

# ── load ─────────────────────────────────────────────────────────────────────
with open(args.labels) as f:
    lj = json.load(f)
gold = {
    int(re.findall(r'\d+', i['id'])[0]): {int(re.findall(r'\d+', c)[0]) for c in i.get('relevant candidates', [])}
    for i in lj['Query Set']
}

def load_raw(folder):
    docs = {}
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith('.txt'): continue
        doc_id = int(re.findall(r'\d+', fn)[0])
        docs[doc_id] = open(os.path.join(folder, fn), errors='ignore').read()
    return docs

def load_plain(folder):
    """Load plain text (no RR labels) — for original docs."""
    docs = {}
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith('.txt'): continue
        doc_id = int(re.findall(r'\d+', fn)[0])
        docs[doc_id] = ' '.join(open(os.path.join(folder, fn), errors='ignore').read().split())
    return docs

print("Loading RR corpus …")
cand_rr  = load_raw(args.cand_dir)
query_rr = load_raw(args.query_dir)
print(f"  RR:  {len(cand_rr)} cands, {len(query_rr)} queries")

orig_available = os.path.isdir(args.orig_cand)
if orig_available:
    print("Loading original corpus …")
    cand_orig  = load_plain(args.orig_cand)
    query_orig = load_plain(args.orig_query)
    print(f"  Orig: {len(cand_orig)} cands, {len(query_orig)} queries")

# ══════════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def extract_roles(text, roles, repeat=1):
    """Return text of sentences belonging to given roles, optionally repeated."""
    lines = []
    for line in text.strip().split('\n'):
        parts = line.split('\t', 1)
        if len(parts) == 2 and parts[0].strip() in roles:
            lines.extend([parts[1].strip()] * repeat)
    return ' '.join(lines)

def extract_all_strip_labels(text):
    """All sentences, label prefix removed."""
    lines = []
    for line in text.strip().split('\n'):
        parts = line.split('\t', 1)
        lines.append(parts[-1].strip())
    return ' '.join(lines)

def extract_with_boost(text, boosted_roles, boost_factor=3):
    """All sentences, but boosted_roles repeated boost_factor times."""
    lines = []
    for line in text.strip().split('\n'):
        parts = line.split('\t', 1)
        if len(parts) != 2:
            lines.append(line.strip())
            continue
        lbl, sent = parts[0].strip(), parts[1].strip()
        count = boost_factor if lbl in boosted_roles else 1
        lines.extend([sent] * count)
    return ' '.join(lines)

def extract_label_boosted(text, boosted_roles, label_repeat=5):
    """All sentences + prepend role label token label_repeat times for boosted roles."""
    lines = []
    for line in text.strip().split('\n'):
        parts = line.split('\t', 1)
        if len(parts) != 2:
            lines.append(line.strip())
            continue
        lbl, sent = parts[0].strip(), parts[1].strip()
        if lbl in boosted_roles:
            prefix = (lbl.lower() + ' ') * label_repeat
            lines.append(prefix + sent)
        else:
            lines.append(sent)
    return ' '.join(lines)

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def micro_f1(ranked_results):
    qdata = []
    for qid, ranked in ranked_results.items():
        if qid in SKIP_IDS: continue
        actual = gold.get(qid)
        if not actual: continue
        qdata.append((actual, [c for c in ranked if c != qid]))
    best_f1, best_k = 0.0, 1
    curve = []
    for k in range(1, 21):
        tp = fp = fn = 0
        for actual, ranked in qdata:
            top_k = set(ranked[:k])
            tp += len(top_k & actual); fp += len(top_k - actual); fn += len(actual - top_k)
        p = tp/(tp+fp) if (tp+fp) else 0.0
        r = tp/(tp+fn) if (tp+fn) else 0.0
        f = 2*p*r/(p+r) if (p+r) else 0.0
        curve.append(f)
        if f > best_f1: best_f1, best_k = f, k
    return best_f1, best_k, curve

def run_tfidf(cand_texts, query_texts, label, ngram=None, scheme="log", verbose=False):
    """Run TF-IDF retrieval and return full metrics dict."""
    if ngram is None: ngram = args.ngram
    c_ids = sorted(cand_texts.keys())
    q_ids = sorted(query_texts.keys())

    c_strs = [' '.join(clean_text(cand_texts[i], remove_stopwords=True, ngram=ngram)) for i in c_ids]
    q_strs = [' '.join(clean_text(query_texts[i], remove_stopwords=True, ngram=ngram)) for i in q_ids]

    # filter truly empty
    empty_c = [i for i,s in enumerate(c_strs) if not s.strip()]
    if empty_c:
        print(f"    WARNING: {len(empty_c)} empty candidate docs for '{label}'")

    vec = TfidfVectorizer(
        analyzer='word', tokenizer=str.split, preprocessor=None, token_pattern=None,
        min_df=args.min_df, max_df=args.max_df,
        sublinear_tf=(scheme == "log"), binary=(scheme == "binary"),
        use_idf=True, norm='l2',
    )
    try:
        C = vec.fit_transform(c_strs)
        Q = vec.transform(q_strs)
    except Exception as e:
        print(f"  SKIPPED '{label}': {e}")
        return None

    scores_mat = (C @ Q.T).toarray()  # (n_cands, n_queries)

    ranked_results = {}
    for qi, qid in enumerate(q_ids):
        top = np.argsort(scores_mat[:, qi])[::-1]
        ranked_results[qid] = [c_ids[i] for i in top]

    # full metrics via evaluate_all (macro)
    filtered = {q: v for q, v in ranked_results.items() if q not in SKIP_IDS}
    relevance = {q: list(v) for q, v in gold.items()}
    m = evaluate_all(filtered, relevance, k_values=K_VALUES, label=label, verbose=verbose)

    # micro F1
    mf1, mk, curve = micro_f1(ranked_results)
    m['_micro_f1'] = mf1
    m['_micro_k']  = mk
    m['_micro_f1_curve'] = curve
    return m

# ══════════════════════════════════════════════════════════════════════════════
# ROLE-SPECIFIC TFIDF + SCORE FUSION
# ══════════════════════════════════════════════════════════════════════════════

def run_role_fusion(cand_rr, query_rr, role_weights, label, ngram=None, scheme="log"):
    """Separate TF-IDF per role, fuse scores with given weights."""
    if ngram is None: ngram = args.ngram
    c_ids = sorted(cand_rr.keys())
    q_ids = sorted(query_rr.keys())

    # build per-role text dicts
    roles = list(role_weights.keys())
    c_role_texts = {r: {cid: extract_roles(t, {r}) for cid,t in cand_rr.items()} for r in roles}
    q_role_texts = {r: {qid: extract_roles(t, {r}) for qid,t in query_rr.items()} for r in roles}

    fused = np.zeros((len(c_ids), len(q_ids)))
    total_w = 0.0

    for r, w in role_weights.items():
        c_strs = [' '.join(clean_text(c_role_texts[r][i], remove_stopwords=True, ngram=ngram)) or '__empty__' for i in c_ids]
        q_strs = [' '.join(clean_text(q_role_texts[r][i], remove_stopwords=True, ngram=ngram)) or '__empty__' for i in q_ids]
        vec = TfidfVectorizer(
            analyzer='word', tokenizer=str.split, preprocessor=None, token_pattern=None,
            min_df=max(1, args.min_df-1), max_df=args.max_df,
            sublinear_tf=(scheme == "log"), use_idf=True, norm='l2',
        )
        try:
            C = vec.fit_transform(c_strs)
            Q = vec.transform(q_strs)
            fused += w * (C @ Q.T).toarray()
            total_w += w
        except Exception:
            pass

    if total_w == 0:
        return None
    fused /= total_w

    ranked_results = {}
    for qi, qid in enumerate(q_ids):
        top = np.argsort(fused[:, qi])[::-1]
        ranked_results[qid] = [c_ids[i] for i in top]

    filtered = {q: v for q, v in ranked_results.items() if q not in SKIP_IDS}
    relevance = {q: list(v) for q, v in gold.items()}
    m = evaluate_all(filtered, relevance, k_values=K_VALUES, label=label, verbose=False)
    mf1, mk, curve = micro_f1(ranked_results)
    m['_micro_f1'] = mf1; m['_micro_k'] = mk; m['_micro_f1_curve'] = curve
    return m

# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════════════
results = []
N = args.ngram

def add(m, name=None):
    if m:
        if name: m['model'] = name
        results.append(m)
        print(f"  → MicroF1={m['_micro_f1']*100:.2f}%@K={m['_micro_k']}  MAP={m['MAP']*100:.2f}%  MRR={m['MRR']*100:.2f}%  NDCG@10={m['NDCG@10']*100:.2f}%")

print(f"\n{'═'*70}")
print(f"  EXPERIMENT A: Role subset combinations  (n={N}, scheme=log)")
print(f"{'═'*70}")

role_combos = [
    ("A0: Baseline — original docs (no RR)", None, None),
    ("A1: All roles (full RR doc)", dict(roles=ALL_ROLES, repeat=1), "subset"),
    ("A2: All roles, label prefix stripped", None, "strip"),
    ("A3: Fact + Argument + Precedent + Ratio", dict(roles={'Fact','Argument','Precedent','RatioOfTheDecision'}, repeat=1), "subset"),
    ("A4: Precedent + RatioOfTheDecision", dict(roles={'Precedent','RatioOfTheDecision'}, repeat=1), "subset"),
    ("A5: Fact + Precedent + Ratio", dict(roles={'Fact','Precedent','RatioOfTheDecision'}, repeat=1), "subset"),
    ("A6: Argument + Precedent + Ratio", dict(roles={'Argument','Precedent','RatioOfTheDecision'}, repeat=1), "subset"),
    ("A7: RatioOfTheDecision only", dict(roles={'RatioOfTheDecision'}, repeat=1), "subset"),
    ("A8: Precedent only", dict(roles={'Precedent'}, repeat=1), "subset"),
    ("A9: Statute + Precedent + Ratio", dict(roles={'Statute','Precedent','RatioOfTheDecision'}, repeat=1), "subset"),
    ("A10: All except RulingByLowerCourt", dict(roles=ALL_ROLES-{'RulingByLowerCourt','RulingByPresentCourt'}, repeat=1), "subset"),
    ("A11: Fact + Argument + Ratio (no Precedent)", dict(roles={'Fact','Argument','RatioOfTheDecision'}, repeat=1), "subset"),
    ("A12: Fact + Argument + Ratio + Judgment", dict(roles={'Fact','Argument','RatioOfTheDecision','RulingByPresentCourt'}, repeat=1), "subset"),
]

for name, cfg, mode in role_combos:
    print(f"\n  {name}")
    if mode is None:
        # original docs
        if orig_available:
            m = run_tfidf(cand_orig, query_orig, name)
            add(m)
    elif mode == "strip":
        c = {cid: extract_all_strip_labels(t) for cid,t in cand_rr.items()}
        q = {qid: extract_all_strip_labels(t) for qid,t in query_rr.items()}
        m = run_tfidf(c, q, name)
        add(m)
    elif mode == "subset":
        roles = cfg['roles']
        c = {cid: extract_roles(t, roles) for cid,t in cand_rr.items()}
        q = {qid: extract_roles(t, roles) for qid,t in query_rr.items()}
        c = {k:v for k,v in c.items() if v.strip()}
        q = {k:v for k,v in q.items() if v.strip()}
        m = run_tfidf(c, q, name)
        add(m)

print(f"\n{'═'*70}")
print(f"  EXPERIMENT B: Role-weighted boosting  (n={N}, scheme=log)")
print("  (all text kept, but important roles repeated N times)")
print(f"{'═'*70}")

boost_combos = [
    ("B1: Boost Precedent+Ratio ×2", {'Precedent','RatioOfTheDecision'}, 2),
    ("B2: Boost Precedent+Ratio ×3", {'Precedent','RatioOfTheDecision'}, 3),
    ("B3: Boost Precedent+Ratio ×5", {'Precedent','RatioOfTheDecision'}, 5),
    ("B4: Boost Ratio ×3", {'RatioOfTheDecision'}, 3),
    ("B5: Boost Precedent ×3", {'Precedent'}, 3),
    ("B6: Boost Fact+Precedent+Ratio ×2", {'Fact','Precedent','RatioOfTheDecision'}, 2),
    ("B7: Boost Fact+Precedent+Ratio ×3", {'Fact','Precedent','RatioOfTheDecision'}, 3),
    ("B8: Boost Argument+Precedent+Ratio ×2", {'Argument','Precedent','RatioOfTheDecision'}, 2),
    ("B9: Downweight RulingByLower (keep others ×1, lower ×0 skip)", None, None),  # handled below
]

for name, roles, factor in boost_combos:
    print(f"\n  {name}")
    if roles is None:
        # B9: exclude RulingByLowerCourt
        c = {cid: extract_roles(t, ALL_ROLES-{'RulingByLowerCourt','RulingByPresentCourt'}, 1) for cid,t in cand_rr.items()}
        q = {qid: extract_roles(t, ALL_ROLES-{'RulingByLowerCourt','RulingByPresentCourt'}, 1) for qid,t in query_rr.items()}
    else:
        c = {cid: extract_with_boost(t, roles, factor) for cid,t in cand_rr.items()}
        q = {qid: extract_with_boost(t, roles, factor) for qid,t in query_rr.items()}
    c = {k:v for k,v in c.items() if v.strip()}
    q = {k:v for k,v in q.items() if v.strip()}
    m = run_tfidf(c, q, name)
    add(m)

print(f"\n{'═'*70}")
print(f"  EXPERIMENT C: Label-as-explicit-signal  (n={N}, scheme=log)")
print("  (prepend role label word many times — boosts label as TF-IDF term)")
print(f"{'═'*70}")

label_boost_combos = [
    ("C1: Label role term ×3 for Precedent+Ratio", {'Precedent','RatioOfTheDecision'}, 3),
    ("C2: Label role term ×5 for Precedent+Ratio", {'Precedent','RatioOfTheDecision'}, 5),
    ("C3: Label role term ×10 for Precedent+Ratio", {'Precedent','RatioOfTheDecision'}, 10),
    ("C4: Label role term ×5 for all roles", ALL_ROLES, 5),
    ("C5: Label role term ×3 for Fact+Prec+Ratio", {'Fact','Precedent','RatioOfTheDecision'}, 3),
]

for name, roles, repeat in label_boost_combos:
    print(f"\n  {name}")
    c = {cid: extract_label_boosted(t, roles, repeat) for cid,t in cand_rr.items()}
    q = {qid: extract_label_boosted(t, roles, repeat) for qid,t in query_rr.items()}
    m = run_tfidf(c, q, name)
    add(m)

print(f"\n{'═'*70}")
print(f"  EXPERIMENT D: Role-specific TF-IDF fusion  (n={N}, scheme=log)")
print("  (separate vectorizer per role, weighted score sum)")
print(f"{'═'*70}")

fusion_combos = [
    ("D1: Equal weight all roles",
     {r: 1.0 for r in ALL_ROLES}),
    ("D2: Precedent×2, Ratio×2, others×1",
     {'Precedent':2.0,'RatioOfTheDecision':2.0,'Fact':1.0,'Argument':1.0,'Statute':1.0,'RulingByLowerCourt':1.0,'RulingByPresentCourt':1.0}),
    ("D3: Precedent×3, Ratio×3, Fact×1, rest×0.5",
     {'Precedent':3.0,'RatioOfTheDecision':3.0,'Fact':1.0,'Argument':1.0,'Statute':0.5,'RulingByLowerCourt':0.5,'RulingByPresentCourt':0.3}),
    ("D4: Only Precedent+Ratio+Fact equal",
     {'Precedent':1.0,'RatioOfTheDecision':1.0,'Fact':1.0}),
    ("D5: Ratio×4, Precedent×2, Fact×1",
     {'RatioOfTheDecision':4.0,'Precedent':2.0,'Fact':1.0}),
]

for name, weights in fusion_combos:
    print(f"\n  {name}")
    m = run_role_fusion(cand_rr, query_rr, weights, name)
    add(m)

print(f"\n{'═'*70}")
print(f"  EXPERIMENT E: Preprocessing variants on best combos")
print(f"{'═'*70}")

# Best from A: A3 with different n-grams and schemes
best_c = {cid: extract_roles(t, {'Fact','Argument','Precedent','RatioOfTheDecision'}) for cid,t in cand_rr.items()}
best_q = {qid: extract_roles(t, {'Fact','Argument','Precedent','RatioOfTheDecision'}) for qid,t in query_rr.items()}
best_c = {k:v for k,v in best_c.items() if v.strip()}
best_q = {k:v for k,v in best_q.items() if v.strip()}

for ng in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    for sc in ["log", "augmented", "binary"]:
        name = f"E: Fact+Arg+Prec+Ratio  n={ng}  scheme={sc}"  # quadgram (n=4) is the standard
        print(f"\n  {name}")
        if sc == "augmented":
            # use CountVectorizer + manual augmented tf
            c_ids = sorted(best_c.keys()); q_ids = sorted(best_q.keys())
            c_strs = [' '.join(clean_text(best_c[i], remove_stopwords=True, ngram=ng)) or '__empty__' for i in c_ids]
            q_strs = [' '.join(clean_text(best_q[i], remove_stopwords=True, ngram=ng)) or '__empty__' for i in q_ids]
            from sklearn.feature_extraction.text import CountVectorizer
            cv = CountVectorizer(analyzer='word', tokenizer=str.split, preprocessor=None, token_pattern=None,
                                 min_df=args.min_df, max_df=args.max_df)
            try:
                raw_c = cv.fit_transform(c_strs).astype(float)
                raw_q = cv.transform(q_strs).astype(float)
                def augment(mat):
                    mat = mat.tocsr()
                    for i in range(mat.shape[0]):
                        s,e = mat.indptr[i], mat.indptr[i+1]
                        if e>s:
                            mx = mat.data[s:e].max()
                            if mx>0: mat.data[s:e] = 0.5 + 0.5*mat.data[s:e]/mx
                    return mat
                aug_c = augment(raw_c.copy()); aug_q = augment(raw_q.copy())
                N2 = aug_c.shape[0]
                df = np.diff(aug_c.tocsc().indptr)
                idf_vec = np.log((1+N2)/(1+df)) + 1.0
                C2 = normalize(aug_c.multiply(idf_vec), norm='l2')
                Q2 = normalize(aug_q.multiply(idf_vec), norm='l2')
                scores_mat = (C2 @ Q2.T).toarray()
                ranked_results = {q_ids[qi]: [c_ids[i] for i in np.argsort(scores_mat[:,qi])[::-1]] for qi in range(len(q_ids))}
                filtered = {q:v for q,v in ranked_results.items() if q not in SKIP_IDS}
                relevance = {q: list(v) for q,v in gold.items()}
                m = evaluate_all(filtered, relevance, k_values=K_VALUES, label=name, verbose=False)
                mf1,mk,curve = micro_f1(ranked_results); m['_micro_f1']=mf1; m['_micro_k']=mk; m['_micro_f1_curve']=curve
                add(m)
            except Exception as ex:
                print(f"    FAILED: {ex}")
        else:
            m = run_tfidf(best_c, best_q, name, ngram=ng, scheme=sc)
            add(m)

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT F: N-gram sweep for top boosting configs
# (fair comparison with Exp E — same n-gram range, same schemes)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print(f"  EXPERIMENT F: N-gram sweep for top boosting configs")
print("  (Boost Prec+Ratio ×5  and  Boost Arg+Prec+Ratio ×2)")
print(f"{'═'*70}")

top_boost_configs = [
    ("F-B3: Boost Prec+Ratio ×5",         {'Precedent','RatioOfTheDecision'},          5),
    ("F-B8: Boost Arg+Prec+Ratio ×2",     {'Argument','Precedent','RatioOfTheDecision'}, 2),
]

for base_name, boost_roles, factor in top_boost_configs:
    bc = {cid: extract_with_boost(t, boost_roles, factor) for cid, t in cand_rr.items()}
    bq = {qid: extract_with_boost(t, boost_roles, factor) for qid, t in query_rr.items()}
    bc = {k: v for k, v in bc.items() if v.strip()}
    bq = {k: v for k, v in bq.items() if v.strip()}
    for ng in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for sc in ["log", "augmented", "binary"]:
            name = f"{base_name}  n={ng}  scheme={sc}"
            print(f"\n  {name}")
            if sc == "augmented":
                c_ids = sorted(bc.keys()); q_ids = sorted(bq.keys())
                c_strs = [' '.join(clean_text(bc[i], remove_stopwords=True, ngram=ng)) or '__empty__' for i in c_ids]
                q_strs = [' '.join(clean_text(bq[i], remove_stopwords=True, ngram=ng)) or '__empty__' for i in q_ids]
                from sklearn.feature_extraction.text import CountVectorizer
                cv_f = CountVectorizer(analyzer='word', tokenizer=str.split, preprocessor=None, token_pattern=None,
                                      min_df=args.min_df, max_df=args.max_df)
                try:
                    raw_cf = cv_f.fit_transform(c_strs).astype(float)
                    raw_qf = cv_f.transform(q_strs).astype(float)
                    aug_cf = augment(raw_cf.copy()); aug_qf = augment(raw_qf.copy())
                    Nf = aug_cf.shape[0]
                    dff = np.diff(aug_cf.tocsc().indptr)
                    idff = np.log((1 + Nf) / (1 + dff)) + 1.0
                    Cf = normalize(aug_cf.multiply(idff), norm='l2')
                    Qf = normalize(aug_qf.multiply(idff), norm='l2')
                    smf = (Cf @ Qf.T).toarray()
                    rrf = {q_ids[qi]: [c_ids[i] for i in np.argsort(smf[:, qi])[::-1]] for qi in range(len(q_ids))}
                    filtf = {q: v for q, v in rrf.items() if q not in SKIP_IDS}
                    relev = {q: list(v) for q, v in gold.items()}
                    m = evaluate_all(filtf, relev, k_values=K_VALUES, label=name, verbose=False)
                    mf1, mk, curve = micro_f1(rrf); m['_micro_f1'] = mf1; m['_micro_k'] = mk; m['_micro_f1_curve'] = curve
                    add(m)
                except Exception as ex:
                    print(f"    FAILED: {ex}")
            else:
                m = run_tfidf(bc, bq, name, ngram=ng, scheme=sc)
                add(m)

# ══════════════════════════════════════════════════════════════════════════════
# BM25 RETRIEVAL HELPER
# (same implementation as Models/BM25/run_script.py)
# ══════════════════════════════════════════════════════════════════════════════
from scipy import sparse as _sparse

class BM25:
    """BM25 retrieval using sklearn TfidfVectorizer for IDF + raw TF counts."""
    def __init__(self, b=0.7, k1=1.6, ngram=4, min_df=2, max_df=0.95):
        self.b = b; self.k1 = k1
        self.vectorizer = TfidfVectorizer(
            analyzer='word', tokenizer=str.split, preprocessor=None, token_pattern=None,
            ngram_range=(1, 1),   # BM25 uses unigrams; n-gram expansion done in clean_text
            min_df=min_df, max_df=max_df, use_idf=True,
        )

    def fit(self, X_strs):
        self.vectorizer.fit(X_strs)
        raw = super(TfidfVectorizer, self.vectorizer).transform(X_strs)
        self._raw_csc = raw.tocsc()
        self._len_X   = raw.sum(1).A1
        self.avdl     = self._len_X.mean()
        self._dl_norm = self.k1 * (1 - self.b + self.b * self._len_X / self.avdl)

    def score_all(self, q_strs):
        """Return (n_cands, n_queries) BM25 score matrix."""
        k1 = self.k1
        scores = np.zeros((self._raw_csc.shape[0], len(q_strs)))
        for qi, qs in enumerate(q_strs):
            q_vec, = super(TfidfVectorizer, self.vectorizer).transform([qs])
            if not _sparse.isspmatrix_csr(q_vec):
                q_vec = q_vec.tocsr()
            if q_vec.nnz == 0:
                continue
            cols = q_vec.indices
            X_sub = self._raw_csc[:, cols]
            idf   = self.vectorizer._tfidf.idf_[None, cols] - 1.0
            numer = X_sub.multiply(np.broadcast_to(idf, X_sub.shape)) * (k1 + 1)
            denom = X_sub + self._dl_norm[:, None]
            scores[:, qi] = (numer / denom).sum(1).A1
        return scores


def run_bm25(cand_texts, query_texts, label, ngram=None, b=0.7, k1=1.6):
    """Run BM25 retrieval (with clean_text n-gram preprocessing) and return metrics."""
    if ngram is None: ngram = args.ngram
    c_ids = sorted(cand_texts.keys())
    q_ids = sorted(query_texts.keys())

    c_strs = [' '.join(clean_text(cand_texts[i], remove_stopwords=True, ngram=ngram)) or '__empty__' for i in c_ids]
    q_strs = [' '.join(clean_text(query_texts[i], remove_stopwords=True, ngram=ngram)) or '__empty__' for i in q_ids]

    try:
        bm = BM25(b=b, k1=k1, min_df=args.min_df, max_df=args.max_df)
        bm.fit(c_strs)
        scores_mat = bm.score_all(q_strs)  # (n_cands, n_queries)
    except Exception as e:
        print(f"  SKIPPED '{label}': {e}")
        return None

    ranked_results = {}
    for qi, qid in enumerate(q_ids):
        top = np.argsort(scores_mat[:, qi])[::-1]
        ranked_results[qid] = [c_ids[i] for i in top]

    filtered  = {q: v for q, v in ranked_results.items() if q not in SKIP_IDS}
    relevance = {q: list(v) for q, v in gold.items()}
    m = evaluate_all(filtered, relevance, k_values=K_VALUES, label=label, verbose=False)
    mf1, mk, curve = micro_f1(ranked_results)
    m['_micro_f1'] = mf1; m['_micro_k'] = mk; m['_micro_f1_curve'] = curve
    return m


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT G: BM25 on RR-filtered / boosted docs  (n-gram sweep)
# Compare BM25 vs TF-IDF on the same text preprocessing pipeline
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print(f"  EXPERIMENT G: BM25 on RR docs  (b=0.7, k1=1.6)")
print(f"  Configs: Baseline  |  A3 subset  |  B8 boost  |  best F-B8 ngram")
print(f"{'═'*70}")

g_configs = [
    ("G0: BM25 Baseline (orig docs)",       cand_orig  if orig_available else None,
                                             query_orig if orig_available else None),
    ("G1: BM25 All RR (strip labels)",
     {cid: extract_all_strip_labels(t) for cid, t in cand_rr.items()},
     {qid: extract_all_strip_labels(t) for qid, t in query_rr.items()}),
    ("G2: BM25 Fact+Arg+Prec+Ratio (A3)",
     {k: v for k, v in {cid: extract_roles(t, {'Fact','Argument','Precedent','RatioOfTheDecision'}) for cid,t in cand_rr.items()}.items() if v.strip()},
     {k: v for k, v in {qid: extract_roles(t, {'Fact','Argument','Precedent','RatioOfTheDecision'}) for qid,t in query_rr.items()}.items() if v.strip()}),
    ("G3: BM25 Boost Arg+Prec+Ratio ×2 (B8)",
     {cid: extract_with_boost(t, {'Argument','Precedent','RatioOfTheDecision'}, 2) for cid, t in cand_rr.items()},
     {qid: extract_with_boost(t, {'Argument','Precedent','RatioOfTheDecision'}, 2) for qid, t in query_rr.items()}),
]

# n-gram sweep for best boosting config (mirrors Exp F-B8)
print(f"\n  --- G fixed configs (n={N}) ---")
for name, gc, gq in g_configs:
    if gc is None:
        print(f"\n  {name}  [SKIPPED — orig docs not available]")
        continue
    gc = {k: v for k, v in gc.items() if v.strip()}
    gq = {k: v for k, v in gq.items() if v.strip()}
    print(f"\n  {name}")
    m = run_bm25(gc, gq, name)
    add(m)

print(f"\n  --- G4: BM25 Boost Arg+Prec+Ratio ×2 — n-gram sweep ---")
g4_c = {cid: extract_with_boost(t, {'Argument','Precedent','RatioOfTheDecision'}, 2) for cid, t in cand_rr.items()}
g4_q = {qid: extract_with_boost(t, {'Argument','Precedent','RatioOfTheDecision'}, 2) for qid, t in query_rr.items()}
g4_c = {k: v for k, v in g4_c.items() if v.strip()}
g4_q = {k: v for k, v in g4_q.items() if v.strip()}
for ng in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    name = f"G4: BM25 Boost Arg+Prec+Ratio ×2  n={ng}"
    print(f"\n  {name}")
    m = run_bm25(g4_c, g4_q, name, ngram=ng)
    add(m)

print(f"\n  --- G5: BM25 Fact+Arg+Prec+Ratio subset — n-gram sweep ---")
g5_c = {k: v for k, v in {cid: extract_roles(t, {'Fact','Argument','Precedent','RatioOfTheDecision'}) for cid,t in cand_rr.items()}.items() if v.strip()}
g5_q = {k: v for k, v in {qid: extract_roles(t, {'Fact','Argument','Precedent','RatioOfTheDecision'}) for qid,t in query_rr.items()}.items() if v.strip()}
for ng in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    name = f"G5: BM25 Fact+Arg+Prec+Ratio  n={ng}"
    print(f"\n  {name}")
    m = run_bm25(g5_c, g5_q, name, ngram=ng)
    add(m)

# ══════════════════════════════════════════════════════════════════════════════
# TF-IDF-F — Early-field-fusion TF-IDF  (TF-IDF analogue of BM25F)
# ---------------------------------------------------------------------------
# Field weights applied to TF *before* IDF multiplication and L2 normalisation.
# Uses the same TF scheme (log/augmented/binary) as all other experiments.
# No text duplication, no IDF skewing, cleaner math than text-repetition hacks.
#
#   Candidate:  wtf(t,d) = Σ_f  w_f · tf_scheme( raw_tf(t, d_f) )
#               score_vec(d) = L2_norm( wtf(d) ⊙ idf )
#   Query:      all active roles concatenated, same shared vocab/IDF
# ══════════════════════════════════════════════════════════════════════════════
def run_tfidf_field(cand_rr_d, query_rr_d, label, field_weights, ngram=None, scheme="log"):
    """TF-IDF-F: per-field TF combined with weights before IDF + L2 norm."""
    if ngram is None: ngram = args.ngram
    c_ids = sorted(cand_rr_d.keys())
    q_ids = sorted(query_rr_d.keys())
    roles = [r for r, w in field_weights.items() if w > 0]

    # Per-role cleaned candidate strings
    c_role_strs = {
        r: [' '.join(clean_text(extract_roles(cand_rr_d.get(cid, ''), {r}),
                                remove_stopwords=True, ngram=ngram))
            for cid in c_ids]
        for r in roles
    }

    # Shared vocab + IDF from all-roles-concatenated candidates
    c_global = [
        ' '.join(clean_text(extract_all_strip_labels(cand_rr_d.get(i, '')),
                             remove_stopwords=True, ngram=ngram)) or '__empty__'
        for i in c_ids
    ]
    shared_vec = TfidfVectorizer(
        analyzer='word', tokenizer=str.split, preprocessor=None,
        token_pattern=None, min_df=args.min_df, max_df=args.max_df,
        sublinear_tf=False, use_idf=True, norm=None,  # raw counts; scheme applied below
    )
    try:
        shared_vec.fit(c_global)
    except Exception as e:
        print(f"  SKIPPED '{label}': {e}")
        return None
    shared_idf = shared_vec._tfidf.idf_   # (V,)

    from sklearn.feature_extraction.text import CountVectorizer as _CFV
    cv = _CFV(analyzer='word', tokenizer=str.split, preprocessor=None,
              token_pattern=None, vocabulary=shared_vec.vocabulary_)

    def apply_tf(strs):
        X = cv.transform(strs).astype(float).tocsr()
        if scheme == "log":
            X.data = np.log1p(X.data)
        elif scheme == "augmented":
            for i in range(X.shape[0]):
                s, e = X.indptr[i], X.indptr[i+1]
                if e > s:
                    mx = X.data[s:e].max()
                    if mx > 0: X.data[s:e] = 0.5 + 0.5 * X.data[s:e] / mx
        elif scheme == "binary":
            X.data[:] = 1.0
        return X

    # Weighted TF sum across fields (candidates)
    wtf = None
    for r, w in field_weights.items():
        if w == 0: continue
        tf_r = apply_tf(c_role_strs[r])
        wtf = w * tf_r if wtf is None else wtf + w * tf_r

    C = normalize(wtf.multiply(shared_idf), norm='l2')

    # Query: concatenate all active role texts, same TF scheme + shared vocab/IDF
    q_strs = [
        ' '.join(clean_text(extract_roles(query_rr_d.get(qid, ''), set(roles)),
                             remove_stopwords=True, ngram=ngram))
        for qid in q_ids
    ]
    Q = normalize(apply_tf(q_strs).multiply(shared_idf), norm='l2')

    scores_mat = (C @ Q.T).toarray()
    ranked_results = {
        q_ids[qi]: [c_ids[i] for i in np.argsort(scores_mat[:, qi])[::-1]]
        for qi in range(len(q_ids))
    }
    filtered  = {q: v for q, v in ranked_results.items() if q not in SKIP_IDS}
    relevance = {q: list(v) for q, v in gold.items()}
    m = evaluate_all(filtered, relevance, k_values=K_VALUES, label=label, verbose=False)
    mf1, mk, curve = micro_f1(ranked_results)
    m['_micro_f1'] = mf1; m['_micro_k'] = mk; m['_micro_f1_curve'] = curve
    return m


# ══════════════════════════════════════════════════════════════════════════════
# REFINED LATE FUSION — shared vocabulary  (fixes Exp D)
# ---------------------------------------------------------------------------
# Exp D failed because each role had its own TF-IDF fitted independently:
#   - Different vocabularies → IDF computed from different document sets
#   - __empty__ token polluted rare-term IDF values
# Fix: fit ONE TF-IDF vectorizer on ALL text, use that shared vocabulary/IDF
# for every per-role matrix.  Empty role docs get zero vectors (correct).
# ══════════════════════════════════════════════════════════════════════════════
def run_late_fusion_shared(cand_rr_d, query_rr_d, label, role_weights,
                            ngram=None, scheme="log"):
    """
    Late fusion: separate L2-normalised TF-IDF per role (shared vocab/IDF),
    then weighted score sum.  Consistent term-space across all role matrices.
    """
    if ngram is None: ngram = args.ngram
    c_ids = sorted(cand_rr_d.keys())
    q_ids = sorted(query_rr_d.keys())
    roles = [r for r, w in role_weights.items() if w > 0]

    # ── 1. Fit shared vocabulary / IDF on all-roles-concatenated candidates ──
    c_global = [
        ' '.join(clean_text(extract_all_strip_labels(cand_rr_d.get(i, '')),
                             remove_stopwords=True, ngram=ngram)) or '__empty__'
        for i in c_ids
    ]
    shared_vec = TfidfVectorizer(
        analyzer='word', tokenizer=str.split, preprocessor=None,
        token_pattern=None, min_df=args.min_df, max_df=args.max_df,
        sublinear_tf=(scheme == "log"), use_idf=True, norm=None,
    )
    shared_vec.fit(c_global)
    shared_idf = shared_vec._tfidf.idf_    # (V,)

    from sklearn.feature_extraction.text import CountVectorizer as _CV3
    cv3 = _CV3(analyzer='word', tokenizer=str.split, preprocessor=None,
               token_pattern=None, vocabulary=shared_vec.vocabulary_)

    def _tf(strs):
        X = cv3.transform(strs).astype(float).tocsr()
        if scheme == "log":
            X.data = np.log1p(X.data)
        return X

    # ── 2. Per-role matrices with shared vocab/IDF, then weighted fusion ─────
    fused   = np.zeros((len(c_ids), len(q_ids)))
    total_w = 0.0
    for r, w in role_weights.items():
        if w == 0:
            continue
        c_r = [' '.join(clean_text(extract_roles(cand_rr_d.get(i, ''), {r}),
                                   remove_stopwords=True, ngram=ngram))
               for i in c_ids]
        q_r = [' '.join(clean_text(extract_roles(query_rr_d.get(i, ''), {r}),
                                   remove_stopwords=True, ngram=ngram))
               for i in q_ids]
        C = normalize(_tf(c_r).multiply(shared_idf), norm='l2')
        Q = normalize(_tf(q_r).multiply(shared_idf), norm='l2')
        fused   += w * (C @ Q.T).toarray()
        total_w += w

    if total_w == 0:
        return None
    fused /= total_w

    ranked_results = {
        q_ids[qi]: [c_ids[i] for i in np.argsort(fused[:, qi])[::-1]]
        for qi in range(len(q_ids))
    }
    filtered  = {q: v for q, v in ranked_results.items() if q not in SKIP_IDS}
    relevance = {q: list(v) for q, v in gold.items()}
    m = evaluate_all(filtered, relevance, k_values=K_VALUES, label=label, verbose=False)
    mf1, mk, curve = micro_f1(ranked_results)
    m['_micro_f1'] = mf1; m['_micro_k'] = mk; m['_micro_f1_curve'] = curve
    return m


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT H: BM25F — Multi-field BM25 (correct alternative to boosting)
# Field weights applied inside BM25 math, not via text duplication.
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print(f"  EXPERIMENT H: TF-IDF-F — early-field-fusion TF-IDF  (n={N})")
print(f"  Field weights combined BEFORE IDF+L2 — no text duplication, no IDF skew")
print(f"{'═'*70}")

h_fixed_configs = [
    ("H1: TF-IDF-F equal weights (all 7 roles)",
     {r: 1.0 for r in ALL_ROLES}),
    ("H2: TF-IDF-F Prec×2 + Ratio×2 + rest×1 (all 7 roles)",
     {'Precedent': 2, 'RatioOfTheDecision': 2, 'Fact': 1, 'Argument': 1,
      'Statute': 1, 'RulingByLowerCourt': 1, 'RulingByPresentCourt': 1}),
    ("H3: TF-IDF-F Prec×3 + Ratio×3 + Fact×1 + Arg×1 (4 roles)",
     {'Precedent': 3, 'RatioOfTheDecision': 3, 'Fact': 1, 'Argument': 1}),
    ("H4: TF-IDF-F Prec×5 + Ratio×3 + Fact×1 + Arg×1",
     {'Precedent': 5, 'RatioOfTheDecision': 3, 'Fact': 1, 'Argument': 1}),
    ("H5: TF-IDF-F Arg×1 + Prec×2 + Ratio×2 (mirrors B8)",
     {'Argument': 1, 'Precedent': 2, 'RatioOfTheDecision': 2}),
    ("H6: TF-IDF-F Arg×1 + Prec×3 + Ratio×3",
     {'Argument': 1, 'Precedent': 3, 'RatioOfTheDecision': 3}),
    ("H7: TF-IDF-F Prec×1 + Ratio×1 only (2 roles)",
     {'Precedent': 1, 'RatioOfTheDecision': 1}),
]

for name, weights in h_fixed_configs:
    print(f"\n  {name}")
    m = run_tfidf_field(cand_rr, query_rr, name, weights)
    add(m)

print(f"\n  --- H-grid: TF-IDF-F weight grid search (Prec × Ratio × Fact × Arg) ---")
_h_best_f1, _h_best_w = 0.0, None
for _wp in [1, 2, 3, 5]:
    for _wr in [1, 2, 3]:
        for _wf in [0, 1]:
            for _wa in [0, 1]:
                if _wf == 0 and _wa == 0:
                    continue
                _hw = {r: v for r, v in [('Precedent', _wp), ('RatioOfTheDecision', _wr),
                                         ('Fact', _wf), ('Argument', _wa)] if v > 0}
                _hname = f"H-grid: TF-IDF-F Prec×{_wp} Ratio×{_wr} Fact×{_wf} Arg×{_wa}"
                print(f"\n  {_hname}")
                m = run_tfidf_field(cand_rr, query_rr, _hname, _hw)
                add(m)
                if m and m['_micro_f1'] > _h_best_f1:
                    _h_best_f1, _h_best_w = m['_micro_f1'], _hw

# n-gram + scheme sweep on best grid weight config
if _h_best_w:
    _hw_tag = '+'.join(f"{r[:4]}×{w}" for r, w in _h_best_w.items())
    print(f"\n  --- H-ngram: TF-IDF-F n-gram × scheme sweep on best weights ({_hw_tag}) ---")
    for _ng in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for _sc in ['log', 'augmented', 'binary']:
            _hname = f"H-ngram: TF-IDF-F {_hw_tag}  n={_ng}  scheme={_sc}"
            print(f"\n  {_hname}")
            m = run_tfidf_field(cand_rr, query_rr, _hname, _h_best_w, ngram=_ng, scheme=_sc)
            add(m)

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT I: Refined Late Fusion — shared vocabulary  (fixes Exp D)
# Each role gets its own TF-IDF, but all share ONE vocabulary and IDF.
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print(f"  EXPERIMENT I: Refined Late Fusion — shared vocabulary  (n={N})")
print(f"  Fixes Exp D: global IDF prevents per-role vocabulary/IDF skew")
print(f"{'═'*70}")

i_fixed_configs = [
    ("I1: Refined fusion equal (Fact+Arg+Prec+Ratio)",
     {'Fact': 1, 'Argument': 1, 'Precedent': 1, 'RatioOfTheDecision': 1}),
    ("I2: Refined fusion Prec×3 + Ratio×3 + Fact×1 + Arg×1",
     {'Fact': 1, 'Argument': 1, 'Precedent': 3, 'RatioOfTheDecision': 3}),
    ("I3: Refined fusion Prec×5 + Ratio×3 + Fact×1 + Arg×1",
     {'Fact': 1, 'Argument': 1, 'Precedent': 5, 'RatioOfTheDecision': 3}),
    ("I4: Refined fusion Arg×1 + Prec×3 + Ratio×3 (no Fact)",
     {'Argument': 1, 'Precedent': 3, 'RatioOfTheDecision': 3}),
    ("I5: Refined fusion all 7 equal",
     {r: 1.0 for r in ALL_ROLES}),
    ("I6: Refined fusion Prec×2 + Ratio×2 + Fact×1 + Arg×1 + Stat×0.5",
     {'Fact': 1, 'Argument': 1, 'Precedent': 2, 'RatioOfTheDecision': 2, 'Statute': 0.5}),
]

for name, weights in i_fixed_configs:
    print(f"\n  {name}")
    m = run_late_fusion_shared(cand_rr, query_rr, name, weights)
    add(m)

print(f"\n  --- I-grid: Refined Fusion weight grid search ---")
_i_best_f1, _i_best_w = 0.0, None
for _wp in [1, 2, 3, 5]:
    for _wr in [1, 2, 3]:
        for _wf in [0, 1]:
            for _wa in [0, 1]:
                if _wf == 0 and _wa == 0:
                    continue
                _iw = {r: v for r, v in [('Precedent', _wp), ('RatioOfTheDecision', _wr),
                                         ('Fact', _wf), ('Argument', _wa)] if v > 0}
                _iname = f"I-grid: Fusion Prec×{_wp} Ratio×{_wr} Fact×{_wf} Arg×{_wa}"
                print(f"\n  {_iname}")
                m = run_late_fusion_shared(cand_rr, query_rr, _iname, _iw)
                add(m)
                if m and m['_micro_f1'] > _i_best_f1:
                    _i_best_f1, _i_best_w = m['_micro_f1'], _iw

# n-gram sweep on best grid weight config
if _i_best_w:
    _iw_tag = '+'.join(f"{r[:4]}×{w}" for r, w in _i_best_w.items())
    print(f"\n  --- I-ngram: Refined Fusion n-gram sweep on best weights ({_iw_tag}) ---")
    for _ng in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        _iname = f"I-ngram: Fusion {_iw_tag}  n={_ng}"
        print(f"\n  {_iname}")
        m = run_late_fusion_shared(cand_rr, query_rr, _iname, _i_best_w, ngram=_ng)
        add(m)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS  (JSON + CSV next to this script)
# ══════════════════════════════════════════════════════════════════════════════
import json as _json, csv as _csv, datetime as _dt

results.sort(key=lambda x: x['_micro_f1'], reverse=True)

_ts   = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results_{_ts}')

# — JSON (full, with curve) —
_json_path = _base + '.json'
with open(_json_path, 'w') as _f:
    _json.dump(results, _f, indent=2, default=str)
print(f"\n  Results saved → {_json_path}")

# — CSV (flat, every metric as a column) —
_csv_path = _base + '.csv'
_scalar_keys = (['model', '_micro_f1', '_micro_k', 'MAP', 'MRR', 'R-Precision', 'n_queries'] +
                [f'{m}@{k}' for k in K_VALUES
                 for m in ('P', 'R', 'F1', 'NDCG', 'MAP', 'MicroF1')])
with open(_csv_path, 'w', newline='') as _f:
    _w = _csv.DictWriter(_f, fieldnames=_scalar_keys, extrasaction='ignore')
    _w.writeheader()
    for r in results:
        _w.writerow({k: round(v * 100, 4) if isinstance(v, float) and not k.startswith('_micro_k')
                       else v
                     for k, v in r.items() if k != '_micro_f1_curve'})
print(f"  Results saved → {_csv_path}")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY TABLE — all metrics from evaluate_all
# ══════════════════════════════════════════════════════════════════════════════
W = 140
print(f"\n\n{'═'*W}")
print(f"  FULL RESULTS — sorted by BestMicroF1".center(W))
print(f"{'═'*W}")

# Compact one-line summary with key metrics
print(f"  {'Rank':<4} {'Config':<50} {'BstMicF1':>9} {'@K':>3} {'MAP':>7} {'MRR':>7} "
      f"{'R-P':>7} {'P@10':>7} {'R@10':>7} {'F1@10':>7} {'NDCG@10':>8} {'MAP@10':>8} {'MicF1@10':>9}")
print(f"  {'─'*136}")
for i, r in enumerate(results):
    marker = " ★" if i == 0 else "  "
    name = r.get('model', '?')[:49]
    print(f"  {i+1:<4} {name:<50} {r['_micro_f1']*100:>8.2f}%  {r['_micro_k']:>2} "
          f"{r['MAP']*100:>6.2f}% "
          f"{r['MRR']*100:>6.2f}% "
          f"{r['R-Precision']*100:>6.2f}% "
          f"{r['P@10']*100:>6.2f}% "
          f"{r['R@10']*100:>6.2f}% "
          f"{r['F1@10']*100:>6.2f}% "
          f"{r['NDCG@10']*100:>7.2f}% "
          f"{r['MAP@10']*100:>7.2f}% "
          f"{r['MicroF1@10']*100:>8.2f}%{marker}")
print(f"{'═'*W}")

# ── Full per-K breakdown for every K value (all keys) — top 5 ──────────────
print(f"\n{'─'*80}")
print("  DETAILED METRICS — Top 5 configs (all K values)")
print(f"{'─'*80}")
for r in results[:5]:
    print(f"\n  {'─'*76}")
    print(f"  #{results.index(r)+1}  {r.get('model', '?')}")
    print(f"  MAP={r['MAP']*100:.2f}%  MRR={r['MRR']*100:.2f}%  "
          f"R-Precision={r['R-Precision']*100:.2f}%  "
          f"BestMicroF1={r['_micro_f1']*100:.2f}%@K={r['_micro_k']}")
    print(f"  {'K':<4} {'P':>7} {'R':>7} {'F1':>7} {'NDCG':>8} {'MAP@K':>8} {'MicroF1':>9}")
    print(f"  {'─'*60}")
    for k in K_VALUES:
        print(f"  {k:<4} {r[f'P@{k}']*100:>6.2f}% "
              f"{r[f'R@{k}']*100:>6.2f}% "
              f"{r[f'F1@{k}']*100:>6.2f}% "
              f"{r[f'NDCG@{k}']*100:>7.2f}% "
              f"{r[f'MAP@{k}']*100:>7.2f}% "
              f"{r[f'MicroF1@{k}']*100:>8.2f}%")

print(f"\n  Best overall: {results[0].get('model', '?')}")
print(f"  BestMicroF1 = {results[0]['_micro_f1']*100:.2f}% @ K={results[0]['_micro_k']}")
print(f"  MAP         = {results[0]['MAP']*100:.2f}%")
print(f"  MRR         = {results[0]['MRR']*100:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS  (saved to plots/ subdirectory next to this script)
# ══════════════════════════════════════════════════════════════════════════════
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(PLOT_DIR, exist_ok=True)

    EXP_COLORS = {'A': '#4C72B0', 'B': '#DD8452', 'C': '#55A868',
                  'D': '#C44E52', 'E': '#8172B2', 'F': '#CCB974',
                  'G': '#64B5CD', 'H': '#E377C2', 'I': '#7F7F7F',
                  '?': '#aaaaaa'}

    def exp_color(name):
        for k, v in EXP_COLORS.items():
            if name.startswith(k): return v
        return EXP_COLORS['?']

    print(f"\n\n{'─'*70}")
    print("  GENERATING PLOTS")
    print(f"{'─'*70}")

    # ── Plot 1: All configs — horizontal BestMicroF1 bar (sorted best at top) ─
    fig, ax = plt.subplots(figsize=(12, max(6, len(results) * 0.35)))
    rev = list(reversed(results))
    ax.barh(range(len(rev)),
            [r['_micro_f1'] * 100 for r in rev],
            color=[exp_color(r.get('model', '?')) for r in rev],
            edgecolor='white', linewidth=0.4)
    ax.set_yticks(range(len(rev)))
    ax.set_yticklabels([r.get('model', '?')[:58] for r in rev], fontsize=7)
    ax.set_xlabel('Best MicroF1 @ optimal K (%)')
    ax.set_title('All configurations — Best MicroF1 (colour = experiment group)')
    ax.axvline(results[0]['_micro_f1'] * 100, color='black',
               linestyle='--', linewidth=0.8, alpha=0.5, label='Best')
    ax.legend(handles=[Patch(facecolor=v, label=f'Exp {k}')
                       for k, v in EXP_COLORS.items() if k != '?'],
              loc='lower right', fontsize=8)
    plt.tight_layout()
    p1 = os.path.join(PLOT_DIR, '1_all_configs_microf1.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  [Plot 1] All-config bar chart          → {p1}")

    # ── Plot 2: MicroF1@K curves — top 6 configs + baseline ───────────────────
    ks = list(range(1, 21))
    baseline_r = next((r for r in results if r.get('model', '').startswith('A0')), None)
    TOP_N = min(6, len(results))
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap10 = plt.get_cmap('tab10')
    if baseline_r and baseline_r.get('_micro_f1_curve'):
        ax.plot(ks, [v * 100 for v in baseline_r['_micro_f1_curve']],
                color='black', linestyle='--', linewidth=1.8,
                label='Baseline (orig docs)', zorder=10)
    for idx, r in enumerate(results[:TOP_N]):
        if not r.get('_micro_f1_curve'): continue
        ax.plot(ks, [v * 100 for v in r['_micro_f1_curve']],
                color=cmap10(idx), linewidth=1.8, marker='o', markersize=3,
                label=r.get('model', '?')[:42])
    ax.set_xlabel('K'); ax.set_ylabel('MicroF1 (%)')
    ax.set_title('MicroF1@K curves — Top 6 configs vs Baseline')
    ax.legend(fontsize=7, loc='upper left',
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(PLOT_DIR, '2_microf1_at_k_curves.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  [Plot 2] MicroF1@K curves (top 6)      → {p2}")

    # ── Plot 3: Multi-metric grouped bar — top 5 configs ──────────────────────
    TOP5 = results[:min(5, len(results))]
    metric_keys   = ['MAP', 'MRR', 'NDCG@10', 'F1@10', '_micro_f1']
    metric_labels = ['MAP', 'MRR', 'NDCG@10', 'F1@10', 'BestMicroF1']
    x3 = np.arange(len(metric_keys)); w3 = 0.15
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap_s2 = plt.get_cmap('Set2')
    for i, r in enumerate(TOP5):
        vals3 = [r[mk] * 100 if mk != '_micro_f1' else r['_micro_f1'] * 100
                 for mk in metric_keys]
        offset = (i - len(TOP5) / 2 + 0.5) * w3
        ax.bar(x3 + offset, vals3, w3,
               label=r.get('model', '?')[:35],
               color=cmap_s2(i), edgecolor='white')
    ax.set_xticks(x3); ax.set_xticklabels(metric_labels)
    ax.set_ylabel('Score (%)')
    ax.set_title('Multi-metric comparison — Top 5 configs')
    ax.legend(fontsize=7, loc='upper left',
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    p3 = os.path.join(PLOT_DIR, '3_multimetric_top5.png')
    plt.savefig(p3, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  [Plot 3] Multi-metric top-5 bar chart  → {p3}")

    # ── Plot 4: Exp E heatmap — n-gram × scheme for Fact+Arg+Prec+Ratio ──────
    e_res = [r for r in results if r.get('model', '').startswith('E:')]
    if e_res:
        ngrams_e = sorted(set(
            int(re.search(r'n=(\d+)', r.get('model','')).group(1))
            for r in e_res if re.search(r'n=(\d+)', r.get('model',''))
        ))
        schemes_e = [sc for sc in ['log', 'augmented', 'binary']
                     if any(sc in r.get('model','') for r in e_res)]
        if ngrams_e and schemes_e:
            hm5 = np.full((len(schemes_e), len(ngrams_e)), np.nan)
            for r in e_res:
                nm = r.get('model', '')
                mg = re.search(r'n=(\d+)', nm)
                if not mg: continue
                ng_v = int(mg.group(1))
                for si, sc in enumerate(schemes_e):
                    if f'scheme={sc}' in nm and ng_v in ngrams_e:
                        hm5[si, ngrams_e.index(ng_v)] = r['_micro_f1'] * 100
            fig, ax = plt.subplots(figsize=(7, 4))
            im5 = ax.imshow(hm5, cmap='Blues', aspect='auto',
                            vmin=np.nanmin(hm5) - 0.3,
                            vmax=np.nanmax(hm5) + 0.1)
            ax.set_xticks(range(len(ngrams_e)))
            ax.set_xticklabels([f'n={v}' for v in ngrams_e])
            ax.set_yticks(range(len(schemes_e)))
            ax.set_yticklabels(schemes_e)
            for si in range(len(schemes_e)):
                for ni in range(len(ngrams_e)):
                    if not np.isnan(hm5[si, ni]):
                        ax.text(ni, si, f'{hm5[si, ni]:.2f}',
                                ha='center', va='center',
                                fontsize=9, fontweight='bold')
            plt.colorbar(im5, ax=ax, label='BestMicroF1 (%)')
            ax.set_title('Exp E (Fact+Arg+Prec+Ratio): BestMicroF1 by n-gram × scheme')
            ax.set_xlabel('n-gram'); ax.set_ylabel('TF weighting scheme')
            plt.tight_layout()
            p4e = os.path.join(PLOT_DIR, '4_expE_heatmap.png')
            plt.savefig(p4e, dpi=150, bbox_inches='tight'); plt.close()
            print(f"  [Plot 4] Exp E n-gram×scheme heatmap   → {p4e}")

    # ── Plot 5: Exp F heatmap — n-gram × scheme for top boosting configs ──────
    for fb_name, fb_prefix in [("F-B3: Boost Prec+Ratio ×5", "F-B3"),
                               ("F-B8: Boost Arg+Prec+Ratio ×2", "F-B8")]:
        f_res = [r for r in results if r.get('model', '').startswith(fb_prefix)]
        if not f_res: continue
        ngrams_f = sorted(set(
            int(re.search(r'n=(\d+)', r.get('model','')).group(1))
            for r in f_res if re.search(r'n=(\d+)', r.get('model',''))
        ))
        schemes_f = [sc for sc in ['log', 'augmented', 'binary']
                     if any(sc in r.get('model','') for r in f_res)]
        if not ngrams_f or not schemes_f: continue
        hmf = np.full((len(schemes_f), len(ngrams_f)), np.nan)
        for r in f_res:
            nm = r.get('model', '')
            mg = re.search(r'n=(\d+)', nm)
            if not mg: continue
            ng_v = int(mg.group(1))
            for si, sc in enumerate(schemes_f):
                if f'scheme={sc}' in nm and ng_v in ngrams_f:
                    hmf[si, ngrams_f.index(ng_v)] = r['_micro_f1'] * 100
        fig, ax = plt.subplots(figsize=(10, 3))
        imf = ax.imshow(hmf, cmap='Oranges', aspect='auto',
                        vmin=np.nanmin(hmf) - 0.3, vmax=np.nanmax(hmf) + 0.1)
        ax.set_xticks(range(len(ngrams_f)))
        ax.set_xticklabels([f'n={v}' for v in ngrams_f])
        ax.set_yticks(range(len(schemes_f)))
        ax.set_yticklabels(schemes_f)
        for si in range(len(schemes_f)):
            for ni in range(len(ngrams_f)):
                if not np.isnan(hmf[si, ni]):
                    ax.text(ni, si, f'{hmf[si, ni]:.2f}',
                            ha='center', va='center', fontsize=8, fontweight='bold')
        plt.colorbar(imf, ax=ax, label='BestMicroF1 (%)')
        ax.set_title(f'Exp F ({fb_name}): BestMicroF1 by n-gram × scheme')
        ax.set_xlabel('n-gram'); ax.set_ylabel('TF scheme')
        plt.tight_layout()
        slug = fb_prefix.replace(':', '').replace(' ', '_')
        p5f = os.path.join(PLOT_DIR, f'5_expF_{slug}_heatmap.png')
        plt.savefig(p5f, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  [Plot 5] Exp F {fb_prefix} heatmap        → {p5f}")

    # ── Plot 5c: Exp G BM25 n-gram sweep — comparison line chart ──────────────
    g4_res = [r for r in results if r.get('model', '').startswith('G4:')]
    g5_res = [r for r in results if r.get('model', '').startswith('G5:')]
    e_log  = [r for r in results if r.get('model', '').startswith('E:') and 'scheme=log' in r.get('model', '')]
    fb8_log = [r for r in results if r.get('model', '').startswith('F-B8:') and 'scheme=log' in r.get('model', '')]
    if g4_res or g5_res:
        fig, ax = plt.subplots(figsize=(10, 5))
        def _sorted_by_n(lst):
            return sorted(lst, key=lambda r: int(re.search(r'n=(\d+)', r.get('model','')).group(1))
                          if re.search(r'n=(\d+)', r.get('model','')) else 0)
        for series, label_s, color_s, ls in [
            (g4_res,  'BM25 Boost Arg+Prec+Ratio ×2', '#64B5CD', '-o'),
            (g5_res,  'BM25 Fact+Arg+Prec+Ratio',     '#1A7DAF', '-s'),
            (e_log,   'TF-IDF Fact+Arg+Prec+Ratio log','#8172B2', '--^'),
            (fb8_log, 'TF-IDF Boost Arg+Prec+Ratio ×2 log','#DD8452', '--D'),
        ]:
            s = _sorted_by_n(series)
            if not s: continue
            ns = [int(re.search(r'n=(\d+)', r.get('model','')).group(1)) for r in s]
            fs = [r['_micro_f1'] * 100 for r in s]
            ax.plot(ns, fs, ls, label=label_s, color=color_s, linewidth=1.8, markersize=5)
        ax.set_xlabel('n-gram'); ax.set_ylabel('BestMicroF1 (%)')
        ax.set_title('BM25 vs TF-IDF: n-gram sweep on RR docs')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p5g = os.path.join(PLOT_DIR, '5c_expG_bm25_vs_tfidf.png')
        plt.savefig(p5g, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  [Plot 5c] BM25 vs TF-IDF n-gram sweep  → {p5g}")

    # ── Plot 6: Exp A — role subset comparison (MAP, MRR, BestMicroF1) ────────
    a_res = [r for r in results if re.match(r'^A\d', r.get('model', ''))]
    if len(a_res) >= 2:
        a_labels = [r.get('model', '?')[4:38] for r in a_res]
        xi6 = np.arange(len(a_res)); w6 = 0.25
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.bar(xi6 - w6, [r['MAP'] * 100         for r in a_res], w6,
               label='MAP',         color='#4C72B0')
        ax.bar(xi6,      [r['MRR'] * 100         for r in a_res], w6,
               label='MRR',         color='#DD8452')
        ax.bar(xi6 + w6, [r['_micro_f1'] * 100   for r in a_res], w6,
               label='BestMicroF1', color='#55A868')
        ax.set_xticks(xi6)
        ax.set_xticklabels(a_labels, rotation=35, ha='right', fontsize=7)
        ax.set_ylabel('Score (%)')
        ax.set_title('Experiment A: Effect of RR Role Subset Selection')
        ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        p5a = os.path.join(PLOT_DIR, '5_expA_role_subsets.png')
        plt.savefig(p5a, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  [Plot 5] Exp A role-subset bar chart   → {p5a}")

    # ── Plot 6: Exp B — boost factor effect (with baseline reference lines) ───
    b_res = [r for r in results if re.match(r'^B\d', r.get('model', ''))]
    if len(b_res) >= 2:
        b_labels = [r.get('model', '?')[4:42] for r in b_res]
        xi7 = np.arange(len(b_res)); w7 = 0.3
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(xi7 - w7 / 2, [r['MAP'] * 100        for r in b_res], w7,
               label='MAP',         color='#4C72B0')
        ax.bar(xi7 + w7 / 2, [r['_micro_f1'] * 100  for r in b_res], w7,
               label='BestMicroF1', color='#55A868')
        if baseline_r:
            ax.axhline(baseline_r['MAP'] * 100, color='#4C72B0',
                       linestyle='--', linewidth=1, alpha=0.6, label='Baseline MAP')
            ax.axhline(baseline_r['_micro_f1'] * 100, color='#55A868',
                       linestyle='--', linewidth=1, alpha=0.6, label='Baseline MicroF1')
        ax.set_xticks(xi7)
        ax.set_xticklabels(b_labels, rotation=35, ha='right', fontsize=7)
        ax.set_ylabel('Score (%)')
        ax.set_title('Experiment B: Role Boosting Effect (dashed = no-RR baseline)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        p6b = os.path.join(PLOT_DIR, '6_expB_boosting.png')
        plt.savefig(p6b, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  [Plot 6] Exp B boosting bar chart      → {p6b}")

    # ── Plot 6h: Exp H BM25F — fixed configs + grid best, MicroF1 bar ────────
    h_fixed_res = [r for r in results if re.match(r'^H\d', r.get('model', ''))]
    h_grid_best = max(
        (r for r in results if r.get('model', '').startswith('H-grid:')),
        key=lambda r: r['_micro_f1'], default=None
    )
    h_ngram_res = [r for r in results if r.get('model', '').startswith('H-ngram:')]
    if h_fixed_res or h_grid_best:
        h_plot_items = h_fixed_res[:]
        if h_grid_best: h_plot_items.append(h_grid_best)
        h_labels = [r.get('model', '?')[3:42] for r in h_plot_items]
        xi_h = np.arange(len(h_plot_items)); w_h = 0.3
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.bar(xi_h - w_h / 2, [r['MAP'] * 100        for r in h_plot_items], w_h,
               label='MAP',         color='#E377C2')
        ax.bar(xi_h + w_h / 2, [r['_micro_f1'] * 100  for r in h_plot_items], w_h,
               label='BestMicroF1', color='#AD2685')
        if baseline_r:
            ax.axhline(baseline_r['MAP'] * 100, color='#E377C2',
                       linestyle='--', linewidth=1, alpha=0.6, label='Baseline MAP')
            ax.axhline(baseline_r['_micro_f1'] * 100, color='#AD2685',
                       linestyle='--', linewidth=1, alpha=0.6, label='Baseline MicroF1')
        ax.set_xticks(xi_h)
        ax.set_xticklabels(h_labels, rotation=35, ha='right', fontsize=7)
        ax.set_ylabel('Score (%)')
        ax.set_title('Experiment H: BM25F Field-Weight Configurations')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        p6h = os.path.join(PLOT_DIR, '6h_expH_bm25f.png')
        plt.savefig(p6h, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  [Plot 6h] Exp H BM25F bar chart        → {p6h}")

    # ── Plot 6h2: BM25F n-gram sweep vs TF-IDF Boost (F-B8) line chart ───────
    fb8_log2 = [r for r in results if r.get('model', '').startswith('F-B8:')
                and 'scheme=log' in r.get('model', '')]
    g4_res2  = [r for r in results if r.get('model', '').startswith('G4:')]
    if h_ngram_res:
        fig, ax = plt.subplots(figsize=(10, 5))

        def _srt(lst):
            return sorted(lst, key=lambda r: int(re.search(r'n=(\d+)', r.get('model', '')).group(1))
                          if re.search(r'n=(\d+)', r.get('model', '')) else 0)

        for series, lbl, col, ls in [
            (h_ngram_res, 'BM25F (best weights) n-gram', '#E377C2', '-o'),
            (fb8_log2,    'TF-IDF Boost Arg+Prec+Ratio ×2 log', '#DD8452', '--D'),
            (g4_res2,     'BM25 Boost Arg+Prec+Ratio ×2',       '#64B5CD', '--s'),
        ]:
            s = _srt(series)
            if not s: continue
            ns = [int(re.search(r'n=(\d+)', r.get('model', '')).group(1)) for r in s]
            fs = [r['_micro_f1'] * 100 for r in s]
            ax.plot(ns, fs, ls, label=lbl, color=col, linewidth=1.8, markersize=5)
        ax.set_xlabel('n-gram'); ax.set_ylabel('BestMicroF1 (%)')
        ax.set_title('BM25F vs TF-IDF Boost vs BM25 Boost: n-gram sweep')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p6h2 = os.path.join(PLOT_DIR, '6h2_expH_ngram_vs_boost.png')
        plt.savefig(p6h2, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  [Plot 6h2] BM25F vs Boost n-gram sweep → {p6h2}")

    # ── Plot 6i: Exp I — Refined Fusion fixed configs bar ─────────────────────
    i_fixed_res  = [r for r in results if re.match(r'^I\d', r.get('model', ''))]
    i_grid_best  = max(
        (r for r in results if r.get('model', '').startswith('I-grid:')),
        key=lambda r: r['_micro_f1'], default=None
    )
    if i_fixed_res or i_grid_best:
        i_plot_items = i_fixed_res[:]
        if i_grid_best: i_plot_items.append(i_grid_best)
        i_labels = [r.get('model', '?')[3:42] for r in i_plot_items]
        xi_i = np.arange(len(i_plot_items))
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.bar(xi_i - w_h / 2, [r['MAP'] * 100        for r in i_plot_items], w_h,
               label='MAP',         color='#7F7F7F')
        ax.bar(xi_i + w_h / 2, [r['_micro_f1'] * 100  for r in i_plot_items], w_h,
               label='BestMicroF1', color='#4F4F4F')
        if baseline_r:
            ax.axhline(baseline_r['MAP'] * 100, color='#7F7F7F',
                       linestyle='--', linewidth=1, alpha=0.8, label='Baseline MAP')
            ax.axhline(baseline_r['_micro_f1'] * 100, color='#4F4F4F',
                       linestyle='--', linewidth=1, alpha=0.8, label='Baseline MicroF1')
        # Reference line: Exp D best (to show the improvement)
        d_best = max((r for r in results if re.match(r'^D\d', r.get('model', ''))),
                     key=lambda r: r['_micro_f1'], default=None)
        if d_best:
            ax.axhline(d_best['_micro_f1'] * 100, color='#C44E52',
                       linestyle=':', linewidth=1.2, alpha=0.8, label='Exp D best MicroF1')
        ax.set_xticks(xi_i)
        ax.set_xticklabels(i_labels, rotation=35, ha='right', fontsize=7)
        ax.set_ylabel('Score (%)')
        ax.set_title('Experiment I: Refined Late Fusion (shared vocab)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        p6i = os.path.join(PLOT_DIR, '6i_expI_refined_fusion.png')
        plt.savefig(p6i, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  [Plot 6i] Exp I Refined Fusion bar     → {p6i}")

    # ── Plot 7: Best result per experiment group — all metrics ───────────────
    exp_groups = {}
    for r in results:
        key = r.get('model', '?')[0]
        if key not in exp_groups or r['_micro_f1'] > exp_groups[key]['_micro_f1']:
            exp_groups[key] = r
    grp_keys = sorted(exp_groups.keys())
    if len(grp_keys) >= 2:
        xi8 = np.arange(len(grp_keys)); w8 = 0.2
        fig, ax = plt.subplots(figsize=(10, 5))
        for oi, (mk, ml, col) in enumerate([
            ('MAP',       'MAP',         '#4C72B0'),
            ('MRR',       'MRR',         '#DD8452'),
            ('NDCG@10',   'NDCG@10',     '#8172B2'),
            ('_micro_f1', 'BestMicroF1', '#55A868'),
        ]):
            vals8 = [exp_groups[k][mk] * 100 for k in grp_keys]
            ax.bar(xi8 + (oi - 1.5) * w8, vals8, w8, label=ml, color=col)
        ax.set_xticks(xi8)
        ax.set_xticklabels([f'Exp {k}\n(best)' for k in grp_keys], fontsize=9)
        ax.set_ylabel('Score (%)')
        ax.set_title('Best result per experiment group — all metrics')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        p7 = os.path.join(PLOT_DIR, '7_best_per_experiment.png')
        plt.savefig(p7, dpi=150, bbox_inches='tight'); plt.close()
        print(f"  [Plot 7] Best-per-group comparison     → {p7}")

    print(f"\n  All plots saved to: {PLOT_DIR}/")

except ImportError:
    print("\n  [Plots skipped — matplotlib not installed. Run: pip install matplotlib]")
except Exception as _plot_err:
    import traceback
    print(f"\n  [Plots failed: {_plot_err}]")
    traceback.print_exc()
