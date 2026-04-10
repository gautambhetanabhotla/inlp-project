"""
eval_tfidf_only.py
==================
Pure TF-IDF n-gram sweep — no SBERT, runs in ~2 min for all configs.

Uses the same preprocessing pipeline as tfidf.py:
  clean_text() → strips XML/citation tags, removes stopwords, lowercases.
Supports all 4 TF schemes: log, raw, binary, augmented.

Usage
-----
    python eval_tfidf_only.py                        # full sweep n_gram=1..10, all schemes
    python eval_tfidf_only.py --max_ngram 5          # sweep up to n_gram=5
    python eval_tfidf_only.py --schemes log          # only log scheme
    python eval_tfidf_only.py --min_df 2 --max_df 0.95
"""

import os, re, json, argparse, time, math
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tfidf_utils import clean_text, evaluate_all

# K values matching tfidf.py
K_VALUES = [5, 6, 7, 8, 9, 10, 11, 15, 20]

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--cand_dir",  default="../BM25/data/corpus/ik_test/candidate")
parser.add_argument("--query_dir", default="../BM25/data/corpus/ik_test/query")
parser.add_argument("--labels",    default="../BM25/data/corpus/ik_test/test.json")
parser.add_argument("--min_df",    type=int,   default=2)
parser.add_argument("--max_df",    type=float, default=0.95)
parser.add_argument("--max_ngram", type=int,   default=10)
parser.add_argument("--schemes",   nargs="+",  default=["log", "raw", "binary", "augmented"])
args = parser.parse_args()

# ── load corpus ───────────────────────────────────────────────────────────────
def load_folder(folder):
    docs = {}
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".txt"):
            continue
        doc_id = int(re.findall(r"\d+", fn)[0])
        with open(os.path.join(folder, fn), errors="ignore") as f:
            docs[doc_id] = f.read()
    return docs

print("Loading corpus …")
candidate_docs = load_folder(args.cand_dir)
query_docs     = load_folder(args.query_dir)
print(f"  Queries: {len(query_docs)}  |  Candidates: {len(candidate_docs)}")

candidate_ids = sorted(candidate_docs.keys())
query_ids     = sorted(query_docs.keys())

# ── load ground truth ─────────────────────────────────────────────────────────
with open(args.labels) as f:
    true_labels = json.load(f)

gold_indexed = {
    int(re.findall(r"\d+", item["id"])[0]): {
        int(re.findall(r"\d+", c)[0])
        for c in item.get("relevant candidates", [])
    }
    for item in true_labels["Query Set"]
}

SKIP_IDS = {1864396, 1508893}

# ── augmented TF transformer for sklearn ──────────────────────────────────────
# sklearn supports sublinear_tf (log) and binary natively.
# For "augmented" (0.5 + 0.5 * tf/max_tf) we use a custom transformer.
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import issparse

class AugmentedTFTransformer(BaseEstimator, TransformerMixin):
    """Apply 0.5 + 0.5*(tf/max_tf_per_doc) to a raw count matrix."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        import scipy.sparse as sp
        X = X.astype(float)
        if issparse(X):
            X = X.tocsr()
            for i in range(X.shape[0]):
                row = X.getrow(i)
                mx  = row.data.max() if row.nnz > 0 else 1.0
                if mx > 0:
                    X.data[X.indptr[i]:X.indptr[i+1]] = (
                        0.5 + 0.5 * X.data[X.indptr[i]:X.indptr[i+1]] / mx
                    )
        return X

# ── evaluation helper ─────────────────────────────────────────────────────────
def evaluate(ranked_results, label=""):
    # Filter SKIP_IDS before passing to evaluate_all
    filtered = {q: v for q, v in ranked_results.items() if q not in SKIP_IDS}
    # evaluate_all expects relevance as {qid -> list}, gold_indexed is {int -> set}
    relevance = {q: list(v) for q, v in gold_indexed.items()}
    m = evaluate_all(filtered, relevance, k_values=K_VALUES, label=label, verbose=False)
    # also compute micro-F1@K=1..20 to stay comparable with hybrid runs
    query_data = []
    for q_id, ranked in filtered.items():
        actual = gold_indexed.get(q_id)
        if not actual:
            continue
        query_data.append((actual, [c for c in ranked if c != q_id]))
    best_micro_f1, best_k = 0.0, 1
    micro_f1_vs_k = []
    for k in range(1, 21):
        tp = fp = fn = 0
        for actual, ranked in query_data:
            top_k = set(ranked[:k])
            tp += len(top_k & actual)
            fp += len(top_k - actual)
            fn += len(actual - top_k)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        micro_f1_vs_k.append(f)
        if f > best_micro_f1:
            best_micro_f1, best_k = f, k
    m["_micro_f1"] = best_micro_f1
    m["_micro_k"]  = best_k
    return m

# ── token cache per n_gram ────────────────────────────────────────────────────
token_cache_c = {}   # n_gram -> list of token lists for candidates
token_cache_q = {}   # n_gram -> list of token lists for queries

def get_tokens(n):
    if n not in token_cache_c:
        print(f"  Tokenising (n_gram={n}) …", flush=True)
        token_cache_c[n] = [clean_text(candidate_docs[i], remove_stopwords=True, ngram=n) for i in candidate_ids]
        token_cache_q[n] = [clean_text(query_docs[i],     remove_stopwords=True, ngram=n) for i in query_ids]
    return token_cache_c[n], token_cache_q[n]

# ── run sweep ─────────────────────────────────────────────────────────────────
results = []  # list of dicts

NGRAMS  = list(range(1, args.max_ngram + 1))
SCHEMES = args.schemes

total = len(NGRAMS) * len(SCHEMES)
done  = 0
t_start = time.time()

print(f"\nRunning {total} configs ({len(NGRAMS)} n-grams × {len(SCHEMES)} schemes) …\n")

for n in NGRAMS:
    cand_tokens, query_tokens = get_tokens(n)
    # join tokens back to strings for sklearn (it will re-tokenise via analyzer=str.split)
    cand_strings  = [" ".join(t) for t in cand_tokens]
    query_strings = [" ".join(t) for t in query_tokens]

    for scheme in SCHEMES:
        done += 1
        label = f"tfidf  n={n:<2}  scheme={scheme:<9}  min_df={args.min_df}  max_df={args.max_df}"

        if scheme == "augmented":
            # fit raw counts then apply augmented transform manually
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import normalize
            import scipy.sparse as sp

            cv = CountVectorizer(
                analyzer="word", tokenizer=str.split, preprocessor=None, token_pattern=None,
                min_df=args.min_df, max_df=args.max_df,
            )
            raw_c = cv.fit_transform(cand_strings).astype(float)
            raw_q = cv.transform(query_strings).astype(float)

            # augmented TF per doc
            def augment(mat):
                mat = mat.tocsr()
                for i in range(mat.shape[0]):
                    s, e = mat.indptr[i], mat.indptr[i+1]
                    if e > s:
                        mx = mat.data[s:e].max()
                        if mx > 0:
                            mat.data[s:e] = 0.5 + 0.5 * mat.data[s:e] / mx
                return mat

            aug_c = augment(raw_c.copy())
            aug_q = augment(raw_q.copy())

            # IDF: same formula as sklearn smooth
            N = aug_c.shape[0]
            df = np.diff(aug_c.tocsc().indptr)
            idf_vec = np.log((1 + N) / (1 + df)) + 1.0

            tfidf_c = aug_c.multiply(idf_vec)
            tfidf_q = aug_q.multiply(idf_vec)

            norm_c = normalize(tfidf_c, norm="l2")
            norm_q = normalize(tfidf_q, norm="l2")

            scores_mat = (norm_c @ norm_q.T).toarray()  # (n_cands, n_queries)

        else:
            vec = TfidfVectorizer(
                analyzer="word", tokenizer=str.split, preprocessor=None, token_pattern=None,
                min_df=args.min_df, max_df=args.max_df,
                sublinear_tf=(scheme == "log"),
                binary=(scheme == "binary"),
                use_idf=True, norm="l2",
            )
            norm_c = vec.fit_transform(cand_strings)   # (n_cands, vocab)
            norm_q = vec.transform(query_strings)       # (n_queries, vocab)
            scores_mat = (norm_c @ norm_q.T).toarray()  # (n_cands, n_queries)

        # build ranked results
        ranked_results = {}
        for qi, q_id in enumerate(query_ids):
            scores = scores_mat[:, qi]
            top_idx = np.argsort(scores)[::-1]
            ranked_results[q_id] = [candidate_ids[i] for i in top_idx]

        m = evaluate(ranked_results, label=label)
        results.append({
            "label": label, "n": n, "scheme": scheme,
            "MAP": m["MAP"], "MRR": m["MRR"], "R-P": m["R-Precision"],
            **{f"F1@{k}": m[f"F1@{k}"]   for k in K_VALUES},
            **{f"P@{k}":  m[f"P@{k}"]    for k in K_VALUES},
            **{f"R@{k}":  m[f"R@{k}"]    for k in K_VALUES},
            **{f"NDCG@{k}": m[f"NDCG@{k}"] for k in K_VALUES},
            **{f"MAP@{k}":  m[f"MAP@{k}"]  for k in K_VALUES},
            **{f"MicroF1@{k}": m[f"MicroF1@{k}"] for k in K_VALUES},
            "micro_f1": m["_micro_f1"], "micro_k": m["_micro_k"],
        })
        print(f"  [{done:>3}/{total}]  {label}")
        print(f"          MAP={m['MAP']*100:.2f}%  MRR={m['MRR']*100:.2f}%  "
              f"MicroF1@10={m['MicroF1@10']*100:.2f}%  "
              f"BestMicroF1={m['_micro_f1']*100:.2f}%@K={m['_micro_k']}")

# ── save results (JSON + CSV) ─────────────────────────────────────────────────
import json as _json, csv as _csv, datetime as _dt
results.sort(key=lambda x: x["micro_f1"], reverse=True)

_ts   = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'tfidf_results_{_ts}')

_json_path = _base + '.json'
with open(_json_path, 'w') as _f:
    _json.dump(results, _f, indent=2, default=str)
print(f"\n  Results saved → {_json_path}")

_csv_path = _base + '.csv'
_csv_keys = (['label', 'n', 'scheme', 'micro_f1', 'micro_k', 'MAP', 'MRR', 'R-P'] +
             [f'{m}@{k}' for k in K_VALUES
              for m in ('P', 'R', 'F1', 'NDCG', 'MAP', 'MicroF1')])
with open(_csv_path, 'w', newline='') as _f:
    _w = _csv.DictWriter(_f, fieldnames=_csv_keys, extrasaction='ignore')
    _w.writeheader()
    for r in results:
        _w.writerow({k: round(v * 100, 4) if isinstance(v, float) and k not in ('micro_k', 'n') else v
                     for k, v in r.items()})
print(f"  Results saved → {_csv_path}")

# ── summary table sorted by BestMicroF1 ──────────────────────────────────────
W = 140
print(f"\n{'═'*W}")
print(f"  RESULTS SUMMARY — sorted by BestMicroF1   (total time: {time.time()-t_start:.0f}s)".center(W))
print(f"{'═'*W}")
print(f"  {'Config':<50} {'BstMicF1':>9} {'@K':>3} {'MAP':>7} {'MRR':>7} "
      f"{'R-P':>7} {'P@10':>7} {'R@10':>7} {'F1@10':>7} {'NDCG@10':>8} {'MAP@10':>8} {'MicF1@10':>9}")
print(f"  {'─'*130}")
for r in results:
    marker = " ★" if r == results[0] else ""
    print(f"  {r['label']:<50} "
          f"{r['micro_f1']*100:>8.2f}%  {r['micro_k']:>2} "
          f"{r['MAP']*100:>6.2f}% "
          f"{r['MRR']*100:>6.2f}% "
          f"{r['R-P']*100:>6.2f}% "
          f"{r['P@10']*100:>6.2f}% "
          f"{r['R@10']*100:>6.2f}% "
          f"{r['F1@10']*100:>6.2f}% "
          f"{r['NDCG@10']*100:>7.2f}% "
          f"{r['MAP@10']*100:>7.2f}% "
          f"{r['MicroF1@10']*100:>8.2f}%{marker}")
print(f"{'═'*W}")

# ── per-scheme best n_gram (by MicroF1) ──────────────────────────────────────
print("\n  Best n_gram per scheme (by MicroF1):")
for scheme in SCHEMES:
    best = max((x for x in results if x["scheme"] == scheme), key=lambda x: x["micro_f1"])
    print(f"    {scheme:<10}  n={best['n']}  MicroF1={best['micro_f1']*100:.2f}%@K={best['micro_k']}  "
          f"MAP={best['MAP']*100:.2f}%  P@10={best['P@10']*100:.2f}%  R@10={best['R@10']*100:.2f}%")

# ── MicroF1 vs n_gram table for log scheme ────────────────────────────────────
log_res = [x for x in sorted(results, key=lambda x: x["n"]) if x["scheme"] == "log"]
if log_res:
    print(f"\n  MicroF1 vs n_gram (scheme=log):")
    print(f"  {'n':<4} {'BstMicF1':>9} {'@K':>3} {'MAP':>7} {'MRR':>7} "
          f"{'P@10':>7} {'R@10':>7} {'F1@10':>7} {'NDCG@10':>8} {'MicF1@10':>9}")
    print(f"  {'─'*82}")
    for r in log_res:
        print(f"  {r['n']:<4} {r['micro_f1']*100:>8.2f}%  {r['micro_k']:>2} "
              f"{r['MAP']*100:>6.2f}% "
              f"{r['MRR']*100:>6.2f}% "
              f"{r['P@10']*100:>6.2f}% "
              f"{r['R@10']*100:>6.2f}% "
              f"{r['F1@10']*100:>6.2f}% "
              f"{r['NDCG@10']*100:>7.2f}% "
              f"{r['MicroF1@10']*100:>8.2f}%")
