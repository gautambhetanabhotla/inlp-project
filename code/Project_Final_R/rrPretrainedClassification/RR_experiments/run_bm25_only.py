"""
run_bm25_only.py
================
Run Experiment G (BM25 on RR-filtered/boosted docs) in isolation.
TF-IDF experiments A-F are skipped — use analyze_rr_full.py for those.

Configs
-------
  G0  BM25 baseline (original plain docs)
  G1  BM25 all RR sentences (labels stripped)
  G2  BM25 Fact+Arg+Prec+Ratio subset (A3 equiv)
  G3  BM25 Boost Arg+Prec+Ratio ×2 (B8 equiv), fixed n=4
  G4  BM25 Boost Arg+Prec+Ratio ×2  — n-gram sweep [2..10]
  G5  BM25 Fact+Arg+Prec+Ratio subset — n-gram sweep [2..10]

Usage
-----
    cd Models/RR
    python run_bm25_only.py
    python run_bm25_only.py --ngram 4 --b 0.7 --k1 1.6
"""

import os, re, json, sys, argparse, datetime, csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from scipy import sparse as _sparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../Transformer-Embeddings'))
from tfidf_utils import clean_text, evaluate_all

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--cand_dir",   default="ik_test_rr/candidate")
parser.add_argument("--query_dir",  default="ik_test_rr/query")
parser.add_argument("--labels",     default="ik_test_rr/test.json")
parser.add_argument("--orig_cand",  default="../BM25/data/corpus/ik_test/candidate")
parser.add_argument("--orig_query", default="../BM25/data/corpus/ik_test/query")
parser.add_argument("--ngram",  type=int,   default=4)
parser.add_argument("--min_df", type=int,   default=2)
parser.add_argument("--max_df", type=float, default=0.95)
parser.add_argument("--b",      type=float, default=0.7,  help="BM25 b param")
parser.add_argument("--k1",     type=float, default=1.6,  help="BM25 k1 param")
args = parser.parse_args()

SKIP_IDS = {1864396, 1508893}
K_VALUES  = [5, 6, 7, 8, 9, 10, 11, 15, 20]

# ── load labels ───────────────────────────────────────────────────────────────
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
        docs[int(re.findall(r'\d+', fn)[0])] = open(os.path.join(folder, fn), errors='ignore').read()
    return docs

def load_plain(folder):
    docs = {}
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith('.txt'): continue
        docs[int(re.findall(r'\d+', fn)[0])] = ' '.join(open(os.path.join(folder, fn), errors='ignore').read().split())
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

# ── text extraction helpers ───────────────────────────────────────────────────
def extract_roles(text, roles):
    lines = []
    for line in text.strip().split('\n'):
        parts = line.split('\t', 1)
        if len(parts) == 2 and parts[0].strip() in roles:
            lines.append(parts[1].strip())
    return ' '.join(lines)

def extract_all_strip_labels(text):
    lines = []
    for line in text.strip().split('\n'):
        parts = line.split('\t', 1)
        lines.append(parts[-1].strip())
    return ' '.join(lines)

def extract_with_boost(text, boosted_roles, boost_factor=2):
    lines = []
    for line in text.strip().split('\n'):
        parts = line.split('\t', 1)
        if len(parts) != 2:
            lines.append(line.strip()); continue
        lbl, sent = parts[0].strip(), parts[1].strip()
        lines.extend([sent] * (boost_factor if lbl in boosted_roles else 1))
    return ' '.join(lines)

# ── eval helpers ──────────────────────────────────────────────────────────────
def micro_f1(ranked_results):
    qdata = []
    for qid, ranked in ranked_results.items():
        if qid in SKIP_IDS: continue
        actual = gold.get(qid)
        if not actual: continue
        qdata.append((actual, [c for c in ranked if c != qid]))
    best_f1, best_k, curve = 0.0, 1, []
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

# ══════════════════════════════════════════════════════════════════════════════
# BM25
# ══════════════════════════════════════════════════════════════════════════════
class BM25:
    """BM25 retrieval using sklearn TfidfVectorizer for IDF + raw TF counts."""
    def __init__(self, b=0.7, k1=1.6, min_df=2, max_df=0.95):
        self.b = b; self.k1 = k1
        self.vectorizer = TfidfVectorizer(
            analyzer='word', tokenizer=str.split, preprocessor=None, token_pattern=None,
            ngram_range=(1, 1),
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
            cols  = q_vec.indices
            X_sub = self._raw_csc[:, cols]
            idf   = self.vectorizer._tfidf.idf_[None, cols] - 1.0
            numer = X_sub.multiply(np.broadcast_to(idf, X_sub.shape)) * (k1 + 1)
            denom = X_sub + self._dl_norm[:, None]
            scores[:, qi] = (numer / denom).sum(1).A1
        return scores


def run_bm25(cand_texts, query_texts, label, ngram=None):
    if ngram is None: ngram = args.ngram
    c_ids = sorted(cand_texts.keys())
    q_ids = sorted(query_texts.keys())

    c_strs = [' '.join(clean_text(cand_texts[i], remove_stopwords=True, ngram=ngram)) or '__empty__' for i in c_ids]
    q_strs = [' '.join(clean_text(query_texts[i], remove_stopwords=True, ngram=ngram)) or '__empty__' for i in q_ids]

    try:
        bm = BM25(b=args.b, k1=args.k1, min_df=args.min_df, max_df=args.max_df)
        bm.fit(c_strs)
        scores_mat = bm.score_all(q_strs)
    except Exception as e:
        print(f"  SKIPPED '{label}': {e}")
        return None

    ranked_results = {q_ids[qi]: [c_ids[i] for i in np.argsort(scores_mat[:, qi])[::-1]]
                      for qi in range(len(q_ids))}

    filtered  = {q: v for q, v in ranked_results.items() if q not in SKIP_IDS}
    relevance = {q: list(v) for q, v in gold.items()}
    m = evaluate_all(filtered, relevance, k_values=K_VALUES, label=label, verbose=False)
    mf1, mk, curve = micro_f1(ranked_results)
    m['_micro_f1'] = mf1; m['_micro_k'] = mk; m['_micro_f1_curve'] = curve
    return m

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT G
# ══════════════════════════════════════════════════════════════════════════════
results = []

def add(m, name=None):
    if m:
        if name: m['model'] = name
        results.append(m)
        print(f"  → MicroF1={m['_micro_f1']*100:.2f}%@K={m['_micro_k']}  "
              f"MAP={m['MAP']*100:.2f}%  MRR={m['MRR']*100:.2f}%  "
              f"NDCG@10={m['NDCG@10']*100:.2f}%")

N = args.ngram
print(f"\n{'═'*70}")
print(f"  EXPERIMENT G: BM25 on RR docs  (b={args.b}, k1={args.k1})")
print(f"  Configs: Baseline | A3 subset | B8 boost | n-gram sweeps [2..10]")
print(f"{'═'*70}")

# ── G0–G3: fixed n-gram ───────────────────────────────────────────────────────
print(f"\n  --- Fixed configs (n={N}) ---")

print(f"\n  G0: BM25 Baseline (original plain docs)")
if orig_available:
    add(run_bm25(cand_orig, query_orig, "G0: BM25 Baseline (orig docs)"))
else:
    print("  [SKIPPED — orig_cand not available]")

print(f"\n  G1: BM25 All RR (strip labels)")
g1_c = {cid: extract_all_strip_labels(t) for cid, t in cand_rr.items()}
g1_q = {qid: extract_all_strip_labels(t) for qid, t in query_rr.items()}
add(run_bm25(g1_c, g1_q, "G1: BM25 All RR (strip labels)"))

print(f"\n  G2: BM25 Fact+Arg+Prec+Ratio (A3 subset)")
g2_c = {k: v for k, v in {cid: extract_roles(t, {'Fact','Argument','Precedent','RatioOfTheDecision'})
                            for cid, t in cand_rr.items()}.items() if v.strip()}
g2_q = {k: v for k, v in {qid: extract_roles(t, {'Fact','Argument','Precedent','RatioOfTheDecision'})
                            for qid, t in query_rr.items()}.items() if v.strip()}
add(run_bm25(g2_c, g2_q, "G2: BM25 Fact+Arg+Prec+Ratio (A3)"))

print(f"\n  G3: BM25 Boost Arg+Prec+Ratio ×2 (B8), n={N}")
g3_c = {cid: extract_with_boost(t, {'Argument','Precedent','RatioOfTheDecision'}, 2) for cid, t in cand_rr.items()}
g3_q = {qid: extract_with_boost(t, {'Argument','Precedent','RatioOfTheDecision'}, 2) for qid, t in query_rr.items()}
g3_c = {k: v for k, v in g3_c.items() if v.strip()}
g3_q = {k: v for k, v in g3_q.items() if v.strip()}
add(run_bm25(g3_c, g3_q, f"G3: BM25 Boost Arg+Prec+Ratio ×2  n={N}"))

# ── G4: n-gram sweep — Boost Arg+Prec+Ratio ×2 ───────────────────────────────
print(f"\n  --- G4: BM25 Boost Arg+Prec+Ratio ×2 — n-gram sweep [2..10] ---")
g4_c = {cid: extract_with_boost(t, {'Argument','Precedent','RatioOfTheDecision'}, 2) for cid, t in cand_rr.items()}
g4_q = {qid: extract_with_boost(t, {'Argument','Precedent','RatioOfTheDecision'}, 2) for qid, t in query_rr.items()}
g4_c = {k: v for k, v in g4_c.items() if v.strip()}
g4_q = {k: v for k, v in g4_q.items() if v.strip()}
for ng in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    name = f"G4: BM25 Boost Arg+Prec+Ratio ×2  n={ng}"
    print(f"\n  {name}")
    add(run_bm25(g4_c, g4_q, name, ngram=ng))

# ── G5: n-gram sweep — Fact+Arg+Prec+Ratio subset ────────────────────────────
print(f"\n  --- G5: BM25 Fact+Arg+Prec+Ratio subset — n-gram sweep [2..10] ---")
g5_c = {k: v for k, v in {cid: extract_roles(t, {'Fact','Argument','Precedent','RatioOfTheDecision'})
                            for cid, t in cand_rr.items()}.items() if v.strip()}
g5_q = {k: v for k, v in {qid: extract_roles(t, {'Fact','Argument','Precedent','RatioOfTheDecision'})
                            for qid, t in query_rr.items()}.items() if v.strip()}
for ng in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    name = f"G5: BM25 Fact+Arg+Prec+Ratio  n={ng}"
    print(f"\n  {name}")
    add(run_bm25(g5_c, g5_q, name, ngram=ng))

# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
results.sort(key=lambda x: x['_micro_f1'], reverse=True)

_ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results_bm25_{_ts}')

with open(_base + '.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n  Results saved → {_base}.json")

_scalar_keys = (['model', '_micro_f1', '_micro_k', 'MAP', 'MRR', 'R-Precision', 'n_queries'] +
                [f'{m}@{k}' for k in K_VALUES for m in ('P', 'R', 'F1', 'NDCG', 'MAP', 'MicroF1')])
with open(_base + '.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=_scalar_keys, extrasaction='ignore')
    w.writeheader()
    for r in results:
        w.writerow({k: round(v * 100, 4) if isinstance(v, float) and k != '_micro_k' else v
                    for k, v in r.items() if k != '_micro_f1_curve'})
print(f"  Results saved → {_base}.csv")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
W = 140
print(f"\n\n{'═'*W}")
print(f"  EXPERIMENT G RESULTS — sorted by BestMicroF1".center(W))
print(f"{'═'*W}")
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

print(f"\n  Best: {results[0].get('model','?')}")
print(f"  BestMicroF1 = {results[0]['_micro_f1']*100:.2f}% @ K={results[0]['_micro_k']}")
print(f"  MAP         = {results[0]['MAP']*100:.2f}%")
print(f"  MRR         = {results[0]['MRR']*100:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ── BM25 n-gram sweep line chart (G4 vs G5) ───────────────────────────────
    def _sorted_by_n(lst):
        return sorted(lst, key=lambda r: int(re.search(r'n=(\d+)', r.get('model','')).group(1))
                      if re.search(r'n=(\d+)', r.get('model','')) else 0)

    g4_res = _sorted_by_n([r for r in results if r.get('model','').startswith('G4:')])
    g5_res = _sorted_by_n([r for r in results if r.get('model','').startswith('G5:')])

    if g4_res or g5_res:
        fig, ax = plt.subplots(figsize=(9, 5))
        for series, lbl, clr, ls in [
            (g4_res, 'BM25 Boost Arg+Prec+Ratio ×2', '#64B5CD', '-o'),
            (g5_res, 'BM25 Fact+Arg+Prec+Ratio',     '#1A7DAF', '-s'),
        ]:
            if not series: continue
            ns = [int(re.search(r'n=(\d+)', r.get('model','')).group(1)) for r in series]
            fs = [r['_micro_f1'] * 100 for r in series]
            ax.plot(ns, fs, ls, label=lbl, linewidth=1.8, markersize=5, color=clr)
        ax.set_xlabel('n-gram'); ax.set_ylabel('BestMicroF1 (%)')
        ax.set_title(f'Exp G: BM25 n-gram sweep (b={args.b}, k1={args.k1})')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(PLOT_DIR, f'G_bm25_ngram_sweep_{_ts}.png')
        plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
        print(f"\n  [Plot] BM25 n-gram sweep → {p}")

    # ── All G configs bar chart ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, max(4, len(results) * 0.4)))
    rev = list(reversed(results))
    ax.barh(range(len(rev)), [r['_micro_f1'] * 100 for r in rev],
            color='#64B5CD', edgecolor='white', linewidth=0.4)
    ax.set_yticks(range(len(rev)))
    ax.set_yticklabels([r.get('model','?')[:55] for r in rev], fontsize=8)
    ax.set_xlabel('Best MicroF1 @ optimal K (%)')
    ax.set_title(f'Experiment G: BM25 on RR docs (b={args.b}, k1={args.k1})')
    ax.axvline(results[0]['_micro_f1'] * 100, color='black',
               linestyle='--', linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    p2 = os.path.join(PLOT_DIR, f'G_bm25_all_configs_{_ts}.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  [Plot] All G configs bar → {p2}")

    print(f"\n  Plots saved to: {PLOT_DIR}/")

except ImportError:
    print("\n  [Plots skipped — matplotlib not installed]")
except Exception as _e:
    import traceback
    print(f"\n  [Plots failed: {_e}]"); traceback.print_exc()
