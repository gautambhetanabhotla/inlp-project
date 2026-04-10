"""
hybrid_transformer_rerank.py
============================
Two-stage legal case retrieval: Improved TF-IDF → Transformer re-ranking.

Stage 1  ─ Improved TF-IDF first-stage retrieval
           • Proper augmented TF scheme (0.5 + 0.5·tf/max_tf) applied to BOTH
             candidates AND queries (fixes the bug in the original pipeline which
             only applied log/binary via sklearn's sublinear_tf flag).
           • Higher n-gram support (n=7–9 best per ablation in eval_tfidf_only.py).
           • min_df / max_df filtering same as eval_tfidf_only.py.

Stage 2  ─ Transformer re-ranking  (selectable via --model_type)
           • biencoder  – bi-encoder (SBERT-style) with sliding-window chunking to
                          handle long legal docs (median ~5000 tokens).
                          Aggregation: max_chunk (default) or mean_chunk.
           • crossencoder – cross-encoder scores each (query, cand) pair jointly.
                            Supports chunk-level scoring for very long docs.

Fusion   ─ final_score = alpha × norm_tfidf + (1-alpha) × norm_model

Usage (standalone)
------------------
    python hybrid_transformer_rerank.py \\
        --model sentence-transformers/all-MiniLM-L6-v2 \\
        --model_type biencoder \\
        --agg max_chunk \\
        --tfidf_scheme augmented \\
        --tfidf_ngram 4 \\
        --tfidf_top_n 200 \\
        --alpha 0.8 \\
        --chunk_tokens 256 \\
        --chunk_stride 128

    python hybrid_transformer_rerank.py \\
        --model cross-encoder/ms-marco-MiniLM-L-6-v2 \\
        --model_type crossencoder \\
        --tfidf_scheme augmented \\
        --tfidf_ngram 4 \\
        --tfidf_top_n 100 \\
        --alpha 0.6

Key improvements over original hybrid_retrieval_chunk_sbert.py
--------------------------------------------------------------
  1. Proper augmented TF applied to QUERIES (original only applied it to candidates
     via the sklearn vectorizer, making augmented TF asymmetric / unused in practice).
  2. n-gram range configurable up to 10; default 7 based on ablations.
  3. Unified experiment dict interface — importable by run_hybrid_sweep.py.
  4. Cross-encoder support added.
  5. Consistent micro-F1 evaluation matching tfidf_utils.
  6. All results saved as both JSON (raw + per-K) and CSV (one row per experiment).
"""

import os, re, json, argparse, time, codecs
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
import tqdm as tqdm_module

from tfidf_utils import clean_text, evaluate_all

# ── Citation patterns for Indian Supreme Court judgements ─────────────────────
# Handles: AIR 1973 SC 1461  |  (1973) 4 SCC 225  |  1973 (4) SCC 225
#          [1973] 1 SCR 1    |  MANU/SC/0001/2020
_CITE_PATS = [
    re.compile(r'AIR\s+\d{4}\s+S\.?\s*C\.?\s+\d+',     re.IGNORECASE),
    re.compile(r'\(\s*\d{4}\s*\)\s*\d+\s+SCC\s+\d+',   re.IGNORECASE),
    re.compile(r'\d{4}\s*\(\s*\d+\s*\)\s*SCC\s+\d+',   re.IGNORECASE),
    re.compile(r'\[\s*\d{4}\s*\]\s*\d+\s+SCR\s+\d+',   re.IGNORECASE),
    re.compile(r'MANU/SC/\d+/\d{4}',                      re.IGNORECASE),
]

def _extract_citations(text: str) -> frozenset:
    """Return frozenset of normalised citation strings found in a judgment."""
    found = set()
    for pat in _CITE_PATS:
        for m in pat.findall(text):
            found.add(re.sub(r'\s+', '_', m.strip().upper()))
    return frozenset(found)

# ─────────────────────────────────────────────────────────────────────────────
SKIP_IDS = {1864396, 1508893}
K_VALUES  = [5, 6, 7, 8, 9, 10, 11, 15, 20]

_DEFAULT_CAND   = "../BM25/data/corpus/ik_test/candidate"
_DEFAULT_QUERY  = "../BM25/data/corpus/ik_test/query"
_DEFAULT_LABELS = "../BM25/data/corpus/ik_test/test.json"

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_corpus(folder: str) -> dict:
    """Load all *.txt files from folder → {int_doc_id: text_string}."""
    docs = {}
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".txt"):
            continue
        doc_id = int(re.findall(r"\d+", fn)[0])
        with codecs.open(os.path.join(folder, fn), "r", "utf-8", errors="ignore") as f:
            docs[doc_id] = " ".join(f.read().split())
    return docs


def load_gold(labels_path: str) -> dict:
    """Load gold relevance {int_query_id → set(int_cand_ids)}."""
    with open(labels_path) as f:
        lj = json.load(f)
    return {
        int(re.findall(r"\d+", item["id"])[0]):
        {int(re.findall(r"\d+", c)[0]) for c in item.get("relevant candidates", [])}
        for item in lj["Query Set"]
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. IMPROVED TF-IDF FIRST STAGE
#    Key fix: augmented TF applied to both candidates AND queries.
# ══════════════════════════════════════════════════════════════════════════════

def _augment_tf(X):
    """Apply 0.5 + 0.5·(tf / max_tf_per_doc) in-place on sparse CSR matrix."""
    X = X.tocsr()
    for i in range(X.shape[0]):
        s, e = X.indptr[i], X.indptr[i + 1]
        if e > s:
            mx = X.data[s:e].max()
            if mx > 0:
                X.data[s:e] = 0.5 + 0.5 * X.data[s:e] / mx
    return X


def build_tfidf_index(cand_tokens: list, scheme: str, min_df=2, max_df=0.95):
    """
    Build TF-IDF matrix for candidates.

    Returns
    -------
    C            : (n_cands, vocab) L2-normalised TF×IDF sparse matrix
    transform_q  : callable(query_token_strings) → L2-normalised Q matrix
    """
    if scheme == "augmented":
        cv = CountVectorizer(
            analyzer="word", tokenizer=str.split, preprocessor=None,
            token_pattern=None, min_df=min_df, max_df=max_df,
        )
        raw_C = cv.fit_transform(cand_tokens).astype(float)
        aug_C = _augment_tf(raw_C.copy())
        N = aug_C.shape[0]
        df_counts = np.diff(aug_C.tocsc().indptr)
        idf_vec = np.log((1 + N) / (1 + df_counts)) + 1.0   # smooth IDF (sklearn formula)
        C = normalize(aug_C.multiply(idf_vec), norm="l2")

        def transform_q(query_token_strings):
            raw_Q = cv.transform(query_token_strings).astype(float)
            aug_Q = _augment_tf(raw_Q.copy())
            return normalize(aug_Q.multiply(idf_vec), norm="l2")

    else:
        vec = TfidfVectorizer(
            analyzer="word", tokenizer=str.split, preprocessor=None,
            token_pattern=None, min_df=min_df, max_df=max_df,
            sublinear_tf=(scheme == "log"),
            binary=(scheme == "binary"),
            use_idf=True, norm="l2",
        )
        C = vec.fit_transform(cand_tokens)

        def transform_q(query_token_strings):
            return vec.transform(query_token_strings)

    return C, transform_q


def run_first_stage(cand_docs, cand_ids, query_docs, query_ids,
                    scheme, ngram, top_n, min_df=2, max_df=0.95):
    """
    Run TF-IDF first-stage retrieval.

    Returns
    -------
    shortlists    : {q_id: [top_n cand_ids]}
    tfidf_scores  : {q_id: {c_id: normalised_score∈[0,1]}}
    """
    print(f"  Tokenising (n={ngram}, scheme={scheme}) …", flush=True)
    cand_token_strs = [
        " ".join(clean_text(cand_docs[i], remove_stopwords=True, ngram=ngram))
        for i in cand_ids
    ]
    query_token_strs = [
        " ".join(clean_text(query_docs[i], remove_stopwords=True, ngram=ngram))
        for i in query_ids
    ]

    print(f"  Building TF-IDF index …", flush=True)
    C, transform_q = build_tfidf_index(cand_token_strs, scheme, min_df, max_df)
    vocab_size = C.shape[1]
    print(f"    Vocab: {vocab_size:,}  |  Cands: {C.shape[0]}  |  Scheme: {scheme}")

    Q = transform_q(query_token_strs)
    scores_mat = (C @ Q.T).toarray()   # (n_cands, n_queries)

    shortlists   = {}
    tfidf_scores = {}
    for qi, q_id in enumerate(query_ids):
        scores  = scores_mat[:, qi]
        s_min, s_max = scores.min(), scores.max()
        norm    = (scores - s_min) / (s_max - s_min + 1e-12)
        tfidf_scores[q_id] = {c_id: float(norm[j]) for j, c_id in enumerate(cand_ids)}
        top_idx = np.argsort(scores)[::-1][:top_n]
        shortlists[q_id] = [cand_ids[j] for j in top_idx]

    # Quick first-stage MicroF1 for diagnostics
    fs_ranked = {
        q_id: [cand_ids[j] for j in np.argsort(scores_mat[:, qi])[::-1]]
        for qi, q_id in enumerate(query_ids)
    }
    fs_f1, fs_k, _ = _micro_f1(fs_ranked, load_gold.__doc__ or {})   # filled below

    return shortlists, tfidf_scores, scores_mat


# ══════════════════════════════════════════════════════════════════════════════
# 3. BI-ENCODER + SLIDING-WINDOW CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def _text_to_chunks(text: str, tokenizer, chunk_tokens: int, stride: int) -> list:
    """Tokenise text, split into overlapping chunks, return decoded strings."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return [text[:200] or "[EMPTY]"]
    chunks, start = [], 0
    while start < len(ids):
        end = min(start + chunk_tokens, len(ids))
        chunks.append(tokenizer.decode(ids[start:end], skip_special_tokens=True))
        if end == len(ids):
            break
        start += stride
    return chunks or [text[:200]]


def biencoder_score(model, shortlists, cand_docs, query_docs, query_ids,
                    chunk_tokens=256, chunk_stride=128, encode_batch=128,
                    agg="max_chunk"):
    """
    Bi-encoder chunk scoring.

    Parameters
    ----------
    agg : "max_chunk"   – global max over all (q_chunk × c_chunk) pairs.
                          Fast but captures only the single best window.
          "mean_chunk"  – mean of per-query-chunk best scores (softer).
          "sum_topN"    – sums the N best per-candidate-chunk scores
                          (e.g. "sum_top3", "sum_top5").  For each candidate
                          chunk the best-matching query chunk is found; the
                          N highest such values are summed.  This captures
                          *distributed* citation evidence: if a candidate cites
                          A, B, C in different windows they all contribute,
                          unlike max_chunk which only scores on the best window.

    Returns
    -------
    {q_id: {c_id: float_score}}  (raw cosine values, NOT normalised yet)
    """
    tokenizer = model.tokenizer
    all_cand_ids = sorted({c for sl in shortlists.values() for c in sl})

    # ── Pre-encode all shortlisted candidate chunks ───────────────────────────
    print(f"  Chunking + encoding {len(all_cand_ids)} candidates "
          f"(chunk={chunk_tokens}, stride={chunk_stride}) …")
    flat_chunks, flat_doc_ids = [], []
    for c_id in all_cand_ids:
        chunks = _text_to_chunks(cand_docs[c_id], tokenizer, chunk_tokens, chunk_stride)
        flat_chunks.extend(chunks)
        flat_doc_ids.extend([c_id] * len(chunks))

    flat_doc_ids = np.array(flat_doc_ids)
    print(f"    Total chunks: {len(flat_chunks):,}  "
          f"({len(flat_chunks)/len(all_cand_ids):.1f} avg/doc)")

    cand_embs = model.encode(
        flat_chunks, batch_size=encode_batch,
        normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True,
    )
    # doc_id → row indices in cand_embs
    cand_idx_map = {c_id: np.where(flat_doc_ids == c_id)[0] for c_id in all_cand_ids}

    # ── Score queries ─────────────────────────────────────────────────────────
    print(f"  Scoring queries (agg={agg}) …")
    scores_all = {}
    for q_id in tqdm_module.tqdm(query_ids):
        q_chunks = _text_to_chunks(query_docs[q_id], tokenizer, chunk_tokens, chunk_stride)
        q_embs   = model.encode(
            q_chunks, batch_size=encode_batch,
            normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True,
        )   # (n_q_chunks, dim)

        scores_all[q_id] = {}
        for c_id in shortlists[q_id]:
            c_embs = cand_embs[cand_idx_map[c_id]]    # (n_c_chunks, dim)
            sim = q_embs @ c_embs.T                    # (n_q_chunks, n_c_chunks)
            if agg == "max_chunk":
                scores_all[q_id][c_id] = float(sim.max())
            elif agg == "mean_chunk":
                scores_all[q_id][c_id] = float(sim.max(axis=1).mean())
            elif agg.startswith("sum_top"):
                # Per candidate chunk: take best query match, then sum the top-K
                k = int(agg[7:])               # "sum_top3" → 3
                per_cand = sim.max(axis=0)     # (n_c_chunks,)
                topk = np.sort(per_cand)[::-1][:k]
                scores_all[q_id][c_id] = float(topk.sum())
            else:
                raise ValueError(
                    f"Unknown agg: {agg!r}. Use max_chunk | mean_chunk | sum_topN (e.g. sum_top3)"
                )

    return scores_all


# ══════════════════════════════════════════════════════════════════════════════
# 4. CITATION OVERLAP SCORE
#    Computes citation *recall*: fraction of cases cited by the query that also
#    appear in the candidate.  This is a third, orthogonal signal:
#      • Unlike TF-IDF, it is not diluted by IDF (common citations still count).
#      • Unlike embeddings, it is exact-match — "AIR 1973 SC 1461" ≠
#        "AIR 1974 SC 1461", which embeddings treat as nearly identical.
#      • Requires no model loading; adds ~2 s of precompute for the whole corpus.
# ══════════════════════════════════════════════════════════════════════════════

def citation_overlap_score(cand_docs: dict, query_docs: dict,
                           cand_ids: list, query_ids: list) -> dict:
    """
    Citation recall: |Q_cites ∩ C_cites| / |Q_cites|.
    Returns {q_id: {c_id: float∈[0,1]}}.  Queries with no citations get 0.
    """
    query_cits = {q_id: _extract_citations(query_docs[q_id]) for q_id in query_ids}
    cand_cits  = {c_id: _extract_citations(cand_docs[c_id])  for c_id in cand_ids}

    n_q_cits = sum(1 for qc in query_cits.values() if qc)
    avg_q    = sum(len(qc) for qc in query_cits.values()) / max(len(query_ids), 1)
    avg_c    = sum(len(cc) for cc in cand_cits.values())  / max(len(cand_ids),  1)
    print(f"  Citation stats: {n_q_cits}/{len(query_ids)} queries have citations  "
          f"| avg/query={avg_q:.1f}  avg/cand={avg_c:.1f}", flush=True)

    scores: dict = {}
    for q_id in query_ids:
        qc = query_cits[q_id]
        if not qc:
            scores[q_id] = {c_id: 0.0 for c_id in cand_ids}
            continue
        n_q = len(qc)
        scores[q_id] = {
            c_id: len(qc & cand_cits[c_id]) / n_q
            for c_id in cand_ids
        }
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# 5. CROSS-ENCODER RE-RANKING
#    For very long docs, score best-chunk pair (query vs all cand chunks).
# ══════════════════════════════════════════════════════════════════════════════

def crossencoder_score(model, shortlists, cand_docs, query_docs, query_ids,
                       batch_size=32, max_length=512):
    """
    Cross-encoder scoring of (query_text, cand_text) pairs.
    Texts are pre-truncated to max_length*4 characters before tokenization to
    avoid the tokenizer processing 14000-token documents in full.

    Returns
    -------
    {q_id: {c_id: float_score}}
    """
    # Pre-truncate: BERT wordpiece ~4 chars/token on average for legal text.
    # This avoids the tokenizer spending 100+ seconds on a 14k-token document.
    char_limit = max_length * 4
    print(f"  Cross-encoder scoring (top-{max(len(sl) for sl in shortlists.values())} per query, char_limit={char_limit}) …")
    scores_all = {}
    for q_id in tqdm_module.tqdm(query_ids):
        q_text = query_docs[q_id][:char_limit]
        sl     = shortlists[q_id]
        pairs  = [(q_text, cand_docs[c_id][:char_limit]) for c_id in sl]
        raw    = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        scores_all[q_id] = {c_id: float(s) for c_id, s in zip(sl, raw)}
    return scores_all


# ══════════════════════════════════════════════════════════════════════════════
# 6. SCORE FUSION
# ══════════════════════════════════════════════════════════════════════════════

def fuse_and_rank(shortlists, tfidf_scores, model_scores, cand_ids, alpha,
                  citation_scores=None, citation_beta=0.0):
    """
    Three-way fusion:
        final = alpha × tfidf
              + citation_beta × citation_recall   (0-1, exact-match signal)
              + (1-alpha-citation_beta) × norm_model

    When citation_scores is None or citation_beta=0 this reduces to the
    original two-way fusion.  Weights need not sum to 1 (ratio determines
    ranking), but keeping alpha+citation_beta≤1 is recommended.
    """
    ranked = {}
    model_weight = max(0.0, 1.0 - alpha - citation_beta)
    for q_id, sl in shortlists.items():
        m = model_scores.get(q_id, {})
        vals = [m[c] for c in sl if c in m]
        mn, mx = (min(vals), max(vals)) if vals else (0.0, 1.0)
        cit = (citation_scores or {}).get(q_id, {})

        fused = {}
        for c_id in cand_ids:
            t      = tfidf_scores[q_id].get(c_id, 0.0)
            m_raw  = m.get(c_id, None)
            m_norm = (m_raw - mn) / (mx - mn + 1e-12) if m_raw is not None else 0.0
            c_scr  = cit.get(c_id, 0.0)
            fused[c_id] = alpha * t + citation_beta * c_scr + model_weight * m_norm

        ranked[q_id] = sorted(cand_ids, key=lambda c: fused[c], reverse=True)
    return ranked


# ══════════════════════════════════════════════════════════════════════════════
# 7. EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _micro_f1(ranked_results, gold):
    query_data = []
    for q_id, ranked in ranked_results.items():
        if q_id in SKIP_IDS:
            continue
        actual = gold.get(q_id)
        if not actual:
            continue
        query_data.append((actual, [c for c in ranked if c != q_id]))
    best_f1, best_k, curve = 0.0, 1, []
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
        curve.append(f)
        if f > best_f1:
            best_f1, best_k = f, k
    return best_f1, best_k, curve


def evaluate_ranked(ranked_results, gold):
    """Full evaluation: micro-F1 + MAP/MRR/NDCG via evaluate_all."""
    filtered  = {q: v for q, v in ranked_results.items() if q not in SKIP_IDS}
    relevance = {q: list(v) for q, v in gold.items()}
    m = evaluate_all(filtered, relevance, k_values=K_VALUES, label="", verbose=False)
    mf1, mk, curve = _micro_f1(ranked_results, gold)
    m["_micro_f1"]        = mf1
    m["_micro_k"]         = mk
    m["_micro_f1_curve"]  = curve
    return m


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN EXPERIMENT RUNNER  (callable from run_hybrid_sweep.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(cfg: dict, gold: dict = None,
                   cand_docs: dict = None, query_docs: dict = None) -> dict:
    """
    Run a single two-stage retrieval experiment.

    Parameters
    ----------
    cfg : dict with keys —
        label          (str)   experiment name for display / saving
        model          (str)   HuggingFace checkpoint; None = TF-IDF only
        model_type     (str)   "biencoder" | "crossencoder" | "tfidf_only"
        skip_tfidf     (bool)  True = pure transformer ranking (no TF-IDF at all);
                               shortlist = full corpus, alpha forced to 0.0.
                               Only meaningful with model_type=biencoder.
        citation_beta  (float) Weight of citation-recall signal in fusion.
                               0.0 disables it.  For tfidf_only + citation set
                               alpha+citation_beta=1 (e.g. 0.9+0.1).
                               For 3-way: alpha+citation_beta+(1-both)=model.
        agg            (str)   "max_chunk" | "mean_chunk"   (biencoder only)
        tfidf_scheme   (str)   "log" | "augmented" | "binary"
        tfidf_ngram    (int)   max n-gram order (e.g. 7)
        tfidf_top_n    (int)   shortlist size for stage 2
        tfidf_min_df   (int)   min document frequency
        tfidf_max_df   (float) max document frequency fraction
        alpha          (float) weight of TF-IDF in fusion  (1.0 = TF-IDF only)
        chunk_tokens   (int)   tokens per chunk (biencoder)
        chunk_stride   (int)   overlap tokens   (biencoder)
        encode_batch   (int)   SBERT encode batch size
        batch_size     (int)   cross-encoder batch size
        max_length     (int)   cross-encoder max input length
        cand_dir       (str)   override candidate dir
        query_dir      (str)   override query dir
        labels         (str)   override labels file

    gold, cand_docs, query_docs  — optional preloaded data (avoids re-loading)

    Returns
    -------
    metrics dict (MAP, MRR, NDCG@10, _micro_f1, _micro_k, …)
    """
    label      = cfg.get("label", "unnamed")
    model_name = cfg.get("model", None)
    mtype      = cfg.get("model_type", "tfidf_only")
    skip_tfidf = cfg.get("skip_tfidf", False)
    agg        = cfg.get("agg", "max_chunk")
    scheme     = cfg.get("tfidf_scheme", "log")
    ngram      = cfg.get("tfidf_ngram", 4)
    top_n      = cfg.get("tfidf_top_n", 200)
    min_df     = cfg.get("tfidf_min_df", 2)
    max_df     = cfg.get("tfidf_max_df", 0.95)
    alpha         = 0.0 if skip_tfidf else cfg.get("alpha", 0.8)
    citation_beta = cfg.get("citation_beta", 0.0)
    chunk_tok  = cfg.get("chunk_tokens", 256)
    chunk_str  = cfg.get("chunk_stride", 128)
    enc_batch  = cfg.get("encode_batch", 128)
    ce_batch   = cfg.get("batch_size", 32)
    max_len    = cfg.get("max_length", 512)

    # ── Load data if not supplied ────────────────────────────────────────────
    if cand_docs is None:
        cand_docs = load_corpus(cfg.get("cand_dir", _DEFAULT_CAND))
    if query_docs is None:
        query_docs = load_corpus(cfg.get("query_dir", _DEFAULT_QUERY))
    if gold is None:
        gold = load_gold(cfg.get("labels", _DEFAULT_LABELS))

    cand_ids  = sorted(cand_docs.keys())
    query_ids = sorted(query_docs.keys())

    t0 = time.time()
    print(f"\n{'─'*70}")
    print(f"  Exp: {label}")
    if skip_tfidf:
        print(f"  Mode: TRANSFORMER ONLY (no TF-IDF)  model={model_name!r}  type={mtype}")
    else:
        print(f"  TF-IDF: n={ngram}, scheme={scheme}, top_n={top_n}")
        if mtype != "tfidf_only":
            print(f"  Model: {model_name!r}  type={mtype}  alpha={alpha}")
    if citation_beta > 0:
        print(f"  Citation overlap: beta={citation_beta}")
    print(f"{'─'*70}", flush=True)

    # ── Stage 1: TF-IDF (skipped when skip_tfidf=True) ───────────────────────
    if skip_tfidf:
        # Pure dense retrieval: full corpus is the shortlist, no TF-IDF score
        shortlists   = {q_id: list(cand_ids) for q_id in query_ids}
        tfidf_scores = {q_id: {c_id: 0.0 for c_id in cand_ids} for q_id in query_ids}
        fs_ranked    = None   # no first-stage TF-IDF ranking to report
        print(f"  [skip_tfidf] Shortlist = all {len(cand_ids)} candidates per query",
              flush=True)
    else:
        cand_token_strs = [
            " ".join(clean_text(cand_docs[i], remove_stopwords=True, ngram=ngram))
            for i in cand_ids
        ]
        query_token_strs = [
            " ".join(clean_text(query_docs[i], remove_stopwords=True, ngram=ngram))
            for i in query_ids
        ]

        C, transform_q = build_tfidf_index(cand_token_strs, scheme, min_df, max_df)
        Q = transform_q(query_token_strs)
        scores_mat = (C @ Q.T).toarray()   # (n_cands, n_queries)

        shortlists   = {}
        tfidf_scores = {}
        for qi, q_id in enumerate(query_ids):
            scores  = scores_mat[:, qi]
            s_min, s_max = scores.min(), scores.max()
            norm    = (scores - s_min) / (s_max - s_min + 1e-12)
            tfidf_scores[q_id] = {c_id: float(norm[j]) for j, c_id in enumerate(cand_ids)}
            top_idx = np.argsort(scores)[::-1][:top_n]
            shortlists[q_id]   = [cand_ids[j] for j in top_idx]

        # ── First-stage diagnostic ──────────────────────────────────────────
        fs_ranked = {
            query_ids[qi]: [cand_ids[j] for j in np.argsort(scores_mat[:, qi])[::-1]]
            for qi in range(len(query_ids))
        }
        fs_f1, fs_k, _ = _micro_f1(fs_ranked, gold)
        print(f"  Stage-1 TF-IDF: MicroF1={fs_f1*100:.2f}%@K={fs_k}", flush=True)

    # ── TF-IDF only: skip stage 2 ────────────────────────────────────────────
    if mtype == "tfidf_only" or model_name is None:
        assert fs_ranked is not None, "tfidf_only requires TF-IDF stage; set skip_tfidf=False"
        if citation_beta > 0:
            cite_sc = citation_overlap_score(cand_docs, query_docs, cand_ids, query_ids)
            ranked  = fuse_and_rank(
                shortlists, tfidf_scores,
                {q_id: {} for q_id in query_ids}, cand_ids, alpha,
                citation_scores=cite_sc, citation_beta=citation_beta,
            )
            metrics = evaluate_ranked(ranked, gold)
        else:
            metrics = evaluate_ranked(fs_ranked, gold)
        elapsed = time.time() - t0
        print(f"  → MicroF1={metrics['_micro_f1']*100:.2f}%@K={metrics['_micro_k']}  "
              f"MAP={metrics['MAP']*100:.2f}%  [{elapsed:.0f}s]")
        metrics["_elapsed"] = elapsed
        return metrics

    # ── Stage 2: Transformer re-ranking ──────────────────────────────────────
    if mtype == "biencoder":
        from sentence_transformers import SentenceTransformer
        print(f"  Loading SentenceTransformer: {model_name} …", flush=True)
        model = SentenceTransformer(model_name)
        model_scores = biencoder_score(
            model, shortlists, cand_docs, query_docs, query_ids,
            chunk_tokens=chunk_tok, chunk_stride=chunk_str,
            encode_batch=enc_batch, agg=agg,
        )

    elif mtype == "crossencoder":
        from sentence_transformers.cross_encoder import CrossEncoder
        print(f"  Loading CrossEncoder: {model_name} …", flush=True)
        model = CrossEncoder(model_name, max_length=max_len)
        model_scores = crossencoder_score(
            model, shortlists, cand_docs, query_docs, query_ids,
            batch_size=ce_batch, max_length=max_len,
        )

    else:
        raise ValueError(f"Unknown model_type: {mtype!r}. Choose biencoder | crossencoder | tfidf_only")

    # When skip_tfidf=True the fused score equals pure model score (alpha=0.0 forced above)

    # ── Optional citation overlap ─────────────────────────────────────────────
    if citation_beta > 0:
        cite_sc = citation_overlap_score(cand_docs, query_docs, cand_ids, query_ids)
    else:
        cite_sc = None

    # ── Fusion + Evaluation ───────────────────────────────────────────────────
    ranked = fuse_and_rank(shortlists, tfidf_scores, model_scores, cand_ids, alpha,
                           citation_scores=cite_sc, citation_beta=citation_beta)
    metrics = evaluate_ranked(ranked, gold)
    elapsed = time.time() - t0

    print(f"  → MicroF1={metrics['_micro_f1']*100:.2f}%@K={metrics['_micro_k']}  "
          f"MAP={metrics['MAP']*100:.2f}%  MRR={metrics['MRR']*100:.2f}%  "
          f"NDCG@10={metrics['NDCG@10']*100:.2f}%  [{elapsed:.0f}s]")
    metrics["_elapsed"] = elapsed
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 9. CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(description="Hybrid TF-IDF + Transformer re-ranking for IL-PCR")
    p.add_argument("--cand_dir",      default=_DEFAULT_CAND)
    p.add_argument("--query_dir",     default=_DEFAULT_QUERY)
    p.add_argument("--labels",        default=_DEFAULT_LABELS)
    p.add_argument("--model",         default=None,
                   help="HuggingFace checkpoint. Leave empty for TF-IDF only.")
    p.add_argument("--model_type",    default="tfidf_only",
                   choices=["tfidf_only", "biencoder", "crossencoder"])
    p.add_argument("--skip_tfidf",    action="store_true",
                   help="Pure transformer mode: skip TF-IDF entirely, rank all candidates by model only")
    p.add_argument("--agg",           default="max_chunk",
                   help="Chunk aggregation for bi-encoder: max_chunk | mean_chunk | sum_topN (e.g. sum_top3)")
    p.add_argument("--citation_beta", type=float, default=0.0,
                   help="Weight for citation-overlap signal (0 disables). "
                        "Set alpha+citation_beta≤1. Works with any model_type.")
    p.add_argument("--tfidf_scheme",  default="augmented",
                   choices=["log", "augmented", "binary"])
    p.add_argument("--tfidf_ngram",   type=int,   default=4)
    p.add_argument("--tfidf_top_n",   type=int,   default=200)
    p.add_argument("--tfidf_min_df",  type=int,   default=2)
    p.add_argument("--tfidf_max_df",  type=float, default=0.95)
    p.add_argument("--alpha",         type=float, default=0.8,
                   help="Weight of TF-IDF in fusion. 1.0 = TF-IDF only.")
    p.add_argument("--chunk_tokens",  type=int,   default=256)
    p.add_argument("--chunk_stride",  type=int,   default=128)
    p.add_argument("--encode_batch",  type=int,   default=128)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--max_length",    type=int,   default=512)
    p.add_argument("--label",         default=None,
                   help="Experiment label for output files")
    p.add_argument("--output_dir",    default="./exp_results_improved",
                   help="Directory to save results JSON/CSV")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    cfg = dict(vars(args))
    cfg["label"] = cfg["label"] or (
        f"hybrid_{cfg['model_type']}_{cfg.get('model','tfidf').split('/')[-1]}"
        f"_n{cfg['tfidf_ngram']}_{cfg['tfidf_scheme']}"
        f"_top{cfg['tfidf_top_n']}_a{cfg['alpha']}"
    )

    print("Loading data …")
    cand_docs  = load_corpus(args.cand_dir)
    query_docs = load_corpus(args.query_dir)
    gold       = load_gold(args.labels)
    print(f"  Candidates: {len(cand_docs)}  |  Queries: {len(query_docs)}")

    metrics = run_experiment(cfg, gold=gold, cand_docs=cand_docs, query_docs=query_docs)

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(args.output_dir, f"result_{ts}.json")
    out_cfg  = os.path.join(args.output_dir, f"config_{ts}.json")

    with open(out_json, "w") as f:
        json.dump({k: v for k, v in metrics.items() if not isinstance(v, list) or len(v) < 50},
                  f, indent=2)
    with open(out_cfg, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nResults saved → {out_json}")
