"""
run_hybrid_sweep.py
===================
Comprehensive TF-IDF + Transformer re-ranking experiment sweep for IL-PCR.

All experiments are defined as a list of config dicts and executed in order
by calling run_experiment() from hybrid_transformer_rerank.py.

Experiment groups
-----------------
Group R  – Reference (TF-IDF only, no transformer)
           Verifies our improved first-stage baseline.

Group I  – Improved-TF-IDF + MiniLM bi-encoder
           Tests whether a better first-stage (augmented n=7-9) fixes the
           original hybrid's under-performance.

Group A  – Alpha sweep
           Sweeps the fusion weight α ∈ {0.5 … 1.0} with the best TF-IDF
           params and MiniLM-L6-v2 to find the optimal fusion weight.

Group G  – Chunk-aggregation strategy
           max_chunk vs mean_chunk with the best config so far.

Group C  – Chunk-size sweep
           chunk={128,256,512} × stride={64,128,256} to trade off coverage
           vs noise when encoding long legal documents.

Group M  – Model sweep (bi-encoder)
           Tests several sentence-transformer and legal-domain models:
           MiniLM, MPNet, InLegalBERT, multi-qa-MiniLM, InCaseLawBERT,
           LegalBERT.

Group CE – Cross-encoder re-ranking
           Uses MS-MARCO cross-encoders (MiniLM, Electra) as stage-2
           models.  Sweeps shortlist size top∈{50,100,200}.

Group TN – Top-N shortlist sweep
           Tests how many candidates to pass to the transformer stage.

Group X  – Combined best (top config per group above)
           Re-runs the overall best configuration identified after reviewing
           results.

Usage
-----
    cd Models/Transformer-Embeddings
    python run_hybrid_sweep.py               # run all experiments
    python run_hybrid_sweep.py --groups R I  # run only selected groups
    python run_hybrid_sweep.py --resume      # skip already-completed labels
    python run_hybrid_sweep.py --dry_run     # print experiment list, no execution
"""

import os, sys, json, time, argparse, csv, traceback
from pathlib import Path
from datetime import datetime

# Make sure we import from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from hybrid_transformer_rerank import (
    load_corpus, load_gold, run_experiment, SKIP_IDS,
)

# ══════════════════════════════════════════════════════════════════════════════
# DATA PATHS
# ══════════════════════════════════════════════════════════════════════════════

_CAND   = "../BM25/data/corpus/ik_test/candidate"
_QUERY  = "../BM25/data/corpus/ik_test/query"
_LABELS = "../BM25/data/corpus/ik_test/test.json"

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def _base(**kwargs):
    """Merge keyword args with sane defaults.

    Defaults are set to the best observed params from previous HYBRID_CHUNK runs:
      chunk_tokens=256, chunk_stride=128, tfidf_top_n=200  (from Apr 2026 sweep)
    """
    defaults = dict(
        model_type   = "tfidf_only",
        model        = None,
        skip_tfidf   = False,   # True = pure transformer (no TF-IDF at all)
        agg          = "max_chunk",
        tfidf_scheme = "log",
        tfidf_ngram  = 4,
        tfidf_top_n  = 200,    # best from previous chunk experiments
        tfidf_min_df = 2,
        tfidf_max_df = 0.95,
        alpha        = 0.80,
        chunk_tokens = 256,    # best from previous chunk experiments
        chunk_stride = 128,    # best from previous chunk experiments
        encode_batch = 128,
        batch_size   = 32,
        max_length   = 512,
        citation_beta = 0.0,   # weight for citation-recall signal (0 = disabled)
    )
    defaults.update(kwargs)
    return defaults


# ─── Group R: Reference TF-IDF only ──────────────────────────────────────────
# Sanity-check / paper-baseline values; no transformer.
# Expected: R_log_n4 ≈ 44.09%,  R_aug_n7 ≈ 44.97%  (from eval_tfidf_only.py)
GROUP_R = [
    _base(label="R_log_n4",    group="R", tfidf_scheme="log",       tfidf_ngram=4),
    _base(label="R_log_n7",    group="R", tfidf_scheme="log",       tfidf_ngram=7),
    _base(label="R_log_n9",    group="R", tfidf_scheme="log",       tfidf_ngram=9),
    _base(label="R_aug_n6",    group="R", tfidf_scheme="augmented", tfidf_ngram=6),
    _base(label="R_aug_n7",    group="R", tfidf_scheme="augmented", tfidf_ngram=7),
    _base(label="R_aug_n9",    group="R", tfidf_scheme="augmented", tfidf_ngram=9),
    _base(label="R_bin_n7",    group="R", tfidf_scheme="binary",    tfidf_ngram=7),
]

# ─── Group I: Improved-TF-IDF + MiniLM bi-encoder ────────────────────────────
# Fixes the original hybrid failure: uses augmented TF + higher n-grams.
# Original best was alpha=0.6, n=4, log → 39.48% (WORSE than TF-IDF alone).
# Hypothesis: alpha≥0.8 + augmented n=7 should beat TF-IDF alone.
_MINILM = "sentence-transformers/all-MiniLM-L6-v2"

GROUP_I = [
    # Baseline-equivalent (original failure config) with improved TF-IDF
    _base(label="I_log_n4_a06",   group="I", model=_MINILM, model_type="biencoder",
          tfidf_scheme="log",       tfidf_ngram=4, alpha=0.6),
    # Improved TF-IDF: alpha 0.6 / 0.8 / 0.9  +  n=9 variant
    _base(label="I_aug_n7_a06",   group="I", model=_MINILM, model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.6),
    _base(label="I_aug_n7_a08",   group="I", model=_MINILM, model_type="biencoder",  # core exp
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.8),
    _base(label="I_aug_n7_a09",   group="I", model=_MINILM, model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.9),
    _base(label="I_aug_n9_a08",   group="I", model=_MINILM, model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=9, alpha=0.8),
]

# ─── Group A: Alpha sweep ─────────────────────────────────────────────────────
# Fill in the gaps NOT already covered by Group I:
#   Group I covers α = 0.6 (I_aug_n7_a06), 0.8 (I_aug_n7_a08), 0.9 (I_aug_n7_a09)
#   So only test: 0.50, 0.70, 0.85, 0.95
GROUP_A = [
    _base(label=f"A_aug_n7_a{round(a*100):03d}", group="A",
          model=_MINILM, model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=a)
    for a in [0.50, 0.70, 0.85, 0.95]
]

# ─── Group G: Aggregation strategy ────────────────────────────────────────────
# max_chunk is the default (already tested as I_aug_n7_a08); only need mean_chunk.
GROUP_G = [
    _base(label="G_aug_n7_a08_meanchunk", group="G",
          model=_MINILM, model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.8, agg="mean_chunk"),
]

# ─── Group C: Chunk-size sweep ────────────────────────────────────────────────
# Indian SC judgments are ~5000 tokens. Test different window granularities.
# chunk=256/stride=128 is the default (= I_aug_n7_a08); only test non-default sizes.
GROUP_C = [
    _base(label="C_chunk128_str64",  group="C",
          model=_MINILM, model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.8,
          chunk_tokens=128, chunk_stride=64),
    _base(label="C_chunk512_str256", group="C",
          model=_MINILM, model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.8,
          chunk_tokens=512, chunk_stride=256),
    _base(label="C_chunk512_str128", group="C",
          model=_MINILM, model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.8,
          chunk_tokens=512, chunk_stride=128),
]

# ─── Group M: Model sweep (bi-encoder) ────────────────────────────────────────
# Tests general + legal-domain sentence transformers.
# Alpha and chunk params reuse the best from Groups A/C.
_MPNET        = "sentence-transformers/all-mpnet-base-v2"
_INLEGAL      = "law-ai/InLegalBERT"
_INCASELAW    = "law-ai/InCaseLawBERT"
_MULTI_QA     = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
_LEGALBERT    = "nlpaueb/legal-bert-base-uncased"
_ROBERTA      = "sentence-transformers/all-roberta-large-v1"  # RoBERTa-large SBERT
_BERT_BASE    = "bert-base-uncased"                            # raw BERT, mean-pool via ST
_SBERT_PARA   = "sentence-transformers/paraphrase-MiniLM-L6-v2"  # classic SBERT paper model

GROUP_M = [
    # ── General sentence transformers (M_minilm skipped: same as I_aug_n7_a08) ─
    _base(label="M_sbert_para",  group="M", model=_SBERT_PARA, model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.8),
    _base(label="M_mpnet",       group="M", model=_MPNET,      model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.8),
    _base(label="M_roberta",     group="M", model=_ROBERTA,    model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.8),
    _base(label="M_bert_base",   group="M", model=_BERT_BASE,  model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.8),
    _base(label="M_multiqa",     group="M", model=_MULTI_QA,   model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.8),
    # ── Legal-domain models ───────────────────────────────────────────────
    _base(label="M_inlegalbert", group="M", model=_INLEGAL,    model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.8),
    _base(label="M_incaselaw",   group="M", model=_INCASELAW,  model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.8),
    _base(label="M_legalbert",   group="M", model=_LEGALBERT,  model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.8),
    # ── Higher-alpha variants for domain models (trust TF-IDF more) ──────
    _base(label="M_inlegalbert_a09", group="M", model=_INLEGAL,   model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.9),
    _base(label="M_incaselaw_a09",   group="M", model=_INCASELAW, model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.9),
    _base(label="M_roberta_a09",     group="M", model=_ROBERTA,   model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.9),
]

# ─── Group CE: Cross-encoder re-ranking ──────────────────────────────────────
# TF-IDF retrieves top-N candidates; cross-encoder scores each (q, c) pair.
# MS-MARCO-trained models generalise best for passage/doc retrieval.
# NOTE: cross-encoder on full corpus (skip_tfidf) is impractical:
#   237 queries × 1727 candidates = 409k pairs → hours of compute.
#   Use TF-IDF shortlist of 100-200 as first stage for all CE experiments.
_CE_MINI    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_CE_ELECTRA = "cross-encoder/ms-marco-electra-base"
_CE_ROBERTA = "cross-encoder/ms-marco-roberta-base"  # RoBERTa cross-encoder

GROUP_CE = [
    # ── MiniLM cross-encoder: top-N sweep ────────────────────────────────
    _base(label="CE_minilm_top50",      group="CE", model=_CE_MINI, model_type="crossencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.6, tfidf_top_n=50),
    _base(label="CE_minilm_top100",     group="CE", model=_CE_MINI, model_type="crossencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.6, tfidf_top_n=100),
    _base(label="CE_minilm_top200",     group="CE", model=_CE_MINI, model_type="crossencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.6, tfidf_top_n=200),
    # ── Alpha sweep for cross-encoder ────────────────────────────────────
    _base(label="CE_minilm_top100_a08", group="CE", model=_CE_MINI, model_type="crossencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.8, tfidf_top_n=100),
    # ── Other cross-encoder architectures ────────────────────────────────
    _base(label="CE_electra_top100",    group="CE", model=_CE_ELECTRA, model_type="crossencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.6, tfidf_top_n=100),
    _base(label="CE_roberta_top100",    group="CE", model=_CE_ROBERTA, model_type="crossencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.6, tfidf_top_n=100),
    _base(label="CE_roberta_top200",    group="CE", model=_CE_ROBERTA, model_type="crossencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.6, tfidf_top_n=200),
]

# ─── Group TN: Top-N shortlist size sweep ─────────────────────────────────────
# At what point does a bigger shortlist add noise vs useful recall?
GROUP_TN = [
    # top200 skipped: same as I_aug_n7_a08 (default)
    _base(label=f"TN_top{n}", group="TN",
          model=_MINILM, model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.8, tfidf_top_n=n)
    for n in [50, 100, 150, 300]
]

# ─── Group T: Transformer-ONLY ranking (no TF-IDF) ───────────────────────────
# skip_tfidf=True → shortlist = full 1727-candidate corpus, alpha forced to 0.0.
# Answers the question: "does transformer alone beat TF-IDF alone (~45%)?".
# Only bi-encoder tested here (cross-encoder on full corpus is impractical).
# 4 representative models: fastest general, strongest general, best Indian legal, 2nd Indian legal.
# T_sbert_para / T_roberta / T_bert_base / T_legalbert dropped:
#   sbert_para is slower MiniLM with no advantage; roberta-large is ~4× slower for marginal gain;
#   bert_base is not fine-tuned for similarity; legalbert is US/EU trained (InLegalBERT is Indian).
GROUP_T = [
    _base(label="T_minilm",      group="T", model=_MINILM,    model_type="biencoder",
          skip_tfidf=True),
    _base(label="T_mpnet",       group="T", model=_MPNET,     model_type="biencoder",
          skip_tfidf=True),
    _base(label="T_inlegalbert", group="T", model=_INLEGAL,   model_type="biencoder",
          skip_tfidf=True),
    _base(label="T_incaselaw",   group="T", model=_INCASELAW, model_type="biencoder",
          skip_tfidf=True),
]

# ─── Group S: sum_topK chunk aggregation ────────────────────────────────────────
# Addresses the "distributed citation evidence" problem: instead of max_chunk
# (only the single best window), sum the top-K per-candidate-chunk scores.
# If a candidate cites A, B, C in separate windows they ALL contribute.
GROUP_S = [
    _base(label="S_sum_top3",          group="S", model=_MINILM,  model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.8, agg="sum_top3"),
    _base(label="S_sum_top5",          group="S", model=_MINILM,  model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.8, agg="sum_top5"),
    _base(label="S_inlegal_sum_top3",  group="S", model=_INLEGAL, model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.8, agg="sum_top3"),
]

# ─── Group CO: Citation overlap ──────────────────────────────────────────────
# Addresses the "exact citation identity" problem: regex-extracts AIR/SCC/SCR/
# MANU citations, computes recall(q_cites, c_cites).  Orthogonal to TF-IDF
# (immune to IDF dilution) and to embeddings (exact-match, not similarity).
# Three sub-experiments:
#   CO pure  – TF-IDF + citation only (no transformer)
#   CO+bi   – 3-way: TF-IDF + citation + bi-encoder
#   CO+CE   – 3-way: TF-IDF + citation + cross-encoder
GROUP_CO = [
    # ── TF-IDF + citation, no transformer (α + β = 1.0) ────────────────
    _base(label="CO_tfidf_b05",   group="CO", model_type="tfidf_only",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.95, citation_beta=0.05),
    _base(label="CO_tfidf_b10",   group="CO", model_type="tfidf_only",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.90, citation_beta=0.10),
    _base(label="CO_tfidf_b20",   group="CO", model_type="tfidf_only",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.80, citation_beta=0.20),
    # ── 3-way: TF-IDF + citation + bi-encoder (α + β + model = 1.0) ─────
    _base(label="CO_minilm_b10",  group="CO", model=_MINILM,  model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.75, citation_beta=0.10),
    _base(label="CO_inlegal_b10", group="CO", model=_INLEGAL, model_type="biencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.75, citation_beta=0.10),
    # ── 3-way: TF-IDF + citation + cross-encoder ──────────────────────
    _base(label="CO_ce_b10",      group="CO", model=_CE_MINI, model_type="crossencoder",
          tfidf_scheme="augmented", tfidf_ngram=4, alpha=0.55, citation_beta=0.10, tfidf_top_n=100),
]

# ─── Group X: Combined best configs (fill in after reviewing results) ─────────
# Placeholder entries; update labels/params after a first sweep run.
GROUP_X = [
    # Example: best model + best alpha + best chunk found above
    # _base(label="X_best", model=..., alpha=..., tfidf_ngram=..., chunk_tokens=...), 
]

# ─── Full experiment registry ────────────────────────────────────────────────
ALL_GROUPS = {
    
    "I":  GROUP_I,
    "A":  GROUP_A,
    "G":  GROUP_G,
    "C":  GROUP_C,
    "M":  GROUP_M,
    "CE": GROUP_CE,
    "TN": GROUP_TN,
    "T":  GROUP_T,
    "S":  GROUP_S,
    "CO": GROUP_CO,
    "X":  GROUP_X,
}

ALL_EXPERIMENTS = [exp for g in ALL_GROUPS.values() for exp in g]


# ══════════════════════════════════════════════════════════════════════════════
# RESULT SAVING
# ══════════════════════════════════════════════════════════════════════════════

def _append_csv(csv_path: str, row: dict):
    fieldnames = [
        "label", "group", "model_type", "skip_tfidf", "model", "tfidf_scheme",
        "tfidf_ngram", "tfidf_top_n", "alpha", "citation_beta", "agg",
        "chunk_tokens", "chunk_stride",
        "micro_f1_pct", "micro_k", "MAP_pct", "MRR_pct", "NDCG10_pct",
        "elapsed_s", "error",
    ]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def _metrics_to_row(cfg: dict, metrics: dict, error: str = "") -> dict:
    return {
        "label":         cfg.get("label", ""),
        "group":         cfg.get("group", ""),
        "model_type":    cfg.get("model_type", ""),
        "skip_tfidf":    cfg.get("skip_tfidf", False),
        "model":         cfg.get("model", ""),
        "tfidf_scheme":  "N/A" if cfg.get("skip_tfidf") else cfg.get("tfidf_scheme", ""),
        "tfidf_ngram":   cfg.get("tfidf_ngram", ""),
        "tfidf_top_n":   cfg.get("tfidf_top_n", ""),
        "alpha":         cfg.get("alpha", ""),
        "citation_beta": cfg.get("citation_beta", 0.0),
        "agg":           cfg.get("agg", ""),
        "chunk_tokens":  cfg.get("chunk_tokens", ""),
        "chunk_stride":  cfg.get("chunk_stride", ""),
        "micro_f1_pct":  round(metrics.get("_micro_f1", 0) * 100, 4) if not error else "",
        "micro_k":       metrics.get("_micro_k", "") if not error else "",
        "MAP_pct":       round(metrics.get("MAP", 0) * 100, 4) if not error else "",
        "MRR_pct":       round(metrics.get("MRR", 0) * 100, 4) if not error else "",
        "NDCG10_pct":    round(metrics.get("NDCG@10", 0) * 100, 4) if not error else "",
        "elapsed_s":     round(metrics.get("_elapsed", 0), 1) if not error else "",
        "error":         error,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PRINTING SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary(results: list):
    header = f"{'Label':<35}  {'Grp':<4}  {'Xfmr?':<6}  {'Model':<40}  {'α':>5}  {'n':>2}  " \
             f"{'Scheme':<9}  {'MicroF1':>8}  {'@K':>3}  {'MAP':>8}  {'MRR':>8}"
    print("\n" + "═" * len(header))
    print(header)
    print("─" * len(header))
    for r in sorted(results, key=lambda x: x.get("micro_f1_pct", 0), reverse=True):
        f1   = r.get("micro_f1_pct", "ERR")
        mk   = r.get("micro_k", "")
        mp   = r.get("MAP_pct", "")
        mrr  = r.get("MRR_pct", "")
        f1s  = f"{f1:.2f}%" if isinstance(f1, float) else f1
        mps  = f"{mp:.2f}%"  if isinstance(mp, float)  else str(mp)
        mrrs = f"{mrr:.2f}%" if isinstance(mrr, float) else str(mrr)
        model = r.get("model", "—") or "—"
        model = model[model.rfind("/")+1:] if "/" in model else model
        only  = "YES" if r.get("skip_tfidf") else "—"
        print(f"{r['label']:<35}  {r.get('group',''):<4}  {only:<6}  {model:<40}  "
              f"{r.get('alpha',''):>5}  {r.get('tfidf_ngram',''):>2}  "
              f"{r.get('tfidf_scheme',''):<9}  {f1s:>8}  {mk:>3}  {mps:>8}  {mrrs:>8}")
    print("═" * len(header))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run full IL-PCR hybrid retrieval experiment sweep"
    )
    parser.add_argument("--groups", nargs="*", default=list(ALL_GROUPS.keys()),
                        help="Experiment group labels to run (default: all). "
                             f"Available: {list(ALL_GROUPS.keys())}")
    parser.add_argument("--resume",  action="store_true",
                        help="Skip experiments whose label is already in the CSV")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print experiments that would run, then exit")
    parser.add_argument("--output_dir", default=None,
                        help="Override output directory (default: exp_results/HYBRID_IMPROVED_<ts>/)")
    args = parser.parse_args()

    # ── Select experiments ────────────────────────────────────────────────────
    selected = []
    for g in args.groups:
        g = g.upper()
        if g not in ALL_GROUPS:
            print(f"WARNING: unknown group '{g}', skipping.")
            continue
        selected.extend(ALL_GROUPS[g])

    if not selected:
        print("No experiments selected. Exiting.")
        return

    # ── Output directory ─────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "exp_results", f"HYBRID_IMPROVED_{ts}"
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path  = os.path.join(out_dir, "all_results.csv")
    json_path = os.path.join(out_dir, "all_results.json")

    print(f"\nOutput directory : {out_dir}")
    print(f"Groups selected  : {args.groups}")
    print(f"Experiments      : {len(selected)}\n")

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"{'Label':<40}  {'Group':<5}  {'Type':<12}  {'Model'}")
        print("-" * 90)
        for e in selected:
            print(f"{e['label']:<40}  {e.get('group',''):<5}  "
                  f"{e.get('model_type',''):<12}  {e.get('model') or '—'}")
        print(f"\nTotal: {len(selected)} experiments  (dry run — nothing executed)")
        return

    # ── Resume: load already-done labels ─────────────────────────────────────
    done_labels = set()
    if args.resume and os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if not row.get("error"):
                    done_labels.add(row["label"])
        print(f"Resuming: {len(done_labels)} experiments already done.")

    # ── Load data once (shared across all experiments) ────────────────────────
    print("Loading corpus …")
    cand_docs  = load_corpus(_CAND)
    query_docs = load_corpus(_QUERY)
    gold       = load_gold(_LABELS)
    print(f"  Candidates : {len(cand_docs)}")
    print(f"  Queries    : {len(query_docs)}")
    print(f"  Gold labels: {len(gold)} queries")

    # ── Run experiments ───────────────────────────────────────────────────────
    all_rows    = []
    errors      = []
    t_sweep_start = time.time()

    for idx, cfg in enumerate(selected, 1):
        label = cfg["label"]
        if label in done_labels:
            print(f"  [{idx}/{len(selected)}] SKIP (done): {label}")
            continue

        print(f"\n[{idx}/{len(selected)}] {label}")
        try:
            metrics = run_experiment(
                cfg, gold=gold, cand_docs=cand_docs, query_docs=query_docs
            )
            row = _metrics_to_row(cfg, metrics)
            # Save individual result JSON
            exp_json = os.path.join(out_dir, f"{label}.json")
            with open(exp_json, "w") as f:
                json.dump({"config": cfg, "metrics": metrics}, f, indent=2,
                          default=lambda o: float(o) if hasattr(o, "__float__") else str(o))

        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR in {label}: {err_msg}")
            traceback.print_exc()
            errors.append((label, err_msg))
            row = _metrics_to_row(cfg, {}, error=err_msg)

        all_rows.append(row)
        _append_csv(csv_path, row)

    # ── Save aggregated JSON ──────────────────────────────────────────────────
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2)

    elapsed = time.time() - t_sweep_start
    print(f"\n\nSweep complete in {elapsed/60:.1f} min  |  {len(all_rows) - len(errors)} succeeded, {len(errors)} failed")
    print(f"Results: {csv_path}")
    if errors:
        print("\nFailed experiments:")
        for lbl, err in errors:
            print(f"  {lbl}: {err}")

    _print_summary(all_rows)


if __name__ == "__main__":
    main()
