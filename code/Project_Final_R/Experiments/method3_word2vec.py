"""
method3_word2vec.py
===================
Method-3 from:
  "A Text Similarity Approach for Precedence Retrieval from Legal Documents"
  Thenmozhi, Kawshik Kannan, Aravindan — IRLeD@FIRE2017

Pipeline
--------
  1. Preprocess: strip punctuation and citation markers
  2. POS-tag each document
  3. Extract NOUNS (NN, NNS, NNP) + VERBS (VB, VBZ, VBN, VBD) as key terms
  4. Lemmatize each key term (WordNet Lemmatizer)
  5. Look up each term in Word2Vec (300-dim vectors)
  6. Document vector = mean of all term vectors found in the vocabulary
  7. Cosine similarity between each query vector and all candidate vectors
  8. Rank candidates by similarity (descending)

Word2Vec model
--------------
  The paper uses GoogleNews-vectors-negative300.bin.gz (pre-trained, 300-dim).
  This is now loaded via gensim.downloader.load("word2vec-google-news-300").

  Alternative: a Word2Vec model trained on the legal corpus itself is also
  offed as a CONFIG option (no external download required).

Run
---
    python3 method3_word2vec.py \
        --data_dir /path/to/dataset/ \
        --split train \
        --w2v_model word2vec-google-news-300

Parameter grid
--------------
  Edit CONFIGS below.  Comment out any line to skip.
"""

import os
import re
import argparse
from typing import Dict, List, Tuple, Optional

import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import gensim.downloader as api  # Added to use best available downloader

from utils import (
    load_split, evaluate_all,
    save_results, print_results_table, save_results_csv,
    cosine_sim_matrix, build_w2v,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS  (edit here)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "./"
SPLIT    = "train"
TOP_K    = 1000
OUTPUT   = "results/method3_results.json"
K_VALUES = [5, 10, 20, 50, 100]

# Identifier for the pre-trained Word2Vec model.
# Set to None to skip the pre-trained model and only run corpus-trained variants.
W2V_PRETRAINED_MODEL = "word2vec-google-news-300"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGS  ← comment out any line to skip
# Each entry: (source, dim, pooling)
#
#   source   : "pretrained"        → GoogleNews vectors (W2V_PRETRAINED_MODEL)
#              "corpus_sg"         → Skip-gram trained on this legal corpus
#              "corpus_cbow"       → CBOW trained on this legal corpus
#   dim      : embedding dimension
#              (for pretrained this is usually 300; updated dynamically)
#   pooling  : "mean"              → mean of term vectors (paper default)
#              "tfidf_mean"        → IDF-weighted mean (extension)
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS: List[Tuple] = [
    # ── Paper Method-3: GoogleNews pretrained, mean pooling ───────────────────
    ("pretrained",   300, "mean"),         # exact paper method

    # ── Corpus-trained variants (no external download) ────────────────────────
    ("corpus_sg",    100, "mean"),
    ("corpus_sg",    200, "mean"),
    ("corpus_sg",    300, "mean"),
    ("corpus_cbow",  100, "mean"),
    ("corpus_cbow",  200, "mean"),
    ("corpus_cbow",  300, "mean"),

    # ── IDF-weighted mean variants ────────────────────────────────────────────
    ("corpus_sg",    100, "tfidf_mean"),
    ("corpus_sg",    200, "tfidf_mean"),
    ("corpus_sg",    300, "tfidf_mean"),
    ("corpus_cbow",  200, "tfidf_mean"),
    ("pretrained",   300, "tfidf_mean"),  # paper pretrained + IDF weighting
]

# ─────────────────────────────────────────────────────────────────────────────
# POS + LEMMATISATION (same as Method-2)
# ─────────────────────────────────────────────────────────────────────────────

NOUN_TAGS = {"NN", "NNS", "NNP"}
VERB_TAGS = {"VB", "VBZ", "VBN", "VBD"}
KEEP_TAGS = NOUN_TAGS | VERB_TAGS

_lemmatizer  = WordNetLemmatizer()
_PUNCT_RE    = re.compile(r'[",\-\'_]')
_CITATION_RE = re.compile(r'\[?\?CITATION\?\]?|<CITATION_\d+>', re.IGNORECASE)


def _wn_pos(tag: str) -> str:
    return "n" if tag in NOUN_TAGS else "v"


def preprocess(text: str) -> str:
    text = _CITATION_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    return text


def extract_key_terms(text: str) -> List[str]:
    """Return lemmatized nouns + verbs (key terms) for a document."""
    text   = preprocess(text)
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    return [
        _lemmatizer.lemmatize(w.lower(), pos=_wn_pos(tag))
        for w, tag in tagged
        if tag in KEEP_TAGS
    ]


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def compute_idf_weights(corpus: Dict[str, List[str]]) -> Dict[str, float]:
    """Smoothed IDF over the corpus of term lists."""
    import math
    from collections import defaultdict
    N  = len(corpus)
    df: Dict[str, int] = defaultdict(int)
    for terms in corpus.values():
        for t in set(terms):
            df[t] += 1
    return {t: math.log((N + 1) / (c + 1)) + 1.0 for t, c in df.items()}


def mean_vector(terms: List[str], wv, dim: int,
                idf: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    Mean (or IDF-weighted mean) of Word2Vec vectors for the given terms.
    Terms not in the vocabulary are silently skipped.
    Returns zero vector if no terms found.
    """
    valid = [(t, wv[t]) for t in terms if t in wv]
    if not valid:
        return np.zeros(dim, dtype=np.float32)

    if idf:
        weights = np.array([idf.get(t, 1.0) for t, _ in valid], dtype=np.float32)
        weights /= weights.sum() + 1e-10
        return np.sum(
            [w * v for (_, v), w in zip(valid, weights)], axis=0
        ).astype(np.float32)

    return np.mean([v for _, v in valid], axis=0).astype(np.float32)


def embed_corpus(corpus: Dict[str, List[str]],
                 wv, dim: int,
                 idf: Optional[Dict[str, float]] = None
                 ) -> Tuple[List[str], np.ndarray]:
    """Embed all documents.  Returns (doc_ids, matrix of shape (N, dim))."""
    ids = list(corpus.keys())
    mat = np.stack(
        [mean_vector(corpus[did], wv, dim, idf) for did in ids], axis=0
    )
    return ids, mat


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  default=DATA_DIR)
    parser.add_argument("--split",     default=SPLIT)
    parser.add_argument("--top_k",     type=int, default=TOP_K)
    parser.add_argument("--output",    default=OUTPUT)
    parser.add_argument(
        "--w2v_model",
        default=W2V_PRETRAINED_MODEL,
        help="Gensim model identifier (e.g. word2vec-google-news-300)"
    )
    args = parser.parse_args()

    queries, candidates, relevance = load_split(args.data_dir, args.split)

    # ── Extract key terms once (POS-tag all documents) ────────────────────────
    print("\nExtracting key terms from candidates (POS-tagging) ...")
    cand_terms: Dict[str, List[str]] = {}
    for i, (cid, text) in enumerate(candidates.items()):
        if i % 200 == 0:
            print(f"  {i}/{len(candidates)}")
        cand_terms[cid] = extract_key_terms(text)

    print("Extracting key terms from queries ...")
    query_terms: Dict[str, List[str]] = {}
    for i, (qid, text) in enumerate(queries.items()):
        if i % 50 == 0:
            print(f"  {i}/{len(queries)}")
        query_terms[qid] = extract_key_terms(text)

    # Pre-compute IDF over candidates (needed for tfidf_mean configs)
    idf_map = compute_idf_weights(cand_terms)

    # ── Pre-train corpus W2V models (cache by source+dim) ────────────────────
    all_sentences = list(cand_terms.values()) + list(query_terms.values())
    all_sentences = [s for s in all_sentences if s]

    corpus_models: Dict[Tuple, object] = {}   # (source, dim) → w2v model

    # Load pretrained using Gensim Downloader API
    pretrained_wv = None
    if args.w2v_model:
        try:
            print(f"\nLoading pretrained Word2Vec model: {args.w2v_model}")
            # This will download the model automatically if not present in ~/gensim-data
            pretrained_wv = api.load(args.w2v_model)
            print("  Loaded.")
        except Exception as e:
            print(f"\n[ERROR] Could not load model '{args.w2v_model}': {e}")
    else:
        print("\n[INFO] No pretrained model identifier provided. Skipping pretrained configs.")

    all_results = []

    for (source, dim, pooling) in CONFIGS:

        # Skip pretrained configs if no model loaded
        if source == "pretrained" and pretrained_wv is None:
            print(f"\n  [SKIP] {source} config — no pretrained model available.")
            continue

        name = f"Method3_W2V_{source}_dim={dim}_{pooling}"
        print(f"\n{'─'*64}\n  {name}\n{'─'*64}")

        # Get or train the appropriate W2V model
        if source == "pretrained":
            wv  = pretrained_wv
            dim = wv.vector_size   # Use actual dimension of the downloaded model
        else:
            model_key = (source, dim)
            if model_key not in corpus_models:
                print(f"  Training Word2Vec ({source}, dim={dim}) ...")
                sg = 1 if source == "corpus_sg" else 0
                w2v_model = build_w2v(
                    all_sentences, vector_size=dim,
                    window=5, min_count=2, sg=sg,
                    workers=4, epochs=5, seed=42
                )
                corpus_models[model_key] = w2v_model
            wv = corpus_models[(source, dim)].wv

        use_idf = (pooling == "tfidf_mean")

        # Embed candidates
        cand_ids, cand_mat = embed_corpus(
            cand_terms, wv, dim, idf=idf_map if use_idf else None
        )

        # Retrieve
        results: Dict[str, List[str]] = {}
        for qid in relevance:
            if qid not in query_terms:
                continue
            qvec  = mean_vector(
                query_terms[qid], wv, dim, idf=idf_map if use_idf else None
            ).reshape(1, -1)
            sims  = cosine_sim_matrix(qvec, cand_mat)[0]
            order = np.argsort(-sims)[:args.top_k]
            results[qid] = [cand_ids[i] for i in order]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    if not all_results:
        print("\n[WARN] No configurations ran — check --w2v_model or enable corpus variants.")
        return

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MAP",
                        title="Method-3 (Word2Vec mean) — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()