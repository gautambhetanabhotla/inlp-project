"""
method1_tfidf_nouns.py
======================
Method-1 from:
  "A Text Similarity Approach for Precedence Retrieval from Legal Documents"
  Thenmozhi, Kawshik Kannan, Aravindan — IRLeD@FIRE2017

Pipeline
--------
  1. Preprocess: strip punctuation and citation markers
  2. POS-tag each document
  3. Extract NOUNS only  (NN, NNS, NNP)
  4. Lemmatize extracted nouns (WordNet Lemmatizer)
  5. Build TF-IDF feature vectors over the noun vocabulary
  6. Compute cosine similarity between each query and every candidate
  7. Rank candidates by similarity score (descending)

Run
---
    python3 method1_tfidf_nouns.py --data_dir /path/to/dataset/ --split train

Parameter grid
--------------
  Edit CONFIGS below.  Comment out any line to skip.
"""

import os
import re
import argparse
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS  (edit here)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "./"
SPLIT    = "train"
TOP_K    = 1000
OUTPUT   = "results/method1_results.json"
K_VALUES = [5, 6, 7, 8, 9, 10, 11, 15, 20]

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGS  ← comment out any line to skip
# Each entry: (min_df, max_df_frac, sublinear_tf)
#   min_df       : minimum document frequency for a term to be kept
#   max_df_frac  : maximum fraction of docs a term may appear in (drop stopwords)
#   sublinear_tf : use log(tf)+1 instead of raw tf (sklearn sublinear_tf param)
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS: List[Tuple] = [
    # ── As described in the paper (plain TF-IDF, nouns only) ─────────────────
    (1,  1.00, False),   # paper default — no frequency filtering
    (2,  1.00, False),
    (2,  0.95, False),
    (2,  0.90, False),
    (5,  0.95, False),
    # ── Sublinear TF variants ─────────────────────────────────────────────────
    (1,  1.00, True),
    (2,  1.00, True),
    (2,  0.95, True),
    (5,  0.95, True),
]

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS & DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────

import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from utils import (
    load_split, read_document, evaluate_all,
    save_results, print_results_table, save_results_csv,
)

def download_nltk_dependencies():
    """Ensure all required NLTK resources are downloaded, including updated tagger names."""
    dependencies = [
        'punkt', 
        'punkt_tab',                    # Added for newer NLTK versions
        'averaged_perceptron_tagger', 
        'averaged_perceptron_tagger_eng', # Added to fix your specific LookupError
        'wordnet',
        'omw-1.4'
    ]
    for dep in dependencies:
        try:
            # We try to find the resource; if it fails, we download it.
            if dep == 'punkt' or dep == 'punkt_tab':
                nltk.data.find(f'tokenizers/{dep}')
            elif 'tagger' in dep:
                nltk.data.find(f'taggers/{dep}')
            else:
                nltk.data.find(f'corpora/{dep}')
        except LookupError:
            print(f"Downloading NLTK dependency: {dep}...")
            nltk.download(dep, quiet=True)

# Run dependency check on import or start
download_nltk_dependencies()

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

# NLTK targets (NN = singular noun, NNS = plural, NNP = proper noun)
NOUN_TAGS = {"NN", "NNS", "NNP"}

_lemmatizer = WordNetLemmatizer()

_PUNCT_RE   = re.compile(r'[",\-\'_]')
_CITATION_RE = re.compile(r'\[?\?CITATION\?\]?|<CITATION_\d+>', re.IGNORECASE)


def preprocess(text: str) -> str:
    """Strip citation markers and punctuation as described in the paper."""
    text = _CITATION_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    return text


def extract_nouns(text: str) -> List[str]:
    """
    POS-tag the text and return lemmatized nouns (NN, NNS, NNP).
    Mirrors the paper's Method-1 feature extraction step.
    """
    text   = preprocess(text)
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    nouns  = [
        _lemmatizer.lemmatize(word.lower(), pos='n')
        for word, tag in tagged
        if tag in NOUN_TAGS
    ]
    return nouns


def doc_to_noun_string(text: str) -> str:
    """Return space-joined noun string for TfidfVectorizer input."""
    return " ".join(extract_nouns(text))


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

    # ── Extract noun strings once (expensive: POS-tag every doc) ─────────────
    print("\nExtracting nouns from candidates ...")
    cand_noun_str: Dict[str, str] = {}
    for i, (cid, text) in enumerate(candidates.items()):
        if i % 200 == 0:
            print(f"  {i}/{len(candidates)}")
        cand_noun_str[cid] = doc_to_noun_string(text)

    print("Extracting nouns from queries ...")
    query_noun_str: Dict[str, str] = {}
    for i, (qid, text) in enumerate(queries.items()):
        if i % 50 == 0:
            print(f"  {i}/{len(queries)}")
        query_noun_str[qid] = doc_to_noun_string(text)

    cand_ids    = list(candidates.keys())
    cand_texts  = [cand_noun_str[c] for c in cand_ids]

    all_results = []

    for (min_df, max_df_frac, sublinear_tf) in CONFIGS:
        name = (f"Method1_TF-IDF_nouns_"
                f"mindf={min_df}_maxdf={max_df_frac}_sub={sublinear_tf}")
        print(f"\n{'─'*64}\n  {name}\n{'─'*64}")

        # Fit TF-IDF on candidate documents only (as in the paper)
        vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df_frac,
            sublinear_tf=sublinear_tf,
            norm="l2",           # cosine-ready (already L2 normalised)
        )
        cand_matrix = vectorizer.fit_transform(cand_texts)  # (C, V)

        # Retrieve
        results: Dict[str, List[str]] = {}
        for qid in relevance:
            if qid not in query_noun_str:
                continue
            qvec  = vectorizer.transform([query_noun_str[qid]])  # (1, V)
            sims  = cosine_similarity(qvec, cand_matrix)[0]      # (C,)
            order = np.argsort(-sims)[:args.top_k]
            results[qid] = [cand_ids[i] for i in order]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MAP",
                        title="Method-1 (Nouns + TF-IDF) — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()