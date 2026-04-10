"""
method2_tfidf_nouns_verbs.py
=============================
Method-2 from:
  "A Text Similarity Approach for Precedence Retrieval from Legal Documents"
  Thenmozhi, Kawshik Kannan, Aravindan — IRLeD@FIRE2017

Pipeline
--------
  1. Preprocess: strip punctuation and citation markers
  2. POS-tag each document
  3. Extract NOUNS (NN, NNS, NNP)  → "concepts"
     Extract VERBS (VB, VBZ, VBN, VBD) → "relations"
  4. Lemmatize extracted terms (WordNet Lemmatizer)
  5. Build TF-IDF feature vectors over the noun+verb vocabulary
  6. Compute cosine similarity between each query and every candidate
  7. Rank candidates by similarity score (descending)

Key difference from Method-1: verbs are added as features alongside nouns.
The paper reports this gives the best MAP (0.2677) and MRR (0.5457).

Run
---
    python3 method2_tfidf_nouns_verbs.py --data_dir /path/to/dataset/ --split train

Parameter grid
--------------
  Edit CONFIGS below.  Comment out any line to skip.
"""

import os
import re
import argparse
from typing import Dict, List, Tuple

import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from utils import (
    load_split, evaluate_all,
    save_results, print_results_table, save_results_csv,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS  (edit here)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "./"
SPLIT    = "train"
TOP_K    = 1000
OUTPUT   = "results/method2_results.json"
K_VALUES = [5, 10, 20, 50, 100]

# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def download_nltk_dependencies():
    """Ensure all required NLTK resources are downloaded, including updated tagger names."""
    dependencies = [
        'punkt', 
        'punkt_tab',                    
        'averaged_perceptron_tagger', 
        'averaged_perceptron_tagger_eng', 
        'wordnet',
        'omw-1.4'
    ]
    for dep in dependencies:
        try:
            if dep in ['punkt', 'punkt_tab']:
                nltk.data.find(f'tokenizers/{dep}')
            elif 'tagger' in dep:
                nltk.data.find(f'taggers/{dep}')
            else:
                nltk.data.find(f'corpora/{dep}')
        except LookupError:
            print(f"Downloading NLTK dependency: {dep}...")
            nltk.download(dep, quiet=True)

# Run dependency check immediately
download_nltk_dependencies()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGS  ← comment out any line to skip
# Each entry: (min_df, max_df_frac, sublinear_tf)
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS: List[Tuple] = [
    # ── Paper default ─────────────────────────────────────────────────────────
    (1,  1.00, False),   # nouns + verbs, plain TF-IDF — paper Method-2
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
# POS TAG SETS (from the paper)
# ─────────────────────────────────────────────────────────────────────────────

# Concepts: all noun forms
NOUN_TAGS = {"NN", "NNS", "NNP"}

# Relations: all verb forms the paper lists
VERB_TAGS = {"VB", "VBZ", "VBN", "VBD"}

KEEP_TAGS = NOUN_TAGS | VERB_TAGS

_lemmatizer  = WordNetLemmatizer()
_PUNCT_RE    = re.compile(r'[",\-\'_]')
_CITATION_RE = re.compile(r'\[?\?CITATION\?\]?|<CITATION_\d+>', re.IGNORECASE)

# Mapping from Penn POS tag prefix → WordNet POS for lemmatizer
_TAG_TO_WN = {
    "NN": "n",    # noun
    "VB": "v",    # verb
}


def _wn_pos(tag: str) -> str:
    """Map Penn Treebank POS tag to WordNet POS for the lemmatizer."""
    if tag in NOUN_TAGS:
        return "n"
    if tag in VERB_TAGS:
        return "v"
    return "n"   # default fallback


def preprocess(text: str) -> str:
    text = _CITATION_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    return text


def extract_concepts_and_relations(text: str) -> List[str]:
    """
    POS-tag and return lemmatized nouns (concepts) + lemmatized verbs (relations).
    Mirrors the paper's Method-2 feature extraction.
    """
    text   = preprocess(text)
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    terms  = [
        _lemmatizer.lemmatize(word.lower(), pos=_wn_pos(tag))
        for word, tag in tagged
        if tag in KEEP_TAGS
    ]
    return terms


def doc_to_feature_string(text: str) -> str:
    return " ".join(extract_concepts_and_relations(text))


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

    # ── Extract noun+verb strings once ───────────────────────────────────────
    print("\nExtracting concepts+relations from candidates ...")
    cand_feat_str: Dict[str, str] = {}
    for i, (cid, text) in enumerate(candidates.items()):
        if i % 200 == 0:
            print(f"  {i}/{len(candidates)}")
        cand_feat_str[cid] = doc_to_feature_string(text)

    print("Extracting concepts+relations from queries ...")
    query_feat_str: Dict[str, str] = {}
    for i, (qid, text) in enumerate(queries.items()):
        if i % 50 == 0:
            print(f"  {i}/{len(queries)}")
        query_feat_str[qid] = doc_to_feature_string(text)

    cand_ids   = list(candidates.keys())
    cand_texts = [cand_feat_str[c] for c in cand_ids]

    all_results = []

    for (min_df, max_df_frac, sublinear_tf) in CONFIGS:
        name = (f"Method2_TF-IDF_nouns+verbs_"
                f"mindf={min_df}_maxdf={max_df_frac}_sub={sublinear_tf}")
        print(f"\n{'─'*64}\n  {name}\n{'─'*64}")

        vectorizer  = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df_frac,
            sublinear_tf=sublinear_tf,
            norm="l2",
        )
        cand_matrix = vectorizer.fit_transform(cand_texts)

        results: Dict[str, List[str]] = {}
        for qid in relevance:
            if qid not in query_feat_str:
                continue
            qvec  = vectorizer.transform([query_feat_str[qid]])
            sims  = cosine_similarity(qvec, cand_matrix)[0]
            order = np.argsort(-sims)[:args.top_k]
            results[qid] = [cand_ids[i] for i in order]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MAP",
                        title="Method-2 (Nouns+Verbs + TF-IDF) — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()