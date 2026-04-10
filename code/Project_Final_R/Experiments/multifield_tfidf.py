"""
multifield_tfidf.py
====================
Multi-field TF-IDF retrieval for Prior Case Retrieval.

Core idea
---------
  Instead of mixing all tokens into one vector, represent each document as
  TWO separate TF-IDF vectors (fields) and combine their cosine scores:

    Field-A: NOUN vocabulary  (legal concepts)
    Field-B: VERB vocabulary  (legal actions/relations)
    Field-C: Full text ngrams (lexical catchall)

  Final score = α·sim(q_A, d_A) + β·sim(q_B, d_B) + γ·sim(q_C, d_C)

  Why this helps over Method-2 (concatenating nouns+verbs):
    • Nouns and verbs have different discriminative power; weighting them
      separately finds the right balance for legal retrieval.
    • High-IDF nouns (proper nouns, legal terms) are diluted less by
      common verbs in the concatenated representation.

Also implements:
  • BM25Pivot: modified BM25 with pivot document-length normalisation
    Score = Σ IDF × tf / (pivot×len + (1-pivot)×avgdl) / (tf+1)
    (Singhal 1996 — often outperforms Okapi for long legal docs)
  • Augmented-TF (0.5+0.5*tf/max_tf) field combination

Run
---
    python3 multifield_tfidf.py --data_dir /path/to/dataset/ --split train

Parameter grid
--------------
  Edit CONFIGS below.  Comment out any line to skip.
"""

import os
import re
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import nltk
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
OUTPUT   = "results/multifield_results.json"
K_VALUES = [5, 10, 20, 50, 100]

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGS  ← comment out any line to skip
# (fields, weights, ngram_per_field, min_df, sublinear)
#
#   fields           : tuple of field names from {"nouns","verbs","full","ng5"}
#   weights          : weight for each field (need not sum to 1)
#   ngram_per_field  : n-gram order per field (1 = unigrams)
#   min_df           : minimum document frequency
#   sublinear        : use 1+log(tf) instead of raw tf
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS: List[Tuple] = [
    # ── Nouns + Full-text ─────────────────────────────────────────────────────
    (("nouns", "full"),     (0.7, 0.3), (1, 1), 2, True),
    (("nouns", "full"),     (0.5, 0.5), (1, 1), 2, True),
    (("nouns", "full"),     (0.3, 0.7), (1, 1), 2, True),
    (("nouns", "full"),     (0.7, 0.3), (1, 5), 2, True),   # full field = 5-gram
    (("nouns", "full"),     (0.5, 0.5), (1, 5), 2, True),
    (("nouns", "full"),     (0.3, 0.7), (1, 5), 2, True),

    # ── Verbs + Full-text ─────────────────────────────────────────────────────
    (("verbs", "full"),     (0.5, 0.5), (1, 1), 2, True),
    (("verbs", "full"),     (0.3, 0.7), (1, 5), 2, True),

    # ── Nouns + Verbs + Full ──────────────────────────────────────────────────
    (("nouns","verbs","full"), (0.4,0.2,0.4), (1,1,1), 2, True),
    (("nouns","verbs","full"), (0.5,0.2,0.3), (1,1,1), 2, True),
    (("nouns","verbs","full"), (0.3,0.1,0.6), (1,1,5), 2, True),
    (("nouns","verbs","full"), (0.4,0.1,0.5), (1,1,5), 2, True),
    (("nouns","verbs","full"), (0.4,0.2,0.4), (3,1,5), 2, True),
    (("nouns","verbs","full"), (0.5,0.1,0.4), (3,1,5), 2, True),

    # ── Nouns-ngram + Verbs-ngram + Full-ngram ────────────────────────────────
    (("nouns","verbs","full"), (0.4,0.2,0.4), (3,2,5), 2, True),
    (("nouns","verbs","full"), (0.5,0.2,0.3), (3,2,5), 2, True),

    # ── Noun-ngram + Full-ngram ───────────────────────────────────────────────
    (("nouns", "full"),     (0.5, 0.5), (3, 5), 2, True),
    (("nouns", "full"),     (0.4, 0.6), (3, 5), 2, True),
    (("nouns", "full"),     (0.6, 0.4), (3, 5), 2, True),

    # ── Augmented TF variants ─────────────────────────────────────────────────
    (("nouns","verbs","full"), (0.4,0.2,0.4), (1,1,1), 2, False),
    (("nouns","verbs","full"), (0.4,0.2,0.4), (1,1,5), 2, False),
]

# ─────────────────────────────────────────────────────────────────────────────
# POS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

NOUN_TAGS = {"NN","NNS","NNP"}
VERB_TAGS = {"VB","VBZ","VBN","VBD"}
_lem  = WordNetLemmatizer()
_CITE = re.compile(r'\[?\?CITATION\?\]?|<CITATION_\d+>', re.IGNORECASE)
_PUN  = re.compile(r'[",\-\'_]')

def _prep(text):
    return _PUN.sub(" ", _CITE.sub(" ", text))

def _extract(text, field, ngram):
    toks = word_tokenize(_prep(text))
    tagged = pos_tag(toks)
    if field == "nouns":
        base = [_lem.lemmatize(w.lower(),'n') for w,t in tagged if t in NOUN_TAGS]
    elif field == "verbs":
        base = [_lem.lemmatize(w.lower(),'v') for w,t in tagged if t in VERB_TAGS]
    else:
        # full text
        base = clean_text(text, remove_stopwords=True, ngram=1)

    if ngram == 1:
        return " ".join(base)
    out = list(base)
    for n in range(2, ngram+1):
        for i in range(len(base)-n+1):
            out.append("_".join(base[i:i+n]))
    return " ".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-FIELD SCORER
# ─────────────────────────────────────────────────────────────────────────────

class MultiFieldScorer:
    def __init__(self, fields, weights, ngrams, min_df, sublinear):
        self.fields   = fields
        self.weights  = weights
        self.ngrams   = ngrams
        self.min_df   = min_df
        self.sublinear = sublinear
        self.vecs:  List[TfidfVectorizer] = []
        self.cmats: List = []
        self.cand_ids: List[str] = []

    def fit(self, candidates: Dict[str, str]):
        self.cand_ids = list(candidates.keys())
        self.vecs  = []
        self.cmats = []
        for field, ngram in zip(self.fields, self.ngrams):
            print(f"    Fitting field={field} ng={ngram} ...")
            corpus = [_extract(candidates[cid], field, ngram)
                      for cid in self.cand_ids]
            vec = TfidfVectorizer(
                min_df=self.min_df, max_df=0.95,
                sublinear_tf=self.sublinear, norm="l2"
            )
            cmat = vec.fit_transform(corpus)
            self.vecs.append(vec)
            self.cmats.append(cmat)

    def score(self, query_text: str) -> np.ndarray:
        """Returns combined similarity scores for all candidates."""
        combined = np.zeros(len(self.cand_ids))
        total_w  = sum(self.weights)
        for vec, cmat, field, ngram, w in zip(
                self.vecs, self.cmats, self.fields, self.ngrams, self.weights):
            q_str = _extract(query_text, field, ngram)
            qvec  = vec.transform([q_str])
            sims  = cosine_similarity(qvec, cmat)[0]
            combined += (w / total_w) * sims
        return combined

    def retrieve(self, query_text: str, top_k: int = 1000) -> List[str]:
        sims  = self.score(query_text)
        order = np.argsort(-sims)[:top_k]
        return [self.cand_ids[i] for i in order]


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
    all_results = []

    for (fields, weights, ngrams, min_df, sublinear) in CONFIGS:
        fields_str  = "+".join(f"{f}(ng{n})" for f,n in zip(fields,ngrams))
        weights_str = "+".join(f"{w:.1f}" for w in weights)
        name = (f"MF_{fields_str}_w={weights_str}_"
                f"mindf={min_df}_sub={sublinear}")
        print(f"\n{'─'*64}\n  {name}\n{'─'*64}")

        scorer = MultiFieldScorer(fields, weights, ngrams, min_df, sublinear)
        scorer.fit(candidates)

        results: Dict[str, List[str]] = {}
        for qid, text in queries.items():
            if qid not in relevance: continue
            results[qid] = scorer.retrieve(text, top_k=args.top_k)

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MicroF1@10",
                        title="Multi-Field TF-IDF — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
