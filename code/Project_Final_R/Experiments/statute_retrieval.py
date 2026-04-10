"""
statute_retrieval.py
=====================
Statute and legal section extraction for Prior Case Retrieval.

Novel approach
--------------
Previous citation overlap used <CITATION_XXXXXXX> tags — case IDs.
This method extracts STATUTORY REFERENCES: specific sections of specific
Acts/Codes/Rules that are mentioned in the text.

Examples of statutory references extracted:
  "Section 498A of the Indian Penal Code"       → 498A_IPC
  "Article 226 of the Constitution"             → Art226_Constitution
  "Section 147 of the Income Tax Act"           → S147_ITA
  "Order XXXIX Rule 1 of the Code of Civil Procedure" → O39R1_CPC
  "Section 11 of the Arbitration Act"           → S11_Arb

Why this is better than citation overlap
-----------------------------------------
• Citation IDs (<CITATION_7530123>) are case references → already tried
• Statute references are TOPIC references: two cases citing Section 498A
  are both about domestic violence / matrimonial cruelty
• More cases share the same statute than the same prior case citation
  → higher coverage, more non-zero matches
• Statute Jaccard is SEMANTICALLY richer than case-ID Jaccard

Implementation
--------------
1. Parse every document with regex patterns for:
   - "Section/Sec/S. NNN [of the] ACT_NAME"
   - "Article NNN [of the] ACT_NAME"
   - "Order [ROMAN] Rule NNN"
   - Common abbreviations: IPC, CrPC, CPC, ITA, Arb, etc.

2. Build an inverted index: statute_id → set of document IDs

3. Similarity: Jaccard, IDF-weighted cosine, overlap count

4. Combination with 5-gram TF-IDF (established best method)

Run
---
    python3 statute_retrieval.py --data_dir /path/to/dataset/ --split train

Parameter grid
--------------
  Edit CONFIGS below.  Comment out any line to skip.
"""

import os
import re
import math
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import (
    load_split, clean_text, evaluate_all,
    save_results, print_results_table, save_results_csv, z_norm,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS  (edit here)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "./"
SPLIT    = "train"
TOP_K    = 1000
OUTPUT   = "results/statute_results.json"
K_VALUES = [5, 6, 7, 8, 9, 10, 11, 15, 20]

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGS  ← comment out any line to skip
# (statute_sim, tfidf_alpha, rerank_depth)
#
#   statute_sim : "jaccard" | "idf_cosine" | "overlap" | "bc"
#   tfidf_alpha : weight of 5-gram TF-IDF in final score
#                 (1-alpha = statute similarity weight)
#   rerank_depth: only rerank this many top TF-IDF results
#                 (1000 = all candidates)
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS: List[Tuple] = [
    # ── Standalone statute similarity ─────────────────────────────────────────
    ("jaccard",    0.0,  1000),
    ("idf_cosine", 0.0,  1000),
    ("bc",         0.0,  1000),
    ("overlap",    0.0,  1000),

    # ── Linear combination with 5-gram TF-IDF ────────────────────────────────
    ("jaccard",    0.9,  1000),
    ("jaccard",    0.8,  1000),
    ("jaccard",    0.7,  1000),
    ("jaccard",    0.6,  1000),
    ("jaccard",    0.5,  1000),
    ("idf_cosine", 0.9,  1000),
    ("idf_cosine", 0.8,  1000),
    ("idf_cosine", 0.7,  1000),
    ("idf_cosine", 0.6,  1000),
    ("idf_cosine", 0.5,  1000),
    ("bc",         0.9,  1000),
    ("bc",         0.8,  1000),
    ("bc",         0.7,  1000),
    ("bc",         0.6,  1000),
    ("overlap",    0.9,  1000),
    ("overlap",    0.8,  1000),
    ("overlap",    0.7,  1000),

    # ── Rerank only top-N from TF-IDF ─────────────────────────────────────────
    ("idf_cosine", 0.8,   50),
    ("idf_cosine", 0.8,  100),
    ("idf_cosine", 0.8,  200),
    ("jaccard",    0.8,   50),
    ("jaccard",    0.8,  100),
    ("jaccard",    0.7,   50),

    # ── Multiplicative boost ─────────────────────────────────────────────────
    ("boost_jac",  0.0,  1000),   # tfidf * (1 + jaccard)
    ("boost_jac",  0.0,   200),   # tfidf * (1 + jaccard), top-200 only
    ("boost_idf",  0.0,  1000),   # tfidf * (1 + idf_cosine)
    ("boost_idf",  0.0,   200),
]


# ─────────────────────────────────────────────────────────────────────────────
# STATUTE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

# Map of act name fragments → canonical abbreviation
_ACT_MAP = [
    (r'indian penal code|i\.p\.c', 'IPC'),
    (r'code of criminal procedure|cr\.?p\.?c', 'CrPC'),
    (r'code of civil procedure|c\.p\.c', 'CPC'),
    (r'income.?tax act', 'ITA'),
    (r'constitution of india|constitution', 'Const'),
    (r'companies act', 'CoAct'),
    (r'arbitration.{0,20}act', 'ArbAct'),
    (r'contract act', 'ContAct'),
    (r'evidence act', 'EvidAct'),
    (r'limitation act', 'LimAct'),
    (r'transfer of property act|t\.p\. act', 'TPA'),
    (r'civil procedure|civil rules', 'CPC'),
    (r'criminal procedure|criminal rules', 'CrPC'),
    (r'customs act', 'CustomsAct'),
    (r'excise act', 'ExciseAct'),
    (r'service act', 'ServiceAct'),
    (r'land acquisition', 'LandAcq'),
    (r'motor vehicles act', 'MVAct'),
]

# Roman numerals for Order extraction
_ROMAN = r'(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})'

# Main extraction patterns
_PATTERNS = [
    # Section NNN of THE ACT NAME
    re.compile(
        r'(?:section|sec\.?|s\.)\s*(\d+[A-Za-z]?(?:\s*\(\d+\))?)'
        r'(?:\s*(?:and|or|,|\s+)\s*(?:section|sec\.?)?\s*\d+[A-Za-z]?)*'
        r'(?:\s+(?:of\s+the\s+|of\s+))?'
        r'([A-Z][A-Za-z\s\.\,]{3,50})',
        re.IGNORECASE
    ),
    # Article NNN of Constitution
    re.compile(
        r'(?:article|art\.?)\s*(\d+[A-Za-z]?)'
        r'(?:\s+(?:of\s+the\s+|of\s+))?'
        r'([A-Z][A-Za-z\s\.\,]{3,50})',
        re.IGNORECASE
    ),
    # Order ROMAN Rule NNN
    re.compile(
        r'(?:order)\s*(' + _ROMAN + r')'
        r'\s+(?:rule|r\.?)\s*(\d+)',
        re.IGNORECASE
    ),
]


def _normalise_act(act_fragment: str) -> str:
    """Map a matched act name to canonical abbreviation."""
    s = act_fragment.strip().lower()
    for pattern, abbr in _ACT_MAP:
        if re.search(pattern, s):
            return abbr
    # Fallback: take first 3 uppercase words
    words = [w for w in act_fragment.split() if w and w[0].isupper()]
    return "_".join(words[:3]) if words else "UnknownAct"


def extract_statutes(text: str) -> Set[str]:
    """
    Extract normalised statute references from raw legal text.
    Returns a set of strings like "S147_ITA", "Art226_Const", "O39R1_CPC".
    """
    # Remove citation tags first
    text = re.sub(r'<[^>]+>', ' ', text)
    found: Set[str] = set()

    for pat in _PATTERNS:
        for m in pat.finditer(text):
            groups = m.groups()
            if len(groups) == 2:
                num_part, act_part = groups
                if num_part and act_part and len(act_part.strip()) > 3:
                    num  = re.sub(r'\s+', '', num_part).upper()
                    act  = _normalise_act(act_part)
                    found.add(f"S{num}_{act}")
            elif len(groups) == 2:
                # Order + Rule
                order, rule = groups
                found.add(f"O{order.strip()}_R{rule.strip()}_CPC")

    # Also extract bare "Section NNN IPC/CrPC/CPC" patterns (very common)
    for m in re.finditer(
            r'(?:section|sec\.?|s\.)\s*(\d+[A-Za-z]?)'
            r'\s+(?:of\s+(?:the\s+)?)?'
            r'(IPC|CrPC|CPC|ITA|CPC)',
            text, re.IGNORECASE):
        num, act = m.groups()
        found.add(f"S{num.upper()}_{act.upper()}")

    return found


# ─────────────────────────────────────────────────────────────────────────────
# SIMILARITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def statute_jaccard(a: Set[str], b: Set[str]) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0

def statute_bc(a: Set[str], b: Set[str]) -> float:
    if not a or not b: return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))

def statute_overlap(a: Set[str], b: Set[str]) -> float:
    return float(len(a & b))

def statute_idf_cosine(a: Set[str], b: Set[str],
                       idf: Dict[str, float]) -> float:
    if not a or not b: return 0.0
    dot = sum(idf.get(s, 1.0)**2 for s in a & b)
    na  = math.sqrt(sum(idf.get(s, 1.0)**2 for s in a))
    nb  = math.sqrt(sum(idf.get(s, 1.0)**2 for s in b))
    return dot / (na * nb) if na * nb else 0.0


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
    cand_ids = list(candidates.keys())
    all_results = []

    # ── Extract statutes ─────────────────────────────────────────────────────
    print("\nExtracting statutory references ...")
    q_stat:  Dict[str, Set[str]] = {}
    c_stat:  Dict[str, Set[str]] = {}
    for qid, text in queries.items():
        q_stat[qid] = extract_statutes(text)
    for cid, text in candidates.items():
        c_stat[cid] = extract_statutes(text)

    n_q = sum(1 for s in q_stat.values() if s)
    n_c = sum(1 for s in c_stat.values() if s)
    all_stats = {s for ss in c_stat.values() for s in ss}
    print(f"  Queries with statutes:    {n_q}/{len(q_stat)}")
    print(f"  Candidates with statutes: {n_c}/{len(c_stat)}")
    print(f"  Unique statute references: {len(all_stats)}")

    # Show most common statutes
    stat_freq = defaultdict(int)
    for ss in c_stat.values():
        for s in ss: stat_freq[s] += 1
    top10 = sorted(stat_freq.items(), key=lambda x: -x[1])[:10]
    print("  Top statutes:")
    for s, f in top10:
        print(f"    {s:<30} appears in {f} candidates")

    # IDF over statute references
    N = len(c_stat)
    stat_df = defaultdict(int)
    for ss in c_stat.values():
        for s in ss: stat_df[s] += 1
    stat_idf: Dict[str, float] = {
        s: math.log((N+1)/(df+1))+1.0 for s, df in stat_df.items()
    }

    # ── Build 5-gram TF-IDF (for hybrid) ─────────────────────────────────────
    print("\nBuilding 5-gram TF-IDF index ...")
    c_sw5 = {cid: " ".join(clean_text(t, True, 1)) for cid, t in candidates.items()}
    q_sw5 = {qid: " ".join(clean_text(t, True, 1)) for qid, t in queries.items()}

    vec5 = TfidfVectorizer(ngram_range=(5,5), min_df=2, max_df=0.95,
                           sublinear_tf=True, norm="l2")
    C5   = vec5.fit_transform([c_sw5[c] for c in cand_ids])

    tfidf5: Dict[str, Dict[str, float]] = {}
    tfidf5_ranked: Dict[str, List[Tuple[str, float]]] = {}
    for qid in relevance:
        if qid not in q_sw5: continue
        qvec = vec5.transform([q_sw5[qid]])
        sims = cosine_similarity(qvec, C5)[0]
        tfidf5[qid] = {cand_ids[i]: float(sims[i]) for i in range(len(cand_ids))}
        order = np.argsort(-sims)
        tfidf5_ranked[qid] = [(cand_ids[i], float(sims[i])) for i in order]

    # ── Run each config ───────────────────────────────────────────────────────
    for (statute_sim, tfidf_alpha, rerank_depth) in CONFIGS:
        boost  = statute_sim.startswith("boost")
        s_type = statute_sim.replace("boost_jac", "jaccard").replace("boost_idf", "idf_cosine")
        name   = (f"Statute_{statute_sim}_a={tfidf_alpha}_depth={rerank_depth}")
        print(f"\n{'─'*64}\n  {name}\n{'─'*64}")

        results: Dict[str, List[str]] = {}
        for qid in relevance:
            if qid not in tfidf5_ranked: continue

            qs   = q_stat.get(qid, set())
            pool = [c for c, _ in tfidf5_ranked[qid]]

            if tfidf_alpha == 0.0 and not boost:
                # Standalone: rank by statute sim only
                stat_sc = {}
                for cid in cand_ids:
                    cs = c_stat.get(cid, set())
                    if s_type == "jaccard":    stat_sc[cid] = statute_jaccard(qs, cs)
                    elif s_type == "bc":       stat_sc[cid] = statute_bc(qs, cs)
                    elif s_type == "overlap":  stat_sc[cid] = statute_overlap(qs, cs)
                    else:                      stat_sc[cid] = statute_idf_cosine(qs, cs, stat_idf)
                ranked = sorted(cand_ids, key=lambda c: stat_sc.get(c,0), reverse=True)
                results[qid] = ranked[:args.top_k]
                continue

            # Hybrid: combine TF-IDF score with statute score
            top_n = pool[:rerank_depth]
            rest  = pool[rerank_depth:]

            # Compute statute scores for pool
            stat_pool: Dict[str, float] = {}
            for cid in top_n:
                cs = c_stat.get(cid, set())
                if s_type == "jaccard":    stat_pool[cid] = statute_jaccard(qs, cs)
                elif s_type == "bc":       stat_pool[cid] = statute_bc(qs, cs)
                elif s_type == "overlap":  stat_pool[cid] = statute_overlap(qs, cs)
                else:                      stat_pool[cid] = statute_idf_cosine(qs, cs, stat_idf)

            if boost:
                # Multiplicative: tfidf * (1 + stat_sim)
                reranked = sorted(
                    top_n,
                    key=lambda c: tfidf5[qid].get(c,0)*(1+stat_pool.get(c,0)),
                    reverse=True
                )
            else:
                # Linear Z-normalised combination
                tf_z  = z_norm({c: tfidf5[qid].get(c,0) for c in top_n})
                st_z  = z_norm(stat_pool)
                combined = {c: tfidf_alpha*tf_z.get(c,0) +
                               (1-tfidf_alpha)*st_z.get(c,0) for c in top_n}
                reranked = sorted(top_n, key=lambda c: combined.get(c,0), reverse=True)

            results[qid] = reranked + rest

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MicroF1@10",
                        title="Statute Extraction — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
