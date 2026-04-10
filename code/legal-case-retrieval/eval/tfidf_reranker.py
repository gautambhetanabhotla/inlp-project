#!/usr/bin/env python3
"""
tfidf_reranker.py - Rerank dense retrieval top-200 candidates using TF-IDF similarity.

This script:
1. Takes dense retrieval top-200 predictions
2. Extracts TF-IDF vectors for query and its candidate pool
3. Reranks candidates using TF-IDF cosine similarity
4. Returns improved ranking
"""

import json
import re
import math
import argparse
import csv
import ast
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def clean_text(text: str, remove_stopwords: bool = True) -> List[str]:
    """Clean and tokenize text."""
    # Lowercase and remove non-alphanumeric
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
    # Split into tokens
    tokens = text.split()
    
    # Remove stopwords
    if remove_stopwords:
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'can', 'may', 'might', 'must', 'as', 'if', 'not',
            'that', 'this', 'which', 'who', 'whom', 'what', 'where', 'when', 'why',
            'how', 'all', 'each', 'every', 'both', 'either', 'neither', 'any',
            'some', 'no', 'nor', 'he', 'she', 'it', 'we', 'they', 'you', 'me', 'us',
        }
        tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
    
    return tokens


def cosine_sim(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Compute cosine similarity between two TF-IDF vectors."""
    dot_product = sum(vec1.get(t, 0.0) * vec2.get(t, 0.0) for t in vec1)
    norm1 = (sum(v * v for v in vec1.values())) ** 0.5
    norm2 = (sum(v * v for v in vec2.values())) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


class TFIDFReranker:
    """TF-IDF reranker for candidate lists."""
    
    def __init__(self):
        self.all_vectors = {}  # {case_id: {term: tfidf_score}}
        self.idf = {}
        self.vocab = set()
    
    def fit(self, corpus: Dict[str, List[str]]):
        """Build TF-IDF index from corpus of tokenized text."""
        # Compute IDF
        df = defaultdict(int)
        raw_tfs = {}
        
        for doc_id, tokens in corpus.items():
            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            raw_tfs[doc_id] = tf
            for t in tf:
                df[t] += 1
        
        N = len(corpus)
        self.idf = {t: math.log((N + 1) / (count + 1)) + 1.0 for t, count in df.items()}
        self.vocab = set(self.idf.keys())
        
        # Compute TF-IDF vectors with L2 normalization
        for doc_id, tf in raw_tfs.items():
            max_tf = max(tf.values()) if tf else 1
            vec = {}
            for t, count in tf.items():
                if t in self.vocab:
                    tf_score = 1.0 + math.log(count) if count > 0 else 0.0  # log normalization
                    vec[t] = tf_score * self.idf[t]
            
            # L2 normalize
            norm = math.sqrt(sum(v * v for v in vec.values()))
            if norm > 0:
                vec = {t: v / norm for t, v in vec.items()}
            self.all_vectors[doc_id] = vec
    
    def get_vector(self, tokens: List[str]) -> Dict[str, float]:
        """Get TF-IDF vector for a list of tokens."""
        tf = defaultdict(int)
        for t in tokens:
            if t in self.vocab:
                tf[t] += 1
        
        vec = {}
        for t, count in tf.items():
            tf_score = 1.0 + math.log(count) if count > 0 else 0.0
            vec[t] = tf_score * self.idf.get(t, 0.0)
        
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            vec = {t: v / norm for t, v in vec.items()}
        
        return vec
    
    def rerank(self, query_tokens: List[str], candidates: List[str]) -> List[Tuple[str, float]]:
        """Rerank candidates using TF-IDF similarity to query."""
        query_vec = self.get_vector(query_tokens)
        if not query_vec:
            return [(c, 0.0) for c in candidates]
        
        scores = []
        for cand_id in candidates:
            if cand_id in self.all_vectors:
                sim = cosine_sim(query_vec, self.all_vectors[cand_id])
                scores.append((cand_id, sim))
            else:
                scores.append((cand_id, 0.0))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


def main():
    parser = argparse.ArgumentParser(description="TF-IDF reranker for dense retrieval candidates")
    parser.add_argument("--dense-pred", default="../retrieval-predictions-dense-top200.json",
                        help="Dense retrieval predictions (top-200)")
    parser.add_argument("--query-dir", default="dataset/ik_test 4/query",
                        help="Directory with query .txt files")
    parser.add_argument("--cand-dir", default="dataset/ik_test 4/candidate",
                        help="Directory with candidate .txt files")
    parser.add_argument("--out", default="../retrieval-predictions-tfidf-rerank.json",
                        help="Output reranked predictions")
    parser.add_argument("--top-k", type=int, default=200,
                        help="Keep top-k after reranking")
    args = parser.parse_args()
    
    # Load dense predictions
    print("Loading dense predictions...")
    with open(args.dense_pred, 'r', encoding='utf-8') as f:
        dense_preds = json.load(f)
    
    # Load query and candidate texts
    print("Loading text files...")
    query_texts = {}
    for p in Path(args.query_dir).glob('*.txt'):
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            query_texts[p.name] = f.read()
    
    cand_texts = {}
    for p in Path(args.cand_dir).glob('*.txt'):
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            cand_texts[p.name] = f.read()
    
    # Build TF-IDF index on all candidates
    print("Building TF-IDF index on candidates...")
    all_tokens = {cand_id: clean_text(text) for cand_id, text in cand_texts.items()}
    reranker = TFIDFReranker()
    reranker.fit(all_tokens)
    
    # Rerank each query's candidates
    print("Reranking candidates for each query...")
    reranked_preds = {}
    for idx, (query_id, dense_candidates) in enumerate(dense_preds.items()):
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(dense_preds)}")
        
        if query_id not in query_texts:
            reranked_preds[query_id] = dense_candidates[:args.top_k]
            continue
        
        query_tokens = clean_text(query_texts[query_id])
        reranked = reranker.rerank(query_tokens, dense_candidates)
        reranked_preds[query_id] = [cand_id for cand_id, _ in reranked[:args.top_k]]
    
    # Save reranked predictions
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(reranked_preds, f, indent=2)
    
    print(f"Reranked predictions saved to: {args.out}")
    print(f"Queries: {len(reranked_preds)}")


if __name__ == "__main__":
    main()
