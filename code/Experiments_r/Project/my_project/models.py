import numpy as np
from rank_bm25 import BM25Okapi, BM25Plus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

class PCRModelEnsemble:
    def __init__(self, use_bm25plus=False, svd_components=100, bm25_weight=0.7, lsa_weight=0.3):
        self.use_bm25plus = use_bm25plus
        self.svd_components = svd_components
        self.bm25_weight = bm25_weight
        self.lsa_weight = lsa_weight
        
        self.bm25_model = None
        self.tfidf_vectorizer = None
        self.svd_model = None
        
        self.candidate_ids = []
        self.candidate_lsa_vectors = None
        
    def fit(self, candidate_docs_tokens):
        """
        candidate_docs_tokens: dict of {doc_id: list_of_tokens}
        """
        self.candidate_ids = list(candidate_docs_tokens.keys())
        tokenized_corpus = [candidate_docs_tokens[doc_id] for doc_id in self.candidate_ids]
        
        print(f"Fitting BM25 model on {len(tokenized_corpus)} candidates...")
        if self.use_bm25plus:
            self.bm25_model = BM25Plus(tokenized_corpus)
        else:
            self.bm25_model = BM25Okapi(tokenized_corpus)
            
        print("Fitting TF-IDF and LSA models...")
        # Join tokens back to strings for sklearn
        text_corpus = [" ".join(tokens) for tokens in tokenized_corpus]
        
        self.tfidf_vectorizer = TfidfVectorizer(max_df=0.85, min_df=2)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_corpus)
        
        # Adjust SVD components if we have very few documents
        n_components = min(self.svd_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        if n_components < 2:
             n_components = 2 # fallback
        
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.candidate_lsa_vectors = self.svd_model.fit_transform(tfidf_matrix)
        print("Models fitted successfully.")

    def _normalize_scores(self, scores):
        """Min-Max normalization of a numpy array to [0, 1]"""
        if len(scores) == 0:
            return scores
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score == 0:
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)
        
    def get_scores(self, query_tokens):
        """
        Computes the ensemble score for a given query against all fitted candidates.
        Returns a sorted list of (candidate_id, score) in descending order.
        """
        # 1. BM25 Scores
        bm25_scores = self.bm25_model.get_scores(query_tokens)
        bm25_scores_norm = self._normalize_scores(bm25_scores)
        
        # 2. LSA Scores
        query_text = " ".join(query_tokens)
        query_tfidf = self.tfidf_vectorizer.transform([query_text])
        query_lsa = self.svd_model.transform(query_tfidf)
        
        # Cosine similarity between query and all candidates
        lsa_scores = cosine_similarity(query_lsa, self.candidate_lsa_vectors)[0]
        # Map from [-1, 1] to [0, 1] for normalization, or just standard min-max
        lsa_scores_norm = self._normalize_scores(lsa_scores)
        
        # 3. Ensemble Scores
        final_scores = (self.bm25_weight * bm25_scores_norm) + (self.lsa_weight * lsa_scores_norm)
        
        # 4. Sort and return
        results = []
        for idx, doc_id in enumerate(self.candidate_ids):
            results.append((doc_id, final_scores[idx]))
            
        # Sort descending by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results
