import os
import re
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from rank_bm25 import BM25Okapi
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import scipy.sparse as sp

try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except:
    os.system("python3 -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

class FastBM25:
    """ Matrix-based BM25 implementation for maximum speed """
    def __init__(self, k1=1.6, b=0.75):
        self.k1 = k1
        self.b = b
        self.vectorizer = TfidfVectorizer(max_df=0.75, min_df=1, use_idf=True, ngram_range=(1,1))
        
    def fit(self, X_text):
        self.vectorizer.fit(X_text)
        y = super(TfidfVectorizer, self.vectorizer).transform(X_text)
        self.avdl = y.sum(1).mean()
        self.X_transformed = y
        self.len_X = self.X_transformed.sum(1).A1

    def get_scores(self, q_text):
        if not q_text.strip():
            return np.zeros(self.X_transformed.shape[0])
            
        q = super(TfidfVectorizer, self.vectorizer).transform([q_text])
        if not sp.isspmatrix_csr(q): q = q.tocsr()
        X = self.X_transformed.tocsc()[:, q.indices]
        denom = X + (self.k1 * (1 - self.b + self.b * self.len_X / self.avdl))[:, None]
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (self.k1 + 1)
        return (numer / denom).sum(1).A1


class TFIDFModel:
    def __init__(self, ngram=(1,2)):
        self.vectorizer = TfidfVectorizer(max_df=0.75, min_df=1, ngram_range=ngram)
        
    def fit(self, X_text):
        self.doc_vectors = self.vectorizer.fit_transform(X_text)
        
    def get_scores(self, q_text):
        q_vec = self.vectorizer.transform([q_text])
        return cosine_similarity(q_vec, self.doc_vectors)[0]


class LSAModel:
    def __init__(self, n_components=100):
        self.tfidf = TfidfVectorizer(max_df=0.75, min_df=2)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        
    def fit(self, X_text):
        tfidf_mat = self.tfidf.fit_transform(X_text)
        self.doc_vectors = self.svd.fit_transform(tfidf_mat)
        
    def get_scores(self, q_text):
        q_vec = self.svd.transform(self.tfidf.transform([q_text]))
        return cosine_similarity(q_vec, self.doc_vectors)[0]


class Doc2VecModel:
    def __init__(self, vector_size=100, epochs=10):
        self.vector_size = vector_size
        self.epochs = epochs
        
    def fit(self, X_text):
        tagged_data = [TaggedDocument(words=doc.lower().split(), tags=[str(i)]) for i, doc in enumerate(X_text)]
        self.model = Doc2Vec(vector_size=self.vector_size, window=5, min_count=2, workers=4, epochs=self.epochs)
        self.model.build_vocab(tagged_data)
        self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        self.doc_vectors = np.array([self.model.dv[str(i)] for i in range(len(X_text))])
        
    def get_scores(self, q_text):
        q_vec = self.model.infer_vector(q_text.lower().split())
        return cosine_similarity([q_vec], self.doc_vectors)[0]


class ConceptTFIDFModel:
    """ Mimics SOTA Event Extraction by keeping only Nouns, Verbs, and Entities """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, ngram_range=(1,2), token_pattern=r"(?u)\b[\w\d_]+\b")

    def fit(self, X_concepts):
        self.doc_vectors = self.vectorizer.fit_transform(X_concepts)
        
    def get_scores(self, q_concepts):
        q_vec = self.vectorizer.transform([q_concepts])
        return cosine_similarity(q_vec, self.doc_vectors)[0]


class GraphDiffusionModel:
    """ Uses TFIDF seed scores and diffuses them through a Candidate Citation Adjacency Graph """
    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.base_model = TFIDFModel(ngram=(1,2))
        self.entity_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w\d_]+\b")
        
    def fit(self, X_text, X_entities):
        self.base_model.fit(X_text)
        
        # Build Transition Matrix
        cand_entity_matrix = self.entity_vectorizer.fit_transform(X_entities)
        W = cosine_similarity(cand_entity_matrix)
        np.fill_diagonal(W, 0) # No self loops
        self.T = normalize(W, norm='l1', axis=1) # Row normalize transition probabilities
        
    def get_scores(self, q_text):
        S_0 = self.base_model.get_scores(q_text)
        return (1 - self.alpha) * S_0 + self.alpha * np.dot(self.T, S_0)
