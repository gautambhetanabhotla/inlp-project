import os
import json
import re
import pickle
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import minmax_scale, normalize
import spacy

try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except:
    os.system("python3 -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# Paths
DATASET_DIR = "../../Dataset"
EVAL_DIR = "../../Eval"
TEST_GOLD_JSON = os.path.join(EVAL_DIR, "ilpcr-test-gold.json")
CAND_LIST_TXT = os.path.join(EVAL_DIR, "ilpcr-test-cand.txt")
CACHE_DIR = "cache"
RESULTS_DIR = "results"

# ---------------------------- MODEL CLASSES ---------------------------- #

class RawBM25:
    """ Fast sparse matrix BM25 implementation """
    def __init__(self, k1=1.6, b=0.75):
        self.k1 = k1
        self.b = b
        self.vectorizer = TfidfVectorizer(max_df=0.65, min_df=1, use_idf=True, ngram_range=(1,1))
        
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
    def __init__(self, ngram_range=(1,2)):
        self.vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, ngram_range=ngram_range)
    
    def fit(self, X_text):
        self.candidate_vectors = self.vectorizer.fit_transform(X_text)
        
    def get_scores(self, q_text):
        q_vec = self.vectorizer.transform([q_text])
        return cosine_similarity(q_vec, self.candidate_vectors)[0]

class BaseTFIDF:
    """ Base TF-IDF model tailored for diffusion """
    def __init__(self, ngram_range=(1,2)):
        self.vectorizer = TfidfVectorizer(max_df=0.75, min_df=1, ngram_range=ngram_range)
        
    def fit(self, X_text):
        self.candidate_vectors = self.vectorizer.fit_transform(X_text)
        
    def get_scores(self, q_text):
        q_vec = self.vectorizer.transform([q_text])
        return cosine_similarity(q_vec, self.candidate_vectors)[0]


class LSAModel:
    def __init__(self, n_components=300):
        self.vectorizer = TfidfVectorizer(max_df=0.8, min_df=2)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        
    def fit(self, X_text):
        tfidf = self.vectorizer.fit_transform(X_text)
        components = min(self.svd.n_components, tfidf.shape[0]-1, tfidf.shape[1]-1)
        if components < 2: components = 2
        self.svd.n_components = components
        self.candidate_vectors = self.svd.fit_transform(tfidf)
        
    def get_scores(self, q_text):
        tfidf = self.vectorizer.transform([q_text])
        q_vec = self.svd.transform(tfidf)
        return cosine_similarity(q_vec, self.candidate_vectors)[0]


# ---------------------------- TEXT PROCESSORS ---------------------------- #

def segment_into_sentences(text):
    lines = text.replace('\n', ' ').replace('\t', ' ')
    lines = lines.split(". ")
    return [l.strip() for l in lines if len(l.strip()) > 10]

def extract_key_concepts(text):
    """ Extracts pure nouns/verbs (Event Concepts) for Sentence Filtering. """
    doc = nlp(text)
    concepts = set()
    for token in doc:
        if token.is_stop or token.is_punct or len(token.lemma_) <= 2: continue
        if token.pos_ in ["NOUN", "PROPN", "VERB"]:
            concepts.add(token.lemma_.lower())
    return concepts

def extract_legal_concepts_unified(text):
    """ Extracts Legal Citations (Regex) + Strong POS Tags (SpaCy) for Hybrid Concept TFIDF. """
    textStr = text.lower()
    concepts = []
    
    citations = re.findall(r'\b\d{4}\s+(?:scc|air|scr|ilr|scale|jt)\s+(?:sc|hc)?\s*\d+\b', textStr)
    statutes = re.findall(r'\b(?:section|sec\.|article|art\.|rule|order)\s+[a-z0-9ivx]+\b', textStr)
    acts = re.findall(r'\b(?:indian penal code|ipc|crpc|cpc|evidence act|constitution(?:\s+of\s+india)?|income tax act)\b', textStr)
    
    concepts.extend([c.replace(" ", "_") for c in citations])
    concepts.extend([s.replace(" ", "_") for s in statutes])
    concepts.extend([a.replace(" ", "_") for a in acts])
    
    max_chunk = 500000 
    text_chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    
    for chunk in text_chunks:
        doc = nlp(chunk)
        for token in doc:
            if token.is_stop or token.is_punct or len(token.lemma_) <= 2: continue
            if token.pos_ in ["NOUN", "PROPN", "VERB"] and not token.lemma_.isnumeric():
                concepts.append(token.lemma_.lower())
                
    return " ".join(concepts)

def extract_legal_entities(text):
    """
    Regex-based Legal Named Entity Recognition (L-NER) strictly for Graphs/Citations.
    """
    text = text.lower()
    entities = []
    
    citations = re.findall(r'\b\d{4}\s+(?:scc|air|scr|ilr|scale|jt)\s+(?:sc|hc)?\s*\d+\b', text)
    entities.extend([c.replace(" ", "_") for c in citations])
    statutes = re.findall(r'\b(?:section|sec\.|article|art\.|rule|order)\s+[a-z0-9ivx]+\b', text)
    entities.extend([s.replace(" ", "_") for s in statutes])
    acts = re.findall(r'\b(?:indian penal code|ipc|crpc|cpc|evidence act|constitution.*?india|income tax act)\b', text)
    entities.extend([a.replace(" ", "_") for a in acts])
    
    return " ".join(entities)

def text_to_standard_tokens(text):
    """ Standard preprocessing: lowercase, remove non-alpha, lemmatize, remove stopwords. """
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    nlp.max_length = max(len(text) + 100, 3000000)
    doc = nlp(text)
    return [t.lemma_ for t in doc if not t.is_stop and len(t.lemma_) > 2]

# ---------------------------- EVALUATION ---------------------------- #

def evaluate_predictions(predictions, gold_dict, valid_cands, k_values=[1, 5, 10, 20], silent=False):
    def get_metrics_at_k(g, p, k):
        gs = set(g)
        ps_list = p[:k]
        ps_set = set(ps_list)
        
        num_hits_micro = len(gs & ps_set)
        num_pred_micro = len(ps_set)
        num_gold_micro = len(gs)
        
        micro_stats = (num_hits_micro, num_pred_micro, num_gold_micro)
        
        if len(gs) == 0:
            return micro_stats, None
            
        hits = [1 if doc in gs else 0 for doc in ps_list]
        num_hits = sum(hits)
        
        prec = num_hits / k if k > 0 else 0.0
        rec = num_hits / len(gs) if len(gs) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        ap = 0.0
        hits_so_far = 0
        for i, h in enumerate(hits):
            if h == 1:
                hits_so_far += 1
                ap += hits_so_far / (i + 1)
        ap /= min(len(gs), k) if min(len(gs), k) > 0 else 1.0
        
        rr = 0.0
        for i, h in enumerate(hits):
            if h == 1:
                rr = 1.0 / (i + 1)
                break
                
        dcg = sum([h / np.log2(i + 2) for i, h in enumerate(hits)])
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(gs), k))])
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        macro_stats = {
            'prec': prec,
            'rec': rec,
            'f1': f1,
            'ap': ap,
            'rr': rr,
            'ndcg': ndcg
        }
        
        return micro_stats, macro_stats

    metrics_by_k = {}
    for k in k_values:
        total_hits = 0
        total_pred = 0
        total_gold = 0
        macro_results = []
        
        for doc_id, cands in gold_dict.items():
            pred = predictions.get(doc_id, [])
            cands = [c for c in cands if c in valid_cands and c != doc_id]
            pred = [c for c in pred if c in valid_cands and c != doc_id]
            
            micro_stats, macro_stats = get_metrics_at_k(cands, pred, k)
            
            total_hits += micro_stats[0]
            total_pred += micro_stats[1]
            total_gold += micro_stats[2]
            
            if macro_stats is not None:
                macro_results.append(macro_stats)
                
        micro_prec = total_hits / total_pred if total_pred > 0 else 0.0
        micro_rec = total_hits / total_gold if total_gold > 0 else 0.0
        micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
        
        if macro_results:
            macro_prec = sum(r['prec'] for r in macro_results) / len(macro_results)
            macro_rec = sum(r['rec'] for r in macro_results) / len(macro_results)
            macro_f1 = sum(r['f1'] for r in macro_results) / len(macro_results)
            map_k = sum(r['ap'] for r in macro_results) / len(macro_results)
            mrr_k = sum(r['rr'] for r in macro_results) / len(macro_results)
            ndcg_k = sum(r['ndcg'] for r in macro_results) / len(macro_results)
        else:
            macro_prec = macro_rec = macro_f1 = map_k = mrr_k = ndcg_k = 0.0
            
        metrics_by_k[k] = {
            'Micro-P': micro_prec,
            'Micro-R': micro_rec,
            'Micro-F1': micro_f1,
            'Macro-P': macro_prec,
            'Macro-R': macro_rec,
            'Macro-F1': macro_f1,
            'MAP': map_k,
            'MRR': mrr_k,
            'NDCG': ndcg_k
        }
        
    if not silent:
        print(f"\n{'K':>2} | {'Micro-F1':>8} | {'Micro-P':>8} | {'Micro-R':>8} | {'Macro-F1':>8} | {'MAP':>8} | {'MRR':>8} | {'NDCG':>8}")
        print("-" * 88)
        for k in k_values:
            m = metrics_by_k[k]
            print(f"{k:2} | {m['Micro-F1']:.4f}   | {m['Micro-P']:.4f}   | {m['Micro-R']:.4f}   | {m['Macro-F1']:.4f}   | {m['MAP']:.4f}   | {m['MRR']:.4f}   | {m['NDCG']:.4f}")
    
    return metrics_by_k


def load_raw_documents(folder_path):
    docs = {}
    filenames = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    for filename in tqdm(filenames, desc=f"Reading {os.path.basename(folder_path)}"):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
            docs[filename] = f.read()
    return docs

def load_preprocessed_documents(directory, cache_path):
    if os.path.exists(cache_path):
        print(f"Loading cached Standard NLP tokens from {cache_path}...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
            
    print(f"Cache {cache_path} not found. Running Standard NLP PREPROCESSING from {directory}...")
    docs, tokens_dict = {}, {}
    filenames = [f for f in os.listdir(directory) if f.endswith(".txt")]
    for filename in tqdm(filenames, desc=f"Processing {os.path.basename(directory)}"):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            docs[filename] = text
            tokens_dict[filename] = text_to_standard_tokens(text)
            
    os.makedirs("cache", exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump((docs, tokens_dict), f)
    return docs, tokens_dict

def generate_or_load_concepts(doc_dict, cache_file, desc):
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
            
    print(f"Cache {cache_file} not found. Running CONCEPT EXTRACTION...")
    concepts_dict = {}
    for doc_id, text in tqdm(doc_dict.items(), desc=desc):
        concepts_dict[doc_id] = extract_legal_concepts_unified(text)
        
    with open(cache_file, "wb") as f:
        pickle.dump(concepts_dict, f)
        
    return concepts_dict

# ---------------------------- MAIN EXECUTION ---------------------------- #

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Load Gold standard and Cands
    cand_dir = os.path.join(DATASET_DIR, "ik_test", "candidate")
    q_dir = os.path.join(DATASET_DIR, "ik_test", "query")
    raw_cand_docs = load_raw_documents(cand_dir)
    raw_q_docs = load_raw_documents(q_dir)
    
    with open(TEST_GOLD_JSON) as f:
        gold_dict = json.load(f)
    with open(CAND_LIST_TXT) as f:
        valid_cands = set(f.read().strip().split("\n"))

    cand_ids = [c for c in raw_cand_docs.keys() if c in valid_cands]
    cand_texts = [raw_cand_docs[c] for c in cand_ids]
    q_ids = list(raw_q_docs.keys())
    q_texts = [raw_q_docs[q] for q in q_ids]
    
    results_summary = {}

    # # ================== EXP 1: PURE RAW TEXT ================== #
    # print("\n\n=============== EXPERIMENT BLOCK 1: PURE RAW TEXT ===============")
    # exp1_models = {
    #     "RawText_BM25_Standard": RawBM25(k1=1.6, b=0.75),
    #     "RawText_TFIDF_Unigram": TFIDFModel(ngram_range=(1,1)),
    #     "RawText_TFIDF_UniBi": TFIDFModel(ngram_range=(1,2))
    # }
    # for ext_name, model_obj in exp1_models.items():
    #     print(f"\n--- Running Experiment: {ext_name} ---")
    #     model_obj.fit(cand_texts)
    #     preds = {}
    #     for q_id, q_text in tqdm(raw_q_docs.items(), desc=f"Scoring {ext_name}"):
    #         scores = model_obj.get_scores(q_text)
    #         ranked = np.argsort(scores)[::-1]
    #         preds[q_id] = [cand_ids[i] for i in ranked]
        
    #     metrics = evaluate_predictions(preds, gold_dict, valid_cands)
    #     results_summary[ext_name] = metrics[20]['Micro-F1']
    #     with open(f"{RESULTS_DIR}/{ext_name}.json", "w") as f: json.dump(preds, f, indent=4)
            

    # # ================== EXP 2: STANDARD PREPROCESSING ================== #
    # print("\n\n=============== EXPERIMENT BLOCK 2: STANDARD PREPROCESSING ===============")
    # c_docs, c_toks_dict = load_preprocessed_documents(cand_dir, "cache/candidates_cache.pkl")
    # q_docs, q_toks_dict = load_preprocessed_documents(q_dir, "cache/queries_cache.pkl")
    
    # c_std_texts = [" ".join(c_toks_dict[k]) for k in cand_ids]
    # q_std_texts = [" ".join(q_toks_dict[k]) for k in q_ids]
    
    # exp2_models = {
    #     "StandardPrep_BM25_Base": RawBM25(k1=1.5, b=0.75),
    #     "StandardPrep_BM25_HighK1": RawBM25(k1=2.0, b=0.75),
    #     "StandardPrep_BM25_LowB": RawBM25(k1=1.5, b=0.5),
    #     "StandardPrep_TFIDF_UniBi": TFIDFModel(ngram_range=(1,2)),
    #     "StandardPrep_LSA_100": LSAModel(n_components=100),
    #     "StandardPrep_LSA_200": LSAModel(n_components=200)
    # }
    
    # for ext_name, model_obj in exp2_models.items():
    #     print(f"\n--- Running Experiment: {ext_name} ---")
    #     model_obj.fit(c_std_texts)
    #     preds = {}
    #     for i, q_id in enumerate(tqdm(q_ids, desc=f"Scoring {ext_name}")):
    #         scores = model_obj.get_scores(q_std_texts[i])
    #         ranked = np.argsort(scores)[::-1]
    #         preds[q_id] = [cand_ids[i] for i in ranked]
            
    #     metrics = evaluate_predictions(preds, gold_dict, valid_cands)
    #     results_summary[ext_name] = metrics[20]['Micro-F1']
    #     with open(f"{RESULTS_DIR}/{ext_name}.json", "w") as f: json.dump(preds, f, indent=4)


    # ================== EXP 3: SENTENCE-LEVEL EVENT FILTERING ================== #
    print("\n\n=============== EXPERIMENT BLOCK 3: SENTENCE-LEVEL EVENT FILTERING ===============")
    
    query_concepts = {}
    query_sentences = {}
    for q_id, text in tqdm(raw_q_docs.items(), desc="Parsing Queries"):
        sents = segment_into_sentences(text)
        query_sentences[q_id] = sents
        concepts = set()
        for s in sents:
            concepts.update(extract_key_concepts(s))
        query_concepts[q_id] = concepts
        
    candidate_sent_concepts = {}
    for c_id in tqdm(cand_ids, desc="Parsing Candidates"):
        sents = segment_into_sentences(raw_cand_docs[c_id])
        parsed = []
        for s in sents:
            parsed.append((s, extract_key_concepts(s)))
        candidate_sent_concepts[c_id] = parsed

    # Hyperparameter tuning matrix for Event Filtering
    exp3_hyperparams = [
        ("EventFilter_BM25_Standard", 1.6, 0.75, 2),  
        ("EventFilter_BM25_Strict", 1.6, 0.75, 3),    
        ("EventFilter_BM25_LowB", 1.6, 0.5, 2),       
        ("EventFilter_BM25_HighK1", 2.0, 0.75, 2)     
    ]
    
    for exp_name, k1, b, overlap_thresh in exp3_hyperparams:
        print(f"\n--- Running Experiment: {exp_name} (overlap>={overlap_thresh}, k1={k1}, b={b}) ---")
        preds = {}
        for q_id in tqdm(raw_q_docs.keys(), desc=f"Rank {exp_name}"):
            q_con = query_concepts[q_id]
            
            filtered_cand_texts = []
            for c_id in cand_ids:
                kept_sents = []
                cand_data = candidate_sent_concepts[c_id]
                for (sent_txt, sent_concepts) in cand_data:
                    if len(q_con.intersection(sent_concepts)) >= overlap_thresh:
                        kept_sents.append(sent_txt)
                filtered_cand_texts.append(" ".join(kept_sents))
                
            model = RawBM25(k1=k1, b=b)
            try:
                model.fit(filtered_cand_texts)
                scores = model.get_scores(" ".join(query_sentences[q_id]))
                ranked = np.argsort(scores)[::-1]
                preds[q_id] = [cand_ids[i] for i in ranked]
            except ValueError:
                preds[q_id] = cand_ids.copy()
            
        metrics = evaluate_predictions(preds, gold_dict, valid_cands)
        results_summary[exp_name] = metrics[20]['Micro-F1']
        with open(f"{RESULTS_DIR}/{exp_name}.json", "w") as f: json.dump(preds, f, indent=4)
            
            
    # # ================== EXP 4: CONCEPT EXTRACTION TF-IDF ================== #
    # print("\n\n=============== EXPERIMENT BLOCK 4: CONCEPT EXTRACTION (LEGAL + EVENTS) ===============")
    # # This matches the script `concept_tfidf_pcr.py`
    # cand_concepts_dict = generate_or_load_concepts(
    #     {c: raw_cand_docs[c] for c in cand_ids}, 
    #     os.path.join(CACHE_DIR, "cand_concepts_unified.pkl"), 
    #     "Extracting Cand Concepts"
    # )
    # cand_texts_filtered = [cand_concepts_dict[c] for c in cand_ids]
    
    # q_concepts_dict = generate_or_load_concepts(
    #     raw_q_docs, 
    #     os.path.join(CACHE_DIR, "query_concepts_unified.pkl"), 
    #     "Extracting Query Concepts"
    # )
    
    # print("\nFitting Denoised TF-IDF Vector Space...")
    # vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, ngram_range=(1,2), token_pattern=r"(?u)\b[\w\d_]+\b")
    # cand_matrix = vectorizer.fit_transform(cand_texts_filtered)
    
    # print(f"\n--- Running Experiment: ConceptExtraction_TFIDF ---")
    # preds = {}
    # for q_id, q_text_filtered in tqdm(q_concepts_dict.items(), desc="TF-IDF Cosine Inference"):
    #     q_vec = vectorizer.transform([q_text_filtered])
    #     scores = cosine_similarity(q_vec, cand_matrix)[0]
    #     ranked = np.argsort(scores)[::-1]
    #     preds[q_id] = [cand_ids[i] for i in ranked]
        
    # metrics = evaluate_predictions(preds, gold_dict, valid_cands)
    # results_summary["ConceptExtraction_TFIDF"] = metrics[20]['Micro-F1']
    # with open(f"{RESULTS_DIR}/ConceptExtraction_TFIDF.json", "w") as f: json.dump(preds, f, indent=4)


    # # ================== EXP 5: BM25 + GRAPH ENSEMBLE ================== #
    # print("\n\n=============== EXPERIMENT BLOCK 5: BM25 + GRAPH ENSEMBLE ===============")
    # print("Extracting L-NER Graph Citation Entities...")
    # cand_graph_entities = [extract_legal_entities(txt) for txt in tqdm(cand_texts, desc="Candidate L-NER")]
    
    # entity_vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w\d_]+\b")
    # cand_entity_matrix = entity_vectorizer.fit_transform(cand_graph_entities)
    
    # print("Fitting Baseline BM25 for Graph Hybrid...")
    # bm25 = RawBM25(k1=1.6, b=0.75)
    # bm25.fit(cand_texts)
    
    # ensemble_hyperparams = [
    #     (0.8, 0.2),
    #     (0.6, 0.4),
    #     (0.4, 0.6)
    # ]
    
    # for bm25_weight, ent_weight in ensemble_hyperparams:
    #     exp_name = f"GraphEnsemble_BM{int(bm25_weight*100)}_Ent{int(ent_weight*100)}"
    #     print(f"\n--- Running Experiment: {exp_name} ---")
    #     preds = {}
        
    #     for q_id, q_text in tqdm(raw_q_docs.items(), desc=f"Re-Ranking ({bm25_weight}/{ent_weight})"):
    #         bm25_scores = bm25.get_scores(q_text)
            
    #         q_ent = extract_legal_entities(q_text)
    #         q_ent_vec = entity_vectorizer.transform([q_ent])
    #         entity_scores = cosine_similarity(q_ent_vec, cand_entity_matrix)[0]
            
    #         if bm25_scores.max() > 0: bm25_norm = minmax_scale(bm25_scores)
    #         else: bm25_norm = bm25_scores
                
    #         if entity_scores.max() > 0: entity_norm = minmax_scale(entity_scores)
    #         else: entity_norm = entity_scores
                
    #         final_scores = (bm25_weight * bm25_norm) + (ent_weight * entity_norm)
    #         ranked = np.argsort(final_scores)[::-1]
    #         preds[q_id] = [cand_ids[i] for i in ranked]
            
    #     metrics = evaluate_predictions(preds, gold_dict, valid_cands)
    #     results_summary[exp_name] = metrics[20]['Micro-F1']
    #     with open(f"{RESULTS_DIR}/{exp_name}.json", "w") as f: json.dump(preds, f, indent=4)


    # # ================== EXP 6: GRAPH DIFFUSION RE-RANKING ================== #
    # print("\n\n=============== EXPERIMENT BLOCK 6: GRAPH DIFFUSION RE-RANKING ===============")
    # print("Fitting Base TF-IDF for Graph Diffusion...")
    # base_model = BaseTFIDF(ngram_range=(1,2))
    # base_model.fit(cand_texts)
    
    # print("Building Bipartite Citation Graph Transition Matrix...")
    # W = cosine_similarity(cand_entity_matrix)
    # np.fill_diagonal(W, 0)
    # T = normalize(W, norm='l1', axis=1)
    
    # diffusion_alphas = [0.15, 0.35, 0.55]
    # for alpha in diffusion_alphas:
    #     exp_name = f"GraphDiffusion_Alpha_{int(alpha*100)}"
    #     print(f"\n--- Running Experiment: {exp_name} ---")
    #     preds = {}
        
    #     for q_id, q_text in tqdm(raw_q_docs.items(), desc=f"Diffusing (Alpha={alpha})"):
    #         S_0 = base_model.get_scores(q_text)
            
    #         q_ent = extract_legal_entities(q_text)
    #         q_ent_vec = entity_vectorizer.transform([q_ent])
    #         direct_ent_sim = cosine_similarity(q_ent_vec, cand_entity_matrix)[0]
            
    #         seed_scores = S_0 + (0.5 * direct_ent_sim)
    #         S_diffused = (1 - alpha) * seed_scores + alpha * np.dot(T, seed_scores)
            
    #         ranked = np.argsort(S_diffused)[::-1]
    #         preds[q_id] = [cand_ids[i] for i in ranked]
            
    #     metrics = evaluate_predictions(preds, gold_dict, valid_cands)
    #     results_summary[exp_name] = metrics[20]['Micro-F1']
    #     with open(f"{RESULTS_DIR}/{exp_name}.json", "w") as f: json.dump(preds, f, indent=4)

    # ----------------------------------------------------------- #
    print("\n\n" + "="*80)
    print(f"{'GLOBAL BASELINE RESULTS SUMMARY (Micro-F1@20)':^80}")
    print("="*80)
    print(f"{'Experiment Variant':<40} | {'Micro-F1@20':>15}")
    print("-" * 80)
    for k, v in sorted(results_summary.items(), key=lambda x: x[1], reverse=True):
        print(f"{k:<40} | {v:>15.4f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
