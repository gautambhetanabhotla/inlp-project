import os
import json
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

from cache_utils import load_raw_documents, get_or_generate_feature, extract_legal_concepts, extract_legal_entities
from metrics import evaluate_predictions_detailed
from models_unsupervised import FastBM25, TFIDFModel, LSAModel, Doc2VecModel, ConceptTFIDFModel, GraphDiffusionModel
from models_supervised import PointwiseLTRModel

DATASET_DIR = "../../Dataset"
EVAL_DIR = "../../Eval"
TEST_GOLD_JSON = os.path.join(EVAL_DIR, "ilpcr-test-gold.json")
TRAIN_GOLD_JSON = os.path.join(DATASET_DIR, "ik_train", "train.json")
CAND_LIST_TXT = os.path.join(EVAL_DIR, "ilpcr-test-cand.txt")
CACHE_DIR = "cache/combined"

random.seed(42)

def main():
    print("====================================")
    print("  PCR Multi-Model Evaluation Suite  ")
    print("====================================\n")
    
    # --- 1. Load Data ---
    print("[1] Loading Datasets...")
    train_cand = load_raw_documents(os.path.join(DATASET_DIR, "ik_train", "candidate"))
    test_cand = load_raw_documents(os.path.join(DATASET_DIR, "ik_test", "candidate"))
    train_q = load_raw_documents(os.path.join(DATASET_DIR, "ik_train", "query"))
    test_q = load_raw_documents(os.path.join(DATASET_DIR, "ik_test", "query"))
    
    all_cands = {**train_cand, **test_cand}
    all_qs = {**train_q, **test_q}
    
    with open(TEST_GOLD_JSON) as f: test_gold = json.load(f)
    with open(TRAIN_GOLD_JSON) as f: train_gold_raw = json.load(f)
    train_gold = {item["query_name"]: item["relevant candidates"] for item in train_gold_raw["Query Set"]}
    with open(CAND_LIST_TXT) as f: valid_test_cands = set(f.read().strip().split("\\n"))
    
    all_gold = {**train_gold, **test_gold}
    valid_train_cands = set(train_cand.keys())
    
    # --- 2. Extract Caches (Concepts & Entities) ---
    print("\n[2] Checking / Building NLP Caches...")
    cand_concepts = get_or_generate_feature(all_cands, f"{CACHE_DIR}/cand_concepts.pkl", extract_legal_concepts, "Extracting Candidate Concepts")
    q_concepts = get_or_generate_feature(all_qs, f"{CACHE_DIR}/q_concepts.pkl", extract_legal_concepts, "Extracting Query Concepts")
    cand_entities = get_or_generate_feature(all_cands, f"{CACHE_DIR}/cand_entities.pkl", extract_legal_entities, "Extracting Candidate Entities")
    q_entities = get_or_generate_feature(all_qs, f"{CACHE_DIR}/q_entities.pkl", extract_legal_entities, "Extracting Query Entities")
    
    # We maintain ordered lists of candidates so matrix indices line up
    cand_ids = list(all_cands.keys())
    cand_texts = [all_cands[c] for c in cand_ids]
    cand_concept_list = [cand_concepts[c] for c in cand_ids]
    cand_entity_list = [cand_entities[c] for c in cand_ids]
    
    # Dictionary to store raw scores for LTR integration and ranking
    # format: scores_dict[model_name][query_id] = array_of_scores_for_all_cands
    scores_dict = {}
    
    # Helper to evaluate Unsupervised models mapping test/train cleanly
    def evaluate_model(model_name, complexity, q_dict, score_func):
        scores_dict[model_name] = {}
        preds_test = {}
        preds_train = {}
        
        print(f"  -> Scoring {model_name}...")
        for q_id, q_data in tqdm(q_dict.items(), desc=model_name, leave=False):
            scores = score_func(q_data)
            scores_dict[model_name][q_id] = scores
            
            # Rank descending
            ranked = np.argsort(scores)[::-1]
            ranked_ids = [cand_ids[i] for i in ranked]
            
            if q_id in test_q:
                preds_test[q_id] = [c for c in ranked_ids if c in valid_test_cands]
            if q_id in train_q:
                preds_train[q_id] = [c for c in ranked_ids if c in valid_train_cands]
                
        metrics_test = evaluate_predictions_detailed(preds_test, test_gold, valid_test_cands, k=20)
        metrics_train = evaluate_predictions_detailed(preds_train, train_gold, valid_train_cands, k=20)
        
        # Combine metrics into a unified Unsupervised metric (Average or Test mostly)
        # We'll report the Test metrics for a fair comparison against supervised!
        return {"Model Name": model_name, "Category": complexity, 
                **metrics_test}
        
    results = []
    
    # --- 3a. Lexical Baselines ---
    print("\n[3a] Executing LEXICAL Models (Unsupervised)...")
    
    m_bm25 = FastBM25(k1=1.6, b=0.75)
    m_bm25.fit(cand_texts)
    results.append(evaluate_model("BM25 (Standard)", "Lexical", all_qs, m_bm25.get_scores))
    
    m_bm25_t = FastBM25(k1=2.0, b=0.8) # Tuned for long text
    m_bm25_t.fit(cand_texts)
    results.append(evaluate_model("BM25 (Tuned)", "Lexical", all_qs, m_bm25_t.get_scores))
    
    m_tfidf2 = TFIDFModel(ngram=(1,2))
    m_tfidf2.fit(cand_texts)
    results.append(evaluate_model("TF-IDF (Unigram+Bigram)", "Lexical", all_qs, m_tfidf2.get_scores))
    
    m_concept = ConceptTFIDFModel()
    m_concept.fit(cand_concept_list)
    results.append(evaluate_model("Concept-Filtered TF-IDF", "Lexical (Advanced)", q_concepts, m_concept.get_scores))
    
    # --- 3b. Semantic Models ---
    print("\n[3b] Executing SEMANTIC Models (Unsupervised)...")
    
    m_lsa = LSAModel(n_components=250)
    m_lsa.fit(cand_texts)
    results.append(evaluate_model("LSA (SVD)", "Semantic", all_qs, m_lsa.get_scores))
    
    # Quick Doc2Vec
    m_d2v = Doc2VecModel(vector_size=100, epochs=5)
    m_d2v.fit(cand_texts)
    results.append(evaluate_model("Doc2Vec (DBOW)", "Semantic", all_qs, m_d2v.get_scores))
    
    # --- 3c. Graph Models ---
    print("\n[3c] Executing GRAPH Models (Unsupervised)...")
    
    m_graph = GraphDiffusionModel(alpha=0.35)
    m_graph.fit(cand_texts, cand_entity_list)
    results.append(evaluate_model("Graph Label Diffusion", "Graph", all_qs, m_graph.get_scores))
    
    # --- 3d. Supervised Machine Learning (LTR) ---
    print("\n[3d] Executing LTR Machine Learning (Supervised)...")
    # Features to use from Unsupervised executions:
    feature_models = ["BM25 (Standard)", "TF-IDF (Unigram+Bigram)", "Concept-Filtered TF-IDF", "LSA (SVD)", "Doc2Vec (DBOW)", "Graph Label Diffusion"]
    
    def build_features(q_id, c_idx):
        return [scores_dict[m][q_id][c_idx] for m in feature_models]
        
    # Build Train Set
    X_train, y_train = [], []
    for q_id, pos_cands in train_gold.items():
        if q_id not in train_q: continue
        
        # Positives
        for p_cand in pos_cands:
            if p_cand in cand_ids:
                X_train.append(build_features(q_id, cand_ids.index(p_cand)))
                y_train.append(1)
                
        # Sample Negatives (5x Positives)
        # Using top BM25 mistakes as hard negatives to make ML learn better
        bm25_scores = scores_dict["BM25 (Standard)"][q_id]
        top_false = [cand_ids[i] for i in np.argsort(bm25_scores)[::-1] if cand_ids[i] not in pos_cands][:len(pos_cands)*5]
        for n_cand in top_false:
            if n_cand in cand_ids:
                X_train.append(build_features(q_id, cand_ids.index(n_cand)))
                y_train.append(0)
    
    ltr = PointwiseLTRModel(n_estimators=100)
    ltr.fit(X_train, y_train)
    
    # Build Test Evaluation
    preds_ltr = {}
    for q_id in tqdm(test_q.keys(), desc="LTR Inference", leave=False):
        q_features = []
        valid_idxs = []
        for i, c_id in enumerate(cand_ids):
            if c_id in valid_test_cands:
                q_features.append(build_features(q_id, i))
                valid_idxs.append(c_id)
        
        scores = ltr.predict_proba(q_features)
        ranked = np.argsort(scores)[::-1]
        preds_ltr[q_id] = [valid_idxs[i] for i in ranked]
        
    metrics_ltr = evaluate_predictions_detailed(preds_ltr, test_gold, valid_test_cands)
    results.append({"Model Name": "RandomForest LTR (Ensemble)", "Category": "Supervised ML", **metrics_ltr})
    
    
    # --- 4. Render Table ---
    print("\n====================================")
    print("        FINAL LEADERBOARD           ")
    print("====================================\n")
    
    df = pd.DataFrame(results)
    
    # Define custom complexity sort order
    cat_order = {"Lexical": 1, "Semantic": 2, "Lexical (Advanced)": 3, "Graph": 4, "Supervised ML": 5}
    df['CatRank'] = df['Category'].map(cat_order)
    df = df.sort_values(by=['CatRank', 'Micro-F1@20'], ascending=[True, False]).drop('CatRank', axis=1)
    
    # Format and print
    col_format = "{:<25} | {:<20} | {:<10} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}"
    print(col_format.format("Model Name", "Category/Type", "µF1@20", "P@20", "R@20", "MAP", "MRR", "nDCG@20"))
    print("-" * 110)
    
    for _, row in df.iterrows():
        print(col_format.format(
            row["Model Name"][:24],
            row["Category"][:19],
            f'{row["Micro-F1@20"]:.4f}',
            f'{row["P@20"]:.4f}',
            f'{row["R@20"]:.4f}',
            f'{row["MAP"]:.4f}',
            f'{row["MRR"]:.4f}',
            f'{row["nDCG@20"]:.4f}'
        ))
        
    # Save CSV
    df.to_csv(os.path.join(DATASET_DIR, "../my_project/combined/pcr_evaluation_report.csv"), index=False)
    print("\nDetailed results saved to `my_project/combined/pcr_evaluation_report.csv`.")
    print("Done!")

if __name__ == "__main__":
    main()
