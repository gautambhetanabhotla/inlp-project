import os
import json
from tqdm import tqdm
from preprocessing import load_documents, preprocess_corpus
from models import PCRModelEnsemble

EVAL_DIR = "../Eval"
DATASET_DIR = "../Dataset/ik_test"
PRED_JSON = "ilpcr-test-pred.json"

def main():
    print("Loading candidate documents...")
    cand_dir = os.path.join(DATASET_DIR, "candidate")
    candidate_docs = load_documents(cand_dir)
    print(f"Loaded {len(candidate_docs)} candidates.")

    # We only want to process candidates that are required by the evaluation script
    eval_cands_path = os.path.join(EVAL_DIR, "ilpcr-test-cand.txt")
    with open(eval_cands_path, "r") as f:
        valid_cands = set(f.read().strip().split('\n'))
    
    # Filter candidates
    candidate_docs = {k: v for k, v in candidate_docs.items() if k in valid_cands}
    print(f"Filtered to {len(candidate_docs)} valid candidates based on Eval script.")

    print("\nLoading query documents...")
    query_dir = os.path.join(DATASET_DIR, "query")
    query_docs = load_documents(query_dir)
    print(f"Loaded {len(query_docs)} queries.")

    # Preprocess all candidates
    print("\nPreprocessing candidate documents...")
    candidate_tokens = preprocess_corpus(candidate_docs)

    # Preprocess all queries
    print("\nPreprocessing query documents...")
    query_tokens = preprocess_corpus(query_docs)

    # Initialize and fit model
    print("\nInitializing Ensemble Model (BM25 + LSA)...")
    # Weights can be tuned. Heavy weight on BM25 usually works best for legal text.
    model = PCRModelEnsemble(use_bm25plus=False, svd_components=100, bm25_weight=0.8, lsa_weight=0.2)
    model.fit(candidate_tokens)

    # Generate predictions
    print("\nGenerating scores for queries...")
    predictions = {}
    for q_id, q_toks in tqdm(query_tokens.items(), desc="Scoring queries"):
        # Returns list of (candidate_id, score) in descending order
        scored_cands = model.get_scores(q_toks)
        # Store just the candidate IDs in ranked order
        predictions[q_id] = [c_id for c_id, score in scored_cands]

    # Save predictions
    output_path = os.path.join(EVAL_DIR, PRED_JSON)
    print(f"\nSaving predictions to {output_path}")
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=4)
        
    print("Done! You can now run the evaluation script.")

if __name__ == "__main__":
    main()
