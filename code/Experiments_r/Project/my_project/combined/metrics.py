import numpy as np

def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

def evaluate_predictions_detailed(predictions, gold_dict, valid_cands, k=20):
    """
    Computes Micro-F1@K, Precision@K, Recall@K, MAP, MRR, nDCG@K
    """
    precisions = []
    recalls = []
    aps = []
    rr = []
    ndcgs = []
    
    # For Micro-F1
    total_C, total_G, total_P = 0, 0, 0

    for q_id, true_cands in gold_dict.items():
        if q_id not in predictions:
            continue
            
        pred_cands = predictions[q_id]
        
        # Filter valid candidates (excluding the query itself and ensuring they exist in the pool)
        true_cands = [c for c in true_cands if c in valid_cands and c != q_id]
        pred_cands = [c for c in pred_cands if c in valid_cands and c != q_id]
        
        if not true_cands:
            continue
            
        pred_at_k = pred_cands[:k]
        
        # --- Micro F1 components ---
        intersection = len(set(true_cands) & set(pred_at_k))
        total_C += intersection
        total_G += len(set(true_cands))
        total_P += len(set(pred_at_k))
        
        # --- Precision & Recall ---
        p_at_k = intersection / k if k > 0 else 0
        r_at_k = intersection / len(set(true_cands)) if len(set(true_cands)) > 0 else 0
        precisions.append(p_at_k)
        recalls.append(r_at_k)
        
        # --- MRR ---
        rank = 0
        for idx, p_cand in enumerate(pred_cands):
            if p_cand in true_cands:
                rank = idx + 1
                break
        rr.append(1.0 / rank if rank > 0 else 0.0)
        
        # --- Average Precision (for MAP) ---
        hits = 0
        sum_precisions = 0
        for idx, p_cand in enumerate(pred_cands):
            if p_cand in true_cands:
                hits += 1
                sum_precisions += hits / (idx + 1.0)
        aps.append(sum_precisions / len(true_cands) if len(true_cands) > 0 else 0.0)
        
        # --- nDCG@K ---
        relevance_vector = [1 if p in true_cands else 0 for p in pred_at_k]
        # Pad with zeros if less than k
        if len(relevance_vector) < k:
            relevance_vector.extend([0] * (k - len(relevance_vector)))
        ndcgs.append(ndcg_at_k(relevance_vector, k))

    # Aggregating
    mean_p = np.mean(precisions) if precisions else 0
    mean_r = np.mean(recalls) if recalls else 0
    map_score = np.mean(aps) if aps else 0
    mrr_score = np.mean(rr) if rr else 0
    mean_ndcg = np.mean(ndcgs) if ndcgs else 0
    
    micro_prec = total_C / total_P if total_P else 0
    micro_rec = total_C / total_G if total_G else 0
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0

    return {
        "Micro-F1@20": micro_f1,
        "P@20": mean_p,
        "R@20": mean_r,
        "MAP": map_score,
        "MRR": mrr_score,
        "nDCG@20": mean_ndcg
    }
