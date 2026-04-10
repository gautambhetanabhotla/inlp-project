import json
import numpy as np
from sklearn.metrics import f1_score

GOLD_FILENAME = "ilpcr-test-gold.json"
PRED_FILENAME = "ilpcr-test-pred.json"
CAND_FILENAME = "ilpcr-test-cand.txt"

with open(GOLD_FILENAME) as fr:
    gold = json.load(fr)

with open(PRED_FILENAME) as fr:
    pred = json.load(fr)

with open(CAND_FILENAME) as fr:
    cv = set(fr.read().strip().split('\n'))

def get_metrics_at_k(g, p, k):
    gs = set(g)
    ps_list = p[:k]
    ps_set = set(ps_list)
    
    num_hits_micro = len(gs & ps_set)
    num_pred_micro = len(ps_set)
    num_gold_micro = len(gs)
    
    micro_stats = (num_hits_micro, num_pred_micro, num_gold_micro)
    
    # Rank metrics are undefined if there's no ground truth (gold)
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

for k in range(1, 21):
    total_hits = 0
    total_pred = 0
    total_gold = 0
    macro_results = []
    
    for id, cands in gold.items():
        if id in pred:
            cands2 = pred[id]
        else:
            cands2 = []
        cands = [c for c in cands if c in cv and c != id]
        cands2 = [c for c in cands2 if c in cv and c != id]
        
        micro_stats, macro_stats = get_metrics_at_k(cands, cands2, k)
        
        c, p, g = micro_stats[0], micro_stats[1], micro_stats[2]
        total_hits += c
        total_pred += p
        total_gold += g
        
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

print(f"{'K':>2} | {'Micro-F1':>8} | {'Micro-P':>8} | {'Micro-R':>8} | {'Macro-F1':>8} | {'MAP':>8} | {'MRR':>8} | {'NDCG':>8}")
print("-" * 88)
for k in range(1, 21):
    m = metrics_by_k[k]
    print(f"{k:2} | {m['Micro-F1']:.4f}   | {m['Micro-P']:.4f}   | {m['Micro-R']:.4f}   | {m['Macro-F1']:.4f}   | {m['MAP']:.4f}   | {m['MRR']:.4f}   | {m['NDCG']:.4f}")






    

    