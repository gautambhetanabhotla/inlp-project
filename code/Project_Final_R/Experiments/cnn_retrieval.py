"""
cnn_retrieval.py
================
TextCNN document encoder for Prior Case Retrieval.

Architecture
------------
  Token embeddings (W2V init or random)
  → Parallel Conv1D filters (multiple kernel sizes)
  → Max-pooling per filter map
  → Concatenated + L2-normalised embedding
  → Cosine similarity retrieval

Training: Triplet loss (anchor=query, pos=relevant, neg=random)
GPU: uses CUDA > MPS > CPU automatically.

Run
---
    python3 cnn_retrieval.py --data_dir /path/to/dataset/ --split train

Parameter grid
--------------
  Edit CNN_GRID below.  Comment out any line to skip that configuration.
"""

import os
import argparse
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import (
    load_split, clean_text, evaluate_all,
    save_results, print_results_table, save_results_csv,
    cosine_sim_matrix, get_device,
    build_vocab, tokens_to_ids, build_w2v, make_embed_matrix,
)

try:
    GENSIM_OK = True
except ImportError:
    GENSIM_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS  (edit here)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "/home/raghavgrover/Desktop/Sem6/INLP/Project_Final/Experiments"
SPLIT    = "train"
TOP_K    = 1000
OUTPUT   = "results/cnn_results.json"
K_VALUES = [5, 6, 7, 8, 9, 10, 11, 15, 20]

MAX_LEN  = 512    # max tokens per document
NEG_PER  = 5      # negative samples per positive pair
BATCH_SZ = 32

# ─────────────────────────────────────────────────────────────────────────────
# CNN_GRID  ← comment out any line to skip
# (embed_dim, filter_sizes, num_filters, dropout, margin, epochs, use_w2v, ngram)
#
#   embed_dim    : token embedding dimension
#   filter_sizes : list of Conv1D kernel sizes (one branch per size)
#   num_filters  : number of filters per kernel size
#   dropout      : dropout rate after pooling
#   margin       : triplet loss margin
#   epochs       : training epochs
#   use_w2v      : initialise embeddings with Word2Vec (True/False)
#   ngram        : tokenisation order (1=unigrams, 2=+bigrams)
# ─────────────────────────────────────────────────────────────────────────────

CNN_GRID: List[Tuple] = [
    # ── Random init, unigrams ─────────────────────────────────────────────────
    (100, [2, 3, 4],    64,  0.3, 0.5, 10, False, 1),
    (100, [3, 4, 5],    64,  0.3, 0.5, 10, False, 1),
    (100, [2, 3, 4, 5], 64,  0.3, 0.5, 10, False, 1),
    (200, [2, 3, 4, 5], 64,  0.3, 0.5, 10, False, 1),
    (100, [2, 3, 4, 5], 128, 0.3, 0.5, 10, False, 1),
    (200, [2, 3, 4, 5], 128, 0.3, 0.5, 10, False, 1),
    (100, [2, 3, 4, 5], 64,  0.5, 0.5, 10, False, 1),   # higher dropout
    (100, [2, 3, 4, 5], 64,  0.3, 0.3, 10, False, 1),   # smaller margin
    (100, [2, 3, 4, 5], 64,  0.3, 0.5, 20, False, 1),   # more epochs
    (200, [2, 3, 4, 5], 128, 0.3, 0.5, 20, False, 1),

    # ── Word2Vec init, unigrams ───────────────────────────────────────────────
    (100, [2, 3, 4, 5], 64,  0.3, 0.5, 10, True,  1),
    (200, [2, 3, 4, 5], 64,  0.3, 0.5, 10, True,  1),
    (200, [2, 3, 4, 5], 128, 0.3, 0.5, 10, True,  1),
    (200, [2, 3, 4, 5], 128, 0.3, 0.5, 20, True,  1),
    (200, [3, 4, 5],    128, 0.3, 0.5, 20, True,  1),

    # ── Bigrams ───────────────────────────────────────────────────────────────
    (100, [2, 3, 4, 5], 64,  0.3, 0.5, 10, False, 2),
    (200, [2, 3, 4, 5], 64,  0.3, 0.5, 10, False, 2),
    (100, [2, 3, 4, 5], 64,  0.3, 0.5, 10, True,  2),
    (200, [2, 3, 4, 5], 128, 0.3, 0.5, 20, True,  2),
]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, filter_sizes,
                 num_filters, dropout=0.3, pretrained=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(pretrained))
        self.convs   = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        e = self.embedding(x).permute(0, 2, 1)
        pooled = [F.relu(c(e)).max(dim=2).values for c in self.convs]
        return F.normalize(self.dropout(torch.cat(pooled, dim=1)), p=2, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# TRIPLET DATASET
# ─────────────────────────────────────────────────────────────────────────────

class TripletDataset(Dataset):
    def __init__(self, q_ids, c_ids, relevance, neg_per=5):
        self.a, self.p, self.n = [], [], []
        all_c = list(c_ids.keys())
        for qid, rel in relevance.items():
            if qid not in q_ids: continue
            rel_set  = set(rel)
            neg_pool = [c for c in all_c if c not in rel_set]
            for pos in rel:
                if pos not in c_ids: continue
                for neg in random.sample(neg_pool, min(neg_per, len(neg_pool))):
                    self.a.append(q_ids[qid])
                    self.p.append(c_ids[pos])
                    self.n.append(c_ids[neg])

    def __len__(self): return len(self.a)

    def __getitem__(self, i):
        return (torch.tensor(self.a[i], dtype=torch.long),
                torch.tensor(self.p[i], dtype=torch.long),
                torch.tensor(self.n[i], dtype=torch.long))


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN + ENCODE
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, dataset, device, epochs, lr=1e-3,
                batch_size=32, margin=0.5):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    sched  = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for a, p, neg in tqdm(loader, desc=f"  Ep {epoch+1}/{epochs}",
                               leave=False):
            a, p, neg = a.to(device), p.to(device), neg.to(device)
            ea, ep, en = model(a), model(p), model(neg)
            loss = F.relu(margin - (ea*ep).sum(1) + (ea*en).sum(1)).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item(); n += 1
        sched.step()
        print(f"  Ep {epoch+1}/{epochs}  loss={total/max(n,1):.4f}")
    return model


@torch.no_grad()
def encode_all(model, id_seqs, device, batch_size=64):
    model.eval()
    ids, vecs = list(id_seqs.keys()), []
    for i in range(0, len(ids), batch_size):
        batch = ids[i: i+batch_size]
        x = torch.tensor([id_seqs[d] for d in batch],
                          dtype=torch.long).to(device)
        vecs.append(model(x).cpu().numpy())
    return ids, np.vstack(vecs)


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

    device = get_device()
    queries, candidates, relevance = load_split(args.data_dir, args.split)
    all_results = []

    tok_cache_q: Dict[int, Dict] = {}
    tok_cache_c: Dict[int, Dict] = {}

    for (embed_dim, filter_sizes, num_filters,
         dropout, margin, epochs, use_w2v, ngram) in CNN_GRID:

        fs_str = "".join(map(str, filter_sizes))
        name   = (f"CNN_dim={embed_dim}_fs={fs_str}_nf={num_filters}"
                  f"_drop={dropout}_m={margin}_ep={epochs}"
                  f"_w2v={use_w2v}_ng={ngram}")
        print(f"\n{'─'*64}\n  {name}\n{'─'*64}")

        if ngram not in tok_cache_q:
            tok_cache_q[ngram] = {
                qid: clean_text(t, True, ngram) for qid, t in queries.items()
            }
            tok_cache_c[ngram] = {
                cid: clean_text(t, True, ngram) for cid, t in candidates.items()
            }
        q_tok, c_tok = tok_cache_q[ngram], tok_cache_c[ngram]
        all_tok = list(q_tok.values()) + list(c_tok.values())

        w2i, vocab = build_vocab(all_tok, min_freq=2)

        pretrained = None
        if use_w2v:
            print(f"  Training W2V (dim={embed_dim}) ...")
            w2v        = build_w2v(all_tok, vector_size=embed_dim,
                                   window=5, min_count=2, sg=1, epochs=5)
            pretrained = make_embed_matrix(vocab, w2i, embed_dim, w2v)

        q_ids = {qid: tokens_to_ids(t, w2i, MAX_LEN)
                 for qid, t in q_tok.items()}
        c_ids = {cid: tokens_to_ids(t, w2i, MAX_LEN)
                 for cid, t in c_tok.items()}

        model   = TextCNN(len(vocab), embed_dim, filter_sizes,
                          num_filters, dropout, pretrained)
        dataset = TripletDataset(q_ids, c_ids, relevance, neg_per=NEG_PER)
        if len(dataset) == 0:
            print("  [SKIP] No training pairs.")
            continue

        model = train_model(model, dataset, device, epochs,
                            batch_size=BATCH_SZ, margin=margin)

        cand_ids, cand_mat = encode_all(model, c_ids, device)

        results: Dict[str, List[str]] = {}
        for qid in relevance:
            if qid not in q_ids: continue
            qx   = torch.tensor([q_ids[qid]], dtype=torch.long).to(device)
            model.eval()
            with torch.no_grad():
                qvec = model(qx).cpu().numpy()
            sims  = cosine_sim_matrix(qvec, cand_mat)[0]
            order = np.argsort(-sims)[:args.top_k]
            results[qid] = [cand_ids[i] for i in order]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MAP",
                        title="TextCNN — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
