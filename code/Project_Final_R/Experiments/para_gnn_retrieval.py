"""
para_gnn_retrieval.py
=====================
Paragraph-GNN document encoder for Prior Case Retrieval.
Adapted from IL-PCSR (Para-GNN).

Architecture
------------
  Document → paragraph chunks (each encoded with mean W2V)
  → Graph: global node + paragraph nodes
  → 2-layer GAT (Graph Attention Network)
  → Global node vector = document embedding
  → Cosine similarity retrieval

Training: Triplet loss.  GPU: CUDA > MPS > CPU.

Run
---
    python3 para_gnn_retrieval.py --data_dir /path/to/dataset/ --split train

Parameter grid
--------------
  Edit PARA_GNN_GRID below.  Comment out any line to skip.
"""

import os
import argparse
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import (
    load_split, clean_text, evaluate_all,
    save_results, print_results_table, save_results_csv,
    cosine_sim_matrix, compute_idf, get_device,
    build_w2v,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS  (edit here)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "/home/raghavgrover/Desktop/Sem6/INLP/Project_Final/Experiments"
SPLIT    = "train"
TOP_K    = 1000
OUTPUT   = "results/para_gnn_results.json"
K_VALUES = [5, 10, 20, 50, 100]

NEG_PER  = 5
BATCH_SZ = 8   # small: each item contains a variable-size graph

# ─────────────────────────────────────────────────────────────────────────────
# PARA_GNN_GRID  ← comment out any line to skip
# (w2v_dim, chunk_size, gat_hidden, n_layers, dropout, margin, epochs)
#
#   w2v_dim    : Word2Vec / paragraph embedding dimension
#   chunk_size : tokens per paragraph chunk
#   gat_hidden : GAT hidden dimension
#   n_layers   : number of stacked GAT layers
#   dropout    : dropout in GAT
#   margin     : triplet loss margin
#   epochs     : training epochs
# ─────────────────────────────────────────────────────────────────────────────

PARA_GNN_GRID: List[Tuple] = [
    # ── 1-layer GAT ───────────────────────────────────────────────────────────
    (100, 100, 64,  1, 0.3, 0.5, 10),
    (100, 100, 128, 1, 0.3, 0.5, 10),
    (200, 100, 64,  1, 0.3, 0.5, 10),
    (200, 100, 128, 1, 0.3, 0.5, 10),
    (100, 200, 64,  1, 0.3, 0.5, 10),
    (200, 200, 128, 1, 0.3, 0.5, 10),

    # ── 2-layer GAT ───────────────────────────────────────────────────────────
    (100, 100, 64,  2, 0.3, 0.5, 10),
    (100, 100, 128, 2, 0.3, 0.5, 10),
    (200, 100, 64,  2, 0.3, 0.5, 10),
    (200, 100, 128, 2, 0.3, 0.5, 10),
    (100, 200, 64,  2, 0.3, 0.5, 10),
    (200, 200, 128, 2, 0.3, 0.5, 10),

    # ── Varied dropout / margin ───────────────────────────────────────────────
    (100, 100, 128, 2, 0.5, 0.5, 10),
    (200, 100, 128, 2, 0.3, 0.3, 10),
    (200, 200, 128, 2, 0.5, 0.3, 10),

    # ── More epochs ───────────────────────────────────────────────────────────
    (100, 100, 128, 2, 0.3, 0.5, 20),
    (200, 100, 128, 2, 0.3, 0.5, 20),
    (200, 200, 128, 2, 0.3, 0.5, 20),
]


# ─────────────────────────────────────────────────────────────────────────────
# PARAGRAPH ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class MeanW2VEncoder:
    def __init__(self, w2v, dim):
        self.w2v = w2v
        self.dim = dim

    def encode(self, tokens: List[str]) -> np.ndarray:
        vecs = [self.w2v.wv[t] for t in tokens if t in self.w2v.wv]
        return np.mean(vecs, axis=0).astype(np.float32) if vecs \
               else np.zeros(self.dim, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT → GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def doc_to_graph(tokens, encoder, chunk_size, dim):
    chunks = [tokens[i: i+chunk_size]
              for i in range(0, len(tokens), chunk_size)
              if tokens[i: i+chunk_size]]
    if not chunks:
        chunks = [[]]
    para_vecs  = np.stack([encoder.encode(c) for c in chunks])
    global_vec = para_vecs.mean(axis=0, keepdims=True)
    node_feats = np.vstack([global_vec, para_vecs]).astype(np.float32)

    N   = len(node_feats)
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(1, N):
        adj[0, i] = adj[i, 0] = 1.0
    for i in range(1, N - 1):
        adj[i, i+1] = adj[i+1, i] = 1.0
    np.fill_diagonal(adj, 1.0)
    return node_feats, adj


# ─────────────────────────────────────────────────────────────────────────────
# GAT MODEL
# ─────────────────────────────────────────────────────────────────────────────

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.W    = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * out_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        Wh  = self.W(h)
        N   = Wh.size(0)
        e   = self.leaky(self.attn(
            torch.cat([Wh.unsqueeze(1).expand(-1,N,-1),
                       Wh.unsqueeze(0).expand(N,-1,-1)], dim=-1)
        ).squeeze(-1))
        e   = e.masked_fill(adj == 0, -1e9)
        alpha = self.drop(torch.softmax(e, dim=1))
        return F.elu(alpha @ Wh)


class ParaGNN(nn.Module):
    def __init__(self, node_dim, gat_hidden, n_layers=2, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(node_dim, gat_hidden)
        self.gat  = nn.ModuleList([
            GATLayer(gat_hidden, gat_hidden, dropout) for _ in range(n_layers)
        ])
        self.drop = nn.Dropout(dropout)

    def forward(self, nf, adj):
        h = F.relu(self.proj(nf))
        for layer in self.gat:
            h = self.drop(layer(h, adj))
        return F.normalize(h[0], p=2, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# TRIPLET DATASET
# ─────────────────────────────────────────────────────────────────────────────

class GNNTripletDataset(Dataset):
    def __init__(self, q_graphs, c_graphs, relevance, neg_per=5):
        self.a, self.p, self.n = [], [], []
        all_c = list(c_graphs.keys())
        for qid, rel in relevance.items():
            if qid not in q_graphs: continue
            rel_set  = set(rel)
            neg_pool = [c for c in all_c if c not in rel_set]
            for pos in rel:
                if pos not in c_graphs: continue
                for neg in random.sample(neg_pool, min(neg_per, len(neg_pool))):
                    self.a.append(q_graphs[qid])
                    self.p.append(c_graphs[pos])
                    self.n.append(c_graphs[neg])

    def __len__(self): return len(self.a)

    def __getitem__(self, i):
        return self.a[i], self.p[i], self.n[i]


def collate_graphs(batch):
    a, p, n = zip(*batch)
    return list(a), list(p), list(n)


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN + ENCODE
# ─────────────────────────────────────────────────────────────────────────────

def train_gnn(model, dataset, device, epochs, lr=1e-3,
              batch_size=8, margin=0.5):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate_graphs)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for anchors, positives, negatives in tqdm(
                loader, desc=f"  Ep {epoch+1}/{epochs}", leave=False):
            ea_l, ep_l, en_l = [], [], []
            for (nf_a, adj_a), (nf_p, adj_p), (nf_n, adj_n) in \
                    zip(anchors, positives, negatives):
                ea_l.append(model(torch.tensor(nf_a,  device=device),
                                  torch.tensor(adj_a, device=device)))
                ep_l.append(model(torch.tensor(nf_p,  device=device),
                                  torch.tensor(adj_p, device=device)))
                en_l.append(model(torch.tensor(nf_n,  device=device),
                                  torch.tensor(adj_n, device=device)))
            ea, ep, en = (torch.stack(x) for x in (ea_l, ep_l, en_l))
            loss = F.relu(margin - (ea*ep).sum(1) + (ea*en).sum(1)).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item(); n += 1
        print(f"  Ep {epoch+1}/{epochs}  loss={total/max(n,1):.4f}")
    return model


@torch.no_grad()
def encode_gnn_all(model, graphs, device):
    model.eval()
    ids, vecs = list(graphs.keys()), []
    for did in ids:
        nf, adj = graphs[did]
        vecs.append(model(torch.tensor(nf,  device=device),
                          torch.tensor(adj, device=device)).cpu().numpy())
    return ids, np.stack(vecs)


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

    q_tok = {qid: clean_text(t, True, 1) for qid, t in queries.items()}
    c_tok = {cid: clean_text(t, True, 1) for cid, t in candidates.items()}
    all_tok = list(q_tok.values()) + list(c_tok.values())

    w2v_cache: Dict[int, object] = {}

    for (w2v_dim, chunk_size, gat_hidden, n_layers,
         dropout, margin, epochs) in PARA_GNN_GRID:

        name = (f"ParaGNN_w2v={w2v_dim}_chunk={chunk_size}"
                f"_gat={gat_hidden}x{n_layers}"
                f"_drop={dropout}_m={margin}_ep={epochs}")
        print(f"\n{'─'*64}\n  {name}\n{'─'*64}")

        if w2v_dim not in w2v_cache:
            print(f"  Training W2V (dim={w2v_dim}) ...")
            w2v_cache[w2v_dim] = build_w2v(
                all_tok, vector_size=w2v_dim, window=5,
                min_count=2, sg=1, epochs=5)
        w2v = w2v_cache[w2v_dim]
        enc = MeanW2VEncoder(w2v, w2v_dim)

        print("  Building document graphs ...")
        q_graphs = {qid: doc_to_graph(t, enc, chunk_size, w2v_dim)
                    for qid, t in tqdm(q_tok.items(), leave=False)}
        c_graphs = {cid: doc_to_graph(t, enc, chunk_size, w2v_dim)
                    for cid, t in tqdm(c_tok.items(), leave=False)}

        model   = ParaGNN(w2v_dim, gat_hidden, n_layers, dropout)
        dataset = GNNTripletDataset(q_graphs, c_graphs, relevance, neg_per=NEG_PER)
        if len(dataset) == 0:
            print("  [SKIP] No training pairs.")
            continue

        model = train_gnn(model, dataset, device, epochs,
                          batch_size=BATCH_SZ, margin=margin)

        cand_ids, cand_mat = encode_gnn_all(model, c_graphs, device)

        results: Dict[str, List[str]] = {}
        for qid in relevance:
            if qid not in q_graphs: continue
            nf, adj = q_graphs[qid]
            model.eval()
            with torch.no_grad():
                qvec = model(torch.tensor(nf,  device=device),
                             torch.tensor(adj, device=device)
                             ).cpu().numpy().reshape(1, -1)
            sims  = cosine_sim_matrix(qvec, cand_mat)[0]
            order = np.argsort(-sims)[:args.top_k]
            results[qid] = [cand_ids[i] for i in order]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MAP",
                        title="Para-GNN — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
