"""
rnn_autoencoder_retrieval.py
=============================
GRU / LSTM Autoencoder document encoder for Prior Case Retrieval.
From: "Prior Case Retrieval for the Court of Cassation of Turkey."

Architecture
------------
  Tokens → W2V embeddings → chunk (100 tokens)
  → Encoder RNN → latent vector
  → Decoder RNN → reconstructed embeddings
  Loss: MSE(original embeddings, reconstructed)
  Document vector: mean of chunk latent vectors

Hybrid variant: BM25 top-N → rerank with autoencoder cosine similarity.
GPU: CUDA > MPS > CPU.

Run
---
    python3 rnn_autoencoder_retrieval.py --data_dir /path/to/dataset/ --split train

Parameter grid
--------------
  Edit RNN_GRID and HYBRID_TOPN below.  Comment out any line to skip.
"""

import os
import argparse
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import (
    load_split, clean_text, evaluate_all,
    save_results, print_results_table, save_results_csv,
    cosine_sim_matrix, get_device, build_w2v, mean_vec,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS  (edit here)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "/home/raghavgrover/Desktop/Sem6/INLP/Project_Final/Experiments"
SPLIT    = "train"
TOP_K    = 1000
OUTPUT   = "results/rnn_ae_results.json"
K_VALUES = [5, 10, 20, 50, 100]

WORKERS  = 4
EPOCHS   = 5

# ─────────────────────────────────────────────────────────────────────────────
# RNN_GRID  ← comment out any line to skip
# (rnn_type, hidden_dim, embed_dim, chunk_size)
#
#   rnn_type   : "GRU" | "LSTM"
#   hidden_dim : encoder/decoder hidden size
#   embed_dim  : W2V token embedding dimension
#   chunk_size : tokens per chunk fed to the RNN
# ─────────────────────────────────────────────────────────────────────────────

RNN_GRID: List[Tuple] = [
    # ── GRU ───────────────────────────────────────────────────────────────────
    ("GRU",  128, 100, 100),
    ("GRU",  256, 100, 100),
    ("GRU",  256, 200, 100),
    ("GRU",  128, 100,  50),
    ("GRU",  256, 100,  50),
    ("GRU",  256, 200,  50),
    ("GRU",  128, 200, 100),
    ("GRU",  512, 200, 100),

    # ── LSTM ──────────────────────────────────────────────────────────────────
    ("LSTM", 128, 100, 100),
    ("LSTM", 256, 100, 100),
    ("LSTM", 256, 200, 100),
    ("LSTM", 128, 100,  50),
    ("LSTM", 256, 100,  50),
    ("LSTM", 256, 200,  50),
    ("LSTM", 128, 200, 100),
    ("LSTM", 512, 200, 100),
]

# BM25 first-stage N values for hybrid experiments
# Each value here adds one hybrid run per RNN config above.
# Comment out values you don't want.
HYBRID_TOPN: List[int] = [
    50,
    100,
    200,
]


# ─────────────────────────────────────────────────────────────────────────────
# RNN AUTOENCODER MODEL
# ─────────────────────────────────────────────────────────────────────────────

class RNNAutoEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, rnn_type="GRU"):
        super().__init__()
        rnn_cls = nn.GRU if rnn_type.upper() == "GRU" else nn.LSTM
        self.encoder = rnn_cls(embed_dim, hidden_dim,
                                batch_first=True, bidirectional=False)
        self.decoder = rnn_cls(hidden_dim, embed_dim,
                                batch_first=True, bidirectional=False)
        self.rnn_type  = rnn_type.upper()
        self.hidden_dim = hidden_dim

    def _get_hidden(self, h):
        return h[0].squeeze(0) if self.rnn_type == "LSTM" else h.squeeze(0)

    def encode(self, x):
        """x: (B, T, E) → (B, H)"""
        _, h = self.encoder(x)
        return self._get_hidden(h)

    def forward(self, x):
        code = self.encode(x)                         # (B, H)
        code_seq = code.unsqueeze(1).expand(-1, x.size(1), -1)
        recon, _ = self.decoder(code_seq)
        return recon, code


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT → CHUNK MATRICES
# ─────────────────────────────────────────────────────────────────────────────

def tokens_to_matrix(tokens, w2v, dim):
    return [list(w2v.wv[t]) for t in tokens if t in w2v.wv]


def chunk_matrix(mat, chunk_size):
    return [mat[i: i+chunk_size]
            for i in range(0, len(mat), chunk_size)
            if mat[i: i+chunk_size]]


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_autoencoder(model, doc_matrices, device, epochs=5, lr=1e-3):
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.to(device)
    random.shuffle(doc_matrices)

    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for mat in tqdm(doc_matrices, desc=f"  Ep {epoch+1}/{epochs}",
                        leave=False):
            for chunk in chunk_matrix(mat, model.hidden_dim):
                if len(chunk) < 2: continue
                x = torch.tensor([chunk], dtype=torch.float32).to(device)
                recon, _ = model(x)
                loss = loss_fn(recon, x)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item(); n += 1
        print(f"  Ep {epoch+1}/{epochs}  loss={total/max(n,1):.5f}")
    return model


@torch.no_grad()
def embed_doc(mat, model, chunk_size, device):
    model.eval()
    chunks = chunk_matrix(mat, chunk_size)
    vecs   = []
    for c in chunks:
        if not c: continue
        x = torch.tensor([c], dtype=torch.float32).to(device)
        vecs.append(model.encode(x).squeeze(0).cpu().numpy())
    if not vecs:
        return np.zeros(model.hidden_dim, dtype=np.float32)
    return np.mean(vecs, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# MINI BM25 (for hybrid)
# ─────────────────────────────────────────────────────────────────────────────

class MiniBM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1; self.b = b

    def fit(self, corpus):
        self.ids  = list(corpus.keys()); self.N = len(self.ids)
        self.df   = defaultdict(int); self.tfs = []; self.lens = []
        for did in self.ids:
            tf = defaultdict(int)
            for t in corpus[did]: tf[t] += 1
            self.tfs.append(dict(tf)); self.lens.append(len(corpus[did]))
            for t in tf: self.df[t] += 1
        self.avgdl = sum(self.lens) / self.N if self.N else 1.0

    def _idf(self, t):
        import math
        d = self.df.get(t, 0)
        return math.log((self.N - d + 0.5) / (d + 0.5) + 1)

    def retrieve(self, qtoks, top_n=200):
        import math
        scores = []
        for i, did in enumerate(self.ids):
            sc = sum(self._idf(t) * self.tfs[i].get(t, 0) * (self.k1+1) / (
                     self.tfs[i].get(t, 0) +
                     self.k1*(1-self.b+self.b*self.lens[i]/self.avgdl))
                     for t in set(qtoks) if self.tfs[i].get(t, 0))
            scores.append((did, sc))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in scores[:top_n]]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--split",    default=SPLIT)
    parser.add_argument("--top_k",    type=int, default=TOP_K)
    parser.add_argument("--output",   default=OUTPUT)
    parser.add_argument("--epochs",   type=int, default=EPOCHS)
    parser.add_argument("--workers",  type=int, default=WORKERS)
    args = parser.parse_args()

    device = get_device()
    queries, candidates, relevance = load_split(args.data_dir, args.split)
    all_results = []

    c_tok = {cid: clean_text(t, True, 1) for cid, t in candidates.items()}
    q_tok = {qid: clean_text(t, True, 1) for qid, t in queries.items()}

    # Pre-build BM25 for hybrid runs
    print("Building BM25 for hybrid ...")
    bm25 = MiniBM25()
    bm25.fit(c_tok)

    w2v_cache: Dict[int, object] = {}

    for (rnn_type, hidden_dim, embed_dim, chunk_size) in RNN_GRID:
        name_base = (f"RNN_{rnn_type}_h={hidden_dim}"
                     f"_emb={embed_dim}_chunk={chunk_size}")
        print(f"\n{'─'*64}\n  {name_base}\n{'─'*64}")

        if embed_dim not in w2v_cache:
            print(f"  Training W2V (dim={embed_dim}) ...")
            all_tok = [t for t in list(c_tok.values())+list(q_tok.values()) if t]
            w2v_cache[embed_dim] = build_w2v(all_tok, vector_size=embed_dim,
                                              window=5, min_count=2, sg=1,
                                              workers=args.workers, epochs=5)
        w2v = w2v_cache[embed_dim]

        c_mats = {cid: tokens_to_matrix(t, w2v, embed_dim)
                  for cid, t in c_tok.items()}
        q_mats = {qid: tokens_to_matrix(t, w2v, embed_dim)
                  for qid, t in q_tok.items()}

        train_mats = [m for m in c_mats.values() if len(m) >= 2]
        model = RNNAutoEncoder(embed_dim, hidden_dim, rnn_type)
        model = train_autoencoder(model, train_mats, device, args.epochs)

        print("  Embedding candidates ...")
        c_vecs = {cid: embed_doc(mat, model, chunk_size, device)
                  for cid, mat in tqdm(c_mats.items(), leave=False)}

        # ── Standalone ──────────────────────────────────────────────────────
        name = f"{name_base}_standalone"
        results: Dict[str, List[str]] = {}
        for qid in relevance:
            if qid not in q_mats: continue
            qmat = q_mats[qid]
            if not qmat: continue
            qvec = embed_doc(qmat, model, chunk_size, device).reshape(1, -1)
            cand_ids = list(c_vecs.keys())
            cand_mat = np.stack([c_vecs[c] for c in cand_ids])
            sims  = cosine_sim_matrix(qvec, cand_mat)[0]
            order = np.argsort(-sims)[:args.top_k]
            results[qid] = [cand_ids[i] for i in order]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

        # ── Hybrid: BM25 top-N → rerank with autoencoder ─────────────────────
        for top_n in HYBRID_TOPN:
            name = f"{name_base}_hybrid_bm25top{top_n}"
            results_h: Dict[str, List[str]] = {}
            for qid in relevance:
                if qid not in q_tok: continue
                bm25_top = bm25.retrieve(q_tok[qid], top_n=top_n)
                if qid not in q_mats or not q_mats[qid]:
                    results_h[qid] = bm25_top
                    continue
                qvec = embed_doc(q_mats[qid], model, chunk_size, device).reshape(1,-1)
                subset_ids = [c for c in bm25_top if c in c_vecs]
                if not subset_ids:
                    results_h[qid] = bm25_top
                    continue
                sub_mat = np.stack([c_vecs[c] for c in subset_ids])
                sims    = cosine_sim_matrix(qvec, sub_mat)[0]
                order   = np.argsort(-sims)
                results_h[qid] = [subset_ids[i] for i in order]

            m = evaluate_all(results_h, relevance, k_values=K_VALUES, label=name)
            all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MAP",
                        title="RNN Autoencoder — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
