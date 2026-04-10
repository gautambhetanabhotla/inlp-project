"""
vae_retrieval.py
================
Variational Autoencoder (VAE) document encoder for Prior Case Retrieval.

Architecture
------------
  BoW or TF-IDF document vector (sparse → dense)
  → Encoder MLP → μ, log_σ²
  → Reparameterised z ~ N(μ, σ²)
  → Decoder MLP → reconstructed document vector
  Loss: ELBO = MSE reconstruction + β × KL divergence

Retrieval: cosine similarity on μ (deterministic at inference).

Variants: standard VAE, β-VAE (β ≠ 1), Denoising VAE (noise on input).
GPU: CUDA > MPS > CPU automatically.

Run
---
    python3 vae_retrieval.py --data_dir /path/to/dataset/ --split train

Parameter grid
--------------
  Edit VAE_GRID below.  Comment out any line to skip.
"""

import os
import math
import argparse
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
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS  (edit here)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR  = "/home/raghavgrover/Desktop/Sem6/INLP/Project_Final/Experiments"
SPLIT     = "train"
TOP_K     = 1000
OUTPUT    = "results/vae_results.json"
K_VALUES  = [5, 10, 20, 50, 100]
MAX_VOCAB = 20000
BATCH_SZ  = 64

# ─────────────────────────────────────────────────────────────────────────────
# VAE_GRID  ← comment out any line to skip
# (input_type, hidden_dims, latent_dim, beta, dropout, epochs, noise)
#
#   input_type  : "bow" | "tfidf"
#   hidden_dims : list of MLP hidden layer sizes (encoder; decoder mirrors)
#   latent_dim  : VAE latent space dimension
#   beta        : KL weight (1.0 = standard VAE, >1 = β-VAE)
#   dropout     : dropout after each hidden layer
#   epochs      : training epochs
#   noise       : std of Gaussian noise added to input (0 = no denoising)
# ─────────────────────────────────────────────────────────────────────────────

VAE_GRID: List[Tuple] = [
    # ── BoW input ─────────────────────────────────────────────────────────────
    ("bow",   [512],        64,  1.0, 0.2, 20, 0.0),
    ("bow",   [512],        128, 1.0, 0.2, 20, 0.0),
    ("bow",   [512],        256, 1.0, 0.2, 20, 0.0),
    ("bow",   [1024],       128, 1.0, 0.2, 20, 0.0),
    ("bow",   [1024],       256, 1.0, 0.2, 20, 0.0),
    ("bow",   [512,  256],  128, 1.0, 0.2, 20, 0.0),
    ("bow",   [1024, 512],  256, 1.0, 0.2, 20, 0.0),
    # β-VAE variants
    ("bow",   [512],        128, 0.1, 0.2, 20, 0.0),   # under-regularised
    ("bow",   [512],        128, 5.0, 0.2, 20, 0.0),   # over-regularised
    ("bow",   [1024],       256, 0.5, 0.2, 20, 0.0),
    # Denoising VAE
    ("bow",   [512],        128, 1.0, 0.2, 20, 0.1),
    ("bow",   [512],        256, 1.0, 0.2, 20, 0.1),
    ("bow",   [1024, 512],  256, 1.0, 0.2, 20, 0.1),
    # More epochs
    ("bow",   [512],        128, 1.0, 0.2, 40, 0.0),
    ("bow",   [1024, 512],  256, 1.0, 0.2, 40, 0.0),
    ("bow",   [512],        128, 1.0, 0.4, 20, 0.0),   # higher dropout
    ("bow",   [1024, 512],  256, 1.0, 0.4, 40, 0.1),

    # ── TF-IDF input ──────────────────────────────────────────────────────────
    ("tfidf", [512],        128, 1.0, 0.2, 20, 0.0),
    ("tfidf", [512],        256, 1.0, 0.2, 20, 0.0),
    ("tfidf", [1024],       256, 1.0, 0.2, 20, 0.0),
    ("tfidf", [1024, 512],  256, 1.0, 0.2, 20, 0.0),
    ("tfidf", [512],        128, 1.0, 0.2, 40, 0.0),
    ("tfidf", [1024, 512],  256, 1.0, 0.2, 40, 0.0),
    ("tfidf", [512],        128, 1.0, 0.2, 20, 0.1),   # denoising
    ("tfidf", [1024, 512],  256, 1.0, 0.4, 40, 0.1),
]


# ─────────────────────────────────────────────────────────────────────────────
# VOCAB + BOW
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab_vae(corpus: Dict[str, List[str]],
                    max_vocab: int = MAX_VOCAB,
                    min_df: int = 2) -> Dict[str, int]:
    df: Dict[str, int] = defaultdict(int)
    for toks in corpus.values():
        for t in set(toks): df[t] += 1
    sorted_t = sorted(
        [(t, c) for t, c in df.items() if c >= min_df],
        key=lambda x: -x[1]
    )[:max_vocab]
    return {t: i for i, (t, _) in enumerate(sorted_t)}


def to_bow(tokens: List[str], vocab: Dict[str, int],
           use_tfidf: bool = False,
           idf: Dict[str, float] = None) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    for t in tokens:
        if t in vocab:
            vec[vocab[t]] += 1.0
    if use_tfidf and idf:
        for t, i in vocab.items():
            vec[i] *= idf.get(t, 1.0)
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout=0.2):
        super().__init__()
        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu   = nn.Linear(prev, latent_dim)
        self.fc_logv = nn.Linear(prev, latent_dim)
        # Decoder
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logv(h)

    def reparameterise(self, mu, logv):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logv)
        return mu

    def forward(self, x):
        mu, logv = self.encode(x)
        z        = self.reparameterise(mu, logv)
        return self.decoder(z), mu, logv

    @torch.no_grad()
    def get_mu(self, x):
        self.eval()
        mu, _ = self.encode(x)
        return F.normalize(mu, p=2, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

class BowDataset(Dataset):
    def __init__(self, vecs: np.ndarray, noise: float = 0.0):
        self.vecs  = torch.tensor(vecs, dtype=torch.float32)
        self.noise = noise

    def __len__(self): return len(self.vecs)

    def __getitem__(self, i):
        x = self.vecs[i]
        if self.noise > 0 and self.training:
            x = x + self.noise * torch.randn_like(x)
        return x, self.vecs[i]


def train_vae(model, vecs, device, epochs, beta, noise, lr=1e-3):
    dataset = BowDataset(vecs, noise)
    loader  = DataLoader(dataset, batch_size=BATCH_SZ, shuffle=True,
                         num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for x_in, x_tgt in tqdm(loader, desc=f"  Ep {epoch+1}/{epochs}",
                                  leave=False):
            x_in, x_tgt = x_in.to(device), x_tgt.to(device)
            recon, mu, logv = model(x_in)
            rec  = F.mse_loss(recon, x_tgt)
            kl   = -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())
            loss = rec + beta * kl
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item(); n += 1
        print(f"  Ep {epoch+1}/{epochs}  loss={total/max(n,1):.4f}")
    return model


@torch.no_grad()
def encode_all(model, vecs, device, batch_size=256):
    model.eval()
    all_mu = []
    for i in range(0, len(vecs), batch_size):
        x  = torch.tensor(vecs[i: i+batch_size],
                           dtype=torch.float32).to(device)
        all_mu.append(model.get_mu(x).cpu().numpy())
    return np.vstack(all_mu)


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

    c_tok = {cid: clean_text(t, True, 1) for cid, t in candidates.items()}
    q_tok = {qid: clean_text(t, True, 1) for qid, t in queries.items()}

    vocab = build_vocab_vae(c_tok, MAX_VOCAB, min_df=2)
    idf   = compute_idf(c_tok)
    V     = len(vocab)
    print(f"Vocabulary: {V} terms")

    cand_bow   = np.stack([to_bow(c_tok[c], vocab, False, None) for c in c_tok])
    query_bow  = np.stack([to_bow(q_tok[q], vocab, False, None) for q in q_tok])
    cand_tfidf = np.stack([to_bow(c_tok[c], vocab, True,  idf)  for c in c_tok])
    query_tfidf= np.stack([to_bow(q_tok[q], vocab, True,  idf)  for q in q_tok])

    cand_ids  = list(c_tok.keys())
    query_ids = list(q_tok.keys())
    qid2idx   = {qid: i for i, qid in enumerate(query_ids)}

    for (inp, hidden_dims, latent_dim, beta, dropout, epochs, noise) in VAE_GRID:
        hd  = "_".join(map(str, hidden_dims))
        name = (f"VAE_{inp}_h={hd}_z={latent_dim}_"
                f"b={beta}_drop={dropout}_ep={epochs}_noise={noise}")
        print(f"\n{'─'*64}\n  {name}\n{'─'*64}")

        tr_mat = cand_bow   if inp == "bow" else cand_tfidf
        q_mat  = query_bow  if inp == "bow" else query_tfidf
        c_mat  = cand_bow   if inp == "bow" else cand_tfidf

        model = VAE(V, hidden_dims, latent_dim, dropout)
        model = train_vae(model, tr_mat, device, epochs, beta, noise)

        cand_mu  = encode_all(model, c_mat, device)
        query_mu = encode_all(model, q_mat, device)

        results: Dict[str, List[str]] = {}
        for qid in relevance:
            if qid not in qid2idx: continue
            qvec  = query_mu[qid2idx[qid]].reshape(1, -1)
            sims  = cosine_sim_matrix(qvec, cand_mu)[0]
            order = np.argsort(-sims)[:args.top_k]
            results[qid] = [cand_ids[i] for i in order]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MAP",
                        title="VAE / β-VAE / Denoising-VAE — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
