"""
siamese_bilstm_retrieval.py
===========================
Siamese BiLSTM + Hierarchical BiLSTM document encoders for PCR.

Flat BiLSTM: pooling options = last / mean / attention
Hier BiLSTM: sentence-level LSTM → document-level LSTM

Training: Triplet loss.  GPU: CUDA > MPS > CPU.

Run
---
    python3 siamese_bilstm_retrieval.py --data_dir /path/to/dataset/ --split train

Parameter grid
--------------
  Edit FLAT_GRID and HIER_GRID below.  Comment out any line to skip.
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
    build_vocab, tokens_to_ids, tokens_to_hier_ids,
    build_w2v, make_embed_matrix,
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS  (edit here)
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = "/home/raghavgrover/Desktop/Sem6/INLP/Project_Final/Experiments"
SPLIT    = "train"
TOP_K    = 1000
OUTPUT   = "results/bilstm_results.json"
K_VALUES = [5, 10, 20, 50, 100]

MAX_LEN   = 512    # flat model: tokens per doc
SENT_LEN  = 64     # hierarchical: tokens per sentence
MAX_SENTS = 16     # hierarchical: sentences per doc
NEG_PER   = 5
BATCH_SZ  = 16
EMBED_DIM = 100    # shared W2V embedding dim

# ─────────────────────────────────────────────────────────────────────────────
# FLAT_GRID  ← comment out any line to skip
# (hidden_dim, num_layers, pooling, dropout, margin, epochs, use_w2v, ngram)
#
#   hidden_dim : BiLSTM total hidden size (split equally fwd/bwd)
#   num_layers : stacked LSTM layers
#   pooling    : "last" | "mean" | "attention"
#   dropout    : dropout rate
#   margin     : triplet loss margin
#   epochs     : training epochs
#   use_w2v    : pre-initialise embeddings from Word2Vec
#   ngram      : tokenisation order
# ─────────────────────────────────────────────────────────────────────────────

FLAT_GRID: List[Tuple] = [
    # ── Random init ───────────────────────────────────────────────────────────
    (128, 1, "mean",      0.3, 0.5, 10, False, 1),
    (256, 1, "mean",      0.3, 0.5, 10, False, 1),
    (256, 2, "mean",      0.3, 0.5, 10, False, 1),
    (128, 1, "attention", 0.3, 0.5, 10, False, 1),
    (256, 1, "attention", 0.3, 0.5, 10, False, 1),
    (256, 2, "attention", 0.3, 0.5, 10, False, 1),
    (128, 1, "last",      0.3, 0.5, 10, False, 1),
    (256, 1, "last",      0.3, 0.5, 10, False, 1),
    (256, 1, "mean",      0.5, 0.5, 10, False, 1),    # higher dropout
    (256, 1, "mean",      0.3, 0.3, 10, False, 1),    # smaller margin
    (256, 1, "mean",      0.3, 0.5, 20, False, 1),    # more epochs
    (256, 1, "attention", 0.3, 0.5, 20, False, 1),

    # ── W2V init ──────────────────────────────────────────────────────────────
    (128, 1, "mean",      0.3, 0.5, 10, True,  1),
    (256, 1, "mean",      0.3, 0.5, 10, True,  1),
    (256, 1, "attention", 0.3, 0.5, 10, True,  1),
    (256, 2, "attention", 0.3, 0.5, 10, True,  1),
    (256, 1, "attention", 0.3, 0.5, 20, True,  1),

    # ── Bigrams ───────────────────────────────────────────────────────────────
    (256, 1, "mean",      0.3, 0.5, 10, False, 2),
    (256, 1, "attention", 0.3, 0.5, 10, False, 2),
    (256, 1, "attention", 0.3, 0.5, 10, True,  2),
]

# ─────────────────────────────────────────────────────────────────────────────
# HIER_GRID  ← comment out any line to skip
# (sent_hidden, doc_hidden, dropout, margin, epochs, use_w2v)
# ─────────────────────────────────────────────────────────────────────────────

HIER_GRID: List[Tuple] = [
    (128, 128, 0.3, 0.5, 10, False),
    (256, 256, 0.3, 0.5, 10, False),
    (128, 256, 0.3, 0.5, 10, False),
    (256, 256, 0.5, 0.5, 10, False),
    (256, 256, 0.3, 0.3, 10, False),
    (256, 256, 0.3, 0.5, 20, False),
    (128, 256, 0.3, 0.5, 10, True),
    (256, 256, 0.3, 0.5, 10, True),
    (256, 256, 0.3, 0.5, 20, True),
]

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        hidden: (B, T, H)
        mask  : (B, T)  1=real token, 0=pad
        """
        scores = self.attn(hidden).squeeze(-1)          # (B, T)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        return (hidden * weights).sum(dim=1)             # (B, H)


# ── FLAT BILSTM ENCODER ──────────────────────────────────────────────────────

class BiLSTMEncoder(nn.Module):
    def __init__(
        self,
        vocab_size, embed_dim, hidden_dim,
        num_layers=1, dropout=0.3,
        pooling="mean",
        pretrained=None,
    ):
        super().__init__()
        self.pooling  = pooling
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(pretrained))

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        if pooling == "attention":
            self.attn_pool = AttentionPooling(hidden_dim)

        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) → (B, out_dim)"""
        mask = (x != 0).float()             # (B, T)
        e    = self.drop(self.embedding(x)) # (B, T, E)
        out, (h, _) = self.lstm(e)          # out:(B,T,H)  h:(2*L, B, H//2)

        if self.pooling == "last":
            # Concatenate forward and backward last hidden states
            vec = torch.cat([h[-2], h[-1]], dim=1)  # (B, H)
        elif self.pooling == "mean":
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            vec = (out * mask.unsqueeze(-1)).sum(dim=1) / lengths
        elif self.pooling == "attention":
            vec = self.attn_pool(out, mask)
        else:
            vec = out.max(dim=1).values

        vec = F.normalize(self.drop(vec), p=2, dim=1)
        return vec


# ── HIERARCHICAL BILSTM ──────────────────────────────────────────────────────

class HierBiLSTMEncoder(nn.Module):
    """
    Sentence BiLSTM → Document BiLSTM.
    Input shape: (B, n_sents, sent_len)
    """
    def __init__(
        self, vocab_size, embed_dim, sent_hidden, doc_hidden,
        dropout=0.3, pretrained=None
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(pretrained))

        self.sent_lstm = nn.LSTM(
            embed_dim, sent_hidden // 2, batch_first=True,
            bidirectional=True
        )
        self.doc_lstm = nn.LSTM(
            sent_hidden, doc_hidden // 2, batch_first=True,
            bidirectional=True
        )
        self.drop    = nn.Dropout(dropout)
        self.out_dim = doc_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, T)"""
        B, S, T = x.shape
        # Reshape to (B*S, T) for sentence encoding
        x_flat = x.view(B * S, T)
        e      = self.drop(self.embedding(x_flat))   # (B*S, T, E)
        _, (h, _) = self.sent_lstm(e)
        # Last hidden: concat fwd/bwd → (B*S, sent_hidden)
        s_vec  = torch.cat([h[-2], h[-1]], dim=1)
        s_vec  = s_vec.view(B, S, -1)               # (B, S, sent_hidden)
        # Doc-level
        _, (hd, _) = self.doc_lstm(self.drop(s_vec))
        doc_vec = torch.cat([hd[-2], hd[-1]], dim=1) # (B, doc_hidden)
        return F.normalize(self.drop(doc_vec), p=2, dim=1)


# ── TRIPLET DATASET ──────────────────────────────────────────────────────────

class TripletDataset(Dataset):
    def __init__(self, q_ids, c_ids, relevance, neg_per=5):
        self.a, self.p, self.n = [], [], []
        all_cands = list(c_ids.keys())
        for qid, rel in relevance.items():
            if qid not in q_ids: continue
            rel_set = set(rel)
            neg_pool = [c for c in all_cands if c not in rel_set]
            for pos in rel:
                if pos not in c_ids: continue
                for neg in random.sample(neg_pool, min(neg_per, len(neg_pool))):
                    self.a.append(q_ids[qid])
                    self.p.append(c_ids[pos])
                    self.n.append(c_ids[neg])

    def __len__(self): return len(self.a)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.a[idx], dtype=torch.long),
            torch.tensor(self.p[idx], dtype=torch.long),
            torch.tensor(self.n[idx], dtype=torch.long),
        )


class HierTripletDataset(Dataset):
    """For hierarchical model: input is (n_sents, sent_len)."""
    def __init__(self, q_ids, c_ids, relevance, neg_per=5):
        self.a, self.p, self.n = [], [], []
        all_cands = list(c_ids.keys())
        for qid, rel in relevance.items():
            if qid not in q_ids: continue
            rel_set = set(rel)
            neg_pool = [c for c in all_cands if c not in rel_set]
            for pos in rel:
                if pos not in c_ids: continue
                for neg in random.sample(neg_pool, min(neg_per, len(neg_pool))):
                    self.a.append(q_ids[qid])
                    self.p.append(c_ids[pos])
                    self.n.append(c_ids[neg])

    def __len__(self): return len(self.a)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.a[idx], dtype=torch.long),
            torch.tensor(self.p[idx], dtype=torch.long),
            torch.tensor(self.n[idx], dtype=torch.long),
        )


# ── TRAINING ─────────────────────────────────────────────────────────────────

def train_model(model, dataset, device, epochs, lr=1e-3, batch_size=16, margin=0.5):
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for a, p, neg in tqdm(loader, desc=f"  Ep {epoch+1}/{epochs}", leave=False):
            a, p, neg = a.to(device), p.to(device), neg.to(device)
            ea, ep, en = model(a), model(p), model(neg)
            loss = F.relu(margin - (ea * ep).sum(1) + (ea * en).sum(1)).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item(); n += 1
        scheduler.step()
        print(f"  Ep {epoch+1}/{epochs}  loss={total/max(n,1):.4f}")
    return model


# ── ENCODE ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_all(model, id_seqs, device, batch_size=32):
    model.eval()
    ids, vecs = list(id_seqs.keys()), []
    for i in range(0, len(ids), batch_size):
        batch = ids[i: i + batch_size]
        x = torch.tensor([id_seqs[d] for d in batch], dtype=torch.long).to(device)
        vecs.append(model(x).cpu().numpy())
    return ids, np.vstack(vecs)


# moved to utils
def _tokens_to_hier_ids_local(tokens, w2i, max_sents, sent_len):
    """Convert flat token list → (max_sents, sent_len) int array."""
    unk = w2i["<UNK>"]
    chunks = []
    for i in range(0, len(tokens), sent_len):
        chunk = tokens[i: i + sent_len]
        ids   = [w2i.get(t, unk) for t in chunk]
        ids  += [0] * (sent_len - len(ids))
        chunks.append(ids)
    # Pad or truncate to max_sents
    while len(chunks) < max_sents:
        chunks.append([0] * sent_len)
    return [c[:sent_len] for c in chunks[:max_sents]]


# ── PARAMETER GRID ───────────────────────────────────────────────────────────





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--split",    default=SPLIT)
    parser.add_argument("--top_k",    type=int, default=TOP_K)
    parser.add_argument("--output",   default=OUTPUT)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    queries, candidates, relevance = load_split(args.data_dir, args.split)
    all_results = []

    # ── Flat tokenisation ─────────────────────────────────────────────────
    tok_cache_q: Dict = {}
    tok_cache_c: Dict = {}

    def get_tok(ngram):
        if ngram not in tok_cache_q:
            tok_cache_q[ngram] = {qid: clean_text(t, True, ngram)
                                   for qid, t in queries.items()}
            tok_cache_c[ngram] = {cid: clean_text(t, True, ngram)
                                   for cid, t in candidates.items()}
        return tok_cache_q[ngram], tok_cache_c[ngram]

    # ─────────────────────────────────────────────────────────────────────
    # FLAT BiLSTM
    # ─────────────────────────────────────────────────────────────────────
    for cfg in FLAT_GRID:
        (hidden, layers, pooling, dropout, margin, epochs, use_w2v, ngram) = cfg
        name = (f"BiLSTM_{hidden}x{layers}_{pooling}_"
                f"drop={dropout}_m={margin}_ep={epochs}_"
                f"w2v={use_w2v}_ng={ngram}")
        print(f"\n{'='*64}\n  {name}\n{'='*64}")

        q_tok, c_tok = get_tok(ngram)
        all_tok = list(q_tok.values()) + list(c_tok.values())
        w2i, vocab = build_vocab(all_tok, min_freq=2)
        V = len(vocab)

        pretrained = None
        if use_w2v and GENSIM_OK:
            w2v = build_w2v(all_tok, vector_size=EMBED_DIM, window=5, min_count=2, sg=1, epochs=5)
            pretrained = make_embed_matrix(vocab, w2i, EMBED_DIM, w2v)

        q_ids = {qid: tokens_to_ids(t, w2i, MAX_LEN) for qid, t in q_tok.items()}
        c_ids = {cid: tokens_to_ids(t, w2i, MAX_LEN) for cid, t in c_tok.items()}

        model   = BiLSTMEncoder(V, EMBED_DIM, hidden, layers, dropout, pooling, pretrained)
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
            qx = torch.tensor([q_ids[qid]], dtype=torch.long).to(device)
            model.eval()
            with torch.no_grad():
                qvec = model(qx).cpu().numpy()
            sims  = cosine_sim_matrix(qvec, cand_mat)[0]
            order = np.argsort(-sims)[: args.top_k]
            results[qid] = [cand_ids[i] for i in order]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        m["model"] = name
        all_results.append(m)

    # ─────────────────────────────────────────────────────────────────────
    # HIERARCHICAL BiLSTM
    # ─────────────────────────────────────────────────────────────────────
    q_tok1, c_tok1 = get_tok(1)   # always unigram for hierarchical
    all_tok1 = list(q_tok1.values()) + list(c_tok1.values())
    w2i_h, vocab_h = build_vocab(all_tok1, min_freq=2)
    V_h = len(vocab_h)

    for cfg in HIER_GRID:
        (s_hid, d_hid, dropout, margin, epochs, use_w2v) = cfg
        name = (f"HierBiLSTM_sh={s_hid}_dh={d_hid}_"
                f"drop={dropout}_m={margin}_ep={epochs}_w2v={use_w2v}")
        print(f"\n{'='*64}\n  {name}\n{'='*64}")

        pretrained_h = None
        if use_w2v and GENSIM_OK:
            w2v = build_w2v(all_tok1, vector_size=EMBED_DIM, window=5, min_count=2, sg=1, epochs=5)
            pretrained_h = make_embed_matrix(vocab_h, w2i_h, EMBED_DIM, w2v)

        # Build hierarchical ID sequences
        q_hier = {qid: tokens_to_hier_ids(t, w2i_h, MAX_SENTS, SENT_LEN)
                  for qid, t in q_tok1.items()}
        c_hier = {cid: tokens_to_hier_ids(t, w2i_h, MAX_SENTS, SENT_LEN)
                  for cid, t in c_tok1.items()}

        model   = HierBiLSTMEncoder(V_h, EMBED_DIM, s_hid, d_hid, dropout, pretrained_h)
        dataset = HierTripletDataset(q_hier, c_hier, relevance, neg_per=NEG_PER)
        if len(dataset) == 0:
            print("  [SKIP] No training pairs.")
            continue

        model = train_model(model, dataset, device, epochs,
                            batch_size=BATCH_SZ, margin=margin)

        cand_ids, cand_mat = encode_all(model, c_hier, device)

        results: Dict[str, List[str]] = {}
        for qid in relevance:
            if qid not in q_hier: continue
            qx = torch.tensor([q_hier[qid]], dtype=torch.long).to(device)
            model.eval()
            with torch.no_grad():
                qvec = model(qx).cpu().numpy()
            sims  = cosine_sim_matrix(qvec, cand_mat)[0]
            order = np.argsort(-sims)[: args.top_k]
            results[qid] = [cand_ids[i] for i in order]

        m = evaluate_all(results, relevance, k_values=K_VALUES, label=name)
        m["model"] = name
        all_results.append(m)

    save_results(all_results, args.output)
    print_results_table(all_results, sort_by="MAP",
                        title="Siamese BiLSTM / Hierarchical BiLSTM — ALL CONFIGURATIONS")
    save_results_csv(all_results, args.output.replace(".json", ".csv"))


if __name__ == "__main__":
    main()
