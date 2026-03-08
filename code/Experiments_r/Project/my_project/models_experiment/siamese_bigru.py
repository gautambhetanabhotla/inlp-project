"""
Siamese BiGRU Network for Prior Case Retrieval
-----------------------------------------------
Architecture:
    1. A shared embedding layer (trained from scratch on the legal corpus).
    2. A shared Bidirectional GRU (BiGRU) encoder.
    3. Cosine similarity between the two encoded vectors as the final score.
    4. Training with Binary Cross-Entropy loss on positive/negative pairs.
    5. At inference, all candidates are pre-encoded; each query encodes once, then we rank.

Usage:
    python3 siamese_bigru.py
"""

import os, json, pickle, time, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

# ─────────────────────────────────────────────────────────────
# CONFIG  (change these to tune the model)
# ─────────────────────────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM     = 64         # Word embedding dimension
HIDDEN_DIM    = 128        # BiGRU hidden size (each direction)
LAYERS        = 1          # Single-layer BiGRU is 2x faster
DROPOUT       = 0.2
MAX_SEQ_LEN   = 150        # Legal first paragraphs are most informative; 150 tokens ≈ 5x speedup vs 512
BATCH_SIZE    = 64         # Larger batch = fewer grad steps, faster epoch
EPOCHS        = 5
LR            = 1e-3
NEGS_PER_POS  = 5          # Fewer negatives = smaller dataset = faster
MIN_FREQ      = 3          # Vocabulary min-count

CACHE_DIR     = "cache"
RESULTS_DIR   = "results"
EVAL_DIR      = "../../Eval"

TRAIN_GOLD_JSON = "../../Dataset/ik_train/train.json"
TEST_GOLD_JSON  = os.path.join(EVAL_DIR, "ilpcr-test-gold.json")
CAND_LIST_TXT   = os.path.join(EVAL_DIR, "ilpcr-test-cand.txt")

# ─────────────────────────────────────────────────────────────
# VOCABULARY
# ─────────────────────────────────────────────────────────────
PAD_IDX = 0
UNK_IDX = 1

class Vocabulary:
    def __init__(self, min_freq=MIN_FREQ):
        self.min_freq  = min_freq
        self.word2idx  = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
        self.idx2word  = {PAD_IDX: "<PAD>", UNK_IDX: "<UNK>"}
        self._counts   = {}

    def add_corpus(self, token_lists):
        for toks in token_lists:
            for t in toks:
                self._counts[t] = self._counts.get(t, 0) + 1
        for word, cnt in self._counts.items():
            if cnt >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx]  = word

    def encode(self, tokens, max_len=MAX_SEQ_LEN):
        ids = [self.word2idx.get(t, UNK_IDX) for t in tokens[:max_len]]
        return ids

    def __len__(self):
        return len(self.word2idx)


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
class PairDataset(Dataset):
    """Each item: (query_tokens, cand_tokens, label 0/1)"""

    def __init__(self, pairs, vocab):
        self.pairs = pairs
        self.vocab = vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        q_toks, c_toks, label = self.pairs[idx]
        q_ids = self.vocab.encode(q_toks)
        c_ids = self.vocab.encode(c_toks)
        return q_ids, c_ids, label


def collate_fn(batch):
    q_seqs, c_seqs, labels = zip(*batch)

    def pad_seqs(seqs):
        lengths = [len(s) for s in seqs]
        max_len = max(lengths)
        padded  = [s + [PAD_IDX] * (max_len - len(s)) for s in seqs]
        return torch.tensor(padded, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)

    q_padded, q_lens = pad_seqs(q_seqs)
    c_padded, c_lens = pad_seqs(c_seqs)
    labels           = torch.tensor(labels, dtype=torch.float)
    return q_padded, q_lens, c_padded, c_lens, labels


# ─────────────────────────────────────────────────────────────
# MODEL: Siamese BiGRU
# ─────────────────────────────────────────────────────────────
class DocumentEncoder(nn.Module):
    """Encodes a variable-length token sequence → fixed-size embedding."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.bigru     = nn.GRU(
            embed_dim, hidden_dim,
            num_layers     = n_layers,
            batch_first    = True,
            bidirectional  = True,
            dropout        = dropout if n_layers > 1 else 0.0
        )
        self.dropout   = nn.Dropout(dropout)
        self.proj      = nn.Linear(hidden_dim * 2, hidden_dim)  # project to hidden_dim

    def forward(self, token_ids, lengths):
        emb    = self.dropout(self.embedding(token_ids))            # (B, T, E)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.bigru(packed)                                 # h_n: (2*layers, B, H)
        # For a single-layer BiGRU, last two hidden states are fwd/bwd
        h_fwd  = h_n[-2]                                            # (B, H)
        h_bwd  = h_n[-1]                                            # (B, H)
        h      = torch.cat([h_fwd, h_bwd], dim=-1)                  # (B, 2H)
        out    = F.relu(self.proj(h))                               # (B, H)
        out    = F.normalize(out, p=2, dim=-1)                      # L2-normalize for cosine
        return out


class SiameseBiGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 n_layers=LAYERS, dropout=DROPOUT):
        super().__init__()
        self.encoder = DocumentEncoder(vocab_size, embed_dim, hidden_dim, n_layers, dropout)

    def forward(self, q_ids, q_lens, c_ids, c_lens):
        q_vec = self.encoder(q_ids, q_lens)
        c_vec = self.encoder(c_ids, c_lens)
        # Cosine similarity in [-1, 1]
        sim   = (q_vec * c_vec).sum(dim=-1)   # dot of L2-normalized vecs = cosine
        return sim

    def encode(self, token_ids, lengths):
        """Encode a single batch of documents — used for pre-computing candidate vectors."""
        self.eval()
        with torch.no_grad():
            return self.encoder(token_ids, lengths)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def get_micro_at_k(g, p, k):
    return len(set(g) & set(p[:k])), len(set(g)), len(set(p[:k]))

def evaluate_predictions(predictions, gold_dict, valid_cands, k=20):
    C, G, P = [], [], []
    for doc_id, cands in gold_dict.items():
        pred  = predictions.get(doc_id, [])
        cands = [c for c in cands if c in valid_cands and c != doc_id]
        pred  = [c for c in pred  if c in valid_cands and c != doc_id]
        c, g, p = get_micro_at_k(cands, pred, k)
        C.append(c); G.append(g); P.append(p)
    prec = sum(C) / sum(P) if sum(P) else 0
    rec  = sum(C) / sum(G) if sum(G) else 0
    return 0 if (prec == 0 or rec == 0) else 2 * prec * rec / (prec + rec)

def encode_in_batches(model, vocab, token_list, batch_size=64):
    """Encode a list of token lists into L2-normalized vectors efficiently."""
    all_vecs = []
    for i in range(0, len(token_list), batch_size):
        batch_toks  = token_list[i : i + batch_size]
        lengths     = [max(len(t), 1) for t in batch_toks]
        max_len     = min(max(lengths), MAX_SEQ_LEN)
        padded      = []
        for t, l in zip(batch_toks, lengths):
            ids = vocab.encode(t)
            ids = ids[:max_len]
            ids = ids + [PAD_IDX] * (max_len - len(ids))
            padded.append(ids)
        ids_t   = torch.tensor(padded, dtype=torch.long).to(DEVICE)
        lens_t  = torch.tensor([min(l, max_len) for l in lengths], dtype=torch.long).to(DEVICE)
        vecs    = model.encode(ids_t, lens_t).cpu().numpy()
        all_vecs.append(vecs)
    return np.vstack(all_vecs)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"[*] Device: {DEVICE}")

    # ── 1. Load Cached Tokens ──────────────────────────────────
    print("\n[1] Loading cached token data...")

    def load_cache(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    tr_c_docs, tr_c_toks = load_cache(f"{CACHE_DIR}/ik_train_candidate_cache.pkl")
    tr_q_docs, tr_q_toks = load_cache(f"{CACHE_DIR}/ik_train_query_cache.pkl")
    te_c_docs, te_c_toks = load_cache(f"{CACHE_DIR}/ik_test_candidate_cache.pkl")
    te_q_docs, te_q_toks = load_cache(f"{CACHE_DIR}/ik_test_query_cache.pkl")

    with open(TRAIN_GOLD_JSON) as f:
        train_gold_list = json.load(f)["Query Set"]
    train_gold = {item["id"]: item["relevant candidates"] for item in train_gold_list}

    with open(TEST_GOLD_JSON) as f:
        test_gold = json.load(f)

    with open(CAND_LIST_TXT) as f:
        valid_test_cands = set(f.read().strip().split("\n"))

    # ── 2. Build Vocabulary ────────────────────────────────────
    print("\n[2] Building vocabulary...")
    vocab = Vocabulary(min_freq=MIN_FREQ)
    all_toks = (list(tr_c_toks.values()) + list(tr_q_toks.values()) +
                list(te_c_toks.values()) + list(te_q_toks.values()))
    vocab.add_corpus(all_toks)
    print(f"    Vocabulary size: {len(vocab):,}")

    # ── 3. Build Pair Dataset ──────────────────────────────────
    print("\n[3] Constructing training pairs (positives + hard negatives)...")
    pairs = []

    # Candidate pool for negative sampling (combine train/test candidates)
    all_c_ids     = list(tr_c_toks.keys()) + [k for k in te_c_toks if k not in tr_c_toks]
    all_c_id_set  = set(all_c_ids)
    random.shuffle(all_c_ids)
    # BM25 pre‑computed token sets for Jaccard hard negatives
    c_tok_lookup  = {k: v for d in (tr_c_toks, te_c_toks) for k, v in d.items()}

    for q_id, q_toks in tqdm(tr_q_toks.items(), desc="Building pairs"):
        positives = [p for p in train_gold.get(q_id, []) if p in c_tok_lookup]
        if not positives:
            continue

        q_set = set(q_toks)

        # Hard negatives: candidates that share many tokens but are NOT positive
        pos_set = set(positives)
        scored  = []
        sample_pool = random.sample(all_c_ids, min(100, len(all_c_ids)))
        for c_id in sample_pool:
            if c_id in pos_set:
                continue
            c_set   = set(c_tok_lookup[c_id])
            overlap = len(q_set & c_set) / max(len(q_set | c_set), 1)
            scored.append((c_id, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        hard_negs = [c for c, _ in scored[:NEGS_PER_POS]]

        for p_id in positives:
            pairs.append((q_toks, c_tok_lookup[p_id], 1))
        for n_id in hard_negs:
            pairs.append((q_toks, c_tok_lookup[n_id], 0))

    random.shuffle(pairs)
    print(f"    Total training pairs: {len(pairs):,}  "
          f"  positives={sum(1 for _,_,l in pairs if l==1):,}")

    dataset    = PairDataset(pairs, vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_fn, num_workers=0)

    # ── 4. Build & Train Model ────────────────────────────────
    print(f"\n[4] Training Siamese BiGRU for {EPOCHS} epochs...")
    model = SiameseBiGRU(vocab_size=len(vocab)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # Cosine-Embedding loss: expects label in {-1, +1}
    criterion = nn.BCEWithLogitsLoss()
    # Map 1→high logit, 0→low logit via cosine scores (already L2-normalized so cosine = dot)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            q_ids, q_lens, c_ids, c_lens, labels = [b.to(DEVICE) for b in batch]
            optimizer.zero_grad()
            sim  = model(q_ids, q_lens, c_ids, c_lens)
            loss = criterion(sim, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"    Epoch {epoch+1}  Loss: {total_loss / len(dataloader):.4f}")

    # Save model
    torch.save(model.state_dict(), f"{RESULTS_DIR}/siamese_bigru.pt")
    print("    Model saved.")

    # ── 5. Pre-encode All Test Candidates ─────────────────────
    print("\n[5] Encoding test candidates...")
    test_c_ids   = [c for c in te_c_toks.keys() if c in valid_test_cands]
    test_c_token_lists = [te_c_toks[c] for c in test_c_ids]
    cand_vectors = encode_in_batches(model, vocab, test_c_token_lists)  # (N_cands, H)
    print(f"    Encoded {cand_vectors.shape[0]} candidates → shape {cand_vectors.shape}")

    # ── 6. Rank & Evaluate ────────────────────────────────────
    print("\n[6] Ranking test queries & evaluating...")
    predictions = {}
    for q_id, q_toks in tqdm(te_q_toks.items(), desc="Querying"):
        max_len = min(len(q_toks), MAX_SEQ_LEN)
        if max_len == 0:
            predictions[q_id] = test_c_ids
            continue
        ids = vocab.encode(q_toks)[:max_len]
        ids_t = torch.tensor([ids], dtype=torch.long).to(DEVICE)
        len_t = torch.tensor([max_len], dtype=torch.long).to(DEVICE)
        q_vec = model.encode(ids_t, len_t).cpu().numpy()             # (1, H)

        scores  = sk_cosine(q_vec, cand_vectors)[0]                  # (N_cands,)
        ranked  = np.argsort(scores)[::-1]
        predictions[q_id] = [test_c_ids[i] for i in ranked]

    with open(f"{RESULTS_DIR}/SiameseBiGRU_pred.json", "w") as f:
        json.dump(predictions, f, indent=4)

    f1 = evaluate_predictions(predictions, test_gold, valid_test_cands)
    print(f"\n{'='*45}")
    print(f"  Siamese BiGRU  Micro-F1@20 : {f1:.4f}")
    print(f"{'='*45}")

if __name__ == "__main__":
    main()
