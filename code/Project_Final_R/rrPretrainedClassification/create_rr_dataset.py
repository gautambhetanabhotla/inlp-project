"""
create_rr_dataset.py
====================
Classifies every sentence in ik_train or ik_test (both candidate/ and query/ dirs)
using the pre-trained MTL rhetorical-role classifier and writes annotated output to
ik_train_rr/ or ik_test_rr/.

Output format (tab-separated, one sentence per line):
    <RR_LABEL>\t<sentence text>

Example:
    Fact    The appellant filed a petition on 7 March 1962.
    Statute Under Section 82 of the Representation of the People Act...

The CSV and JSON mapping files are copied unchanged.

Progress is tracked in rr_dataset_progress.json so the run can be safely interrupted
and resumed without reprocessing already-completed files.

Usage
-----
    python create_rr_dataset.py train   # process ik_train  → ik_train_rr
    python create_rr_dataset.py test    # process ik_test   → ik_test_rr
"""

import sys
import os
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import nltk
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Download NLTK data silently
# ──────────────────────────────────────────────────────────────────────────────
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
CHECKPOINT_FILE = "rr_dataset_progress.json"

INPUT_DIRS  = {"train": "ik_train",    "test": "ik_test"}
OUTPUT_DIRS = {"train": "ik_train_rr", "test": "ik_test_rr"}
SUBDIRS     = ["candidate", "query"]

BERT_BATCH_SIZE = 24        # sentences per BERT forward pass (increase if GPU/RAM allows)
MODEL_PATH  = "model_state.tar"
TAG2IDX_PATH = "tag2idx.json"

SKIP_LABELS = {"<pad>", "<start>", "<end>"}


# ──────────────────────────────────────────────────────────────────────────────
# Model classes (copied exactly from test.py)
# ──────────────────────────────────────────────────────────────────────────────
class LSTM_Emitter_Binary(nn.Module):
    def __init__(self, n_tags, emb_dim, hidden_dim, drop=0.5, device="cuda"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(emb_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(drop)
        self.hidden2tag = nn.Linear(hidden_dim, n_tags)
        self.device = device

    def init_hidden(self, batch_size):
        return (
            torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device),
            torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device),
        )

    def forward(self, sequences):
        self.hidden = self.init_hidden(sequences.shape[0])
        x, self.hidden = self.lstm(sequences, self.hidden)
        x_new = self.dropout(x)
        x_new = self.hidden2tag(x_new)
        return x_new, x


class LSTM_Emitter(nn.Module):
    def __init__(self, n_tags, emb_dim, hidden_dim, drop=0.5, device="cuda"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(emb_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(drop)
        self.hidden2tag = nn.Linear(2 * hidden_dim, n_tags)
        self.device = device

    def init_hidden(self, batch_size):
        return (
            torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device),
            torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device),
        )

    def forward(self, sequences, hidden_binary):
        self.hidden = self.init_hidden(sequences.shape[0])
        x, self.hidden = self.lstm(sequences, self.hidden)
        final = torch.zeros((x.shape[0], x.shape[1], 2 * x.shape[2])).to(self.device)
        for b, doc in enumerate(x):
            for i, sent in enumerate(doc):
                final[b][i] = torch.cat((x[b][i], hidden_binary[b][i]), 0)
        final = self.dropout(final)
        final = self.hidden2tag(final)
        return final


class CRF(nn.Module):
    def __init__(self, n_tags, sos_tag_idx, eos_tag_idx, pad_tag_idx=None):
        super().__init__()
        self.n_tags = n_tags
        self.SOS_TAG_IDX = sos_tag_idx
        self.EOS_TAG_IDX = eos_tag_idx
        self.PAD_TAG_IDX = pad_tag_idx
        self.transitions = nn.Parameter(torch.empty(self.n_tags, self.n_tags))
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        self.transitions.data[:, self.SOS_TAG_IDX] = -1e6
        self.transitions.data[self.EOS_TAG_IDX, :] = -1e6
        if self.PAD_TAG_IDX is not None:
            self.transitions.data[self.PAD_TAG_IDX, :] = -1e6
            self.transitions.data[:, self.PAD_TAG_IDX] = -1e6
            self.transitions.data[self.PAD_TAG_IDX, self.EOS_TAG_IDX] = 0.0
            self.transitions.data[self.PAD_TAG_IDX, self.PAD_TAG_IDX] = 0.0

    def decode(self, emissions, mask=None):
        if mask is None:
            mask = torch.ones(emissions.shape[:2])
        _, sequences = self._viterbi_decode(emissions, mask)
        return _, sequences

    def _viterbi_decode(self, emissions, mask):
        batch_size, seq_len, n_tags = emissions.shape
        alphas = self.transitions[self.SOS_TAG_IDX, :].unsqueeze(0) + emissions[:, 0]
        backpointers = []
        for i in range(1, seq_len):
            e_scores = emissions[:, i].unsqueeze(1)
            t_scores = self.transitions.unsqueeze(0)
            a_scores = alphas.unsqueeze(2)
            scores = e_scores + t_scores + a_scores
            max_scores, max_tags = torch.max(scores, dim=1)
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * max_scores + (1 - is_valid) * alphas
            backpointers.append(max_tags.t())
        last_transition = self.transitions[:, self.EOS_TAG_IDX]
        end_scores = alphas + last_transition.unsqueeze(0)
        _, max_final_tags = torch.max(end_scores, dim=1)
        best_sequences = []
        emission_lengths = mask.int().sum(dim=1)
        for i in range(batch_size):
            sample_length = emission_lengths[i].item()
            sample_final_tag = max_final_tags[i].item()
            sample_backpointers = backpointers[: sample_length - 1]
            sample_path = self._find_best_path(i, sample_final_tag, sample_backpointers)
            best_sequences.append(sample_path)
        return None, best_sequences

    def _find_best_path(self, sample_id, best_tag, backpointers):
        best_path = [best_tag]
        for bpt in reversed(backpointers):
            best_tag = bpt[best_tag][sample_id].item()
            best_path.insert(0, best_tag)
        return best_path


class MTL_Classifier(nn.Module):
    def __init__(self, n_tags, sent_emb_dim, sos_tag_idx, eos_tag_idx, pad_tag_idx, device="cuda"):
        super().__init__()
        self.device = device
        self.emitter = LSTM_Emitter(n_tags, sent_emb_dim, sent_emb_dim, 0.5, device).to(device)
        self.crf = CRF(n_tags, sos_tag_idx, eos_tag_idx, pad_tag_idx).to(device)
        self.emitter_binary = LSTM_Emitter_Binary(5, 3 * sent_emb_dim, sent_emb_dim, 0.5, device).to(device)
        self.crf_binary = CRF(5, sos_tag_idx, eos_tag_idx, pad_tag_idx).to(device)

    def forward(self, x, x_binary):
        batch_size = len(x)
        seq_lengths = [len(doc) for doc in x]
        max_seq_len = max(seq_lengths)

        tensor_x = [torch.tensor(doc, dtype=torch.float) for doc in x]
        tensor_x_binary = [torch.tensor(doc, dtype=torch.float) for doc in x_binary]

        tensor_x = nn.utils.rnn.pad_sequence(tensor_x, batch_first=True).to(self.device)
        tensor_x_binary = nn.utils.rnn.pad_sequence(tensor_x_binary, batch_first=True).to(self.device)

        mask = torch.zeros(batch_size, max_seq_len).to(self.device)
        for i, sl in enumerate(seq_lengths):
            mask[i, :sl] = 1

        emissions_binary, hidden_binary = self.emitter_binary(tensor_x_binary)
        emissions = self.emitter(tensor_x, hidden_binary)

        _, path = self.crf.decode(emissions, mask=mask)
        return path


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {
        "ik_train": {"candidate": [], "query": []},
        "ik_test":  {"candidate": [], "query": []},
    }


def save_checkpoint(progress: dict):
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp, CHECKPOINT_FILE)   # atomic write


# ──────────────────────────────────────────────────────────────────────────────
# BERT batched embedding
# ──────────────────────────────────────────────────────────────────────────────
def embed_sentences(sentences: list[str], tokenizer, bert_model, device, batch_size=BERT_BATCH_SIZE) -> list[np.ndarray]:
    """Return list of 768-dim CLS embeddings, one per sentence."""
    all_embs = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        encoded = tokenizer(
            batch,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            out = bert_model(**encoded)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()  # (B, 768)
        all_embs.extend(cls)
    return all_embs


# ──────────────────────────────────────────────────────────────────────────────
# Core classification
# ──────────────────────────────────────────────────────────────────────────────
def classify_document(
    text: str,
    tokenizer,
    bert_model,
    mtl_model,
    idx2tag: dict,
    device,
) -> list[tuple[str, str]]:
    """
    Split text into sentences and classify each with an RR label.
    Returns list of (sentence, label) excluding special tokens.
    """
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return []

    gen_embs = embed_sentences(sentences, tokenizer, bert_model, device)

    # Build 3× concatenated embeddings for the binary/shift module
    n = len(gen_embs)
    zeros = np.zeros(768, dtype=np.float32)
    s_embs = []
    for i in range(n):
        prev = gen_embs[i - 1] if i > 0 else zeros
        curr = gen_embs[i]
        nxt  = gen_embs[i + 1] if i < n - 1 else zeros
        s_embs.append(np.concatenate([prev, curr, nxt]))

    with torch.inference_mode():
        pred = mtl_model([gen_embs], [s_embs])   # batch of 1

    results = []
    for sent, tag_idx in zip(sentences, pred[0]):
        label = idx2tag[tag_idx]
        if label not in SKIP_LABELS:
            results.append((sent, label))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# File processing
# ──────────────────────────────────────────────────────────────────────────────
def process_file(in_path: str, out_path: str, tokenizer, bert_model, mtl_model, idx2tag, device):
    """Classify a single document and write the annotated output."""
    with open(in_path, encoding="utf-8", errors="ignore") as f:
        text = f.read()

    pairs = classify_document(text, tokenizer, bert_model, mtl_model, idx2tag, device)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for sentence, label in pairs:
            # Escape any tab characters in the sentence itself
            safe_sent = sentence.replace("\t", " ").replace("\n", " ")
            f.write(f"{label}\t{safe_sent}\n")


def process_split(
    split_key: str,          # "train" or "test"
    split_name: str,         # "ik_train" or "ik_test"
    input_dir: str,
    output_dir: str,
    tokenizer,
    bert_model,
    mtl_model,
    idx2tag: dict,
    device,
    progress: dict,
):
    os.makedirs(output_dir, exist_ok=True)

    for subdir in SUBDIRS:
        in_sub  = os.path.join(input_dir,  subdir)
        out_sub = os.path.join(output_dir, subdir)
        os.makedirs(out_sub, exist_ok=True)

        if not os.path.isdir(in_sub):
            print(f"  [!] Subdir not found: {in_sub}, skipping.")
            continue

        all_files = sorted(
            f for f in os.listdir(in_sub)
            if f.endswith(".txt") and f != ".gitkeep"
        )
        done  = set(progress[split_name][subdir])
        pending = [f for f in all_files if f not in done]

        print(
            f"\n  [{split_name}/{subdir}]  "
            f"{len(all_files)} total | {len(done)} done | {len(pending)} remaining"
        )

        if not pending:
            print("  ✓ All files already processed, skipping.")
            continue

        pbar = tqdm(pending, desc=f"  {split_name}/{subdir}", unit="doc", ncols=90)
        for fname in pbar:
            in_path  = os.path.join(in_sub,  fname)
            out_path = os.path.join(out_sub, fname)
            try:
                process_file(in_path, out_path, tokenizer, bert_model, mtl_model, idx2tag, device)
                progress[split_name][subdir].append(fname)
                save_checkpoint(progress)
            except Exception as e:
                pbar.write(f"  [ERROR] {fname}: {e}")
                if "--debug" in sys.argv:
                    traceback.print_exc()

    # Copy mapping files (CSV, JSON) unchanged
    for fname in os.listdir(input_dir):
        if fname.endswith(".csv") or fname.endswith(".json"):
            src = os.path.join(input_dir, fname)
            dst = os.path.join(output_dir, fname)
            shutil.copy2(src, dst)
            print(f"  Copied mapping file: {fname}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "test"):
        print("Usage: python create_rr_dataset.py <train|test>")
        print("  train  → processes ik_train  → ik_train_rr")
        print("  test   → processes ik_test   → ik_test_rr")
        sys.exit(1)

    split_key  = sys.argv[1]                  # "train" or "test"
    input_dir  = INPUT_DIRS[split_key]        # e.g. "ik_train"
    output_dir = OUTPUT_DIRS[split_key]       # e.g. "ik_train_rr"
    split_name = input_dir                    # used as checkpoint key

    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input directory not found: {input_dir}")
        sys.exit(1)

    print("=" * 70)
    print(f"  RR Dataset Creator")
    print(f"  Input  : {input_dir}")
    print(f"  Output : {output_dir}")
    print("=" * 70)

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")

    # ── Load tag maps ────────────────────────────────────────────────────────
    with open(TAG2IDX_PATH) as f:
        tag2idx = json.load(f)
    idx2tag = {v: k for k, v in tag2idx.items()}
    print(f"  Tags   : {[k for k in tag2idx if k not in SKIP_LABELS]}")

    # ── Load BERT ────────────────────────────────────────────────────────────
    print("\n  Loading BERT tokenizer and model...")
    tokenizer  = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
    bert_model.eval()

    # ── Load MTL classifier ──────────────────────────────────────────────────
    print("  Loading MTL classifier...")
    mtl_model = MTL_Classifier(
        n_tags       = len(tag2idx),
        sent_emb_dim = 768,
        sos_tag_idx  = tag2idx["<start>"],
        eos_tag_idx  = tag2idx["<end>"],
        pad_tag_idx  = tag2idx["<pad>"],
        device       = str(device),
    ).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    mtl_model.load_state_dict(checkpoint["state_dict"])
    mtl_model.eval()
    print("  ✅ Models loaded successfully.\n")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    progress = load_checkpoint()
    # Ensure keys exist for this split
    if split_name not in progress:
        progress[split_name] = {"candidate": [], "query": []}
    for sub in SUBDIRS:
        if sub not in progress[split_name]:
            progress[split_name][sub] = []

    # ── Process ──────────────────────────────────────────────────────────────
    t0 = time.time()
    process_split(
        split_key, split_name, input_dir, output_dir,
        tokenizer, bert_model, mtl_model, idx2tag, device, progress
    )
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"  Done!  Elapsed: {elapsed/60:.1f} min")
    print(f"  Output written to: {output_dir}/")
    print(f"  Progress saved to: {CHECKPOINT_FILE}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
