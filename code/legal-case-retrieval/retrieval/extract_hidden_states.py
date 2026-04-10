#!/usr/bin/env python3
"""
extract_hidden_states.py - Extract hidden state vectors from HierBERT model for dense retrieval.

Outputs a JSON file mapping case_id → hidden_state_vector (list of floats).
These vectors can then be used with cosine similarity for retrieval instead of label matching.
"""

import json
import os
import re
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoTokenizer, AutoModel

# ─────────────────────────── CONFIG ──────────────────────────────
MODEL_SRC       = "law-ai/InLegalBERT"
WEIGHTS_PATH    = "pytorch_model.bin"
MAX_SEGMENTS    = 128
MAX_SEG_SIZE    = 128
# ─────────────────────────────────────────────────────────────────


# ══════════════════  Model architecture (mirrors lsi.py & test.py)  ════════

class LstmAttn(nn.Module):
    def __init__(self, hidden_size, drop=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size // 2,
                            batch_first=True, bidirectional=True)
        self.attn_fc   = nn.Linear(hidden_size, hidden_size)
        self.context   = nn.Parameter(torch.rand(hidden_size))
        self.dropout   = nn.Dropout(drop)

    def forward(self, inputs, attention_mask=None, dynamic_context=None):
        if attention_mask is None:
            attention_mask = torch.ones(inputs.shape[:2],
                                        dtype=torch.bool, device=inputs.device)
        lengths = attention_mask.float().sum(dim=1)
        packed  = pack_padded_sequence(inputs,
                                       torch.clamp(lengths, min=1).cpu(),
                                       enforce_sorted=False, batch_first=True)
        outputs = pad_packed_sequence(self.lstm(packed)[0], batch_first=True)[0]

        activated = torch.tanh(self.dropout(self.attn_fc(outputs)))
        ctx       = (dynamic_context if dynamic_context is not None
                     else self.context.expand(inputs.size(0), self.hidden_size))
        scores    = torch.bmm(activated, ctx.unsqueeze(2)).squeeze(2)
        scores    = F.softmax(scores.masked_fill(~attention_mask, -1e32), dim=1)
        hidden    = (outputs * scores.unsqueeze(2)).sum(dim=1)
        return outputs, hidden


class HierBert(nn.Module):
    def __init__(self, encoder, fragment_size=32, drop=0.5):
        super().__init__()
        self.bert_encoder    = encoder
        self.hidden_size     = encoder.config.hidden_size
        self.fragment_size   = fragment_size
        self.segment_encoder = LstmAttn(self.hidden_size, drop=drop)
        self.dropout         = nn.Dropout(drop)

    def _encoder_forward(self, input_ids, attention_mask, dummy):
        return self.dropout(
            self.bert_encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        )[:, 0, :]

    def forward(self, input_ids=None, attention_mask=None, encoder_outputs=None):
        if input_ids is not None:
            batch_size, max_num_segments, max_segment_size = input_ids.shape
        else:
            batch_size, max_num_segments = encoder_outputs.shape[:2]

        if input_ids is not None:
            flat_ids  = input_ids.view(-1, max_segment_size)
            flat_mask = attention_mask.view(-1, max_segment_size)
            dummy     = torch.ones(1, dtype=torch.float, requires_grad=True)

            enc_outs = []
            for i in range(0, batch_size * max_num_segments, self.fragment_size):
                enc_outs.append(
                    self._encoder_forward(flat_ids [i:i+self.fragment_size],
                                          flat_mask[i:i+self.fragment_size],
                                          dummy)
                )
            encoder_outputs = torch.cat(enc_outs, dim=0).view(
                batch_size, max_num_segments, self.hidden_size)
            attention_mask  = attention_mask.any(dim=2)

        outputs, hidden = self.segment_encoder(
            inputs=encoder_outputs, attention_mask=attention_mask)
        return outputs, hidden


class HierBertForTextClassification(nn.Module):
    def __init__(self, hier_encoder, num_labels, drop=0.5):
        super().__init__()
        self.hidden_size   = hier_encoder.hidden_size
        self.num_labels    = num_labels
        self.hier_encoder  = hier_encoder
        self.classifier_fc = nn.Linear(hier_encoder.hidden_size, num_labels)
        self.dropout       = nn.Dropout(drop)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        hidden = self.dropout(
            self.hier_encoder(input_ids=input_ids,
                              attention_mask=attention_mask)[1])
        return hidden


# ══════════════════════════  Helpers  ════════════════════════════

def build_batch(sentences, tokenizer):
    """Tokenise a list of sentences → (input_ids, attention_mask) tensors."""
    encoded  = tokenizer(
        sentences,
        return_token_type_ids=False,
        padding=False,
        truncation=True,
        max_length=MAX_SEG_SIZE,
    )
    n        = min(len(sentences), MAX_SEGMENTS)
    max_tok  = min(max(len(s) for s in encoded["input_ids"][:n]), MAX_SEG_SIZE)

    input_ids = torch.zeros(1, n, max_tok, dtype=torch.long).fill_(
        tokenizer.pad_token_id)
    for i, ids in enumerate(encoded["input_ids"][:n]):
        ids = ids[:MAX_SEG_SIZE]
        input_ids[0, i, :len(ids)] = torch.tensor(ids)

    attention_mask = input_ids != tokenizer.pad_token_id
    return input_ids, attention_mask


def split_into_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ["No textual content available for this legal case."]
    chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+|\n+", text) if chunk.strip()]
    return chunks if chunks else [text]


def load_cases_from_dataset(dataset_root, query_dir):
    query_path = Path(dataset_root) / query_dir
    if not query_path.exists():
        raise FileNotFoundError(f"Query directory not found: {query_path}")

    cases = {}
    empty_count = 0
    for txt_path in sorted(query_path.glob("*.txt")):
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if not text:
            empty_count += 1
        cases[txt_path.name] = split_into_sentences(text)

    if empty_count > 0:
        print(f"Warning: found {empty_count} empty case files in {query_path}")

    return cases


@torch.no_grad()
def extract_hidden_state(model, tokenizer, sentences, device):
    """Extract hidden state vector for a single document."""
    input_ids, attention_mask = build_batch(sentences, tokenizer)
    input_ids      = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    hidden = model(input_ids=input_ids, attention_mask=attention_mask)
    # hidden shape: (1, hidden_size)
    return hidden.cpu().numpy()[0].tolist()


def main():
    parser = argparse.ArgumentParser(description="Extract hidden state vectors from HierBERT for dense retrieval.")
    parser.add_argument("--dataset-root", default="dataset/ik_test 4", help="Dataset split root")
    parser.add_argument("--query-dir", default="query", help="Query directory name inside dataset root")
    parser.add_argument("--cand-dir", default="", help="Optional candidate directory (if different from query-dir)")
    parser.add_argument("--hidden-out", default="hidden_states.json", help="Output path for hidden states JSON")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-document logging")
    args = parser.parse_args()

    # Detect device: MPS for Mac M2, CUDA for NVIDIA, CPU fallback
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SRC)
    bert_encoder = AutoModel.from_pretrained(MODEL_SRC)
    hier_bert = HierBert(bert_encoder)
    model = HierBertForTextClassification(hier_bert, num_labels=227)
    
    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading weights from {WEIGHTS_PATH}...")
        state_dict = torch.load(WEIGHTS_PATH, map_location=device)
        # Filter out classifier head (won't match due to different num_labels)
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                                if not k.startswith('classifier_fc') and not k.startswith('loss_fct')}
        model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded {len(filtered_state_dict)} weight keys (skipped classifier_fc)")
    else:
        print(f"Warning: weights file not found at {WEIGHTS_PATH}")

    model = model.to(device)
    model.eval()

    # Extract hidden states
    print(f"Extracting hidden states from {args.dataset_root}/{args.query_dir}...")
    cases = load_cases_from_dataset(args.dataset_root, args.query_dir)
    
    hidden_states = {}
    for idx, (case_id, sentences) in enumerate(cases.items()):
        if not args.quiet and (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(cases)}")
        
        vector = extract_hidden_state(model, tokenizer, sentences, device)
        hidden_states[case_id] = vector

    # Also extract from candidate dir if specified
    if args.cand_dir:
        print(f"Extracting hidden states from {args.dataset_root}/{args.cand_dir}...")
        cand_cases = load_cases_from_dataset(args.dataset_root, args.cand_dir)
        for idx, (case_id, sentences) in enumerate(cand_cases.items()):
            if not args.quiet and (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(cand_cases)}")
            
            vector = extract_hidden_state(model, tokenizer, sentences, device)
            hidden_states[case_id] = vector

    # Save
    with open(args.hidden_out, "w", encoding="utf-8") as f:
        json.dump(hidden_states, f)

    print(f"Saved {len(hidden_states)} hidden state vectors to {args.hidden_out}")


if __name__ == "__main__":
    main()
