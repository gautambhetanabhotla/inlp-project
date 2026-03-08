#!/usr/bin/env python
# coding: utf-8
"""
test.py - Run the pretrained LSI model on sample text.

Usage:
    python test.py

Expected directory layout (same as your tree):
    .
    ├── Eval/
    │   └── label_vocab.json
    ├── lsi.py
    └── pytorch_model.bin   <-- pretrained weights

The script will print the predicted IPC sections for each sample.
"""

import json
import torch
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoTokenizer, AutoModel, BatchEncoding
from transformers.file_utils import ModelOutput

# ─────────────────────────── CONFIG ──────────────────────────────
MODEL_SRC       = "law-ai/InLegalBERT"   # HuggingFace model ID (or local path)
WEIGHTS_PATH    = "pytorch_model.bin"    # pretrained weights
LABEL_VOCAB     = "Eval/label_vocab.json"
THRESHOLD       = 0.5                    # confidence threshold for a positive prediction
MAX_SEGMENTS    = 128                    # max sentences per document
MAX_SEG_SIZE    = 128                    # max tokens per sentence
CACHE_DIR       = "Cache"
# ─────────────────────────────────────────────────────────────────

# ── Sample texts ── (replace / extend as needed) ──────────────────
SAMPLES = {
    "sample_murder_dowry": [
        "The appellant herein was convicted by the Sessions Court under Section 302 read with Section 34 of the Indian Penal Code for the murder of his wife.",
        "The deceased was married to the accused-appellant approximately two years prior to the date of the incident.",
        "The prosecution alleged that the accused, along with his mother and brother, subjected the deceased to persistent cruelty on account of inadequate dowry.",
        "The first informant, the father of the deceased, lodged the FIR stating that his daughter had telephoned him two days before her death complaining of harassment.",
        "Post-mortem examination revealed ligature marks around the neck of the deceased, and the cause of death was certified as asphyxia due to strangulation.",
        "The defence contended that the deceased had committed suicide by hanging and that no offence under Section 302 IPC was made out against the accused.",
        "The Sessions Court rejected this contention holding that the medical evidence was inconsistent with suicide and that the accused had failed to explain the circumstances of death.",
        "The learned Sessions Judge further held that the accused persons had common intention within the meaning of Section 34 IPC to commit the offence.",
        "It was urged on behalf of the appellant before this Court that the prosecution had not established its case beyond reasonable doubt and that the benefit of doubt ought to be extended to the accused.",
        "The High Court on appeal affirmed the conviction under Section 304B IPC in addition to Section 498A IPC, modifying the conviction under Section 302.",
        "The prosecution relied on circumstantial evidence, the last seen theory, and the conduct of the accused immediately after the death.",
        "This Court finds that the chain of circumstances is complete and points unerringly to the guilt of the accused and that no innocent explanation is forthcoming.",
        "The conviction under Section 304B read with Section 498A of the Indian Penal Code is accordingly confirmed.",
    ],

    "sample_robbery_hurt": [
        "This appeal arises out of a judgment of conviction passed by the Additional Sessions Judge against the accused persons for offences punishable under Sections 392, 394 and 397 of the Indian Penal Code.",
        "The prosecution case, briefly stated, is that on the night in question the accused persons formed an unlawful assembly armed with deadly weapons including iron rods and a country-made pistol.",
        "The complainant, a shopkeeper, was returning home after closing his establishment when the accused persons intercepted him on the public road.",
        "They demanded cash and, on the complainant's resistance, one of the accused struck him on the head with an iron rod causing a grievous lacerated wound.",
        "The complainant fell unconscious and was admitted to the Government Hospital where he was treated for grievous hurt caused by a dangerous weapon.",
        "The accused were apprehended near the scene of occurrence with the looted cash and the weapons used in the commission of the offence.",
        "The learned Sessions Judge convicted the accused under Section 394 read with Section 397 IPC, holding that robbery had been committed with the use of a deadly weapon.",
        "The defence counsel argued that the identification of the accused by the complainant was unreliable as the incident took place in darkness.",
        "However, the trial court relied on the corroborating testimony of an independent eyewitness and the recovery of the stolen articles from the possession of the accused.",
        "The charge under Section 120B IPC for criminal conspiracy was also framed as the accused persons had allegedly assembled in furtherance of a pre-planned scheme.",
        "Medical evidence confirmed that the nature of the injuries sustained by the complainant was consistent with a blow from a blunt weapon.",
        "This Court holds that the offence under Section 397 IPC is clearly made out, as robbery was committed with the use of a deadly weapon causing grievous hurt.",
        "The sentence of rigorous imprisonment of seven years awarded by the trial court is confirmed.",
    ],

    "sample_cheating_forgery": [
        "The accused, a former bank manager, was charged under Sections 420, 467, 468 and 471 of the Indian Penal Code for dishonestly inducing the complainant company to part with a sum of Rs. 25 lakhs.",
        "The prosecution alleged that the accused fabricated loan sanction letters bearing forged signatures of senior officials of the bank.",
        "These forged documents were presented to the complainant as genuine instruments to induce it to make an advance payment towards a fictitious infrastructure project.",
        "The accused had dishonestly represented to the complainant that the bank had sanctioned a loan of Rs. 2 crores in favour of the project company.",
        "On the faith of this representation the complainant deposited a sum of Rs. 25 lakhs as margin money, which was thereafter misappropriated by the accused.",
        "The handwriting expert examined the disputed documents and opined that the signatures thereon did not match the specimens of the bank officials.",
        "The investigating officer recovered several incriminating documents, stamp papers, and a forged rubber seal from the residential premises of the accused.",
        "The trial court convicted the accused under Section 420 IPC for cheating, Section 467 IPC for forgery of a valuable security, and Section 471 IPC for using forged documents as genuine.",
        "The accused preferred an appeal contending that the prosecution had not established dishonest intention at the time of the transaction and that there was only a civil dispute.",
        "This Court rejects that contention observing that the fabrication of bank documents goes far beyond a mere breach of contract and constitutes a criminal act of forgery.",
        "The element of mens rea required under Section 415 IPC has been duly established through the conduct of the accused and the recovery of fabricated instruments.",
        "The conviction under Sections 467 and 468 IPC, both of which carry a maximum sentence of seven years, is confirmed by this Court.",
        "The sentences awarded by the trial court shall run concurrently as they arise out of the same transaction.",
    ],

    "sample_kidnapping_assault": [
        "The accused was charged under Sections 363, 366, 376(2) and 506 of the Indian Penal Code for kidnapping a minor girl and committing rape.",
        "The prosecutrix, aged fourteen years as established by her school certificate, was lured by the accused on the pretext of offering employment to her family.",
        "The accused took her to a rented accommodation in a different district and kept her confined there against her will for a period of three days.",
        "During her confinement the accused committed rape upon her on multiple occasions and also threatened her with dire consequences if she disclosed the matter to anyone.",
        "On her release the prosecutrix disclosed the incident to her mother who lodged a First Information Report at the local police station.",
        "Medical examination conducted by the Civil Surgeon confirmed the age of the prosecutrix to be consistent with that of a minor and found evidence corroborating her account.",
        "The accused pleaded that the prosecutrix had accompanied him of her own free will and that the relationship between them was consensual.",
        "This Court holds that consent of a minor is no defence to the charge of kidnapping under Section 363 IPC or to the charge of rape under Section 376 IPC.",
        "The testimony of the prosecutrix is cogent, consistent and duly corroborated by medical evidence and the evidence of her mother.",
        "The accused is also found guilty under Section 366A IPC for inducing a minor girl to go from one place to another for purposes of illicit intercourse.",
        "The act of threatening the prosecutrix constitutes the offence under Section 506 IPC for criminal intimidation.",
        "The accused is accordingly convicted on all charges and sentenced to rigorous imprisonment of ten years under Section 376(2) IPC which shall not run concurrently with other sentences.",
        "The Sessions Court's order of conviction and sentence is affirmed by this Court in all respects.",
    ],
}

# ─────────────────────────────────────────────────────────────────


# ══════════════════  Model architecture (mirrors lsi.py)  ════════

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


@dataclass
class TextClassifierOutput(ModelOutput):
    loss: torch.Tensor = None
    logits: torch.Tensor = None
    hidden_states: torch.Tensor = None


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
        self.loss_fct      = nn.BCEWithLogitsLoss(torch.ones(num_labels))
        self.dropout       = nn.Dropout(drop)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        hidden = self.dropout(
            self.hier_encoder(input_ids=input_ids,
                              attention_mask=attention_mask)[1])
        logits = self.dropout(self.classifier_fc(hidden))
        loss   = self.loss_fct(logits, labels) if labels is not None else None
        return TextClassifierOutput(loss=loss,
                                    logits=torch.sigmoid(logits),
                                    hidden_states=hidden)


# ══════════════════════════  Helpers  ════════════════════════════

def build_batch(sentences, tokenizer):
    """Tokenise a list of sentences → (input_ids, attention_mask) tensors."""
    encoded  = tokenizer(sentences, return_token_type_ids=False,
                         padding=False, truncation=False)
    n        = min(len(sentences), MAX_SEGMENTS)
    max_tok  = min(max(len(s) for s in encoded["input_ids"][:n]), MAX_SEG_SIZE)

    input_ids = torch.zeros(1, n, max_tok, dtype=torch.long).fill_(
        tokenizer.pad_token_id)
    for i, ids in enumerate(encoded["input_ids"][:n]):
        ids = ids[:MAX_SEG_SIZE]
        input_ids[0, i, :len(ids)] = torch.tensor(ids)

    attention_mask = input_ids != tokenizer.pad_token_id
    return input_ids, attention_mask


@torch.no_grad()
def predict(model, tokenizer, sentences, device, threshold=THRESHOLD):
    """Return list of predicted statute labels for a single document."""
    input_ids, attention_mask = build_batch(sentences, tokenizer)
    input_ids      = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    output = model(input_ids=input_ids, attention_mask=attention_mask)
    probs  = output.logits.cpu().numpy()[0]          # shape: (num_labels,)
    return probs


# ══════════════════════════  Main  ═══════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ── Load label vocab ──────────────────────────────────────────
    with open(LABEL_VOCAB) as f:
        label_vocab = json.load(f)
    idx_to_label = {v: k for k, v in label_vocab.items()}

    # ── Load tokeniser ────────────────────────────────────────────
    print(f"Loading tokenizer from '{MODEL_SRC}' …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SRC, cache_dir=CACHE_DIR)

    # ── Build model ───────────────────────────────────────────────
    print(f"Loading base model from '{MODEL_SRC}' …")
    bert      = AutoModel.from_pretrained(MODEL_SRC, cache_dir=CACHE_DIR)
    hier_bert = HierBert(bert)
    model     = HierBertForTextClassification(
        hier_bert, num_labels=len(label_vocab)).to(device)

    # ── Load pretrained weights ───────────────────────────────────
    print(f"Loading weights from '{WEIGHTS_PATH}' …\n")
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"Ignored unexpected keys: {unexpected}")
    if missing:
        print(f"Warning – missing keys: {missing}")
    model.eval()

    # ── Run inference on each sample ──────────────────────────────
    results = {}
    for doc_id, sentences in SAMPLES.items():
        probs      = predict(model, tokenizer, sentences, device)
        predicted  = [idx_to_label[i]
                      for i, p in enumerate(probs) if p >= THRESHOLD]

        results[doc_id] = predicted

        print(f"{'─'*60}")
        print(f"Document : {doc_id}")
        print(f"Input    : {len(sentences)} sentence(s)")
        if predicted:
            print("Predicted statutes:")
            for stat in predicted:
                idx   = label_vocab[stat]
                print(f"  [{probs[idx]:.3f}]  {stat}")
        else:
            print("  (no statute predicted above threshold)")

        # Show top-5 predictions regardless of threshold
        top5_idx = np.argsort(probs)[::-1][:5]
        print("Top-5 predictions (by confidence):")
        for i in top5_idx:
            print(f"  [{probs[i]:.3f}]  {idx_to_label[i]}")
        print()

    # ── Optionally save predictions in eval-compatible format ─────
    out_path = "ilsi-test-pred.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Predictions saved to '{out_path}'")


if __name__ == "__main__":
    main()