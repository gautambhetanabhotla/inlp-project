"""
Rhetorical Role (RR) segmentation module.

Classifies sentences in legal documents into semantic roles:
  Facts, Arguments, Ratio Decidendi, Ruling by Lower Court,
  Ruling by Present Court, Statutes, Precedents.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from config import Config, RRConfig
from data_loader import (
    ContextualSentenceDataset,
    LegalDocument,
    SentenceClassificationDataset,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RhetoricalRoleClassifier(nn.Module):
    """Sentence-level classifier on top of a legal BERT encoder."""

    def __init__(self, cfg: RRConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(cfg.model_name)
        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_size // 2, cfg.num_labels),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}


# ---------------------------------------------------------------------------
# Hierarchical model variant  (sentence + document context via BiLSTM)
# ---------------------------------------------------------------------------

class HierarchicalRRClassifier(nn.Module):
    """Two-level classifier: sentence encoder → BiLSTM over document → classifier.

    Captures inter-sentence dependencies which are crucial for RR labeling
    (e.g. a sentence after "Facts" is likely also "Facts").
    """

    def __init__(self, cfg: RRConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(cfg.model_name)
        hidden_size = self.encoder.config.hidden_size

        # Freeze lower encoder layers to save memory
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        for layer in self.encoder.encoder.layer[:8]:
            for param in layer.parameters():
                param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden_size, cfg.num_labels),
        )

    def encode_sentences(self, input_ids, attention_mask):
        """Encode a batch of sentences → [B, H]."""
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # [CLS]

    def forward(self, sentence_embeddings, labels=None):
        """
        Args:
            sentence_embeddings: [1, num_sentences, H]
            labels:              [1, num_sentences]
        """
        lstm_out, _ = self.lstm(sentence_embeddings)  # [1, S, H]
        logits = self.classifier(lstm_out)             # [1, S, C]

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, self.cfg.num_labels), labels.view(-1)
            )
        return {"loss": loss, "logits": logits}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class RRTrainer:
    """Train and evaluate a Rhetorical Role classifier."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rr_cfg = cfg.rr
        self.device = cfg.device
        self.tokenizer = AutoTokenizer.from_pretrained(self.rr_cfg.model_name)
        self.model = RhetoricalRoleClassifier(self.rr_cfg).to(self.device)
        self.label2id = {name: i for i, name in enumerate(self.rr_cfg.label_names)}
        self.id2label = {i: name for name, i in self.label2id.items()}

    # ----- dataset preparation -----

    def prepare_dataset(
        self,
        documents: list[list[str]],
        labels: list[list[str]],
    ) -> DataLoader:
        """Convert lists of document-sentences + string-labels into a DataLoader."""
        int_labels = [
            [self.label2id.get(l, 0) for l in doc_labels]
            for doc_labels in labels
        ]
        return self._make_loader(documents, int_labels)

    def prepare_dataset_from_ids(
        self,
        documents: list[list[str]],
        labels: list[list[int]],
    ) -> DataLoader:
        """Convert lists of document-sentences + integer-labels into a DataLoader.

        Use this when labels are already ints (e.g. from the HuggingFace IL-TUR dataset).
        """
        return self._make_loader(documents, labels)

    def _make_loader(
        self,
        documents: list[list[str]],
        int_labels: list[list[int]],
    ) -> DataLoader:
        if self.rr_cfg.use_context:
            dataset = ContextualSentenceDataset(
                documents,
                int_labels,
                self.tokenizer,
                max_length=self.rr_cfg.max_seq_length,
                context_window=self.rr_cfg.context_window,
            )
        else:
            flat_sents = [s for doc in documents for s in doc]
            flat_labels = [l for doc in int_labels for l in doc]
            dataset = SentenceClassificationDataset(
                flat_sents,
                flat_labels,
                self.tokenizer,
                max_length=self.rr_cfg.max_seq_length,
            )

        return DataLoader(
            dataset,
            batch_size=self.rr_cfg.batch_size,
            shuffle=True,
            num_workers=0,
        )

    # ----- training loop -----

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.rr_cfg.learning_rate,
            weight_decay=self.rr_cfg.weight_decay,
        )
        total_steps = len(train_loader) * self.rr_cfg.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.rr_cfg.warmup_ratio),
            num_training_steps=total_steps,
        )

        best_val_f1 = 0.0
        for epoch in range(self.rr_cfg.num_epochs):
            self.model.train()
            total_loss = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask, labels=labels)
                loss = outputs["loss"]

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logger.info("Epoch %d/%d — train loss: %.4f", epoch + 1, self.rr_cfg.num_epochs, avg_loss)

            if val_loader is not None:
                metrics = self.evaluate(val_loader)
                logger.info("  val accuracy: %.4f  |  macro-F1: %.4f", metrics["accuracy"], metrics["macro_f1"])
                if metrics["macro_f1"] > best_val_f1:
                    best_val_f1 = metrics["macro_f1"]
                    self.save(self.cfg.paths.models_dir / "rr_best.pt")

        return self.model

    # ----- evaluation -----

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_preds, all_labels = [], []

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"]

            logits = self.model(input_ids, attention_mask)["logits"]
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        from sklearn.metrics import accuracy_score, f1_score

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        return {
            "accuracy": accuracy_score(all_labels, all_preds),
            "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
            "per_class_f1": f1_score(all_labels, all_preds, average=None, zero_division=0).tolist(),
        }

    # ----- inference -----

    @torch.no_grad()
    def predict(self, sentences: list[str]) -> list[str]:
        """Predict rhetorical roles for a list of sentences."""
        self.model.eval()
        dataset = SentenceClassificationDataset(
            sentences,
            [0] * len(sentences),  # dummy labels
            self.tokenizer,
            max_length=self.rr_cfg.max_seq_length,
        )
        loader = DataLoader(dataset, batch_size=self.rr_cfg.batch_size, shuffle=False)

        predictions = []
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            logits = self.model(input_ids, attention_mask)["logits"]
            preds = logits.argmax(dim=-1).cpu().tolist()
            predictions.extend([self.id2label[p] for p in preds])

        return predictions

    def segment_document(self, doc: LegalDocument) -> LegalDocument:
        """Add rhetorical role labels to a LegalDocument in-place."""
        doc.rhetorical_roles = self.predict(doc.sentences)
        return doc

    # ----- persistence -----

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info("Saved RR model to %s", path)

    def load(self, path: Path):
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        logger.info("Loaded RR model from %s", path)
