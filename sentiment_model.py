"""
sentiment_model.py
------------------
Fine-tunes DistilBERT for 3-class sentiment classification
(Negative=0, Neutral=1, Positive=2) and evaluates on a held-out test set.

Usage:
    # Train
    python src/sentiment_model.py --train --data data/processed.csv

    # Predict on new CSV
    python src/sentiment_model.py --predict --data data/new_reviews.csv \
        --model-dir models/distilbert-sentiment
"""

import argparse
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (DistilBertTokenizerFast,
                          DistilBertForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

LABELS     = ["Negative", "Neutral", "Positive"]
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 128
BATCH_SIZE = 32
EPOCHS     = 4
LR         = 2e-5
SEED       = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")


# ── Dataset ────────────────────────────────────────────────────────────────────
class ReviewDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ── Training ───────────────────────────────────────────────────────────────────
def train(data_path: str, model_dir: str = "models/distilbert-sentiment") -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(data_path)
    df.dropna(subset=["model_input", "label_id"], inplace=True)
    texts  = df["model_input"].tolist()
    labels = df["label_id"].astype(int).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train
    )
    log.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Tokenizer & datasets ───────────────────────────────────────────────────
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_ds  = ReviewDataset(X_train, y_train, tokenizer)
    val_ds    = ReviewDataset(X_val,   y_val,   tokenizer)
    test_ds   = ReviewDataset(X_test,  y_test,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_f1, best_epoch = 0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out  = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                labels         = batch["labels"].to(device)
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()

        avg_loss = total_loss / len(train_loader)

        # ── Validation ────────────────────────────────────────────────────────
        val_preds, val_true = _predict_loader(model, val_loader)
        val_f1 = f1_score(val_true, val_preds, average="weighted")
        log.info(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            log.info(f"  ✅ Best model saved to {model_dir}")

    log.info(f"\nBest model from epoch {best_epoch} (Val F1: {best_val_f1:.4f})")

    # ── Test evaluation ────────────────────────────────────────────────────────
    log.info("Running final evaluation on test set…")
    model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
    test_preds, test_true = _predict_loader(model, test_loader)
    acc = accuracy_score(test_true, test_preds)
    log.info(f"\nTest Accuracy: {acc:.4f}")
    log.info("\n" + classification_report(test_true, test_preds, target_names=LABELS))

    # Save confusion matrix data for visualization
    cm = confusion_matrix(test_true, test_preds)
    np.save(f"{model_dir}/confusion_matrix.npy", cm)

    # Save test predictions
    pd.DataFrame({
        "text":       X_test,
        "true_label": [LABELS[i] for i in test_true],
        "pred_label": [LABELS[i] for i in test_preds],
    }).to_csv("data/test_predictions.csv", index=False)
    log.info("Test predictions saved to data/test_predictions.csv")


# ── Inference ──────────────────────────────────────────────────────────────────
def predict(data_path: str, model_dir: str = "models/distilbert-sentiment",
            output_path: str = "data/predictions.csv") -> pd.DataFrame:
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    df = pd.read_csv(data_path)
    texts = df.get("model_input", df.get("review_text", df.iloc[:, 0])).astype(str).tolist()

    dataset = ReviewDataset(texts, [0] * len(texts), tokenizer)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE)

    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device)
            ).logits
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_probs.extend(probs.tolist())

    df["predicted_sentiment"] = [LABELS[p] for p in all_preds]
    df["confidence"]          = [max(p) for p in all_probs]
    df.to_csv(output_path, index=False)
    log.info(f"Predictions saved to {output_path}")
    return df


def predict_single(text: str, model_dir: str = "models/distilbert-sentiment") -> dict:
    """Predict sentiment for a single review string — used by Azure ML scorer."""
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    padding="max_length", max_length=MAX_LEN)
    with torch.no_grad():
        logits = model(
            input_ids      = enc["input_ids"].to(device),
            attention_mask = enc["attention_mask"].to(device)
        ).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred  = int(np.argmax(probs))
    return {
        "sentiment":   LABELS[pred],
        "confidence":  round(float(probs[pred]), 4),
        "scores": {
            "Negative": round(float(probs[0]), 4),
            "Neutral":  round(float(probs[1]), 4),
            "Positive": round(float(probs[2]), 4),
        }
    }


# ── Helper ─────────────────────────────────────────────────────────────────────
def _predict_loader(model, loader) -> tuple[list, list]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device)
            ).logits
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            trues.extend(batch["labels"].cpu().tolist())
    return preds, trues


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",     action="store_true")
    parser.add_argument("--predict",   action="store_true")
    parser.add_argument("--data",      default="data/processed.csv")
    parser.add_argument("--model-dir", default="models/distilbert-sentiment")
    parser.add_argument("--output",    default="data/predictions.csv")
    args = parser.parse_args()

    if args.train:
        train(args.data, args.model_dir)
    elif args.predict:
        predict(args.data, args.model_dir, args.output)
    else:
        parser.print_help()
