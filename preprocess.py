"""
preprocess.py
-------------
Cleans and prepares raw e-commerce review CSV data for sentiment
classification and topic modeling.

Usage:
    python src/preprocess.py --input data/sample_reviews.csv --output data/processed.csv
"""

import argparse
import re
import string
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

# Download required NLTK data on first run
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ── Label mapping ──────────────────────────────────────────────────────────────
def rating_to_label(rating: int) -> str:
    """Map star rating → sentiment label."""
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

def rating_to_id(rating: int) -> int:
    """Map star rating → integer label for model training."""
    if rating >= 4:
        return 2
    elif rating == 3:
        return 1
    else:
        return 0

# ── Text cleaning ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Full cleaning pipeline:
      1. Lowercase
      2. Remove URLs, HTML tags, special characters
      3. Remove punctuation & digits
      4. Remove stopwords
      5. Lemmatize
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)          # URLs
    text = re.sub(r"<.*?>", "", text)                      # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                  # non-alpha chars
    text = re.sub(r"\s+", " ", text).strip()               # extra whitespace
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def clean_text_light(text: str) -> str:
    """
    Light cleaning for the transformer model — keeps sentence structure intact
    (transformers benefit from punctuation and casing context).
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:512]  # DistilBERT max practical length

# ── Main preprocessing ─────────────────────────────────────────────────────────
def preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    log.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)

    log.info(f"Raw shape: {df.shape}")

    # ── Standardise column names ───────────────────────────────────────────────
    col_map = {
        "reviewText": "review_text",
        "overall":    "rating",
        "asin":       "product_id",
        "reviewerID": "reviewer_id",
        "unixReviewTime": "review_date",
        "summary":    "review_summary",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # ── Drop rows with no review text ─────────────────────────────────────────
    before = len(df)
    df.dropna(subset=["review_text", "rating"], inplace=True)
    df["review_text"] = df["review_text"].astype(str)
    df = df[df["review_text"].str.strip().str.len() > 10]
    log.info(f"Dropped {before - len(df)} rows with missing/short text")

    # ── Rating → labels ────────────────────────────────────────────────────────
    df["rating"] = df["rating"].astype(int)
    df["sentiment_label"] = df["rating"].apply(rating_to_label)
    df["label_id"]        = df["rating"].apply(rating_to_id)

    # ── Cleaned text columns ──────────────────────────────────────────────────
    log.info("Cleaning text — this may take a minute on large datasets…")
    df["clean_text"]  = df["review_text"].apply(clean_text)        # for LDA
    df["model_input"] = df["review_text"].apply(clean_text_light)  # for BERT

    # ── Remove duplicate reviews ──────────────────────────────────────────────
    df.drop_duplicates(subset=["review_text"], inplace=True)

    # ── Parse date ────────────────────────────────────────────────────────────
    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], unit="s", errors="coerce")

    # ── Final column selection ────────────────────────────────────────────────
    keep_cols = [c for c in [
        "review_id", "product_id", "reviewer_id", "review_date",
        "rating", "sentiment_label", "label_id",
        "review_text", "model_input", "clean_text"
    ] if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    log.info(f"Processed shape: {df.shape}")
    log.info(f"Label distribution:\n{df['sentiment_label'].value_counts()}")

    df.to_csv(output_path, index=False)
    log.info(f"Saved to {output_path}")
    return df


def generate_sample_dataset(n: int = 500, output_path: str = "data/sample_reviews.csv") -> None:
    """Generate a synthetic sample dataset for testing the pipeline."""
    import random
    np.random.seed(42)
    random.seed(42)

    positive_snippets = [
        "Absolutely love this product! Works perfectly and the quality is outstanding.",
        "Best purchase I've made this year. Fast shipping and exactly as described.",
        "The battery life is incredible — lasts all day easily.",
        "Great value for money. Would definitely recommend to anyone.",
        "Setup was easy and performance is top notch. Very happy customer.",
    ]
    neutral_snippets = [
        "It's okay for the price. Does what it says but nothing special.",
        "Average product. Shipping was fast but packaging was a bit disappointing.",
        "Not bad, not great. It works as expected but has some minor issues.",
        "Decent quality. I'd probably buy again if the price dropped a bit.",
    ]
    negative_snippets = [
        "Completely stopped working after two weeks. Total waste of money.",
        "The description was very misleading. Nothing like the pictures shown.",
        "Terrible customer service and the product broke on first use.",
        "Poor build quality — feels very cheap and plasticky.",
        "Arrived damaged and the return process was a nightmare.",
    ]

    rows = []
    for i in range(n):
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.10, 0.10, 0.15, 0.30, 0.35])
        if rating >= 4:
            text = random.choice(positive_snippets) + f" Review #{i}"
        elif rating == 3:
            text = random.choice(neutral_snippets) + f" Review #{i}"
        else:
            text = random.choice(negative_snippets) + f" Review #{i}"

        rows.append({
            "review_id":   f"R{i:05d}",
            "product_id":  f"B{np.random.randint(1000,9999):04d}XYZ",
            "reviewer_id": f"U{np.random.randint(100,999):03d}",
            "review_date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(np.random.randint(0, 700))),
            "rating":      int(rating),
            "review_text": text,
        })

    pd.DataFrame(rows).to_csv(output_path, index=False)
    log.info(f"Sample dataset saved to {output_path} ({n} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/sample_reviews.csv")
    parser.add_argument("--output", default="data/processed.csv")
    parser.add_argument("--generate-sample", action="store_true",
                        help="Generate a synthetic sample dataset first")
    args = parser.parse_args()

    if args.generate_sample:
        generate_sample_dataset(output_path=args.input)

    preprocess(args.input, args.output)
