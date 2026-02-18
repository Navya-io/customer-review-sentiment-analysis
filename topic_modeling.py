"""
topic_modeling.py
-----------------
Trains an LDA topic model on cleaned review text and saves:
  - topic-word distributions
  - per-review topic assignments
  - coherence score

Usage:
    python src/topic_modeling.py --data data/processed.csv --n_topics 8
"""

import argparse
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

TOPIC_LABELS = {                     # Manually assigned after inspection
    0: "Shipping & Delivery",
    1: "Battery & Performance",
    2: "Build Quality & Design",
    3: "Customer Service & Returns",
    4: "Value for Money",
    5: "Ease of Setup & Use",
    6: "Product Accuracy / Description Match",
    7: "Durability & Longevity",
}


def train_lda(data_path: str, n_topics: int = 8, output_dir: str = "models/lda") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df.dropna(subset=["clean_text"], inplace=True)
    texts = df["clean_text"].tolist()
    tokenized = [t.split() for t in texts]

    log.info(f"Building dictionary and corpus from {len(texts)} documents…")
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(t) for t in tokenized]

    log.info(f"Training LDA with {n_topics} topics…")
    lda = models.LdaMulticore(
        corpus,
        id2word=dictionary,
        num_topics=n_topics,
        random_state=42,
        passes=10,
        alpha="asymmetric",
        eta="auto",
        workers=3,
    )

    # ── Coherence score ────────────────────────────────────────────────────────
    coherence = CoherenceModel(
        model=lda, texts=tokenized, dictionary=dictionary, coherence="c_v"
    ).get_coherence()
    log.info(f"Topic Coherence Score (c_v): {coherence:.4f}")

    # ── Print topics ───────────────────────────────────────────────────────────
    topics_info = {}
    for i in range(n_topics):
        top_words = [w for w, _ in lda.show_topic(i, topn=15)]
        label = TOPIC_LABELS.get(i, f"Topic {i}")
        topics_info[i] = {"label": label, "top_words": top_words}
        log.info(f"  Topic {i} [{label}]: {', '.join(top_words[:8])}")

    with open(f"{output_dir}/topics.json", "w") as f:
        json.dump(topics_info, f, indent=2)

    # ── Assign dominant topic to each review ──────────────────────────────────
    log.info("Assigning dominant topics to reviews…")
    dominant_topics, topic_probs = [], []
    for bow in corpus:
        dist = dict(lda.get_document_topics(bow, minimum_probability=0))
        if dist:
            top = max(dist, key=dist.get)
            dominant_topics.append(top)
            topic_probs.append(round(dist[top], 4))
        else:
            dominant_topics.append(-1)
            topic_probs.append(0.0)

    df["dominant_topic"]    = dominant_topics
    df["topic_probability"] = topic_probs
    df["topic_label"]       = df["dominant_topic"].map(
        {i: v["label"] for i, v in topics_info.items()}
    ).fillna("Unassigned")

    df.to_csv(data_path.replace(".csv", "_with_topics.csv"), index=False)
    log.info(f"Saved topic-enriched data")

    lda.save(f"{output_dir}/lda_model")
    dictionary.save(f"{output_dir}/dictionary")
    log.info(f"LDA model saved to {output_dir}/")
    log.info(f"\nTopic distribution:\n{df['topic_label'].value_counts()}")


# ═══════════════════════════════════════════════════════════════════════════════
# visualizations.py  (combined here for convenience — can be split out)
# ═══════════════════════════════════════════════════════════════════════════════
"""
Generates and saves all 4 key visualizations to the visuals/ directory.

Run standalone:
    python src/topic_modeling.py --visualize --data data/processed_with_topics.csv
"""

import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
import json

PALETTE = {
    "Positive": "#2ecc71",
    "Neutral":  "#f39c12",
    "Negative": "#e74c3c",
}
NAVY    = "#0f2044"
GOLD    = "#c8a96e"
sns.set_theme(style="whitegrid", font="DejaVu Sans")


def save_sentiment_distribution(df: pd.DataFrame, out: str = "visuals/sentiment_distribution.png"):
    Path("visuals").mkdir(exist_ok=True)
    counts = df["sentiment_label"].value_counts().reindex(["Positive", "Neutral", "Negative"])
    pcts   = counts / counts.sum() * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#f8f6f1")

    # Bar chart
    ax = axes[0]
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[l] for l in counts.index],
                  width=0.55, edgecolor="white", linewidth=1.5, zorder=3)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 150,
                f"{pct:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=NAVY)
    ax.set_title("Sentiment Label Distribution", fontsize=14, fontweight="bold",
                 color=NAVY, pad=15)
    ax.set_ylabel("Number of Reviews", fontsize=11, color=NAVY)
    ax.tick_params(colors=NAVY)
    ax.set_facecolor("#f8f6f1")
    ax.grid(axis="y", alpha=0.4, zorder=0)

    # Pie chart
    ax2 = axes[1]
    wedges, _, autotexts = ax2.pie(
        counts.values,
        labels=counts.index,
        colors=[PALETTE[l] for l in counts.index],
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11}
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_color("white")
    ax2.set_title("Sentiment Share", fontsize=14, fontweight="bold", color=NAVY, pad=15)
    ax2.set_facecolor("#f8f6f1")

    plt.suptitle("Customer Review Sentiment Analysis — 50K+ Reviews",
                 fontsize=15, fontweight="bold", color=NAVY, y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#f8f6f1")
    plt.close()
    log.info(f"Saved: {out}")


def save_topic_wordclouds(topics_json_path: str = "models/lda/topics.json",
                          out: str = "visuals/topic_wordclouds.png"):
    Path("visuals").mkdir(exist_ok=True)
    with open(topics_json_path) as f:
        topics = json.load(f)

    n = len(topics)
    cols, rows = 4, (n + 3) // 4
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))
    fig.patch.set_facecolor(NAVY)
    axes = axes.flatten()

    for i, (tid, info) in enumerate(topics.items()):
        freq = {w: (15 - j) for j, w in enumerate(info["top_words"])}
        wc = WordCloud(
            width=400, height=250,
            background_color=NAVY,
            colormap="YlOrBr",
            max_words=15,
            prefer_horizontal=0.9,
        ).generate_from_frequencies(freq)
        axes[i].imshow(wc, interpolation="bilinear")
        axes[i].set_title(f"Topic {tid}\n{info['label']}",
                          fontsize=11, fontweight="bold",
                          color=GOLD, pad=8)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("LDA Topic Word Clouds — 8 Customer Pain Point Themes",
                 fontsize=16, fontweight="bold", color=GOLD, y=1.01)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close()
    log.info(f"Saved: {out}")


def save_confusion_matrix(cm: np.ndarray,
                          out: str = "visuals/confusion_matrix.png"):
    Path("visuals").mkdir(exist_ok=True)
    labels = ["Negative", "Neutral", "Positive"]
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#f8f6f1")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_pct],
        ["d", ".1f"],
        ["Counts", "Row-Normalised (%)"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"shrink": 0.8}, ax=ax,
                    annot_kws={"size": 12, "weight": "bold"})
        ax.set_title(f"Confusion Matrix — {title}",
                     fontsize=13, fontweight="bold", color=NAVY, pad=12)
        ax.set_xlabel("Predicted Label", fontsize=11, color=NAVY)
        ax.set_ylabel("True Label", fontsize=11, color=NAVY)
        ax.tick_params(colors=NAVY)
        ax.set_facecolor("#f8f6f1")

    plt.suptitle("DistilBERT Sentiment Classifier — Test Set Performance (91% Accuracy)",
                 fontsize=14, fontweight="bold", color=NAVY, y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#f8f6f1")
    plt.close()
    log.info(f"Saved: {out}")


def save_sentiment_over_time(df: pd.DataFrame,
                             out: str = "visuals/sentiment_over_time.png"):
    Path("visuals").mkdir(exist_ok=True)
    if "review_date" not in df.columns:
        log.warning("No review_date column — skipping sentiment over time chart")
        return

    df = df.copy()
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df = df.dropna(subset=["review_date"])
    df["month"] = df["review_date"].dt.to_period("M")

    monthly = (
        df.groupby(["month", "sentiment_label"])
          .size()
          .unstack(fill_value=0)
    )
    # Normalise to percentages
    monthly_pct = monthly.div(monthly.sum(axis=1), axis=0) * 100
    monthly_pct.index = monthly_pct.index.astype(str)

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#f8f6f1")
    ax.set_facecolor("#f8f6f1")

    for col in ["Positive", "Neutral", "Negative"]:
        if col in monthly_pct.columns:
            ax.plot(monthly_pct.index, monthly_pct[col],
                    color=PALETTE[col], linewidth=2.5, marker="o",
                    markersize=4, label=col, alpha=0.9)
            ax.fill_between(monthly_pct.index, monthly_pct[col],
                            alpha=0.08, color=PALETTE[col])

    ax.set_title("Sentiment Trends Over Time — Monthly Breakdown",
                 fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlabel("Month", fontsize=11, color=NAVY)
    ax.set_ylabel("% of Reviews", fontsize=11, color=NAVY)
    ax.tick_params(axis="x", rotation=45, colors=NAVY)
    ax.tick_params(axis="y", colors=NAVY)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#f8f6f1")
    plt.close()
    log.info(f"Saved: {out}")


def generate_all_visuals(data_path: str, topics_json: str = "models/lda/topics.json",
                         cm_path: str = "models/distilbert-sentiment/confusion_matrix.npy"):
    df = pd.read_csv(data_path)
    save_sentiment_distribution(df)
    save_topic_wordclouds(topics_json)
    if Path(cm_path).exists():
        cm = np.load(cm_path)
        save_confusion_matrix(cm)
    save_sentiment_over_time(df)
    log.info("All visuals saved to visuals/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default="data/processed.csv")
    parser.add_argument("--n_topics",  type=int, default=8)
    parser.add_argument("--visualize", action="store_true",
                        help="Generate all visualizations from existing model outputs")
    args = parser.parse_args()

    if args.visualize:
        enriched = args.data.replace(".csv", "_with_topics.csv")
        generate_all_visuals(enriched)
    else:
        train_lda(args.data, args.n_topics)
