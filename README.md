# 🛍️ Customer Review Sentiment & Topic Analysis

> Fine-tuned DistilBERT for sentiment classification (91% accuracy) + LDA topic modeling on 50K+ e-commerce reviews, deployed via Azure ML with real-time CRM connectivity.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![AzureML](https://img.shields.io/badge/Azure-ML-0078D4?logo=microsoftazure)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

This end-to-end NLP pipeline processes and analyzes over **50,000 e-commerce customer reviews** to:
- Classify sentiment (Positive / Neutral / Negative) using a fine-tuned **DistilBERT** model
- Surface emerging product themes and pain points using **LDA Topic Modeling**
- Serve real-time predictions via **Azure ML REST endpoints** connected to CRM systems

**Business Impact:**
| Metric | Result |
|--------|--------|
| Sentiment Classification Accuracy | **91%** |
| Manual QA Review Time Reduction | **60%** |
| Model Latency (Azure endpoint) | **<200ms** |
| Topics Identified | **8 distinct product themes** |

---

## 📊 Visualizations

### Sentiment Distribution
![Sentiment Distribution](visuals/sentiment_distribution.png)

### LDA Topic Word Clouds
![Topic Word Clouds](visuals/topic_wordclouds.png)

### Confusion Matrix
![Confusion Matrix](visuals/confusion_matrix.png)

### Sentiment Trend Over Time
![Sentiment Over Time](visuals/sentiment_over_time.png)

---

## 🗂️ Dataset

Using the **Amazon Product Reviews** dataset (subset: Electronics category).

| Field | Description |
|-------|-------------|
| `review_id` | Unique review identifier |
| `product_id` | Product ASIN |
| `rating` | Star rating (1–5) |
| `review_text` | Raw review text |
| `review_date` | Date of submission |
| `verified_purchase` | Boolean |

**To get the full dataset:**
```bash
# Option 1 — Hugging Face datasets
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('McAuley-Lab/Amazon-Reviews-2023', 'raw_review_Electronics', trust_remote_code=True); ds['full'].to_csv('data/reviews_full.csv')"

# Option 2 — Kaggle
# Download from: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
```
A **500-row sample** (`data/sample_reviews.csv`) is included for immediate testing.

---

## 🚀 Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/navya-manjunatha/customer-review-sentiment-analysis.git
cd customer-review-sentiment-analysis
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
```bash
# Step 1 — Preprocess
python src/preprocess.py --input data/sample_reviews.csv --output data/processed.csv

# Step 2 — Fine-tune DistilBERT
python src/sentiment_model.py --train --data data/processed.csv

# Step 3 — Topic Modeling
python src/topic_modeling.py --data data/processed.csv --n_topics 8

# Step 4 — Generate Visualizations
jupyter nbconvert --to notebook --execute notebooks/04_Visualizations.ipynb
```

### 3. Run Notebooks in Order
```
notebooks/01_EDA_and_Preprocessing.ipynb
notebooks/02_Sentiment_Classification_DistilBERT.ipynb
notebooks/03_LDA_Topic_Modeling.ipynb
notebooks/04_Visualizations.ipynb
```

---

## 🧠 Model Architecture

```
Raw Review Text
      │
      ▼
Text Preprocessing (cleaning, tokenization)
      │
      ▼
DistilBERT Tokenizer (max_length=128)
      │
      ▼
DistilBERT Base Uncased (fine-tuned, 3-class head)
      │
      ▼
Sentiment Label: Positive / Neutral / Negative
      │
      ▼ (parallel)
LDA Topic Modeling (8 topics, TF-IDF vectorized)
      │
      ▼
Topic Tags per Review
```

---

## ☁️ Azure ML Deployment

```bash
# Authenticate
az login

# Deploy endpoint
python azure_ml/deploy.py \
  --model-path models/distilbert-sentiment \
  --endpoint-name navya-sentiment-ep \
  --instance-type Standard_DS3_v2

# Test the live endpoint
curl -X POST https://<your-endpoint>.inference.ml.azure.com/score \
  -H "Authorization: Bearer <your-key>" \
  -H "Content-Type: application/json" \
  -d '{"text": "The battery life on this product is incredible!"}'
```

**Response:**
```json
{
  "sentiment": "Positive",
  "confidence": 0.97,
  "topic": "Battery & Performance",
  "inference_time_ms": 143
}
```

---

## 📁 Project Structure

```
├── data/               ← Raw and processed datasets
├── notebooks/          ← Step-by-step Jupyter notebooks
├── src/                ← Modular Python source code
├── azure_ml/           ← Azure ML scoring + deployment scripts
├── visuals/            ← Generated charts and plots
├── models/             ← Model weights (see instructions inside)
└── requirements.txt
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| NLP / ML | HuggingFace Transformers, Scikit-learn, NLTK, Gensim |
| Data | Pandas, NumPy, SQL (SQLite) |
| Visualization | Matplotlib, Seaborn, WordCloud, Plotly |
| Cloud | Azure ML, Azure Container Instances |
| Deployment | REST API, FastAPI (local), Azure ML Endpoints |

---

## 📬 Contact

**Navya Manjunatha** · Data Analyst  
📧 manjunatha.navya10@gmail.com  
🔗 [linkedin.com/in/navya-manjunatha](https://www.linkedin.com/in/navya-manjunatha/)

---

*⭐ If this project helped you, consider giving it a star!*
