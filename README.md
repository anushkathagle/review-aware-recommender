# Review-Grounded NLP System for Beauty Product Recommendation and Summarization

An end-to-end deep learning system that recommends beauty products and generates faithful, evidence-grounded pros-and-cons summaries from Amazon reviews.

---

## 📋 Project Overview

Given a user query (e.g., "recommend a shampoo"), the system:
1. **Asks clarification questions** if the query is too generic
2. **Retrieves relevant products** using semantic similarity
3. **Generates faithful pros-and-cons summaries** grounded in actual review text (no hallucination)

The key innovation: We extract and rank the **most important actual review sentences** instead of generating summaries, ensuring 100% faithfulness to source evidence.

---

## 🎯 Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Hallucination in Summaries** | Extractive design: rank real review sentences instead of generating text. **Achieved 100% grounding score.** |
| **Retrieval Quality** | BLAIR dense embeddings + sentiment-aware filtering. **Recall@20 = 0.9457**, outperforming TF-IDF by 2.2x. |
| **Generic Query Handling** | BiLSTM classifier detects vague queries and generates 2–3 targeted clarification questions before retrieval. |

---

## 🛠 System Architecture

```
OFFLINE PHASE (Pre-computation)
├── Data Preprocessing → segment reviews into sentences
├── Text Encoding (BLAIR) → 768-dimensional embeddings
├── Sentiment Classification (CNN) → label Pro/Con
└── Vector Indexing (FAISS HNSW) → fast ANN search

ONLINE PHASE (Query Processing)
├── Sub-task 1: Clarification Check (BiLSTM)
│   └── If generic query → ask clarifying questions
├── Sub-task 2: Sentiment-Aware Retrieval (BLAIR + FAISS)
│   └── Match query to products, retrieve Pro & Con pools
└── Sub-task 3: Evidence-Grounded Summarization (BiLSTM)
    └── Rank review sentences, select top-K per sentiment
```

---

## 📊 Results

| Component | Metric | Score |
|-----------|--------|-------|
| **CNN Sentiment Classifier** | Test Macro F1 | 0.8198 |
| **BiLSTM Clarification Classifier** | 5-Fold CV F1 | 0.9726 ± 0.0146 |
| **BLAIR Retrieval vs. TF-IDF** | Recall@20 | **0.9457** vs 0.3466 |
| **BiLSTM Extractive Summarizer** | ROUGE-1 | 0.3567 |
| **System Faithfulness** | Grounding Score | **1.0000 (100%)** |

---

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/anushka-thagle/nlp-beauty-recommendation.git
cd nlp-beauty-recommendation
pip install -r requirements.txt

# Run notebooks
jupyter notebook notebooks/Phase0_to_3_Colab_r1.ipynb    # Train sentiment classifier
jupyter notebook notebooks/Phase4_to_10_Colab_r1.ipynb   # Full pipeline
```

---

outputs



📄 Full Project Report

View Complete Technical Report (PDF)

The 50-page report includes:


Literature review (6 key papers)
Detailed data preprocessing & EDA
Model architecture & hyperparameter justification
Training curves & convergence analysis
Qualitative sanity checks & error analysis
Faithfulness evaluation methodology
Realistic use cases & deployment considerations
Full code base & execution instructions
