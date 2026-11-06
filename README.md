# Academic AI Assistant — Knowledge Retrieval Challenge

A document retrieval solution developed for the [Academic AI Assistant: Knowledge Retrieval Challenge](https://www.kaggle.com/competitions/machine-deep-learning) on Kaggle.

## Overview

The objective of the competition is to build an information retrieval system that identifies the most relevant knowledge base document (`document_id`) for a given user query.

The dataset represents a fictional academic institution and contains structured information about faculty, courses, examination schedules, attendance policies, academic calendars, campus services, placement rules, hostels, holidays, and FAQs.

The solution follows the baseline approach provided by the competition, with one additional component: **Cross-Encoder reranking**.

## Approach

The retrieval pipeline consists of two main stages:

1. **Candidate Retrieval — TF-IDF + Cosine Similarity**
   - Merge and preprocess the provided JSON knowledge base.
   - Convert documents and queries into TF-IDF representations.
   - Calculate cosine similarity between each query and the knowledge base documents.
   - Retrieve the Top-K most relevant candidate documents.

2. **Reranking — Cross-Encoder**
   - Evaluate query-document pairs using a Cross-Encoder.
   - Obtain relevance scores for the retrieved candidates.
   - Rerank candidates based on these scores.
   - Select the highest-ranked document as the final prediction.

The Cross-Encoder is used as an additional reranking stage rather than replacing the initial TF-IDF retrieval mechanism.

## Pipeline

```text
User Query
    |
    v
TF-IDF Vectorization
    |
    v
Cosine Similarity
    |
    v
Top-K Candidate Retrieval
    |
    v
Cross-Encoder Reranking
    |
    v
Best Matching Document
    |
    v
Predicted document_id
```

## Dataset

The competition provides:

- **Knowledge Base:** 190 structured documents stored across 10 JSON files.
- **Training Set:** 110 labeled queries.
- **Test Set:** 50 queries without target labels.
- **Target:** `document_id` — the identifier of the most relevant knowledge base document.

## Evaluation

The competition uses **Accuracy** as its evaluation metric.

A prediction is considered correct only when the predicted `document_id` exactly matches the ground-truth document identifier.

## Repository Structure

```text
.
└── Academic RAG.ipynb    # Complete baseline implementation with Cross-Encoder reranking
```

The notebook contains the complete workflow, from data preparation and retrieval to candidate reranking and submission generation.

## Technologies

- Python
- Pandas & NumPy
- Scikit-learn
- TF-IDF
- Cosine Similarity
- Sentence Transformers / Cross-Encoder
- Information Retrieval
- Natural Language Processing (NLP)

## Competition

[Kaggle — Academic AI Assistant: Knowledge Retrieval Challenge](https://www.kaggle.com/competitions/machine-deep-learning)
