# Semantic Question Similarity Detection System

A Machine Learning / NLP project that detects whether two questions are **semantically similar (duplicate)** or **different**.

This project uses the **Quora Question Pairs dataset** and compares two approaches:

1. TF-IDF + Logistic Regression (Baseline model)
2. Sentence Transformers (MiniLM) for semantic similarity

The system calculates a **similarity score** between two questions and predicts whether they are duplicates.

---

## Project Overview

Many Q&A platforms such as **Quora** contain multiple questions asking the same thing in different words. Detecting such duplicate questions helps:

- Reduce redundant content
- Improve search quality
- Provide faster answers
- Improve knowledge organization

This project builds an NLP system that determines whether two questions have the **same meaning**.

---

## Features

- Text preprocessing pipeline
- TF-IDF + Logistic Regression baseline model
- Transformer-based semantic similarity model
- Cosine similarity based duplicate detection
- Interactive question comparison
- Model evaluation with accuracy, precision, recall and F1 score

---

## Model Performance

| Model | Accuracy |
|------|------|
| TF-IDF + Logistic Regression | ~0.76 |
| Sentence-BERT (MiniLM) | ~0.74 |

---

## Project Structure
```
Semantic-Question-Similarity-Detection-System
│
├── src
│ ├── preprocessing.py
│ ├── tfidf_model.py
│ ├── transformer_model.py
│ └── evaluation.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

- Python
- Scikit-learn
- Sentence Transformers
- PyTorch
- NumPy
- Natural Language Processing (NLP)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/KrishVerma6797/Semantic-Question-Similarity-Detection-System.git
cd Semantic-Question-Similarity-Detection-System
```

## Install dependencies:
```
pip install -r requirements.txt
```

## Running the Project
```
Run the main application:

python app.py

Example interaction:

Enter question 1: How can I learn Python?
Enter question 2: What is the best way to learn Python?

Similarity score: 0.92
Duplicate question
Enter question 1: What is France?
Enter question 2: What is kheer?

Similarity score: 0.14
Different question
```
## How It Works
1. Baseline Model
Convert text to TF-IDF vectors
Train Logistic Regression classifier
Predict duplicate or non-duplicate questions

2. Transformer Model
Use SentenceTransformer (MiniLM-L6-v2)
Convert questions into sentence embeddings
Compute cosine similarity
Apply a similarity threshold to detect duplicates

## Dataset

The project uses the Quora Question Pairs dataset, which contains pairs of questions labeled as:
1 → duplicate question
0 → different question

## Future Improvements

Add a web interface using Streamlit
Use FAISS vector search for faster similarity lookup
Train a fine-tuned transformer model
Deploy as an API
