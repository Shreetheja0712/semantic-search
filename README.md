# Semantic Search

A simple semantic search system built from scratch using **Gemini Embeddings** and **Cosine Similarity**.

## What is implemented

- Reads text documents from `documents.txt`
- Generates embeddings using `gemini-embedding-001`
- Generates an embedding for the user's search query
- Calculates cosine similarity between the query and each document
- Ranks documents based on semantic similarity
- Returns the Top-K most relevant results

## Tech Stack

- Python
- Google Gemini Embeddings
- NumPy
- python-dotenv

## Architecture

```text
Documents
    ↓
Gemini Embeddings
    ↓
Document Vectors
    ↓
User Query
    ↓
Query Embedding
    ↓
Cosine Similarity
    ↓
Rank Results
    ↓
Top-K Results