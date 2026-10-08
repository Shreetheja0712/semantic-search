# Semantic Search with Gemini and PostgreSQL

A small semantic-search project that uses Google Gemini embeddings and PostgreSQL with the `pgvector` extension. Each non-empty line in `documents.txt` is embedded and stored in the database. A search query is embedded in the same vector space, then the database returns the three closest documents by cosine distance.

## How it works

```text
documents.txt
    -> Gemini `gemini-embedding-001`
    -> 768-dimensional vectors
    -> PostgreSQL / pgvector

search query
    -> Gemini embedding
    -> pgvector cosine-distance search (HNSW index)
    -> top 3 results with similarity scores
```

## Requirements

- Python 3.10+
- A Gemini API key
- PostgreSQL with the `pgvector` extension enabled

Install the Python dependencies:

```bash
pip install -r requirements.txt
pip install "psycopg[binary]"
```

## Configuration

Create a `.env` file in this directory:

```env
GEMINI_API_KEY=your_gemini_api_key
DATABASE_URL=postgresql://username:password@host:5432/database_name
```

The `.env` file is ignored by Git. Never commit real API keys or database credentials.

## Database setup

Run this SQL once in the configured PostgreSQL database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL
);
```

The `VECTOR(768)` dimension must match the `output_dimensionality=768` setting used by the scripts.

## Usage

Run the database connectivity check:

```bash
python test_db.py
```

Generate embeddings for every non-empty line in `documents.txt` and insert them into PostgreSQL:

```bash
python data_input.py
```

Run semantic search and enter a query when prompted:

```bash
python semantic_search.py
```

Example query:

```text
Search: software development
```

The search script prints the top three documents and their cosine similarity scores. Running `data_input.py` repeatedly inserts duplicate rows; clear the table first if you want to rebuild the index:

```sql
TRUNCATE TABLE documents RESTART IDENTITY;
```

## Project files

| File | Purpose |
| --- | --- |
| `documents.txt` | Source documents, one document per line |
| `data_input.py` | Generates and stores document embeddings |
| `semantic_search.py` | Embeds a query and searches PostgreSQL with pgvector |
| `test_db.py` | Verifies the `DATABASE_URL` connection |
| `semantic_search_v1.py` | Earlier in-memory version using NumPy cosine similarity |
| `requirements.txt` | Python dependencies |

## Tech stack

- Python
- Google Gemini API (`google-genai`)
- PostgreSQL and `pgvector`
- NumPy and `python-dotenv`
- Psycopg 3 (`psycopg`)
