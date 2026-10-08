from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
import numpy as np
import psycopg


load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

conn = psycopg.connect(
    os.getenv("DATABASE_URL")
)

cursor = conn.cursor()

#COSINE SIMILARITY

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    return np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)





file_path = "documents.txt"

try:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()

except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    exit()


# Generate document embeddings


try:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=lines,
        config=types.EmbedContentConfig(
            task_type="SEMANTIC_SIMILARITY",
            output_dimensionality=768
        )
    )

except Exception as e:
    print(f"An error occurred while embedding documents: {e}")
    exit()


# Store documents + embeddings


try:
    for line, embedding in zip(lines, response.embeddings):
        cursor.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (line, embedding.values)
        )
except Exception as e:
    print(f"An error occurred while storing documents and embeddings: {e}")
    conn.rollback()
    exit()

print(f"Generated embeddings for documents.")

conn.commit()

cursor.close()
conn.close()


