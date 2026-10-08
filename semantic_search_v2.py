import os
import psycopg
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Get user query


query = input("Search: ").strip()

# Generate query embedding

try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=[query],
            config=types.EmbedContentConfig(
                task_type="SEMANTIC_SIMILARITY",
                output_dimensionality=768
            )
        )

        query_embedding = response.embeddings[0].values

except Exception as e:
    print(f"An error occurred while generating the query embedding: {e}")
    exit()

try:
    conn = psycopg.connect(
        os.getenv("DATABASE_URL")
    )
    cursor = conn.cursor()
except Exception as e:
    print(f"An error occurred while connecting to the database: {e}")
    exit()

try:
    cursor.execute(
        """
        SELECT
            content,
            1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT 3;
        """,
        (query_embedding, query_embedding)
    )

    results = cursor.fetchall()
except Exception as e:
    print(f"An error occurred while executing the query: {e}")
    cursor.close()
    conn.close()
    exit()

# Display results

print("\nTop results:\n")

for content, similarity in results:
    print(f"Similarity: {similarity:.4f}")
    print(f"Document: {content}")
    print()


cursor.close()
conn.close()