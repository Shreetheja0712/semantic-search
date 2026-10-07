from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
import numpy as np

#COSINE SIMILARITY

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    return np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)



load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)



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


documents = []

for line, embedding in zip(lines, response.embeddings):
    documents.append({
        "text": line,
        "embedding": embedding.values
    })

print(f"Generated embeddings for {len(documents)} documents.")



# Get query


query = input("\nEnter your search query: ").strip()

if not query:
    print("Query cannot be empty.")
    exit()


# Generate query embedding


try:
    query_response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[query],
        config=types.EmbedContentConfig(
            task_type="SEMANTIC_SIMILARITY",
            output_dimensionality=768
        )
    )

except Exception as e:
    print(f"An error occurred while embedding the query: {e}")
    exit()


query_embedding = query_response.embeddings[0].values


# Calculate similarities


results = []

for document in documents:

    similarity = cosine_similarity(
        query_embedding,
        document["embedding"]
    )

    results.append({
        "text": document["text"],
        "similarity": similarity
    })


# Sort by similarity


results.sort(
    key=lambda x: x["similarity"],
    reverse=True
)

# Display Top K results


TOP_K = 3

print(f"\nQuery: {query}")
print(f"\nTop {TOP_K} results:\n")

for i, result in enumerate(results[:TOP_K], start=1):

    print(f"{i}. Similarity: {result['similarity']:.4f}")
    print(f"   {result['text']}")
    print()