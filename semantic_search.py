from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    return dot_product / (norm_vec1 * norm_vec2)

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

file_path = "documents.txt"

try:
    with open(file_path,"r",encoding="utf-8") as f:
        lines = [line.strip() for line in f]
except Exception as e:
    print(f"An error occurred while reading the file: {e}")

try :
    response = client.models.embed_content(
        model = "gemini-embedding-001",
        contents = lines,
        config = types.EmbedContentConfig(task_type = "SEMANTIC_SIMILARITY" , output_dimensionality=768)
    )
except Exception as e:
    print(f"An error occurred while embedding the content: {e}")

embedding_obj = [e.values for e in response.embeddings]

print("Embeddings generated successfully.")

try:
    query = "I like Coding"
    query_response = client.models.embed_content(
        model = "gemini-embedding-001",
        contents = [query],
        config = types.EmbedContentConfig(task_type = "SEMANTIC_SIMILARITY" , output_dimensionality=768)
    )
except Exception as e:
    print(f"An error occurred while embedding the query: {e}")

query_embedding = query_response.embeddings[0].values
print("Query embedding generated successfully.")

print("Calculating cosine similarities between the query and document embeddings...")
similarities = [cosine_similarity(query_embedding, doc_embedding) for doc_embedding in embedding_obj]

print("Similarties for each document statement:\n")
print("Query: " + query + "\n")
for i, similarity in enumerate(similarities):
    print("Document statement: " + lines[i])
    print("Similarity: " + str(similarity))
