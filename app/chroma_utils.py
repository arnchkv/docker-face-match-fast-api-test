import chromadb # type: ignore
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction # type: ignore
import numpy as np

client = chromadb.Client()
collection = client.get_or_create_collection(name="faces")

# Optional: replace with real metadata later
def find_best_match(query_embedding, threshold=0.6):
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    if not result["ids"][0]:
        return "No match"

    distance = result["distances"][0][0]
    name = result["metadatas"][0][0].get("name", "Unknown")

    return name if distance < threshold else "No match"

def add_face(name, embedding):
    collection.add(
        ids=[name],
        embeddings=[embedding],
        metadatas=[{"name": name}]
    )
