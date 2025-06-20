import chromadb  # type: ignore
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction  # type: ignore
import numpy as np

# Initialize ChromaDB client
client = chromadb.Client()

# Get or create the "faces" collection
collection = client.get_or_create_collection(name="faces")

# Find the best match using distance threshold
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

# Add a face embedding
def add_face(name, embedding):
    collection.add(
        ids=[name],
        embeddings=[embedding],
        metadatas=[{"name": name}]
    )
    # client.persist()

# Delete a face from the collection
def delete_face(face_id: str):
    collection.delete(ids=[face_id])
    # client.persist()
