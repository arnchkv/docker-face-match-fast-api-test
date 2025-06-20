import chromadb  # type: ignore
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction  # type: ignore
from chromadb import PersistentClient
import numpy as np
import os

# Persistent directory (make sure it is mounted if using Docker)
CHROMA_DIR = "/chroma_storage"
os.makedirs(CHROMA_DIR, exist_ok=True)

# Initialize persistent client
client = PersistentClient(path=CHROMA_DIR)

# Get or create the "faces" collection
collection = client.get_or_create_collection(name="faces")

def find_best_match(query_embedding, threshold=0.6):
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        include=["distances", "metadatas"]
    )

    # Check if we got any match at all
    if not result["ids"] or not result["ids"][0]:
        return "No match"

    # Extract the first (top) match info
    distance = result["distances"][0][0]
    metadata = result["metadatas"][0][0]
    name = metadata.get("name", "Unknown")

    return name if distance < threshold else "No match"

# --- Add a new face embedding ---
def add_face(name, embedding):
    collection.add(
        ids=[name],
        embeddings=[embedding],
        metadatas=[{"name": name}]
    )
    # No need to call client.persist() in newer versions

# --- Delete a face from the collection ---
def delete_face(face_id: str):
    collection.delete(ids=[face_id])

# --- List all faces stored in Chroma ---
def list_all_faces():
    result = collection.get(include=["metadatas"])  # "ids" is included automatically

    faces = []
    for id_, metadata in zip(result["ids"], result["metadatas"]):
        faces.append({
            "id": id_,
            "name": metadata.get("name", "Unknown")
        })

    return faces
