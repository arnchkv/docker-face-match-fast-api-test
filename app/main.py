from fastapi import FastAPI, UploadFile, File  # type: ignore
import os
from app.face_utils import extract_face_embedding
from app.chroma_utils import find_best_match, add_face

app = FastAPI()

SAVE_DIR = "/app/registered_faces"

@app.post("/match/")
async def match_face(file: UploadFile = File(...)):
    image = await file.read()
    embedding = extract_face_embedding(image)
    if embedding is None:
        return {"match": "No face detected"}

    match = find_best_match(embedding)
    return {"match": match}


@app.post("/register/")
async def register_face(name: str, file: UploadFile = File(...)):
    img = await file.read()
    embedding = extract_face_embedding(img)
    if embedding is None:
        return {"error": "No face found"}

    # Generate unique filename
    unique_id = str(uuid.uuid4())
    filename_only = f"{name}_{unique_id}.jpg"
    full_path = os.path.join(SAVE_DIR, filename_only)

    # Save image persistently
    with open(full_path, "wb") as f:
        f.write(img)

    # Save to ChromaDB with filename as ID
    add_face(filename_only, embedding)

    return {
        "status": "registered",
        "filename": filename_only,
        "path": full_path
    }