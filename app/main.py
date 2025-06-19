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

    add_face(name, embedding)

    # Save the image persistently
    os.makedirs(SAVE_DIR, exist_ok=True)
    filename = os.path.join(SAVE_DIR, f"{name}.jpg")
    with open(filename, "wb") as f:
        f.write(img)

    return {"status": f"{name} registered and image saved"}
