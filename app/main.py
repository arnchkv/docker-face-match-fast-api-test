from fastapi import FastAPI, UploadFile, File  # type: ignore
import os
from app.face_utils import extract_face_embedding
from app.chroma_utils import find_best_match, add_face
import uuid
from fastapi.responses import JSONResponse, FileResponse
from fastapi import HTTPException
from app.chroma_utils import delete_face

app = FastAPI()

SAVE_DIR = "/registered_faces"

@app.get("/list/")
async def list_faces():
    try:
        files = os.listdir(SAVE_DIR)
        images = [f for f in files if f.endswith(".jpg")]
        return {"registered_faces": images}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/view/{filename}")
async def view_face(filename: str):
    file_path = os.path.join(SAVE_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path, media_type="image/jpeg")

@app.delete("/delete/{filename}")
async def delete_face_image(filename: str):
    file_path = os.path.join(SAVE_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        os.remove(file_path)
        delete_face(filename)  # You must implement this in chroma_utils
        return {"status": f"{filename} deleted"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

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