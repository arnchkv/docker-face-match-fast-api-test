import face_recognition
import numpy as np
from PIL import Image
import io

def extract_face_embedding(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_image = np.array(image)
    encodings = face_recognition.face_encodings(np_image)
    return encodings[0].tolist() if encodings else None
