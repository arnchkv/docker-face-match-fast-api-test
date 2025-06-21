import os
import uuid
from shutil import copyfile

from face_utils import extract_face_embedding
from chroma_utils import add_face

INPUT_DIR = "/input_images"
SAVE_DIR = "/registered_faces"

def register_all_faces():
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input folder not found: {INPUT_DIR}")
        return

    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            src_path = os.path.join(INPUT_DIR, filename)
            print(f"📷 Processing: {src_path}")

            try:
                with open(src_path, "rb") as f:
                    image_bytes = f.read()

                embedding = extract_face_embedding(image_bytes)
                if embedding is None:
                    print(f"❌ No face detected in {filename}")
                    continue

                face_id = f"{os.path.splitext(filename)[0]}_{str(uuid.uuid4())}.jpg"
                add_face(face_id, embedding)

                dest_path = os.path.join(SAVE_DIR, face_id)
                copyfile(src_path, dest_path)

                print(f"✅ Registered and saved: {dest_path}")
            except Exception as e:
                print(f"⚠️ Error processing {filename}: {e}")

if __name__ == "__main__":
    register_all_faces()
