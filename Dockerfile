import os
import logging
import io
import base64

import requests
from PIL import Image
from pymilvus import MilvusClient
from dotenv import load_dotenv

# ---------- ENV ----------
load_dotenv()
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vision-rag")

# ---------- PATHS ----------
BASE_DIR = "/home/clouduser01/jaswanth"
DOCS_DIR = os.path.join(BASE_DIR, "Docs")
MILVUS_DB = os.path.join(BASE_DIR, "sample_milvus_rag.db")

COLLECTION = "rag_images"
DIM = 8  # required by Milvus Lite

milvus = MilvusClient(uri=MILVUS_DB)

# ---------- SUPPORTED FILE TYPES ----------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TEXT_EXTENSIONS = {".txt", ".html", ".htm", ".pdf", ".ppt", ".pptx"}

# ---------- IMAGE UTILS ----------
def image_to_base64(path: str):
    with Image.open(path) as img:
        buf = io.BytesIO()
        img = img.convert("RGB")
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return b64

# ---------- TEXT EXTRACTION (LIGHTWEIGHT FALLBACK) ----------
def extract_text_fallback(path: str) -> str:
    try:
        with open(path, "r", errors="ignore") as f:
            return f.read()[:5000]
    except Exception:
        return ""

# ---------- MILVUS INIT ----------
def ensure_collection():
    if not milvus.has_collection(COLLECTION):
        milvus.create_collection(
            collection_name=COLLECTION,
            dimension=DIM,
            metric_type="L2",
            auto_id=True,
        )
        logger.info("Milvus Lite collection created")
    else:
        logger.info("Milvus collection already exists")

# ---------- INGEST FILES ----------
def ingest_files(file_paths):
    for path in file_paths:
        path = os.path.abspath(path)

        if not os.path.isfile(path):
            logger.error(f"Invalid file path skipped: {path}")
            continue

        ext = os.path.splitext(path)[-1].lower()

        if ext in IMAGE_EXTENSIONS:
            img_b64 = image_to_base64(path)
            milvus.insert(
                collection_name=COLLECTION,
                data=[{
                    "vector": [0.0] * DIM,
                    "image_base64": img_b64,
                }],
            )
            logger.info(f"Image ingested: {path}")

        elif ext in TEXT_EXTENSIONS:
            text = extract_text_fallback(path)
            milvus.insert(
                collection_name=COLLECTION,
                data=[{
                    "vector": [0.0] * DIM,
                    "text": text,
                }],
            )
            logger.info(f"Text-based file ingested: {path}")

        else:
            logger.warning(f"Unsupported file skipped: {path}")

# ---------- RETRIEVE IMAGE ----------
def retrieve_image():
    res = milvus.query(
        collection_name=COLLECTION,
        limit=1,
        output_fields=["image_base64"],
    )

    if not res or "image_base64" not in res[0]:
        return None

    return res[0]["image_base64"]

# ---------- VISION QA ----------
def vision_rag(query: str) -> str:
    img_b64 = retrieve_image()

    if not img_b64:
        return "No image found to answer this question."

    image_url = f"data:image/jpeg;base64,{img_b64}"

    response = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "This is an official document."},
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {
                            "type": "text",
                            "text": f"Question: {query}\nAnswer clearly."
                        },
                    ],
                }
            ],
            "max_tokens": 300,
            "temperature": 0.4,
        },
        timeout=120,
    )

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ---------- MAIN ----------
if __name__ == "__main__":
    ensure_collection()

    files = [
        os.path.join(DOCS_DIR, "Singapore_NID_F 1.jpeg"),
        os.path.join(DOCS_DIR, "Singapore_NID_B 1.jpeg"),
        # PDFs / PPT / HTML can be added here
    ]

    ingest_files(files)

    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break
        print("A:", vision_rag(query))
