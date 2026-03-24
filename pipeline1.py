import os
import logging
import socket
import time
from typing import List, Optional
import requests
from pymilvus import MilvusClient
# from sentence_transformers import SentenceTransformer

# ---------- Load .env ----------
from dotenv import load_dotenv
load_dotenv()

# ---------- NV-Ingest ----------
from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import (
    run_pipeline,
    PipelineCreationSchema,
)
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------- Env ----------
if "NVIDIA_API_KEY" not in os.environ:
    raise RuntimeError("Set NVIDIA_API_KEY in .env")
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]

# ---------- Milvus ----------
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
COLLECTION = "rag_documents"
DIM = 1024
milvus = MilvusClient(uri=MILVUS_URI)

# ---------- Wait for broker ----------
def wait_for_broker(host="localhost", port=7671, timeout=120):
    logger.info(f"Waiting for broker {host}:{port}…")
    start = time.time()
    while time.time() - start < timeout:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        if s.connect_ex((host, port)) == 0:
            s.close()
            logger.info("Broker ready!")
            return
        s.close()
        time.sleep(0.5)
    raise RuntimeError("Broker timeout")

# ---------- Embedding ----------
def embed_nvidia(texts: List[str]) -> List[List[float]]:
    url = "https://integrate.api.nvidia.com/v1/embeddings"
    payload = {
        "model": "nvidia/nv-embedqa-e5-v5",
        "input": texts,
        "input_type": "query",
    }
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}"}
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    return [e["embedding"] for e in r.json()["data"]]

def embed_fallback(texts: List[str]) -> List[List[float]]:
    model = SentenceTransformer("intfloat/e5-large")
    return model.encode(texts, normalize_embeddings=True).tolist()

# ---------- INGESTION (NO SHUTDOWN NEEDED) ----------
def ingest_document(file_paths: List[str], output_dir: Optional[str] = None) -> List[dict]:
    logger.info(f"Ingesting: {file_paths}")

    # Start pipeline (background)
    cfg = PipelineCreationSchema()
    run_pipeline(cfg, block=False, disable_dynamic_scaling=True, run_in_subprocess=True)
    logger.info("Pipeline started...")

    wait_for_broker()

    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost",
    )

    ingestor = (
        Ingestor(client=client)
        .files(file_paths)
        .load()
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=True,
            extract_infographics=True,
            table_output_format="markdown",
            text_depth="page",
        )
        .split(
            tokenizer="meta-llama/Llama-3.2-1B",
            chunk_size=512,
            chunk_overlap=50,
            params={"split_source_types": ["text", "table", "chart"]},
        )
        .caption(
            endpoint_url="https://integrate.api.nvidia.com/v1/chat/completions",
            model_name="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
            api_key=NVIDIA_API_KEY,
        )
        .embed()
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ingestor = ingestor.save_to_disk(output_directory=output_dir, cleanup=True)

    ingestor = ingestor.vdb_upload(
        collection_name=COLLECTION,
        milvus_uri=MILVUS_URI,
        dense_dim=DIM,
    )

    # Run ingestion
    results_lazy, failures = ingestor.ingest(show_progress=True, return_failures=True)
    results = list(results_lazy) # Convert to real list

    if failures:
        logger.warning(f"{len(failures)} failures")

    # Extract all chunks
    flat = []
    for doc in results:
        if hasattr(doc, "chunks"):
            for chunk in doc.chunks:
                flat.append({
                    "text": chunk.get("content", ""),
                    "embedding": chunk.get("embedding", [])
                })

    logger.info(f"Extracted {len(flat)} chunks → Milvus")
    return flat

# ---------- RETRIEVAL ----------
def retrieve(query: str, top_k: int = 5) -> List[str]:
    try:
        q_emb = embed_nvidia([query])[0]
    except Exception as e:
        logger.warning(f"Embed failed: {e}, using fallback")
        q_emb = embed_fallback([query])[0]

    hits = milvus.search(
        collection_name=COLLECTION,
        data=[q_emb],
        limit=top_k,
        output_fields=["text"],
    )[0]
    return [h.entity.get("text") for h in hits]

# ---------- RAG (FIXED LLM ENDPOINT) ----------
def rag_chatbot(query: str) -> str:
    ctx = retrieve(query)
    if not ctx:
        return "No relevant info."

    prompt = f"Context:\n{' '.join(ctx)}\n\nQuestion: {query}\nAnswer:"

    try:
        r = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions", 
            json={
                "model": "meta/llama-3.2-90b-vision-instruct", 
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.7,
            },
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"LLM error: {e}"

# ---------- MILVUS INIT ----------
def ensure_collection():
    if not milvus.has_collection(COLLECTION):
        milvus.create_collection(
            collection_name=COLLECTION,
            dimension=DIM,
            metric_type="L2",
            auto_id=True,
        )
        logger.info("Created collection")
    else:
        logger.info("Collection exists")

# ---------- MAIN ----------
if __name__ == "__main__":
    ensure_collection()

    pdf = "C:\\Users\\pabbidi\\Downloads\\NV-Ingest\\docks\\minion-tech.pdf"
    chunks = ingest_document([pdf], output_dir="./temp_ingest")

    print(f"\nIngested {len(chunks)} chunks")
    for c in chunks[:2]:
        print(" •", c["text"][:100].replace("\n", " ") + "...")

    queries = [
        "When was the agreement made?"
       
    ]
    for q in queries:
        print(f"\nQ: {q}")
        print(f"A: {rag_chatbot(q)}")