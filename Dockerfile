import os
# Must be set BEFORE any grpc/ray imports — fixes fork() crash in pipeline subprocess
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
os.environ["GRPC_POLL_STRATEGY"] = "poll"

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

import logging
import socket
import time
from typing import List, Optional
import requests
from pymilvus import MilvusClient

# ---------- NV-Ingest ----------
from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import (
    run_pipeline,
    PipelineCreationSchema,
)
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# ---------- Env ----------
if "NVIDIA_API_KEY" not in os.environ:
    raise RuntimeError("Set NVIDIA_API_KEY in env")
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]

# ---------- Milvus ----------
MILVUS_DB = "./milvus_rag.db"
COLLECTION = "rag_documents"
DIM = 1024  # nv-embedqa-e5-v5 = 1024
milvus = MilvusClient(uri=MILVUS_DB)

# ---------- Wait for broker ----------
def wait_for_broker(host="localhost", port=7671, timeout=180):
    logger.info(f"Waiting for broker {host}:{port}...")
    start = time.time()
    while time.time() - start < timeout:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        if s.connect_ex((host, port)) == 0:
            s.close()
            logger.info("Broker ready!")
            return
        s.close()
        time.sleep(1)
    raise RuntimeError("Broker timeout — pipeline subprocess likely crashed. Check stderr above.")

# ---------- Embedding (for RAG retrieval queries) ----------
def embed_nvidia(texts: List[str]) -> List[List[float]]:
    r = requests.post(
        "https://integrate.api.nvidia.com/v1/embeddings",
        json={
            "model": "nvidia/nv-embedqa-e5-v5",
            "input": texts,
            "input_type": "query",
            "encoding_format": "float",
            "truncate": "END",
        },
        headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
        timeout=60,
    )
    r.raise_for_status()
    return [e["embedding"] for e in r.json()["data"]]

# ---------- INGESTION ----------
def ingest_document(file_paths: List[str], output_dir: Optional[str] = None) -> List[dict]:
    logger.info(f"Ingesting: {file_paths}")

    cfg = PipelineCreationSchema()
    run_pipeline(cfg, block=False, disable_dynamic_scaling=True, run_in_subprocess=True)
    logger.info("Pipeline subprocess launched...")

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
            params={"split_source_types": ["PDF", "text", "html", "mp3", "docx", "pptx"]},
        )
        .embed(
            endpoint_url="https://integrate.api.nvidia.com/v1/embeddings",
            model_name="nvidia/nv-embedqa-e5-v5",
            api_key=NVIDIA_API_KEY,
        )
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ingestor = ingestor.save_to_disk(output_directory=output_dir, cleanup=True)

    ingestor = ingestor.vdb_upload(
        collection_name=COLLECTION,
        milvus_uri=MILVUS_DB,
        dense_dim=DIM,
    )

    results_lazy, failures = ingestor.ingest(show_progress=True, return_failures=True)
    results = list(results_lazy)

    if failures:
        logger.warning(f"{len(failures)} failures during ingestion")

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
    q_emb = embed_nvidia([query])[0]
    hits = milvus.search(
        collection_name=COLLECTION,
        data=[q_emb],
        limit=top_k,
        output_fields=["text"],
    )[0]
    return [h["entity"].get("text") for h in hits]

# ---------- RAG ----------
def rag_chatbot(query: str) -> str:
    ctx = retrieve(query)
    if not ctx:
        return "No relevant info found."
    prompt = f"Context:\n{' '.join(ctx)}\n\nQuestion: {query}\nAnswer:"
    try:
        r = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            json={
                "model": "meta/llama-3.3-70b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
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
            metric_type="COSINE",
            auto_id=True,
        )
        logger.info("Created Milvus collection")
    else:
        logger.info("Milvus collection exists")

# ---------- MAIN ----------
if __name__ == "__main__":
    ensure_collection()

    pdf = "./Docs/multimodal_test.pdf"
    chunks = ingest_document([pdf], output_dir="./temp_ingest")

    print(f"\nIngested {len(chunks)} chunks")
    for c in chunks[:2]:
        print(" •", c["text"][:120].replace("\n", " ") + "...")

    print("\n--- RAG Chatbot Ready ---")
    print("Type your question (or 'exit' to quit)\n")

    while True:
        try:
            q = input("Ask a question: ").strip()
            if q.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            if not q:
                continue
            answer = rag_chatbot(q)
            print(f"\nA: {answer}\n")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
