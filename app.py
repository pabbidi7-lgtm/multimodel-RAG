"""
NV-Ingest 25.9.0 Library Mode RAG Pipeline
===========================================
This is the REAL NV-Ingest pipeline using:
  - run_pipeline()         → starts Ray subprocess
  - SimpleClient broker    → port 7671
  - Ingestor chain         → .load() .extract() .split() .caption() .embed() .vdb_upload()
  - Milvus Lite            → local file DB (milvus_rag.db)
  - NVIDIA cloud NIMs      → no GPU needed

Milvus Lite explanation:
  - milvus_uri="milvus_rag.db" creates a LOCAL FILE database
  - No Docker needed, no separate Milvus server
  - .vdb_upload() in the Ingestor chain stores embeddings INTO this file
  - pymilvus MilvusClient reads FROM this same file for retrieval
  - It's the same Milvus, just embedded mode (like SQLite vs PostgreSQL)

API Keys needed:
  - NVIDIA_API_KEY      → from https://build.nvidia.com (free)
  - NVIDIA_BUILD_API_KEY → SAME key as above (set both to same value)
  Both are needed because different NIM endpoints check different env vars.

Install (on Linux with uv):
  uv venv --python 3.12 nvingest
  source nvingest/bin/activate
  uv pip install nv-ingest==25.9.0 nv-ingest-api==25.9.0 nv-ingest-client==25.9.0 milvus-lite==2.4.12

Run:
  export NVIDIA_API_KEY=nvapi-xxxxx
  export NVIDIA_BUILD_API_KEY=nvapi-xxxxx   # same key
  taskset -c 0-7 python rag_pipeline_NVINGEST.py
"""

import os
import logging
import socket
import time
from typing import List, Optional
import requests
from pymilvus import MilvusClient

# ---------- Load .env ----------
from dotenv import load_dotenv
load_dotenv()

# ---------- NV-Ingest 25.9.0 imports ----------
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
    raise RuntimeError("Set NVIDIA_API_KEY in .env or environment")
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]

# ---------- Milvus Lite (local file DB - NO Docker needed) ----------
# This creates a file called "milvus_rag.db" in your project folder
# Both .vdb_upload() and MilvusClient read/write to this SAME file
MILVUS_URI = os.environ.get("MILVUS_URI", "./milvus_rag.db")
COLLECTION = "rag_documents"
DIM = 1024  # nvidia/nv-embedqa-e5-v5 outputs 1024-dim vectors
milvus = MilvusClient(uri=MILVUS_URI)


# =========================================================================
#  WAIT FOR BROKER (same as lead's code)
# =========================================================================

def wait_for_broker(host="localhost", port=7671, timeout=120):
    """Wait for the NV-Ingest Ray pipeline broker to be ready on port 7671."""
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


# =========================================================================
#  EMBEDDING (for retrieval queries - calls NVIDIA cloud API directly)
# =========================================================================

def embed_nvidia(texts: List[str]) -> List[List[float]]:
    """Embed texts using NVIDIA NIM cloud endpoint for retrieval."""
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


# =========================================================================
#  INGESTION (the real NV-Ingest pipeline)
# =========================================================================

def ingest_document(file_paths: List[str], output_dir: Optional[str] = None) -> List[dict]:
    """
    Ingest documents through the NV-Ingest Ray pipeline.

    What happens inside:
      1. run_pipeline() starts a Ray subprocess (background)
      2. wait_for_broker() waits for port 7671 to open
      3. Ingestor chain processes the PDF:
         .load()       → load file bytes
         .extract()    → extract text/tables/charts/images via cloud NIMs
         .split()      → chunk text using Llama tokenizer
         .caption()    → generate image captions via VLM NIM
         .embed()      → embed all chunks via NVIDIA embedding NIM
         .vdb_upload() → store embeddings in Milvus Lite file
    """
    logger.info(f"Ingesting: {file_paths}")

    # Start Ray pipeline (background subprocess)
    cfg = PipelineCreationSchema()
    run_pipeline(cfg, block=False, disable_dynamic_scaling=True, run_in_subprocess=True)
    logger.info("Pipeline started...")

    # Wait for broker to be ready
    wait_for_broker()

    # Extra wait for Ray actors to fully initialize
    time.sleep(5)

    # Connect client to the broker
    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost",
    )

    # Build the Ingestor chain (same structure as lead's code)
    # No API keys hardcoded - they come from env vars automatically
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

    # Optional: save extracted content to disk
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ingestor = ingestor.save_to_disk(output_directory=output_dir, cleanup=True)

    # Upload embeddings to Milvus Lite
    # This writes to the SAME milvus_rag.db file that MilvusClient reads from
    ingestor = ingestor.vdb_upload(
        collection_name=COLLECTION,
        milvus_uri=MILVUS_URI,
        dense_dim=DIM,
    )

    # Run the full pipeline
    results_lazy, failures = ingestor.ingest(show_progress=True, return_failures=True)
    results = list(results_lazy)

    if failures:
        logger.warning(f"{len(failures)} failures")
        for i, f in enumerate(failures[:3]):
            logger.warning(f"  Failure [{i}]: {f}")

    # Extract chunks from results
    flat = []
    for doc in results:
        if hasattr(doc, "chunks"):
            for chunk in doc.chunks:
                flat.append({
                    "text": chunk.get("content", ""),
                    "embedding": chunk.get("embedding", [])
                })

    logger.info(f"Extracted {len(flat)} chunks → Milvus Lite ({MILVUS_URI})")
    return flat


# =========================================================================
#  RETRIEVAL (reads from the SAME Milvus Lite file)
# =========================================================================

def retrieve(query: str, top_k: int = 10) -> List[str]:
    """Retrieve relevant text chunks from Milvus Lite using vector similarity."""
    try:
        q_emb = embed_nvidia([query])[0]
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return []

    hits = milvus.search(
        collection_name=COLLECTION,
        data=[q_emb],
        limit=top_k,
        output_fields=["text"],
    )[0]
    return [h.entity.get("text") for h in hits if h.entity.get("text")]


# =========================================================================
#  RAG CHATBOT
# =========================================================================

def rag_chatbot(query: str) -> str:
    """Retrieve context and generate answer using NVIDIA LLM NIM."""
    ctx = retrieve(query)
    if not ctx:
        return "No relevant info."

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


# =========================================================================
#  MILVUS COLLECTION SETUP
# =========================================================================

def ensure_collection():
    """Create Milvus Lite collection if it doesn't exist."""
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


# =========================================================================
#  MAIN
# =========================================================================

if __name__ == "__main__":
    ensure_collection()

    # --- Your PDF path ---
    pdf = "./Docs/PK0016.pdf"
    chunks = ingest_document([pdf], output_dir="./temp_ingest")

    print(f"\nIngested {len(chunks)} chunks")
    for c in chunks[:2]:
        print(" •", c["text"][:100].replace("\n", " ") + "...")

    # --- Queries ---
    queries = [
        "What are all the test results that are outside the normal biological reference interval?",
        "Based on the kidney function test results and the eGFR classification table, what is the patient's GFR category?",
        "What is the patient's HbA1c value and does this patient fall in the non-diabetic, prediabetic, or diabetic range?",
        "Summarize all findings from the ultrasound whole abdomen including the impression and what further tests were advised?",
        "What are the patient's lipid profile results and classify each lipid parameter as optimal, borderline high, or high?",
    ]
    print("\n" + "=" * 60)
    for q in queries:
        print(f"\nQ: {q}")
        print(f"A: {rag_chatbot(q)}")
        print("-" * 60)
