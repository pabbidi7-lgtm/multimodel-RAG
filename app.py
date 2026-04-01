"""
NV-Ingest 25.9.0 + LangGraph RAG Agent — Single-file implementation
====================================================================
Architecture Layers:
  0 — Configuration
  1 — NV-Ingest pipeline (.vdb_upload REMOVED)
  2 — Content-type router (replaces vdb_upload)
  3 — Milvus Lite (3 collections: text / table / image)
  4 — Metadata enrichment (attached at router stage)
  5 — LangGraph RAG agent (5 nodes + retry loops)
  6 — RAGResponse (answer · confidence · sources · retry_count · latency_ms)
"""

import os
import sys
import json
import time
import uuid
import socket
import asyncio
import hashlib
import logging
from enum import Enum
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
)

import requests
from dotenv import load_dotenv
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

# ---------- NV-Ingest 25.9.0 imports ----------
from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import (
    run_pipeline,
    PipelineCreationSchema,
)
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient

# ---------- LangGraph ----------
from langgraph.graph import StateGraph, END

# ╔═══════════════════════════════════════════════════════════════════╗
# ║  LAYER 0 — CONFIGURATION                                        ║
# ╚═══════════════════════════════════════════════════════════════════╝

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("nv_ingest_rag")

# --- Required env ---
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise RuntimeError("Set NVIDIA_API_KEY in .env")


# --- Milvus Lite ---
MILVUS_DB = os.environ.get(
    "MILVUS_DB", os.path.join(os.path.dirname(__file__), "milvus_rag.db")
)
DIM = 1024  # nv-embedqa-e5-v5 dimension

# --- Broker ---
BROKER_HOST = os.environ.get("BROKER_HOST", "localhost")
BROKER_PORT = int(os.environ.get("BROKER_PORT", 7671))
BROKER_TIMEOUT = int(os.environ.get("BROKER_TIMEOUT", 120))

# --- Collection names ---
TEXT_COLLECTION = "text_collection"
TABLE_COLLECTION = "table_collection"
IMAGE_COLLECTION = "image_collection"

# --- LLM endpoints (NVIDIA API Catalog) ---
EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
RERANKER_URL = "https://integrate.api.nvidia.com/v1/ranking"
RERANKER_MODEL = "nvidia/nv-rerankqa-mistral-4b-v3"
LLM_PRIMARY_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
LLM_PRIMARY_MODEL = "meta/llama-3.3-70b-instruct"
LLM_FALLBACK_MODEL = "nvidia/nemotron-70b-instruct"
CAPTION_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
CAPTION_MODEL = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"

# --- Pipeline version tag ---
PIPELINE_VERSION = "nv-ingest-25.9.0-rag-agent-v1"

logger.info("Layer 0 — Configuration loaded")
logger.info(f"  MILVUS_DB   = {MILVUS_DB}")
logger.info(f"  DIM         = {DIM}")
logger.info(f"  BROKER      = {BROKER_HOST}:{BROKER_PORT}")


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  LAYER 3 — MILVUS LITE (3 collections)                          ║
# ╚═══════════════════════════════════════════════════════════════════╝
# Created before ingestion so Layer 2 router can insert.

milvus = MilvusClient(uri=MILVUS_DB)


def _create_text_collection():
    """text_collection: vector FLOAT 1024, text · page · section_title + metadata"""
    if milvus.has_collection(TEXT_COLLECTION):
        logger.info(f"  Collection '{TEXT_COLLECTION}' exists")
        return
    schema = CollectionSchema(fields=[
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema("text", DataType.VARCHAR, max_length=65535),
        FieldSchema("page", DataType.INT32),
        FieldSchema("section_title", DataType.VARCHAR, max_length=1024),
        FieldSchema("doc_id", DataType.VARCHAR, max_length=256),
        FieldSchema("chunk_id", DataType.VARCHAR, max_length=256),
        FieldSchema("ingested_at", DataType.VARCHAR, max_length=64),
        FieldSchema("pipeline_version", DataType.VARCHAR, max_length=128),
        FieldSchema("bbox", DataType.VARCHAR, max_length=256),
        FieldSchema("bbox_page_dims", DataType.VARCHAR, max_length=256),
        FieldSchema("word_count", DataType.INT32),
        FieldSchema("source_filename", DataType.VARCHAR, max_length=512),
    ])
    milvus.create_collection(
        collection_name=TEXT_COLLECTION,
        schema=schema,
    )
    milvus.create_index(
        collection_name=TEXT_COLLECTION,
        field_name="vector",
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )
    logger.info(f"  Created '{TEXT_COLLECTION}'")


def _create_table_collection():
    """table_collection: vector FLOAT 1024, text · headers · row_count + metadata"""
    if milvus.has_collection(TABLE_COLLECTION):
        logger.info(f"  Collection '{TABLE_COLLECTION}' exists")
        return
    schema = CollectionSchema(fields=[
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema("text", DataType.VARCHAR, max_length=65535),
        FieldSchema("headers", DataType.VARCHAR, max_length=4096),
        FieldSchema("row_count", DataType.INT32),
        FieldSchema("col_count", DataType.INT32),
        FieldSchema("table_id", DataType.VARCHAR, max_length=256),
        FieldSchema("doc_id", DataType.VARCHAR, max_length=256),
        FieldSchema("chunk_id", DataType.VARCHAR, max_length=256),
        FieldSchema("ingested_at", DataType.VARCHAR, max_length=64),
        FieldSchema("pipeline_version", DataType.VARCHAR, max_length=128),
        FieldSchema("bbox", DataType.VARCHAR, max_length=256),
        FieldSchema("bbox_page_dims", DataType.VARCHAR, max_length=256),
        FieldSchema("page", DataType.INT32),
        FieldSchema("source_filename", DataType.VARCHAR, max_length=512),
    ])
    milvus.create_collection(
        collection_name=TABLE_COLLECTION,
        schema=schema,
    )
    milvus.create_index(
        collection_name=TABLE_COLLECTION,
        field_name="vector",
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )
    logger.info(f"  Created '{TABLE_COLLECTION}'")


def _create_image_collection():
    """image_collection: vector FLOAT 1024, caption · image_path · bbox + metadata"""
    if milvus.has_collection(IMAGE_COLLECTION):
        logger.info(f"  Collection '{IMAGE_COLLECTION}' exists")
        return
    schema = CollectionSchema(fields=[
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema("caption", DataType.VARCHAR, max_length=65535),
        FieldSchema("image_path", DataType.VARCHAR, max_length=1024),
        FieldSchema("image_type", DataType.VARCHAR, max_length=64),
        FieldSchema("bbox", DataType.VARCHAR, max_length=256),
        FieldSchema("doc_id", DataType.VARCHAR, max_length=256),
        FieldSchema("chunk_id", DataType.VARCHAR, max_length=256),
        FieldSchema("ingested_at", DataType.VARCHAR, max_length=64),
        FieldSchema("pipeline_version", DataType.VARCHAR, max_length=128),
        FieldSchema("bbox_page_dims", DataType.VARCHAR, max_length=256),
        FieldSchema("page", DataType.INT32),
        FieldSchema("source_filename", DataType.VARCHAR, max_length=512),
    ])
    milvus.create_collection(
        collection_name=IMAGE_COLLECTION,
        schema=schema,
    )
    milvus.create_index(
        collection_name=IMAGE_COLLECTION,
        field_name="vector",
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )
    logger.info(f"  Created '{IMAGE_COLLECTION}'")


def ensure_collections():
    """Layer 3 init — create all 3 collections if they don't exist."""
    logger.info("Layer 3 — Ensuring Milvus collections …")
    _create_text_collection()
    _create_table_collection()
    _create_image_collection()


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  HELPERS — Broker wait + NVIDIA embedding                       ║
# ╚═══════════════════════════════════════════════════════════════════╝

def wait_for_broker(
    host: str = BROKER_HOST,
    port: int = BROKER_PORT,
    timeout: int = BROKER_TIMEOUT,
):
    logger.info(f"Waiting for broker {host}:{port} …")
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
    raise RuntimeError(f"Broker timeout after {timeout}s")


def embed_nvidia(texts: List[str], input_type: str = "query") -> List[List[float]]:
    """Call NVIDIA NV-EmbedQA E5-v5 endpoint. Returns list of 1024-dim vectors."""
    payload = {
        "model": EMBED_MODEL,
        "input": texts,
        "input_type": input_type,
    }
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}"}
    r = requests.post(EMBED_URL, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    return [e["embedding"] for e in r.json()["data"]]


def _generate_doc_id(file_path: str) -> str:
    """Deterministic doc ID from file path."""
    return hashlib.sha256(file_path.encode()).hexdigest()[:16]


def _generate_chunk_id() -> str:
    return uuid.uuid4().hex[:16]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  LAYER 1 — NV-INGEST PIPELINE (.vdb_upload REMOVED)             ║
# ║  .load() → .extract() → .split()+.caption() → .embed()         ║
# ╚═══════════════════════════════════════════════════════════════════╝

def run_nv_ingest_pipeline(
    file_paths: List[str],
    output_dir: Optional[str] = None,
) -> List[dict]:
    """
    Layer 1 — Run NV-Ingest 25.9.0 pipeline WITHOUT .vdb_upload().
    Returns raw result list from ingestor.ingest().
    """
    logger.info(f"Layer 1 — NV-Ingest pipeline for {len(file_paths)} file(s)")

    # Start Ray pipeline in background
    cfg = PipelineCreationSchema()
    run_pipeline(cfg, block=False, disable_dynamic_scaling=True, run_in_subprocess=True)
    logger.info("  Pipeline started (background)")

    wait_for_broker()

    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=BROKER_PORT,
        message_client_hostname=BROKER_HOST,
    )

    # Build ingestor chain — NOTE: NO .vdb_upload()
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
            params={
                "split_source_types": ["text", "table", "chart"],
                
            },
        )
        .caption(
            endpoint_url=CAPTION_URL,
            model_name=CAPTION_MODEL,
            api_key=NVIDIA_API_KEY,
        )
        .embed(
            endpoint_url="https://integrate.api.nvidia.com/v1",
            model_name=EMBED_MODEL,
            api_key=NVIDIA_API_KEY,
        )
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ingestor = ingestor.save_to_disk(output_directory=output_dir, cleanup=True)

    # NO .vdb_upload() — we handle routing ourselves in Layer 2

    results_lazy, failures = ingestor.ingest(show_progress=True, return_failures=True)
    results = list(results_lazy)

    if failures:
        logger.warning(f"  {len(failures)} ingestion failure(s)")
        for f in failures[:5]:
            logger.warning(f"    {f}")

    logger.info(f"  Pipeline produced {len(results)} result doc(s)")
    return results


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  LAYER 2 — CONTENT-TYPE ROUTER  (replaces .vdb_upload)          ║
# ║  Reads metadata["content_type"] + metadata["embedding"]         ║
# ║  Routes → text_collection | table_collection | image_collection ║
# ╚═══════════════════════════════════════════════════════════════════╝

# ╔═══════════════════════════════════════════════════════════════════╗
# ║  LAYER 4 — METADATA ENRICHMENT  (attached at router stage)      ║
# ║  Common: doc_id · chunk_id · ingested_at · pipeline_version     ║
# ║          bbox · bbox_page_dims                                   ║
# ║  TEXT:   section_title · word_count                              ║
# ║  TABLE:  headers · row_count · col_count · table_id             ║
# ║  IMAGE:  image_type · image_path (S3/MinIO)                     ║
# ╚═══════════════════════════════════════════════════════════════════╝

def _safe_str(val: Any, max_len: int = 65535) -> str:
    if val is None:
        return ""
    s = str(val)
    return s[:max_len]


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _extract_table_headers(text: str) -> str:
    """Extract first row from markdown table as headers."""
    lines = text.strip().split("\n")
    if lines and "|" in lines[0]:
        headers = [c.strip() for c in lines[0].split("|") if c.strip()]
        return json.dumps(headers)
    return "[]"


def _count_table_rows(text: str) -> int:
    lines = text.strip().split("\n")
    # skip header + separator
    data_lines = [l for l in lines if "|" in l and not set(l.replace("|", "").strip()).issubset({"-", " "})]
    return max(0, len(data_lines) - 1)  # minus header


def _count_table_cols(text: str) -> int:
    lines = text.strip().split("\n")
    if lines and "|" in lines[0]:
        return len([c for c in lines[0].split("|") if c.strip()])
    return 0


def content_type_router(
    results: List[dict],
    file_paths: List[str],
    image_output_dir: Optional[str] = None,
) -> Dict[str, int]:
    """
    Layers 2 + 4 combined.
    Iterates over NV-Ingest results, reads content_type + embedding from
    each chunk's metadata, enriches with metadata, and inserts into the
    correct Milvus collection.

    Returns counts: {"text": N, "table": N, "image": N}
    """
    logger.info("Layer 2+4 — Content-type routing + metadata enrichment")
    now = _now_iso()
    counts = {"text": 0, "table": 0, "image": 0}

    # Build doc_id map from file paths
    doc_id_map = {}
    for fp in file_paths:
        doc_id_map[os.path.basename(fp)] = _generate_doc_id(fp)

    for doc_idx, doc in enumerate(results):
        # NV-Ingest results can be a list of dicts or objects — normalise
        if isinstance(doc, dict):
            items = doc.get("data", [doc])
        elif isinstance(doc, list):
            items = doc
        elif hasattr(doc, "data"):
            items = doc.data if isinstance(doc.data, list) else [doc.data]
        else:
            items = [doc]

        for item in items:
            # Normalise to dict
            if not isinstance(item, dict):
                if hasattr(item, "__dict__"):
                    item = item.__dict__
                else:
                    continue

            metadata = item.get("metadata", item)
            content = metadata.get("content", item.get("content", item.get("text", "")))
            content_type = metadata.get("content_type", metadata.get("type", "text"))
            embedding = metadata.get("embedding", item.get("embedding", []))

            # Skip if no embedding
            if not embedding or len(embedding) != DIM:
                logger.debug(f"  Skipping chunk — embedding missing or wrong dim")
                continue

            # Determine source filename
            source_name = metadata.get("source_name", metadata.get("source_filename", ""))
            if not source_name and doc_idx < len(file_paths):
                source_name = os.path.basename(file_paths[doc_idx])

            doc_id = doc_id_map.get(source_name, _generate_doc_id(source_name or str(doc_idx)))
            chunk_id = _generate_chunk_id()

            # Common metadata (Layer 4)
            bbox_raw = metadata.get("bbox", metadata.get("bounding_box", ""))
            bbox = json.dumps(bbox_raw) if isinstance(bbox_raw, (list, dict)) else _safe_str(bbox_raw, 256)
            bbox_page_dims_raw = metadata.get("bbox_page_dims", metadata.get("page_dimensions", ""))
            bbox_page_dims = json.dumps(bbox_page_dims_raw) if isinstance(bbox_page_dims_raw, (list, dict)) else _safe_str(bbox_page_dims_raw, 256)
            page = _safe_int(metadata.get("page", metadata.get("page_number", 0)))

            content_str = _safe_str(content)
            ct_lower = str(content_type).lower()

            # ---- ROUTE ----
            if ct_lower in ("table", "chart", "structured"):
                # TABLE collection
                row = {
                    "vector": embedding,
                    "text": content_str,
                    "headers": _extract_table_headers(content_str),
                    "row_count": _count_table_rows(content_str),
                    "col_count": _count_table_cols(content_str),
                    "table_id": f"tbl_{chunk_id}",
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "ingested_at": now,
                    "pipeline_version": PIPELINE_VERSION,
                    "bbox": bbox,
                    "bbox_page_dims": bbox_page_dims,
                    "page": page,
                    "source_filename": _safe_str(source_name, 512),
                }
                milvus.insert(collection_name=TABLE_COLLECTION, data=[row])
                counts["table"] += 1

            elif ct_lower in ("image", "infographic", "figure", "chart_image"):
                # IMAGE collection
                image_path = _safe_str(
                    metadata.get("image_path", metadata.get("artifact_path", "")),
                    1024,
                )
                image_type = _safe_str(
                    metadata.get("image_type", metadata.get("subtype", "unknown")),
                    64,
                )
                caption = content_str  # caption from .caption() stage
                row = {
                    "vector": embedding,
                    "caption": caption,
                    "image_path": image_path,
                    "image_type": image_type,
                    "bbox": bbox,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "ingested_at": now,
                    "pipeline_version": PIPELINE_VERSION,
                    "bbox_page_dims": bbox_page_dims,
                    "page": page,
                    "source_filename": _safe_str(source_name, 512),
                }
                milvus.insert(collection_name=IMAGE_COLLECTION, data=[row])
                counts["image"] += 1

            else:
                # TEXT collection (default)
                section_title = _safe_str(
                    metadata.get("section_title", metadata.get("title", "")),
                    1024,
                )
                word_count = len(content_str.split()) if content_str else 0
                row = {
                    "vector": embedding,
                    "text": content_str,
                    "page": page,
                    "section_title": section_title,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "ingested_at": now,
                    "pipeline_version": PIPELINE_VERSION,
                    "bbox": bbox,
                    "bbox_page_dims": bbox_page_dims,
                    "word_count": word_count,
                    "source_filename": _safe_str(source_name, 512),
                }
                milvus.insert(collection_name=TEXT_COLLECTION, data=[row])
                counts["text"] += 1

    logger.info(f"  Routed → text={counts['text']}, table={counts['table']}, image={counts['image']}")
    return counts


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  FULL INGESTION ENTRY POINT (Layer 0→1→2→3→4)                   ║
# ╚═══════════════════════════════════════════════════════════════════╝

def ingest_documents(
    file_paths: List[str],
    output_dir: Optional[str] = None,
) -> Dict[str, int]:
    """
    End-to-end ingestion:
      1. Ensure Milvus collections exist (Layer 3)
      2. Run NV-Ingest pipeline (Layer 1)
      3. Route + enrich + insert into Milvus (Layers 2 + 4)
    """
    ensure_collections()
    results = run_nv_ingest_pipeline(file_paths, output_dir)
    counts = content_type_router(results, file_paths, output_dir)
    return counts


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  LAYER 5 — LANGGRAPH RAG AGENT (5 nodes)                        ║
# ║  Node 1: query_classifier                                       ║
# ║  Node 2: parallel_retriever  (dense + BM25 → RRF)               ║
# ║  Node 3: result_merger       (dynamic weights · dedup · top 30)  ║
# ║  Node 4: reranker            (nv-rerankqa-mistral-4b-v3)         ║
# ║  Node 5: generator           (llama-3.3-70b + nemotron fallback) ║
# ║                                                                  ║
# ║  Loops: all LOW → retry ×1  |  hallucination → retry ×1         ║
# ╚═══════════════════════════════════════════════════════════════════╝

# --- Layer 6: RAGResponse ---

@dataclass
class RAGResponse:
    answer: str = ""
    confidence: str = "LOW"  # HIGH / MEDIUM / LOW
    sources: List[Dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0
    latency_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --- LangGraph State ---

class RAGState(TypedDict):
    query: str
    # Node 1 outputs
    intent: str
    collection_targets: List[str]
    collection_weights: Dict[str, float]
    # Node 2 outputs
    raw_results: List[Dict[str, Any]]
    # Node 3 outputs
    merged_results: List[Dict[str, Any]]
    # Node 4 outputs
    reranked_results: List[Dict[str, Any]]
    reranker_scores: Dict[str, str]  # chunk_id → HIGH/MEDIUM/LOW
    all_low: bool
    # Node 5 outputs
    answer: str
    confidence: str
    is_hallucination: bool
    # Control
    retry_count: int
    max_retries: int
    latency_ms: Dict[str, float]
    sources: List[Dict[str, Any]]


# ── Node 1: query_classifier ──

def node_query_classifier(state: RAGState) -> RAGState:
    """
    Classify query intent → determine collection targets + dynamic weights.
    Uses LLM to classify what kind of information the query needs.
    """
    t0 = time.time()
    query = state["query"]

    classification_prompt = f"""Classify this query into one or more content types it needs.
Return ONLY a valid JSON object with no extra text.

Query: "{query}"

Return format:
{{"intent": "text|table|image|mixed", "collections": {{"text_collection": 0.0-1.0, "table_collection": 0.0-1.0, "image_collection": 0.0-1.0}}}}

Rules:
- If query asks about data, numbers, comparisons → high weight on table_collection
- If query asks about images, diagrams, figures → high weight on image_collection
- If query is general text question → high weight on text_collection
- For mixed queries, distribute weights. Weights should sum to 1.0.
- Always include at least text_collection with weight >= 0.2
"""

    try:
        r = requests.post(
            LLM_PRIMARY_URL,
            json={
                "model": LLM_PRIMARY_MODEL,
                "messages": [{"role": "user", "content": classification_prompt}],
                "max_tokens": 200,
                "temperature": 0.0,
            },
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
            timeout=30,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        # Strip markdown code fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        intent = parsed.get("intent", "text")
        weights = parsed.get("collections", {"text_collection": 1.0})
    except Exception as e:
        logger.warning(f"  query_classifier fallback: {e}")
        intent = "text"
        weights = {TEXT_COLLECTION: 0.7, TABLE_COLLECTION: 0.2, IMAGE_COLLECTION: 0.1}

    # Ensure all collections present
    for col in [TEXT_COLLECTION, TABLE_COLLECTION, IMAGE_COLLECTION]:
        if col not in weights:
            weights[col] = 0.0

    targets = [col for col, w in weights.items() if w > 0.05]

    elapsed = (time.time() - t0) * 1000
    logger.info(f"  Node 1 query_classifier: intent={intent}, targets={targets} ({elapsed:.0f}ms)")

    state["intent"] = intent
    state["collection_targets"] = targets
    state["collection_weights"] = weights
    state["latency_ms"]["query_classifier"] = round(elapsed, 1)
    return state


# ── Node 2: parallel_retriever ──

def _dense_search(collection: str, query_embedding: List[float], top_k: int = 20) -> List[dict]:
    """Dense vector search on a single collection."""
    # Determine output fields based on collection
    if collection == TEXT_COLLECTION:
        output_fields = ["text", "page", "section_title", "doc_id", "chunk_id", "source_filename", "word_count"]
    elif collection == TABLE_COLLECTION:
        output_fields = ["text", "headers", "row_count", "col_count", "table_id", "doc_id", "chunk_id", "page", "source_filename"]
    elif collection == IMAGE_COLLECTION:
        output_fields = ["caption", "image_path", "image_type", "doc_id", "chunk_id", "page", "source_filename"]
    else:
        output_fields = ["text", "doc_id", "chunk_id"]

    try:
        hits = milvus.search(
            collection_name=collection,
            data=[query_embedding],
            limit=top_k,
            output_fields=output_fields,
        )[0]
    except Exception as e:
        logger.warning(f"  Dense search failed on {collection}: {e}")
        return []

    results = []
    for rank, hit in enumerate(hits):
        entity = hit.entity if hasattr(hit, "entity") else hit
        text_content = ""
        if isinstance(entity, dict):
            text_content = entity.get("text", entity.get("caption", ""))
            entity_dict = entity
        elif hasattr(entity, "get"):
            text_content = entity.get("text", entity.get("caption", ""))
            entity_dict = {f: entity.get(f, "") for f in output_fields}
        else:
            continue

        results.append({
            "text": text_content,
            "collection": collection,
            "dense_rank": rank + 1,
            "dense_score": hit.distance if hasattr(hit, "distance") else hit.get("distance", 0),
            "chunk_id": entity_dict.get("chunk_id", ""),
            "doc_id": entity_dict.get("doc_id", ""),
            "metadata": entity_dict,
        })
    return results


def _bm25_search(collection: str, query: str, top_k: int = 20) -> List[dict]:
    """
    BM25-style keyword search via Milvus query + client-side scoring.
    Milvus Lite doesn't have native BM25, so we do a simple keyword filter.
    """
    # Determine text field
    text_field = "caption" if collection == IMAGE_COLLECTION else "text"

    # Simple keyword extraction
    keywords = [w.lower() for w in query.split() if len(w) > 2]
    if not keywords:
        return []

    # Build filter — check if any keyword is in the text
    # Milvus Lite supports basic string matching
    filters = []
    for kw in keywords[:3]:  # limit to 3 keywords for filter
        filters.append(f'{text_field} like "%{kw}%"')

    filter_str = " or ".join(filters)

    if collection == TEXT_COLLECTION:
        output_fields = ["text", "page", "section_title", "doc_id", "chunk_id", "source_filename"]
    elif collection == TABLE_COLLECTION:
        output_fields = ["text", "headers", "row_count", "doc_id", "chunk_id", "page", "source_filename"]
    elif collection == IMAGE_COLLECTION:
        output_fields = ["caption", "image_path", "image_type", "doc_id", "chunk_id", "page", "source_filename"]
    else:
        output_fields = ["text", "doc_id", "chunk_id"]

    try:
        hits = milvus.query(
            collection_name=collection,
            filter=filter_str,
            output_fields=output_fields,
            limit=top_k,
        )
    except Exception as e:
        logger.debug(f"  BM25 query failed on {collection}: {e}")
        return []

    results = []
    for rank, hit in enumerate(hits):
        text_content = hit.get("text", hit.get("caption", ""))
        # Simple BM25-like score: count keyword matches
        text_lower = text_content.lower()
        score = sum(1 for kw in keywords if kw in text_lower) / max(len(keywords), 1)
        results.append({
            "text": text_content,
            "collection": collection,
            "bm25_rank": rank + 1,
            "bm25_score": score,
            "chunk_id": hit.get("chunk_id", ""),
            "doc_id": hit.get("doc_id", ""),
            "metadata": hit,
        })

    # Sort by score desc
    results.sort(key=lambda x: x["bm25_score"], reverse=True)
    return results[:top_k]


def node_parallel_retriever(state: RAGState) -> RAGState:
    """
    Node 2 — Parallel retrieval: dense + BM25 across targeted collections.
    Uses asyncio.gather() pattern for parallelism.
    """
    t0 = time.time()
    query = state["query"]
    targets = state["collection_targets"]

    # Get query embedding
    try:
        q_emb = embed_nvidia([query], input_type="query")[0]
    except Exception as e:
        logger.error(f"  Embedding failed: {e}")
        state["raw_results"] = []
        state["latency_ms"]["parallel_retriever"] = round((time.time() - t0) * 1000, 1)
        return state

    all_results = []

    # Run dense + BM25 for each target collection
    for col in targets:
        dense_hits = _dense_search(col, q_emb, top_k=20)
        bm25_hits = _bm25_search(col, query, top_k=20)
        all_results.extend(dense_hits)
        all_results.extend(bm25_hits)

    elapsed = (time.time() - t0) * 1000
    logger.info(f"  Node 2 parallel_retriever: {len(all_results)} raw hits ({elapsed:.0f}ms)")

    state["raw_results"] = all_results
    state["latency_ms"]["parallel_retriever"] = round(elapsed, 1)
    return state


# ── Node 3: result_merger ──

def _reciprocal_rank_fusion(results: List[dict], k: int = 60) -> List[dict]:
    """RRF scoring: score = Σ 1/(k + rank) across systems."""
    chunk_scores: Dict[str, float] = {}
    chunk_map: Dict[str, dict] = {}

    for r in results:
        cid = r.get("chunk_id", id(r))
        if cid not in chunk_map:
            chunk_map[cid] = r
            chunk_scores[cid] = 0.0

        if "dense_rank" in r:
            chunk_scores[cid] += 1.0 / (k + r["dense_rank"])
        if "bm25_rank" in r:
            chunk_scores[cid] += 1.0 / (k + r["bm25_rank"])

    # Sort by RRF score descending
    sorted_ids = sorted(chunk_scores, key=chunk_scores.get, reverse=True)
    fused = []
    for cid in sorted_ids:
        entry = chunk_map[cid].copy()
        entry["rrf_score"] = chunk_scores[cid]
        fused.append(entry)
    return fused


def node_result_merger(state: RAGState) -> RAGState:
    """
    Node 3 — Merge results with dynamic weights, dedup by chunk_id, top 30.
    """
    t0 = time.time()
    raw = state["raw_results"]
    weights = state.get("collection_weights", {})

    # Apply collection weights to scores
    for r in raw:
        col = r.get("collection", "")
        w = weights.get(col, 1.0)
        if "dense_score" in r:
            # For L2, lower is better — invert for weighting
            r["dense_rank"] = r.get("dense_rank", 999)
        if "bm25_score" in r:
            r["bm25_score"] = r.get("bm25_score", 0) * w

    # RRF fusion
    merged = _reciprocal_rank_fusion(raw)

    # Apply collection weights to RRF scores
    for m in merged:
        col = m.get("collection", "")
        w = weights.get(col, 1.0)
        m["rrf_score"] = m.get("rrf_score", 0) * w

    # Re-sort after weighting
    merged.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)

    # Dedup by chunk_id — keep first (highest score)
    seen = set()
    deduped = []
    for m in merged:
        cid = m.get("chunk_id", "")
        if cid and cid in seen:
            continue
        seen.add(cid)
        deduped.append(m)

    # Top 30
    top30 = deduped[:30]

    elapsed = (time.time() - t0) * 1000
    logger.info(f"  Node 3 result_merger: {len(raw)} → {len(top30)} after dedup+top30 ({elapsed:.0f}ms)")

    state["merged_results"] = top30
    state["latency_ms"]["result_merger"] = round(elapsed, 1)
    return state


# ── Node 4: reranker ──

def node_reranker(state: RAGState) -> RAGState:
    """
    Node 4 — Rerank with nv-rerankqa-mistral-4b-v3.
    Classifies each result as HIGH / MEDIUM / LOW.
    If ALL results are LOW → set all_low flag for retry.
    """
    t0 = time.time()
    query = state["query"]
    merged = state["merged_results"]

    if not merged:
        state["reranked_results"] = []
        state["reranker_scores"] = {}
        state["all_low"] = True
        state["latency_ms"]["reranker"] = 0
        return state

    # Prepare passages for reranker
    passages = [{"text": r.get("text", "")} for r in merged]

    try:
        r = requests.post(
            RERANKER_URL,
            json={
                "model": RERANKER_MODEL,
                "query": {"text": query},
                "passages": passages[:40],  # API limit
            },
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
            timeout=60,
        )
        r.raise_for_status()
        rankings = r.json().get("rankings", [])
    except Exception as e:
        logger.warning(f"  Reranker API failed: {e}, using RRF order")
        # Fallback: keep RRF order, mark all as MEDIUM
        for m in merged:
            m["reranker_label"] = "MEDIUM"
            m["reranker_score"] = m.get("rrf_score", 0)
        state["reranked_results"] = merged[:15]
        state["reranker_scores"] = {m.get("chunk_id", ""): "MEDIUM" for m in merged}
        state["all_low"] = False
        state["latency_ms"]["reranker"] = round((time.time() - t0) * 1000, 1)
        return state

    # Map reranker scores back to merged results
    score_map = {}  # original_index → logit
    for rank_item in rankings:
        idx = rank_item.get("index", 0)
        logit = rank_item.get("logit", 0.0)
        score_map[idx] = logit

    # Assign HIGH / MEDIUM / LOW labels
    reranker_scores = {}
    for i, m in enumerate(merged):
        logit = score_map.get(i, -10.0)
        m["reranker_score"] = logit
        if logit > 0:
            label = "HIGH"
        elif logit > -3:
            label = "MEDIUM"
        else:
            label = "LOW"
        m["reranker_label"] = label
        cid = m.get("chunk_id", str(i))
        reranker_scores[cid] = label

    # Sort by reranker score desc
    sorted_results = sorted(merged, key=lambda x: x.get("reranker_score", -999), reverse=True)

    # Filter: keep HIGH + MEDIUM only
    filtered = [r for r in sorted_results if r.get("reranker_label") in ("HIGH", "MEDIUM")]

    all_low = len(filtered) == 0

    if all_low:
        # Keep top 5 even if LOW for retry attempt
        filtered = sorted_results[:5]

    elapsed = (time.time() - t0) * 1000
    logger.info(f"  Node 4 reranker: {len(filtered)} kept (all_low={all_low}) ({elapsed:.0f}ms)")

    state["reranked_results"] = filtered
    state["reranker_scores"] = reranker_scores
    state["all_low"] = all_low
    state["latency_ms"]["reranker"] = round(elapsed, 1)
    return state


# ── Node 5: generator ──

INPUT_GUARDRAILS_PROMPT = """You are a factual RAG assistant. Rules:
1. Answer ONLY based on the provided context. Do not use prior knowledge.
2. If the context does not contain enough information, say "I don't have enough information to answer this based on the available documents."
3. Cite which source/page the information comes from when possible.
4. Be concise and precise.
5. Do not fabricate information.
"""

OUTPUT_GUARDRAILS_CHECK_PROMPT = """Given this context and answer, determine if the answer is hallucinated.

Context:
{context}

Answer:
{answer}

Respond with ONLY a JSON object: {{"is_hallucination": true/false, "confidence": "HIGH"/"MEDIUM"/"LOW", "reason": "brief explanation"}}
"""


def _call_llm(prompt: str, model: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
    """Call NVIDIA LLM endpoint."""
    r = requests.post(
        LLM_PRIMARY_URL,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": INPUT_GUARDRAILS_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def node_generator(state: RAGState) -> RAGState:
    """
    Node 5 — Generate answer with context formatting, input + output guardrails.
    Primary: llama-3.3-70b  |  Fallback: nemotron-70b
    Includes hallucination check.
    """
    t0 = time.time()
    query = state["query"]
    results = state["reranked_results"]

    if not results:
        state["answer"] = "No relevant information found in the documents."
        state["confidence"] = "LOW"
        state["is_hallucination"] = False
        state["sources"] = []
        state["latency_ms"]["generator"] = 0
        return state

    # Format context inline (as per architecture)
    context_parts = []
    sources = []
    for i, r in enumerate(results):
        text = r.get("text", "")
        collection = r.get("collection", "unknown")
        page = r.get("metadata", {}).get("page", "?")
        source_file = r.get("metadata", {}).get("source_filename", "unknown")
        label = r.get("reranker_label", "?")

        context_parts.append(
            f"[Source {i+1} | {collection} | Page {page} | Relevance: {label}]\n{text}"
        )
        sources.append({
            "chunk_id": r.get("chunk_id", ""),
            "collection": collection,
            "page": page,
            "source_filename": source_file,
            "reranker_label": label,
            "reranker_score": r.get("reranker_score", 0),
        })

    context_block = "\n\n---\n\n".join(context_parts)

    prompt = f"""Context:
{context_block}

Question: {query}

Answer based strictly on the context above. Cite source numbers [Source N] when referencing information."""

    # Primary LLM: llama-3.3-70b
    answer = ""
    model_used = LLM_PRIMARY_MODEL
    try:
        answer = _call_llm(prompt, model=LLM_PRIMARY_MODEL)
    except Exception as e:
        logger.warning(f"  Primary LLM failed: {e}, trying fallback")
        try:
            answer = _call_llm(prompt, model=LLM_FALLBACK_MODEL)
            model_used = LLM_FALLBACK_MODEL
        except Exception as e2:
            logger.error(f"  Fallback LLM also failed: {e2}")
            answer = f"LLM generation failed: {e2}"
            state["answer"] = answer
            state["confidence"] = "LOW"
            state["is_hallucination"] = False
            state["sources"] = sources
            state["latency_ms"]["generator"] = round((time.time() - t0) * 1000, 1)
            return state

    # ── Hallucination check ──
    is_hallucination = False
    confidence = "MEDIUM"
    try:
        check_prompt = OUTPUT_GUARDRAILS_CHECK_PROMPT.format(
            context=context_block[:3000],  # Truncate for check
            answer=answer,
        )
        check_r = requests.post(
            LLM_PRIMARY_URL,
            json={
                "model": LLM_PRIMARY_MODEL,
                "messages": [{"role": "user", "content": check_prompt}],
                "max_tokens": 150,
                "temperature": 0.0,
            },
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
            timeout=30,
        )
        check_r.raise_for_status()
        check_raw = check_r.json()["choices"][0]["message"]["content"].strip()
        check_raw = check_raw.replace("```json", "").replace("```", "").strip()
        check_parsed = json.loads(check_raw)
        is_hallucination = check_parsed.get("is_hallucination", False)
        confidence = check_parsed.get("confidence", "MEDIUM")
    except Exception as e:
        logger.warning(f"  Hallucination check failed: {e}")
        confidence = "MEDIUM"

    elapsed = (time.time() - t0) * 1000
    logger.info(f"  Node 5 generator: model={model_used}, halluc={is_hallucination}, conf={confidence} ({elapsed:.0f}ms)")

    state["answer"] = answer
    state["confidence"] = confidence
    state["is_hallucination"] = is_hallucination
    state["sources"] = sources
    state["latency_ms"]["generator"] = round(elapsed, 1)
    return state


# ── Routing logic (retry loops) ──

def should_retry_after_reranker(state: RAGState) -> str:
    """After Node 4: if all LOW and retries remain → go back to Node 2."""
    if state["all_low"] and state["retry_count"] < state["max_retries"]:
        state["retry_count"] += 1
        logger.info(f"  RETRY (all LOW) — attempt {state['retry_count']}")
        return "retry_retriever"
    return "generator"


def should_retry_after_generator(state: RAGState) -> str:
    """After Node 5: if hallucination detected and retries remain → go back to Node 2."""
    if state["is_hallucination"] and state["retry_count"] < state["max_retries"]:
        state["retry_count"] += 1
        logger.info(f"  RETRY (hallucination) — attempt {state['retry_count']}")
        return "retry_retriever"
    return "end"


# ── Build the LangGraph ──

def build_rag_graph() -> StateGraph:
    """Construct the 5-node LangGraph RAG agent with retry loops."""

    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("query_classifier", node_query_classifier)
    graph.add_node("parallel_retriever", node_parallel_retriever)
    graph.add_node("result_merger", node_result_merger)
    graph.add_node("reranker", node_reranker)
    graph.add_node("generator", node_generator)

    # Set entry point
    graph.set_entry_point("query_classifier")

    # Edges: linear flow
    graph.add_edge("query_classifier", "parallel_retriever")
    graph.add_edge("parallel_retriever", "result_merger")
    graph.add_edge("result_merger", "reranker")

    # Conditional: after reranker → generator or retry
    graph.add_conditional_edges(
        "reranker",
        should_retry_after_reranker,
        {
            "generator": "generator",
            "retry_retriever": "parallel_retriever",
        },
    )

    # Conditional: after generator → end or retry
    graph.add_conditional_edges(
        "generator",
        should_retry_after_generator,
        {
            "end": END,
            "retry_retriever": "parallel_retriever",
        },
    )

    return graph


# ── Compiled agent ──

_compiled_graph = None


def get_rag_agent():
    global _compiled_graph
    if _compiled_graph is None:
        graph = build_rag_graph()
        _compiled_graph = graph.compile()
    return _compiled_graph


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  LAYER 6 — RAGResponse  (public API)                            ║
# ╚═══════════════════════════════════════════════════════════════════╝

def rag_query(query: str, max_retries: int = 1) -> RAGResponse:
    """
    Run the full RAG agent and return a RAGResponse.

    Args:
        query: User question
        max_retries: Max retry attempts for all-LOW or hallucination (default 1)

    Returns:
        RAGResponse with answer, confidence, sources, retry_count, latency_ms
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"RAG Query: {query}")
    logger.info(f"{'='*60}")

    agent = get_rag_agent()

    initial_state: RAGState = {
        "query": query,
        "intent": "",
        "collection_targets": [],
        "collection_weights": {},
        "raw_results": [],
        "merged_results": [],
        "reranked_results": [],
        "reranker_scores": {},
        "all_low": False,
        "answer": "",
        "confidence": "LOW",
        "is_hallucination": False,
        "retry_count": 0,
        "max_retries": max_retries,
        "latency_ms": {},
        "sources": [],
    }

    t0 = time.time()
    final_state = agent.invoke(initial_state)
    total_ms = (time.time() - t0) * 1000

    final_state["latency_ms"]["total"] = round(total_ms, 1)

    response = RAGResponse(
        answer=final_state.get("answer", ""),
        confidence=final_state.get("confidence", "LOW"),
        sources=final_state.get("sources", []),
        retry_count=final_state.get("retry_count", 0),
        latency_ms=final_state.get("latency_ms", {}),
    )

    logger.info(f"\nRAGResponse:")
    logger.info(f"  confidence  = {response.confidence}")
    logger.info(f"  retries     = {response.retry_count}")
    logger.info(f"  sources     = {len(response.sources)}")
    logger.info(f"  latency     = {response.latency_ms}")
    logger.info(f"  answer      = {response.answer[:200]}...")

    return response


# ╔═══════════════════════════════════════════════════════════════════╗
# ║  MAIN — CLI entry point                                         ║
# ╚═══════════════════════════════════════════════════════════════════╝

def main():
    import argparse

    parser = argparse.ArgumentParser(description="NV-Ingest 25.9.0 + LangGraph RAG Agent")
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # Ingest command
    ingest_p = sub.add_parser("ingest", help="Ingest documents into Milvus")
    ingest_p.add_argument("files", nargs="+", help="PDF/document file paths")
    ingest_p.add_argument("--output-dir", default=None, help="Save extracted artifacts to disk")

    # Query command
    query_p = sub.add_parser("query", help="Run RAG query")
    query_p.add_argument("question", help="Question to ask")
    query_p.add_argument("--max-retries", type=int, default=1, help="Max retry attempts")

    # Interactive mode
    sub.add_parser("interactive", help="Interactive Q&A loop")

    args = parser.parse_args()

    if args.command == "ingest":
        # Validate files
        for f in args.files:
            if not os.path.isfile(f):
                print(f"ERROR: File not found: {f}")
                sys.exit(1)

        counts = ingest_documents(args.files, output_dir=args.output_dir)
        print(f"\nIngestion complete!")
        print(f"  text chunks:  {counts['text']}")
        print(f"  table chunks: {counts['table']}")
        print(f"  image chunks: {counts['image']}")

    elif args.command == "query":
        ensure_collections()
        resp = rag_query(args.question, max_retries=args.max_retries)
        print(f"\n{'='*60}")
        print(f"Answer:     {resp.answer}")
        print(f"Confidence: {resp.confidence}")
        print(f"Retries:    {resp.retry_count}")
        print(f"Sources:    {len(resp.sources)}")
        print(f"Latency:    {resp.latency_ms}")
        print(f"{'='*60}")

    elif args.command == "interactive":
        ensure_collections()
        print("\nNV-Ingest + LangGraph RAG Agent")
        print("Type 'quit' to exit, 'ingest <path>' to add documents\n")
        while True:
            try:
                user_input = input("Q: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break
            if user_input.lower().startswith("ingest "):
                path = user_input[7:].strip()
                if os.path.isfile(path):
                    counts = ingest_documents([path])
                    print(f"  Ingested: text={counts['text']}, table={counts['table']}, image={counts['image']}")
                else:
                    print(f"  File not found: {path}")
                continue

            resp = rag_query(user_input)
            print(f"\nA: {resp.answer}")
            print(f"   [confidence={resp.confidence}, retries={resp.retry_count}, "
                  f"sources={len(resp.sources)}, total_ms={resp.latency_ms.get('total', '?')}]\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
