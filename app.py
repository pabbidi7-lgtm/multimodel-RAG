"""
Enterprise Multimodal RAG Pipeline with NV-Ingest 25.9.0
=========================================================
Architecture:
  Layer 1: NV-Ingest (extract + split + caption + embed + vdb_upload)
  Layer 2: Metadata Enrichment (bbox, content_type, nearby_objects)
  Layer 3: Retrieval + Reranking + Confidence Scoring
  Layer 4: LLM Generation with Fallback + Guardrails
  Layer 5: FastAPI (POST /ingest, POST /query, GET /health)

Models (all NVIDIA cloud NIMs — no GPU):
  - YOLOX page elements (table/chart/image detection with bbox)
  - PaddleOCR (text from detected elements)
  - nvidia/llama-3.1-nemotron-nano-vl-8b-v1 (image captioning)
  - nvidia/nv-embedqa-e5-v5 (embedding, 1024-dim)
  - nvidia/nv-rerankqa-mistral-4b-v3 (reranking)
  - meta/llama-3.3-70b-instruct (primary LLM)
  - nvidia/llama-3.1-nemotron-70b-instruct (fallback LLM)

Run:
  export NVIDIA_API_KEY=nvapi-xxxxx
  export NVIDIA_BUILD_API_KEY=nvapi-xxxxx
  taskset -c 0-7 python enterprise_rag.py
"""

import os
import json
import time
import socket
import logging
import hashlib
import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from contextlib import asynccontextmanager
from enum import Enum

import httpx
from pymilvus import MilvusClient
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

class Config:
    """Centralized configuration — all tunables in one place."""
    # API
    NVIDIA_API_KEY: str = os.environ.get("NVIDIA_API_KEY", "")
    NVIDIA_BUILD_API_KEY: str = os.environ.get("NVIDIA_BUILD_API_KEY", "")

    # Milvus
    MILVUS_URI: str = os.environ.get("MILVUS_URI", "./enterprise_rag.db")
    COLLECTION: str = os.environ.get("COLLECTION", "enterprise_docs")

    # Embedding
    EMBED_MODEL: str = "nvidia/nv-embedqa-e5-v5"
    EMBED_DIM: int = 1024
    EMBED_URL: str = "https://integrate.api.nvidia.com/v1/embeddings"
    EMBED_BATCH_SIZE: int = 50

    # Reranker
    RERANK_MODEL: str = "nvidia/nv-rerankqa-mistral-4b-v3"
    RERANK_URL: str = "https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-rerankqa-mistral-4b-v3/reranking"

    # LLMs
    PRIMARY_LLM: str = "meta/llama-3.3-70b-instruct"
    FALLBACK_LLM: str = "nvidia/llama-3.1-nemotron-70b-instruct"
    LLM_URL: str = "https://integrate.api.nvidia.com/v1/chat/completions"

    # Captioning VLM
    CAPTION_MODEL: str = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
    CAPTION_URL: str = "https://integrate.api.nvidia.com/v1/chat/completions"

    # NV-Ingest
    BROKER_HOST: str = "localhost"
    BROKER_PORT: int = 7671
    BROKER_TIMEOUT: int = 120
    VDB_DENSE_DIM: int = 2048  # NV-Ingest default embedder dimension

    # Retrieval
    INITIAL_TOP_K: int = 30   # broad vector search
    RERANK_TOP_K: int = 10    # after reranking
    MAX_CONTEXT_CHUNKS: int = 8

    # Confidence thresholds (reranker logit scores)
    HIGH_CONFIDENCE: float = 2.0
    MEDIUM_CONFIDENCE: float = 0.0

    # Guardrails
    MAX_QUERY_LENGTH: int = 2000
    BLOCKED_PATTERNS: List[str] = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"forget\s+(all\s+)?previous",
        r"you\s+are\s+now",
        r"system\s*prompt",
        r"<\s*script",
    ]

    # Uploads
    UPLOAD_DIR: str = "./uploads"
    SUPPORTED_EXTENSIONS: set = {".pdf", ".docx", ".pptx", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

    # Timeouts
    HTTP_TIMEOUT: int = 120
    RETRY_ATTEMPTS: int = 3
    RETRY_DELAY: float = 2.0


# ═══════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("enterprise_rag")


# ═══════════════════════════════════════════════════════════════════
#  DATA MODELS
# ═══════════════════════════════════════════════════════════════════

class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    CHART = "chart"
    IMAGE = "image"
    INFOGRAPHIC = "infographic"
    UNKNOWN = "unknown"

@dataclass
class EnrichedChunk:
    """Rich metadata container for each extracted chunk."""
    chunk_id: str = ""
    text: str = ""
    content_type: str = "text"
    page_number: int = -1
    source_file: str = ""
    bbox: List[float] = field(default_factory=list)        # [x1, y1, x2, y2]
    bbox_max_dims: List[float] = field(default_factory=list)  # [page_w, page_h]
    caption: str = ""
    nearby_text: List[str] = field(default_factory=list)
    nearby_tables: List[str] = field(default_factory=list)
    nearby_images: List[str] = field(default_factory=list)

class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class RetrievedChunk:
    """Chunk with retrieval + reranking scores."""
    text: str = ""
    content_type: str = "text"
    page_number: int = -1
    source_file: str = ""
    vector_score: float = 0.0
    rerank_score: float = 0.0
    confidence: str = "low"

@dataclass
class RAGResponse:
    """Complete RAG response with provenance."""
    answer: str = ""
    model_used: str = ""
    fallback_used: bool = False
    confidence: str = "low"
    sources: List[Dict] = field(default_factory=list)
    guardrail_flags: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str
    collection: Optional[str] = None
    top_k: Optional[int] = None
    content_type_filter: Optional[str] = None

class IngestRequest(BaseModel):
    collection: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════
#  SINGLETON NIM REGISTRY
# ═══════════════════════════════════════════════════════════════════

class NIMRegistry:
    """Lazy singleton — each NIM client initialized once, reused everywhere.
    Uses async httpx for non-blocking I/O. Thread-safe via GIL.
    """
    _instance: Optional["NIMRegistry"] = None
    _http: Optional[httpx.Client] = None
    _milvus: Optional[MilvusClient] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def http(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(
                timeout=Config.HTTP_TIMEOUT,
                headers={
                    "Authorization": f"Bearer {Config.NVIDIA_API_KEY}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
            logger.info("HTTP client initialized")
        return self._http

    @property
    def milvus(self) -> MilvusClient:
        if self._milvus is None:
            self._milvus = MilvusClient(uri=Config.MILVUS_URI)
            logger.info(f"Milvus client initialized: {Config.MILVUS_URI}")
        return self._milvus

    def _call_nim(self, url: str, payload: dict, description: str = "") -> dict:
        """Unified NIM API call with retry + exponential backoff."""
        for attempt in range(Config.RETRY_ATTEMPTS):
            try:
                r = self.http.post(url, json=payload)
                if r.status_code == 429:
                    wait = Config.RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"{description}: rate limited, waiting {wait}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"{description}: HTTP {e.response.status_code} on attempt {attempt+1}")
                if attempt == Config.RETRY_ATTEMPTS - 1:
                    raise
                time.sleep(Config.RETRY_DELAY * (2 ** attempt))
            except httpx.ConnectError as e:
                logger.error(f"{description}: connection error on attempt {attempt+1}: {e}")
                if attempt == Config.RETRY_ATTEMPTS - 1:
                    raise
                time.sleep(Config.RETRY_DELAY * (2 ** attempt))
        return {}

    # ── Embedding ──
    def embed(self, texts: List[str], input_type: str = "passage") -> List[List[float]]:
        """Batch embed texts via NVIDIA NIM. Handles batching internally."""
        all_embeddings = []
        for i in range(0, len(texts), Config.EMBED_BATCH_SIZE):
            batch = texts[i:i + Config.EMBED_BATCH_SIZE]
            # Skip empty texts
            batch = [t if t.strip() else "empty" for t in batch]
            data = self._call_nim(
                Config.EMBED_URL,
                {"model": Config.EMBED_MODEL, "input": batch, "input_type": input_type},
                description=f"embed batch {i//Config.EMBED_BATCH_SIZE + 1}"
            )
            all_embeddings.extend([e["embedding"] for e in data.get("data", [])])
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query (uses input_type='query' for asymmetric search)."""
        return self.embed([query], input_type="query")[0]

    # ── Reranker ──
    def rerank(self, query: str, passages: List[str]) -> List[Dict]:
        """Rerank passages using NVIDIA reranker NIM.
        Returns list of {index, logit} sorted by relevance descending.
        """
        if not passages:
            return []
        data = self._call_nim(
            Config.RERANK_URL,
            {
                "model": Config.RERANK_MODEL,
                "query": {"text": query},
                "passages": [{"text": p} for p in passages],
                "truncate": "END",
            },
            description="rerank"
        )
        return data.get("rankings", [])

    # ── LLM Generation ──
    def generate(self, prompt: str, model: str = None, max_tokens: int = 1024,
                 temperature: float = 0.7) -> Tuple[str, str]:
        """Generate text with primary LLM, fallback on failure.
        Returns (answer, model_used).
        """
        model = model or Config.PRIMARY_LLM
        models_to_try = [model]
        if model == Config.PRIMARY_LLM:
            models_to_try.append(Config.FALLBACK_LLM)

        for m in models_to_try:
            try:
                data = self._call_nim(
                    Config.LLM_URL,
                    {
                        "model": m,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    description=f"LLM:{m.split('/')[-1]}"
                )
                answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if answer.strip():
                    return answer, m
            except Exception as e:
                logger.warning(f"LLM {m} failed: {e}")
                continue

        return "Unable to generate answer. Both LLMs failed.", "none"

    def close(self):
        if self._http:
            self._http.close()


# ═══════════════════════════════════════════════════════════════════
#  GUARDRAILS
# ═══════════════════════════════════════════════════════════════════

class Guardrails:
    """Input validation + output grounding checks."""

    @staticmethod
    def check_input(query: str) -> List[str]:
        """Validate query. Returns list of flags (empty = clean)."""
        flags = []
        if len(query) > Config.MAX_QUERY_LENGTH:
            flags.append("query_too_long")
        if not query.strip():
            flags.append("empty_query")
        for pattern in Config.BLOCKED_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                flags.append("prompt_injection_detected")
                break
        return flags

    @staticmethod
    def check_output(answer: str, context_texts: List[str]) -> List[str]:
        """Check if LLM answer is grounded in provided context."""
        flags = []
        hallucination_phrases = [
            "based on my knowledge",
            "as an ai",
            "i don't have access to",
            "in general",
            "based on my training",
            "i cannot access",
        ]
        answer_lower = answer.lower()
        for phrase in hallucination_phrases:
            if phrase in answer_lower:
                flags.append("possible_hallucination")
                break

        # Check if answer mentions something not in any context chunk
        if not context_texts:
            flags.append("no_context_provided")
        elif "not in the context" in answer_lower or "cannot find" in answer_lower:
            flags.append("answer_not_grounded")

        return flags


# ═══════════════════════════════════════════════════════════════════
#  METADATA ENRICHER
# ═══════════════════════════════════════════════════════════════════

class MetadataEnricher:
    """Parses NV-Ingest raw JSON results into EnrichedChunks.
    Preserves bbox, content_type, nearby_objects, captions.
    """

    @staticmethod
    def enrich_results(raw_results: list, source_file: str = "") -> List[EnrichedChunk]:
        """Parse NV-Ingest extraction results into enriched chunks."""
        chunks = []
        for item in raw_results:
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except (json.JSONDecodeError, TypeError):
                    continue

            metadata = item if isinstance(item, dict) else {}
            if "metadata" in metadata:
                metadata = metadata["metadata"]

            content = metadata.get("content", "")
            if not content or not content.strip():
                continue

            # Content metadata
            cm = metadata.get("content_metadata", {})
            content_type = cm.get("type", "text")
            subtype = cm.get("subtype", "")
            page_number = cm.get("page_number", -1)

            # Map to our enum
            if content_type == "structured":
                if subtype == "chart":
                    ct = ContentType.CHART
                else:
                    ct = ContentType.TABLE
            elif content_type == "image":
                ct = ContentType.IMAGE
            else:
                ct = ContentType.TEXT

            # Hierarchy / nearby objects
            hierarchy = cm.get("hierarchy", {})
            nearby = hierarchy.get("nearby_objects", {})
            nearby_text = nearby.get("text", {}).get("content", [])
            nearby_tables = nearby.get("structured", {}).get("content", [])
            nearby_images = nearby.get("images", {}).get("content", [])

            # BBox from image_metadata or table_metadata
            bbox = []
            bbox_max = []
            img_meta = metadata.get("image_metadata")
            if img_meta:
                bbox = img_meta.get("image_location", [])
                bbox_max = img_meta.get("image_location_max_dimensions", [])

            # Caption
            caption = ""
            if img_meta:
                caption = img_meta.get("caption", "")
            table_meta = metadata.get("table_metadata")
            if table_meta:
                caption = table_meta.get("caption", caption)

            # Source
            src = metadata.get("source_metadata", {})
            src_name = src.get("source_name", source_file)

            # Chunk ID
            chunk_id = hashlib.md5(
                f"{src_name}:{page_number}:{content[:50]}".encode()
            ).hexdigest()[:12]

            chunks.append(EnrichedChunk(
                chunk_id=chunk_id,
                text=content,
                content_type=ct.value,
                page_number=page_number,
                source_file=os.path.basename(src_name),
                bbox=bbox,
                bbox_max_dims=bbox_max,
                caption=caption,
                nearby_text=nearby_text[:3],   # limit stored nearby
                nearby_tables=nearby_tables[:3],
                nearby_images=nearby_images[:3],
            ))

        logger.info(f"Enriched {len(chunks)} chunks from {source_file}")
        return chunks


# ═══════════════════════════════════════════════════════════════════
#  NV-INGEST ENGINE
# ═══════════════════════════════════════════════════════════════════

class NVIngestEngine:
    """Manages the NV-Ingest Ray pipeline lifecycle.
    Wraps run_pipeline + broker + Ingestor chain.
    """
    _pipeline_started: bool = False

    @staticmethod
    def wait_for_broker(host: str = None, port: int = None, timeout: int = None):
        host = host or Config.BROKER_HOST
        port = port or Config.BROKER_PORT
        timeout = timeout or Config.BROKER_TIMEOUT
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
            time.sleep(0.5)
        raise RuntimeError(f"Broker timeout after {timeout}s")

    @classmethod
    def start_pipeline(cls):
        """Start NV-Ingest Ray pipeline (idempotent)."""
        if cls._pipeline_started:
            logger.info("Pipeline already running")
            return

        from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import (
            run_pipeline, PipelineCreationSchema,
        )
        cfg = PipelineCreationSchema()
        run_pipeline(cfg, block=False, disable_dynamic_scaling=True, run_in_subprocess=True)
        cls._pipeline_started = True
        logger.info("NV-Ingest pipeline started")
        cls.wait_for_broker()

    @classmethod
    def ingest(cls, file_paths: List[str], collection: str = None) -> List[EnrichedChunk]:
        """Full NV-Ingest pipeline: extract → split → caption → embed → vdb_upload.
        Returns enriched chunks with rich metadata.
        """
        collection = collection or Config.COLLECTION

        from nv_ingest_client.client import Ingestor, NvIngestClient
        from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
        from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

        cls.start_pipeline()

        client = NvIngestClient(
            message_client_allocator=SimpleClient,
            message_client_port=Config.BROKER_PORT,
            message_client_hostname=Config.BROKER_HOST,
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
                endpoint_url=Config.CAPTION_URL,
                model_name=Config.CAPTION_MODEL,
                api_key=Config.NVIDIA_API_KEY,
            )
            .embed()
            .vdb_upload(
                collection_name=collection,
                milvus_uri=Config.MILVUS_URI,
                sparse=False,
                dense_dim=Config.VDB_DENSE_DIM,
            )
        )

        logger.info(f"Starting ingestion: {file_paths}")
        t0 = time.time()
        results, failures = ingestor.ingest(show_progress=True, return_failures=True)
        elapsed = time.time() - t0
        logger.info(f"Ingestion complete in {elapsed:.1f}s | results={len(results)} failures={len(failures)}")

        if failures:
            for i, f in enumerate(failures[:3]):
                logger.warning(f"Failure [{i}]: {str(f)[:200]}")

        # Enrich results with metadata
        all_chunks = []
        for doc_result in results:
            try:
                blob = ingest_json_results_to_blob(doc_result)
                # The blob is text, but we need structured data
                # Parse the raw result JSON instead
                if isinstance(doc_result, list):
                    for item in doc_result:
                        chunks = MetadataEnricher.enrich_results([item], file_paths[0])
                        all_chunks.extend(chunks)
                elif isinstance(doc_result, dict):
                    chunks = MetadataEnricher.enrich_results([doc_result], file_paths[0])
                    all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Enrichment error: {e}")

        logger.info(f"Total enriched chunks: {len(all_chunks)}")
        return all_chunks


# ═══════════════════════════════════════════════════════════════════
#  RETRIEVAL ENGINE
# ═══════════════════════════════════════════════════════════════════

class RetrievalEngine:
    """Dense search → Rerank → Confidence scoring."""

    def __init__(self, registry: NIMRegistry):
        self.reg = registry

    def _classify_confidence(self, logit: float) -> str:
        if logit >= Config.HIGH_CONFIDENCE:
            return ConfidenceLevel.HIGH
        elif logit >= Config.MEDIUM_CONFIDENCE:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    def retrieve(self, query: str, collection: str = None,
                 top_k: int = None, content_type_filter: str = None) -> List[RetrievedChunk]:
        """Full retrieval pipeline: embed → search → rerank → score."""
        collection = collection or Config.COLLECTION
        initial_k = top_k or Config.INITIAL_TOP_K

        # 1. Embed query
        q_emb = self.reg.embed_query(query)

        # 2. Dense vector search in Milvus
        try:
            search_results = self.reg.milvus.search(
                collection_name=collection,
                data=[q_emb],
                limit=initial_k,
                output_fields=["text"],
            )[0]
        except Exception as e:
            logger.error(f"Milvus search failed: {e}")
            return []

        if not search_results:
            logger.info("No search results from Milvus")
            return []

        # 3. Extract texts and scores
        candidates = []
        for hit in search_results:
            entity = hit["entity"] if isinstance(hit, dict) else hit.entity
            text = entity.get("text", "") if isinstance(entity, dict) else getattr(entity, "text", "")
            score = hit.get("distance", 0.0) if isinstance(hit, dict) else getattr(hit, "distance", 0.0)
            if text and text.strip():
                candidates.append({
                    "text": text,
                    "vector_score": score,
                    "content_type": "text",  # Milvus may not store this field
                    "page_number": -1,
                    "source_file": "",
                })

        if not candidates:
            return []

        # 4. Rerank with NVIDIA reranker
        rerank_k = Config.RERANK_TOP_K
        try:
            passages = [c["text"] for c in candidates]
            rankings = self.reg.rerank(query, passages)

            reranked = []
            for rank in rankings[:rerank_k]:
                idx = rank.get("index", 0)
                logit = rank.get("logit", 0.0)
                c = candidates[idx]
                confidence = self._classify_confidence(logit)
                reranked.append(RetrievedChunk(
                    text=c["text"],
                    content_type=c["content_type"],
                    page_number=c["page_number"],
                    source_file=c["source_file"],
                    vector_score=c["vector_score"],
                    rerank_score=logit,
                    confidence=confidence,
                ))
            return reranked

        except Exception as e:
            logger.warning(f"Reranker failed ({e}), falling back to vector scores only")
            # Fallback: return top results by vector score without reranking
            return [
                RetrievedChunk(
                    text=c["text"],
                    content_type=c["content_type"],
                    page_number=c["page_number"],
                    source_file=c["source_file"],
                    vector_score=c["vector_score"],
                    rerank_score=0.0,
                    confidence="low",
                )
                for c in candidates[:rerank_k]
            ]


# ═══════════════════════════════════════════════════════════════════
#  RAG ENGINE
# ═══════════════════════════════════════════════════════════════════

class RAGEngine:
    """Orchestrates retrieval → guardrails → generation → output validation."""

    SYSTEM_PROMPT = """You are a precise document analysis assistant. Answer questions using ONLY the provided context.

Rules:
- If the answer is in the context, provide it with specific details (numbers, names, dates).
- If the answer is NOT in the context, say "The provided documents do not contain this information."
- Never invent or assume information not present in the context.
- When citing figures, always specify the exact values from the context.
- If multiple chunks are relevant, synthesize information across them."""

    def __init__(self, registry: NIMRegistry, retrieval: RetrievalEngine):
        self.reg = registry
        self.retrieval = retrieval

    def query(self, query: str, collection: str = None,
              top_k: int = None, content_type_filter: str = None) -> RAGResponse:
        """Full RAG pipeline with guardrails."""
        t0 = time.time()

        # ── Input guardrails ──
        input_flags = Guardrails.check_input(query)
        if "prompt_injection_detected" in input_flags:
            return RAGResponse(
                answer="This query has been flagged and cannot be processed.",
                guardrail_flags=input_flags,
                latency_ms=(time.time() - t0) * 1000,
            )
        if "empty_query" in input_flags:
            return RAGResponse(
                answer="Please provide a query.",
                guardrail_flags=input_flags,
                latency_ms=(time.time() - t0) * 1000,
            )

        # ── Retrieve ──
        chunks = self.retrieval.retrieve(
            query, collection=collection,
            top_k=top_k, content_type_filter=content_type_filter
        )
        if not chunks:
            return RAGResponse(
                answer="No relevant documents found for this query.",
                confidence="low",
                latency_ms=(time.time() - t0) * 1000,
            )

        # ── Build context ──
        context_chunks = chunks[:Config.MAX_CONTEXT_CHUNKS]
        context_parts = []
        sources = []
        for i, c in enumerate(context_chunks):
            header = f"[Chunk {i+1} | type={c.content_type} | relevance={c.confidence}]"
            context_parts.append(f"{header}\n{c.text}")
            sources.append({
                "chunk_index": i + 1,
                "content_type": c.content_type,
                "page_number": c.page_number,
                "source_file": c.source_file,
                "rerank_score": round(c.rerank_score, 4),
                "confidence": c.confidence,
            })

        context = "\n\n---\n\n".join(context_parts)

        # ── Generate ──
        prompt = f"""{self.SYSTEM_PROMPT}

Context:
{context}

Question: {query}
Answer:"""

        answer, model_used = self.reg.generate(prompt)
        fallback_used = model_used != Config.PRIMARY_LLM

        # ── Output guardrails ──
        output_flags = Guardrails.check_output(answer, [c.text for c in context_chunks])

        # ── Overall confidence ──
        high_count = sum(1 for c in context_chunks if c.confidence == "high")
        if high_count >= 2:
            overall = "high"
        elif high_count >= 1 or any(c.confidence == "medium" for c in context_chunks):
            overall = "medium"
        else:
            overall = "low"

        all_flags = input_flags + output_flags

        elapsed = (time.time() - t0) * 1000
        logger.info(
            f"RAG query completed | model={model_used.split('/')[-1]} | "
            f"chunks={len(context_chunks)} | confidence={overall} | "
            f"flags={all_flags} | {elapsed:.0f}ms"
        )

        return RAGResponse(
            answer=answer,
            model_used=model_used,
            fallback_used=fallback_used,
            confidence=overall,
            sources=sources,
            guardrail_flags=all_flags,
            latency_ms=round(elapsed, 1),
        )


# ═══════════════════════════════════════════════════════════════════
#  MILVUS COLLECTION MANAGER
# ═══════════════════════════════════════════════════════════════════

class CollectionManager:
    """Manages Milvus collections — create, list, delete."""

    def __init__(self, registry: NIMRegistry):
        self.reg = registry

    def ensure_collection(self, name: str = None, dim: int = None):
        name = name or Config.COLLECTION
        dim = dim or Config.EMBED_DIM
        if not self.reg.milvus.has_collection(name):
            self.reg.milvus.create_collection(
                collection_name=name,
                dimension=dim,
                metric_type="L2",
                auto_id=True,
            )
            logger.info(f"Created collection: {name}")
        else:
            logger.info(f"Collection exists: {name}")

    def list_collections(self) -> List[str]:
        return self.reg.milvus.list_collections()

    def collection_stats(self, name: str = None) -> dict:
        name = name or Config.COLLECTION
        try:
            stats = self.reg.milvus.get_collection_stats(name)
            return {"collection": name, "stats": stats}
        except Exception as e:
            return {"collection": name, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
#  FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════

registry = NIMRegistry()
retrieval_engine = RetrievalEngine(registry)
rag_engine = RAGEngine(registry, retrieval_engine)
collection_manager = CollectionManager(registry)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("Starting Enterprise RAG API")
    os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
    yield
    logger.info("Shutting down")
    registry.close()

app = FastAPI(
    title="Enterprise Multimodal RAG API",
    description="NV-Ingest + Reranker + LLM Fallback + Guardrails",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """System health check."""
    return {
        "status": "ok",
        "milvus": Config.MILVUS_URI,
        "collections": collection_manager.list_collections(),
        "models": {
            "embedder": Config.EMBED_MODEL,
            "reranker": Config.RERANK_MODEL,
            "primary_llm": Config.PRIMARY_LLM,
            "fallback_llm": Config.FALLBACK_LLM,
        }
    }


@app.post("/ingest")
async def ingest_endpoint(files: List[UploadFile] = File(...), collection: str = None):
    """Upload and ingest documents via NV-Ingest pipeline."""
    collection = collection or Config.COLLECTION
    saved_paths = []

    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in Config.SUPPORTED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type: {ext}")
        path = os.path.join(Config.UPLOAD_DIR, f.filename)
        content = await f.read()
        with open(path, "wb") as fh:
            fh.write(content)
        saved_paths.append(path)
        logger.info(f"Saved upload: {path} ({len(content)} bytes)")

    try:
        chunks = NVIngestEngine.ingest(saved_paths, collection=collection)
        return {
            "status": "success",
            "files_ingested": len(saved_paths),
            "chunks_enriched": len(chunks),
            "collection": collection,
        }
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(500, f"Ingestion failed: {str(e)}")


@app.post("/query")
def query_endpoint(req: QueryRequest):
    """RAG query with retrieval, reranking, guardrails, and LLM generation."""
    response = rag_engine.query(
        query=req.query,
        collection=req.collection,
        top_k=req.top_k,
        content_type_filter=req.content_type_filter,
    )
    return asdict(response)


@app.get("/collections")
def list_collections():
    """List all collections and stats."""
    names = collection_manager.list_collections()
    results = []
    for name in names:
        results.append(collection_manager.collection_stats(name))
    return {"collections": results}


# ═══════════════════════════════════════════════════════════════════
#  CLI MODE (for testing without FastAPI)
# ═══════════════════════════════════════════════════════════════════

def cli_mode():
    """Run pipeline from command line for testing."""
    import sys

    if not Config.NVIDIA_API_KEY:
        print("ERROR: Set NVIDIA_API_KEY environment variable")
        sys.exit(1)

    print("=" * 60)
    print("Enterprise Multimodal RAG Pipeline")
    print("=" * 60)

    # Check for PDF argument
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "./Docs/invoice-0-4.pdf"

    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        print("Usage: python enterprise_rag.py <path_to_pdf>")
        sys.exit(1)

    # Ingest
    print(f"\n[1/3] Ingesting: {pdf_path}")
    chunks = NVIngestEngine.ingest([pdf_path])
    print(f"      Enriched chunks: {len(chunks)}")
    for c in chunks[:3]:
        print(f"      - [{c.content_type}] page={c.page_number} len={len(c.text)} bbox={c.bbox[:4] if c.bbox else 'none'}")

    # Initialize Milvus client lazily (after ingestion to avoid gRPC conflict)
    print(f"\n[2/3] Milvus collection: {Config.COLLECTION}")
    stats = collection_manager.collection_stats()
    print(f"      Stats: {stats}")

    # Query
    queries = [
        "What is the total amount due?",
        "Who is the salesperson?",
        "List all items with quantity 25 or more",
    ]

    print(f"\n[3/3] Running {len(queries)} queries with reranking + guardrails")
    print("=" * 60)

    for q in queries:
        response = rag_engine.query(q)
        print(f"\nQ: {q}")
        print(f"A: {response.answer[:500]}")
        print(f"   Model: {response.model_used} | Confidence: {response.confidence} | "
              f"Fallback: {response.fallback_used} | Latency: {response.latency_ms}ms")
        if response.guardrail_flags:
            print(f"   Flags: {response.guardrail_flags}")
        print(f"   Sources: {len(response.sources)} chunks used")
        print("-" * 60)


if __name__ == "__main__":
    cli_mode()
