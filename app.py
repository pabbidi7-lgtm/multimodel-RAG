"""
Enterprise Multimodal RAG Pipeline — Best Models 2025
=======================================================
All models sourced from build.nvidia.com (March 2026)

Model Registry:
  Extraction (NV-Ingest 25.9.0 via NVIDIA API):
    - nemoretriever-page-elements-v2     : YOLOX page element detection
    - nemoretriever-graphic-elements-v1  : YOLOX chart/graphic detection  
    - nemoretriever-table-structure-v1   : YOLOX table structure (rows/cols/cells)
    - nemoretriever-ocr-v1               : Best OCR (upgraded from PaddleOCR)
    - llama-3.1-nemotron-nano-vl-8b-v1   : Best VLM for image captioning

  Retrieval:
    - nvidia/llama-3.2-nv-embedqa-1b-v2  : Best embedding (2048-dim, multilingual, 8192 tokens)
    - nvidia/llama-3.2-nv-rerankqa-1b-v2 : Best reranker (pairs with above embedding)

  Generation:
    - meta/llama-3.3-70b-instruct        : Primary LLM
    - nvidia/llama-3.1-nemotron-70b-instruct : Fallback LLM
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import socket
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

# ═══════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("enterprise_rag")


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — ALL MODELS UPDATED TO BEST 2025 VERSIONS
# ═══════════════════════════════════════════════════════════════════

class Config:

    NVIDIA_API_KEY: str = os.environ.get("NVIDIA_API_KEY", "")

    # ── Milvus ──
    MILVUS_URI: str = os.environ.get("MILVUS_URI", "./enterprise_rag.db")
    COLLECTION: str = os.environ.get("COLLECTION", "enterprise_docs")

    # ── Embedding — UPGRADED: llama-3.2-nv-embedqa-1b-v2 (2048-dim) ──
    # Reason: multilingual, 8192 token context, dynamic embedding size
    EMBED_MODEL: str = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    EMBED_DIM: int = 2048                    # ← changed from 1024
    EMBED_URL: str = "https://integrate.api.nvidia.com/v1/embeddings"
    EMBED_BATCH_SIZE: int = 32

    # ── Reranker — UPGRADED: llama-3.2-nv-rerankqa-1b-v2 ──
    # Reason: pairs perfectly with llama-3.2 embedder, multilingual, 3.5x faster
    RERANK_MODEL: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
    RERANK_URL: str = (
        "https://ai.api.nvidia.com/v1/retrieval/nvidia/"
        "llama-3.2-nv-rerankqa-1b-v2/reranking"
    )

    # ── LLMs — unchanged, still best ──
    PRIMARY_LLM: str = "meta/llama-3.3-70b-instruct"
    FALLBACK_LLM: str = "nvidia/llama-3.1-nemotron-70b-instruct"
    LLM_URL: str = "https://integrate.api.nvidia.com/v1/chat/completions"

    # ── VLM captioning — unchanged, still best ──
    CAPTION_MODEL: str = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
    CAPTION_URL: str = "https://integrate.api.nvidia.com/v1/chat/completions"
    CAPTION_MIN_WORDS: int = 10

    # ── NV-Ingest extraction NIMs — best 2025 endpoints ──
    # Page elements detection (YOLOX) — latest v2
    YOLOX_PAGE_ENDPOINT: str = (
        "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v2"
    )
    # Graphic/chart element detection
    YOLOX_GRAPHIC_ENDPOINT: str = (
        "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-graphic-elements-v1"
    )
    # Table structure (rows, columns, cells)
    YOLOX_TABLE_ENDPOINT: str = (
        "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-table-structure-v1"
    )
    # OCR — UPGRADED: nemoretriever-ocr-v1 (replaces PaddleOCR)
    OCR_ENDPOINT: str = (
        "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr-v1"
    )
    # NeMoRetriever Parse — advanced VLM-based layout + table OCR
    PARSE_ENDPOINT: str = "https://integrate.api.nvidia.com/v1/chat/completions"

    # ── NV-Ingest broker ──
    BROKER_HOST: str = "localhost"
    BROKER_PORT: int = 7671
    BROKER_TIMEOUT: int = 120

    # ── Layout chunker ──
    LAYOUT_MAX_CHARS: int = 1500
    LAYOUT_OVERLAP_CHARS: int = 150

    # ── Table chunker ──
    TABLE_STORE_ROWS: bool = True

    # ── Hybrid search ──
    DENSE_TOP_K: int = 30
    SPARSE_TOP_K: int = 30
    RERANK_TOP_K: int = 10
    MAX_CONTEXT_CHUNKS: int = 8
    RRF_K: int = 60

    # ── Confidence thresholds ──
    HIGH_CONFIDENCE: float = 2.0
    MEDIUM_CONFIDENCE: float = 0.0

    # ── Guardrails ──
    MAX_QUERY_LENGTH: int = 2000
    BLOCKED_PATTERNS: List[str] = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"forget\s+(all\s+)?previous",
        r"you\s+are\s+now",
        r"system\s*prompt",
        r"<\s*script",
        r"jailbreak",
    ]

    # ── Uploads ──
    UPLOAD_DIR: str = "./uploads"
    SUPPORTED_EXTENSIONS: set = {
        ".pdf", ".docx", ".pptx",
        ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
    }

    # ── HTTP ──
    HTTP_TIMEOUT: int = 120
    RETRY_ATTEMPTS: int = 3
    RETRY_DELAY: float = 2.0


# ═══════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════

class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    TABLE_ROW = "table_row"
    CHART = "chart"
    IMAGE = "image"
    INFOGRAPHIC = "infographic"
    UNKNOWN = "unknown"


@dataclass
class EnrichedChunk:
    chunk_id: str = ""
    text: str = ""
    content_type: str = ContentType.TEXT
    page_number: int = -1
    source_file: str = ""
    bbox: List[float] = field(default_factory=list)
    bbox_max_dims: List[float] = field(default_factory=list)
    caption: str = ""
    image_b64: str = ""
    nearby_text: List[str] = field(default_factory=list)
    nearby_tables: List[str] = field(default_factory=list)
    table_headers: List[str] = field(default_factory=list)
    row_index: int = -1


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class RetrievedChunk:
    text: str = ""
    content_type: str = "text"
    page_number: int = -1
    source_file: str = ""
    caption: str = ""
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    confidence: str = "low"


@dataclass
class RAGResponse:
    answer: str = ""
    model_used: str = ""
    fallback_used: bool = False
    confidence: str = "low"
    sources: List[Dict] = field(default_factory=list)
    guardrail_flags: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    retrieval_stats: Dict = field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str
    collection: Optional[str] = None
    top_k: Optional[int] = None
    content_type_filter: Optional[str] = None
    page_filter: Optional[int] = None
    source_filter: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════
# NIM REGISTRY
# ═══════════════════════════════════════════════════════════════════

class NIMRegistry:

    _instance: Optional[NIMRegistry] = None
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
            logger.info(f"Milvus client connected: {Config.MILVUS_URI}")
        return self._milvus

    def _call_nim(self, url: str, payload: dict, description: str = "") -> dict:
        for attempt in range(Config.RETRY_ATTEMPTS):
            try:
                r = self.http.post(url, json=payload)
                if r.status_code == 429:
                    wait = Config.RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"{description}: rate-limited, waiting {wait:.1f}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as exc:
                logger.error(f"{description}: HTTP {exc.response.status_code} attempt {attempt+1}")
                if attempt == Config.RETRY_ATTEMPTS - 1:
                    raise
                time.sleep(Config.RETRY_DELAY * (2 ** attempt))
            except httpx.ConnectError as exc:
                logger.error(f"{description}: connect error attempt {attempt+1}: {exc}")
                if attempt == Config.RETRY_ATTEMPTS - 1:
                    raise
                time.sleep(Config.RETRY_DELAY * (2 ** attempt))
        return {}

    # ── Embedding — llama-3.2-nv-embedqa-1b-v2 (2048-dim) ──────────

    def embed(self, texts: List[str], input_type: str = "passage") -> List[List[float]]:
        results: List[List[float]] = []
        for i in range(0, len(texts), Config.EMBED_BATCH_SIZE):
            batch = texts[i : i + Config.EMBED_BATCH_SIZE]
            batch = [t.strip() if t.strip() else "<empty>" for t in batch]
            data = self._call_nim(
                Config.EMBED_URL,
                {
                    "model": Config.EMBED_MODEL,
                    "input": batch,
                    "input_type": input_type,
                    "encoding_format": "float",
                    "truncate": "END",    # llama-3.2 supports up to 8192 tokens
                },
                description=f"embed[{input_type}] batch {i // Config.EMBED_BATCH_SIZE + 1}",
            )
            results.extend(item["embedding"] for item in data.get("data", []))
        return results

    def embed_query(self, query: str) -> List[float]:
        return self.embed([query], input_type="query")[0]

    # ── VLM Captioning — llama-3.1-nemotron-nano-vl-8b-v1 ──────────

    def caption_image(self, image_b64: str, context_hint: str = "") -> str:
        system = (
            "You are a precise document image analyst. "
            "Describe the image content in detail: identify charts (axes, legends, values), "
            "tables (headers, data), diagrams (components, labels), or photographs. "
            "If it is a vector diagram, list all text labels and structural components. "
            "Be specific and factual."
        )
        user_parts: List[dict] = []
        if context_hint:
            user_parts.append({"type": "text", "text": f"Context: {context_hint}\n"})
        user_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
        })
        user_parts.append({"type": "text", "text": "Describe this image in detail."})

        try:
            data = self._call_nim(
                Config.CAPTION_URL,
                {
                    "model": Config.CAPTION_MODEL,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_parts},
                    ],
                    "max_tokens": 400,
                    "temperature": 0.2,
                },
                description="VLM caption",
            )
            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
        except Exception as exc:
            logger.warning(f"VLM captioning failed: {exc}")
            return ""

    # ── Reranker — llama-3.2-nv-rerankqa-1b-v2 ─────────────────────

    def rerank(self, query: str, passages: List[str]) -> List[Dict]:
        if not passages:
            return []
        data = self._call_nim(
            Config.RERANK_URL,
            {
                "model": Config.RERANK_MODEL,
                "query": {"text": query},
                "passages": [{"text": p[:2000]} for p in passages],
                "truncate": "END",
            },
            description="rerank",
        )
        return data.get("rankings", [])

    # ── LLM Generation — llama-3.3-70b-instruct ─────────────────────

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> Tuple[str, str]:
        model = model or Config.PRIMARY_LLM
        candidates = [model]
        if model == Config.PRIMARY_LLM:
            candidates.append(Config.FALLBACK_LLM)

        for m in candidates:
            try:
                data = self._call_nim(
                    Config.LLM_URL,
                    {
                        "model": m,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    description=f"LLM:{m.split('/')[-1]}",
                )
                answer = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                if answer:
                    return answer, m
            except Exception as exc:
                logger.warning(f"LLM {m} failed: {exc}")

        return "Unable to generate answer. Both LLMs failed.", "none"

    def close(self) -> None:
        if self._http:
            self._http.close()


# ═══════════════════════════════════════════════════════════════════
# GUARDRAILS
# ═══════════════════════════════════════════════════════════════════

class Guardrails:
    HALLUCINATION_PHRASES = [
        "based on my knowledge", "as an ai", "i don't have access to",
        "in general", "based on my training", "i cannot access", "my knowledge cutoff",
    ]

    @staticmethod
    def check_input(query: str) -> List[str]:
        flags: List[str] = []
        if not query.strip():
            flags.append("empty_query")
            return flags
        if len(query) > Config.MAX_QUERY_LENGTH:
            flags.append("query_too_long")
        for pattern in Config.BLOCKED_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                flags.append("prompt_injection_detected")
                break
        return flags

    @classmethod
    def check_output(cls, answer: str, context_texts: List[str]) -> List[str]:
        flags: List[str] = []
        a = answer.lower()
        for phrase in cls.HALLUCINATION_PHRASES:
            if phrase in a:
                flags.append("possible_hallucination")
                break
        if not context_texts:
            flags.append("no_context_provided")
        elif "not in the context" in a or "cannot find" in a:
            flags.append("answer_not_grounded")
        return flags


# ═══════════════════════════════════════════════════════════════════
# PHASE 2A — LAYOUT CHUNKER
# ═══════════════════════════════════════════════════════════════════

class LayoutChunker:

    @staticmethod
    def chunk(elements: List[Dict], source_file: str) -> List[EnrichedChunk]:
        def sort_key(e: Dict) -> Tuple:
            cm = e.get("content_metadata", {})
            pg = cm.get("page_number", 0)
            bbox = e.get("image_metadata", {}).get("image_location", [0, 0, 0, 0])
            y1 = bbox[1] if len(bbox) >= 2 else 0
            return (pg, y1)

        elements = sorted(elements, key=sort_key)
        pages: Dict[int, List[str]] = defaultdict(list)
        page_meta: Dict[int, Dict] = {}

        for e in elements:
            cm = e.get("content_metadata", {})
            pg = cm.get("page_number", -1)
            text = e.get("content", "").strip()
            if text:
                pages[pg].append(text)
                if pg not in page_meta:
                    page_meta[pg] = e

        chunks: List[EnrichedChunk] = []
        for pg, paragraphs in sorted(pages.items()):
            full_text = "\n\n".join(paragraphs)
            sub_chunks = LayoutChunker._split_text(
                full_text, Config.LAYOUT_MAX_CHARS, Config.LAYOUT_OVERLAP_CHARS
            )
            meta = page_meta.get(pg, {})
            src = meta.get("source_metadata", {})
            src_name = src.get("source_name", source_file)
            bbox = meta.get("image_metadata", {}).get("image_location", [])

            for idx, text in enumerate(sub_chunks):
                cid = hashlib.md5(
                    f"{src_name}:{pg}:text:{idx}:{text[:40]}".encode()
                ).hexdigest()[:12]
                chunks.append(EnrichedChunk(
                    chunk_id=cid, text=text, content_type=ContentType.TEXT,
                    page_number=pg, source_file=os.path.basename(src_name), bbox=bbox,
                ))

        logger.info(f"LayoutChunker produced {len(chunks)} text chunks")
        return chunks

    @staticmethod
    def _split_text(text: str, max_chars: int, overlap: int) -> List[str]:
        if len(text) <= max_chars:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chars
            if end >= len(text):
                chunks.append(text[start:])
                break
            split_at = text.rfind(". ", start, end)
            if split_at == -1 or split_at <= start:
                split_at = text.rfind("\n", start, end)
            if split_at == -1 or split_at <= start:
                split_at = end
            else:
                split_at += 1
            chunks.append(text[start:split_at].strip())
            start = max(start + 1, split_at - overlap)
        return [c for c in chunks if c.strip()]


# ═══════════════════════════════════════════════════════════════════
# PHASE 2B — TABLE CHUNKER
# ═══════════════════════════════════════════════════════════════════

class TableChunker:

    @staticmethod
    def chunk(elements: List[Dict], source_file: str) -> List[EnrichedChunk]:
        chunks: List[EnrichedChunk] = []
        for e in elements:
            cm = e.get("content_metadata", {})
            pg = cm.get("page_number", -1)
            content = e.get("content", "").strip()
            if not content:
                continue

            src = e.get("source_metadata", {})
            src_name = src.get("source_name", source_file)
            table_meta = e.get("table_metadata", {})
            caption = table_meta.get("caption", "")
            img_meta = e.get("image_metadata", {})
            bbox = img_meta.get("image_location", []) if img_meta else []
            bbox_max = img_meta.get("image_location_max_dimensions", []) if img_meta else []
            hierarchy = cm.get("hierarchy", {})
            nearby = hierarchy.get("nearby_objects", {})
            nearby_text = nearby.get("text", {}).get("content", [])

            searchable = f"[Table Caption: {caption}]\n{content}" if caption else content
            cid = hashlib.md5(f"{src_name}:{pg}:table:{content[:40]}".encode()).hexdigest()[:12]
            chunks.append(EnrichedChunk(
                chunk_id=cid, text=searchable, content_type=ContentType.TABLE,
                page_number=pg, source_file=os.path.basename(src_name),
                bbox=bbox, bbox_max_dims=bbox_max, caption=caption, nearby_text=nearby_text[:3],
            ))

            if Config.TABLE_STORE_ROWS and content.startswith("|"):
                chunks.extend(TableChunker._parse_markdown_rows(
                    content, pg, src_name, bbox, caption, nearby_text
                ))

        logger.info(f"TableChunker produced {len(chunks)} table chunks")
        return chunks

    @staticmethod
    def _parse_markdown_rows(
        markdown: str, page: int, src_name: str,
        bbox: List[float], caption: str, nearby_text: List[str],
    ) -> List[EnrichedChunk]:
        lines = [l.strip() for l in markdown.splitlines() if l.strip()]
        if len(lines) < 3:
            return []
        headers = [c.strip() for c in lines[0].strip("|").split("|")]
        row_chunks: List[EnrichedChunk] = []
        for row_idx, row_line in enumerate(lines[2:]):
            cells = [c.strip() for c in row_line.strip("|").split("|")]
            parts = [f"{h}: {v}" for h, v in zip(headers, cells) if h and v]
            if not parts:
                continue
            row_text = " | ".join(parts)
            if caption:
                row_text = f"[From table: {caption}] {row_text}"
            cid = hashlib.md5(
                f"{src_name}:{page}:row:{row_idx}:{row_text[:40]}".encode()
            ).hexdigest()[:12]
            row_chunks.append(EnrichedChunk(
                chunk_id=cid, text=row_text, content_type=ContentType.TABLE_ROW,
                page_number=page, source_file=os.path.basename(src_name),
                bbox=bbox, caption=caption, nearby_text=nearby_text[:2],
                table_headers=headers, row_index=row_idx,
            ))
        return row_chunks


# ═══════════════════════════════════════════════════════════════════
# PHASE 2C — IMAGE PROCESSOR
# ═══════════════════════════════════════════════════════════════════

class ImageProcessor:

    def __init__(self, registry: NIMRegistry):
        self.reg = registry

    def process(self, elements: List[Dict], source_file: str) -> List[EnrichedChunk]:
        chunks: List[EnrichedChunk] = []
        for e in elements:
            cm = e.get("content_metadata", {})
            pg = cm.get("page_number", -1)
            subtype = cm.get("subtype", "image")
            content_type_raw = cm.get("type", "image")

            if content_type_raw == "structured" and subtype == "chart":
                ct = ContentType.CHART
            elif content_type_raw == "infographic":
                ct = ContentType.INFOGRAPHIC
            else:
                ct = ContentType.IMAGE

            src = e.get("source_metadata", {})
            src_name = src.get("source_name", source_file)
            img_meta = e.get("image_metadata", {}) or {}
            bbox = img_meta.get("image_location", [])
            bbox_max = img_meta.get("image_location_max_dimensions", [])
            existing_caption = img_meta.get("caption", "").strip()
            hierarchy = cm.get("hierarchy", {})
            nearby = hierarchy.get("nearby_objects", {})
            nearby_text_list: List[str] = nearby.get("text", {}).get("content", [])

            image_b64 = img_meta.get("image_content", "")
            if not image_b64:
                raw_content = e.get("content", "")
                if raw_content and self._looks_like_b64(raw_content):
                    image_b64 = raw_content

            caption = self._resolve_caption(
                existing_caption, image_b64, nearby_text_list, src_name, pg
            )

            nearby_prefix = ""
            if nearby_text_list:
                nearby_prefix = "Nearby text: " + " ".join(nearby_text_list[:2][:200])
            searchable = f"{caption}\n{nearby_prefix}".strip()
            if not searchable:
                searchable = f"[{ct.value} on page {pg} of {os.path.basename(src_name)}]"

            cid = hashlib.md5(
                f"{src_name}:{pg}:{ct.value}:{caption[:40]}".encode()
            ).hexdigest()[:12]
            chunks.append(EnrichedChunk(
                chunk_id=cid, text=searchable, content_type=ct.value,
                page_number=pg, source_file=os.path.basename(src_name),
                bbox=bbox, bbox_max_dims=bbox_max, caption=caption,
                image_b64=image_b64, nearby_text=nearby_text_list[:3],
            ))

        logger.info(f"ImageProcessor produced {len(chunks)} image/chart chunks")
        return chunks

    def _resolve_caption(
        self, existing: str, image_b64: str,
        nearby_text: List[str], src_name: str, pg: int,
    ) -> str:
        if existing and len(existing.split()) >= Config.CAPTION_MIN_WORDS:
            return existing
        if image_b64:
            context_hint = " ".join(nearby_text[:2]) if nearby_text else ""
            svg_text = self._extract_svg_text(image_b64)
            if svg_text:
                vlm_caption = f"[Vector diagram] Labels: {svg_text}"
                if len(image_b64) < 500_000:
                    vlm = self.reg.caption_image(image_b64, context_hint)
                    if vlm and len(vlm.split()) >= Config.CAPTION_MIN_WORDS:
                        return vlm
                return vlm_caption
            vlm = self.reg.caption_image(image_b64, context_hint)
            if vlm and len(vlm.split()) >= Config.CAPTION_MIN_WORDS:
                return vlm
            if vlm:
                context = " ".join(nearby_text[:2]) if nearby_text else ""
                return f"{vlm}. Context: {context}".strip(". ")
        if nearby_text:
            return (
                f"[Image/chart on page {pg}, file {os.path.basename(src_name)}] "
                f"Nearby text: {' '.join(nearby_text[:3])}"
            )
        return f"[Image/chart on page {pg} of {os.path.basename(src_name)}]"

    @staticmethod
    def _looks_like_b64(s: str) -> bool:
        return bool(re.match(r"^[A-Za-z0-9+/\r\n]+=*$", s.strip()[:100]))

    @staticmethod
    def _extract_svg_text(b64_or_svg: str) -> str:
        try:
            decoded = base64.b64decode(b64_or_svg).decode("utf-8", errors="ignore")
        except Exception:
            decoded = b64_or_svg
        if "<svg" not in decoded.lower():
            return ""
        texts = re.findall(r"<text[^>]*>([^<]+)</text>", decoded, re.IGNORECASE)
        tspan = re.findall(r"<tspan[^>]*>([^<]+)</tspan>", decoded, re.IGNORECASE)
        all_text = [t.strip() for t in texts + tspan if t.strip()]
        return " | ".join(all_text[:20]) if all_text else ""


# ═══════════════════════════════════════════════════════════════════
# PHASE 1 — NV-INGEST ENGINE (extraction only, NV-Ingest 25.9.0)
# ═══════════════════════════════════════════════════════════════════

class NVIngestEngine:

    _pipeline_started: bool = False

    @staticmethod
    def wait_for_broker(
        host: str = Config.BROKER_HOST,
        port: int = Config.BROKER_PORT,
        timeout: int = Config.BROKER_TIMEOUT,
    ) -> None:
        logger.info(f"Waiting for NV-Ingest broker {host}:{port} ...")
        deadline = time.time() + timeout
        while time.time() < deadline:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            if s.connect_ex((host, port)) == 0:
                s.close()
                logger.info("Broker ready!")
                return
            s.close()
            time.sleep(0.5)
        raise RuntimeError(f"NV-Ingest broker not reachable after {timeout}s")

    @classmethod
    def start_pipeline(cls) -> None:
        if cls._pipeline_started:
            return

        from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import (
            PipelineCreationSchema,
            run_pipeline,
        )

        # ── Explicitly pass best 2025 NIM endpoints + API key ──
        # This prevents the 403 Forbidden error on YOLOX/OCR endpoints
        cfg = PipelineCreationSchema(
            # Page element detection (YOLOX v2 — latest)
            yolox_http_endpoint=Config.YOLOX_PAGE_ENDPOINT,
            yolox_infer_protocol="http",
            yolox_auth_token=Config.NVIDIA_API_KEY,

            # Graphic/chart element detection
            yolox_graphic_elements_http_endpoint=Config.YOLOX_GRAPHIC_ENDPOINT,
            yolox_graphic_elements_infer_protocol="http",

            # Table structure detection
            yolox_table_structure_http_endpoint=Config.YOLOX_TABLE_ENDPOINT,
            yolox_table_structure_infer_protocol="http",

            # OCR — upgraded to nemoretriever-ocr-v1
            paddle_http_endpoint=Config.OCR_ENDPOINT,
            paddle_infer_protocol="http",

            # NeMoRetriever Parse (VLM-based OCR/layout)
            nemoretriever_parse_http_endpoint=Config.PARSE_ENDPOINT,

            # Global auth token for all endpoints
            auth_token=Config.NVIDIA_API_KEY,
        )

        run_pipeline(
            cfg,
            block=False,
            disable_dynamic_scaling=True,
            run_in_subprocess=True,
        )
        cls._pipeline_started = True
        logger.info("NV-Ingest pipeline started (NV-Ingest 25.9.0)")
        cls.wait_for_broker()

    @classmethod
    def extract_only(
        cls,
        file_paths: List[str],
        registry: NIMRegistry,
    ) -> List[EnrichedChunk]:
        from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
        from nv_ingest_client.client import Ingestor, NvIngestClient

        cls.start_pipeline()

        client = NvIngestClient(
            message_client_allocator=SimpleClient,
            message_client_port=Config.BROKER_PORT,
            message_client_hostname=Config.BROKER_HOST,
        )

        # Phase 1: extract + caption only
        # No .split() .embed() .vdb_upload() — owned by this code
        ingestor = (
            Ingestor(client=client)
            .files(file_paths)
            .extract(
                extract_text=True,
                extract_tables=True,
                extract_charts=True,
                extract_images=True,
                extract_infographics=True,
                table_output_format="markdown",
                text_depth="page",
            )
            .caption(
                endpoint_url=Config.CAPTION_URL,
                model_name=Config.CAPTION_MODEL,
                api_key=Config.NVIDIA_API_KEY,
            )
        )

        logger.info(f"Phase 1 extraction started: {file_paths}")
        t0 = time.time()
        results, failures = ingestor.ingest(show_progress=True, return_failures=True)
        elapsed = time.time() - t0
        logger.info(
            f"Phase 1 complete in {elapsed:.1f}s | "
            f"results={len(results)} failures={len(failures)}"
        )
        for i, f in enumerate(failures[:3]):
            logger.warning(f"Failure [{i}]: {str(f)[:200]}")

        # Bucket by content type
        text_elements: List[Dict] = []
        table_elements: List[Dict] = []
        image_elements: List[Dict] = []

        for doc_result in results:
            items = doc_result if isinstance(doc_result, list) else [doc_result]
            for item in items:
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except Exception:
                        continue
                if not isinstance(item, dict):
                    continue
                meta = item.get("metadata", item)
                cm = meta.get("content_metadata", {})
                ct = cm.get("type", "text")
                sub = cm.get("subtype", "")

                if ct == "structured":
                    table_elements.append(meta)
                elif ct in ("image", "infographic") or (ct == "structured" and sub == "chart"):
                    image_elements.append(meta)
                else:
                    text_elements.append(meta)

        logger.info(
            f"Bucketed: text={len(text_elements)} "
            f"tables={len(table_elements)} images={len(image_elements)}"
        )

        source = file_paths[0] if file_paths else ""
        image_processor = ImageProcessor(registry)

        all_chunks: List[EnrichedChunk] = []
        all_chunks.extend(LayoutChunker.chunk(text_elements, source))
        all_chunks.extend(TableChunker.chunk(table_elements, source))
        all_chunks.extend(image_processor.process(image_elements, source))

        logger.info(f"Phase 2 total enriched chunks: {len(all_chunks)}")
        return all_chunks


# ═══════════════════════════════════════════════════════════════════
# MILVUS STORE — 2048-dim schema (matches llama-3.2-nv-embedqa-1b-v2)
# ═══════════════════════════════════════════════════════════════════

class MilvusStore:

    def __init__(self, registry: NIMRegistry, collection: str = None):
        self.reg = registry
        self.collection_name = collection or Config.COLLECTION

    def ensure_collection(self) -> None:
        if self.reg.milvus.has_collection(self.collection_name):
            logger.info(f"Collection exists: {self.collection_name}")
            return

        fields = [
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=Config.EMBED_DIM),  # 2048
            FieldSchema("sparse_vector", DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema("text", DataType.VARCHAR, max_length=65535),
            FieldSchema("content_type", DataType.VARCHAR, max_length=32),
            FieldSchema("page_number", DataType.INT64),
            FieldSchema("source_file", DataType.VARCHAR, max_length=512),
            FieldSchema("caption", DataType.VARCHAR, max_length=4096),
            FieldSchema("bbox", DataType.VARCHAR, max_length=256),
            FieldSchema("chunk_id", DataType.VARCHAR, max_length=32),
        ]
        schema = CollectionSchema(fields, description="Enterprise RAG multimodal 2048-dim")

        connections.connect("default", uri=Config.MILVUS_URI)
        col = Collection(self.collection_name, schema)

        col.create_index("embedding", {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 256},
        })
        col.create_index("sparse_vector", {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
        })
        col.create_index("content_type")
        col.create_index("page_number")
        col.create_index("source_file")

        logger.info(f"Created collection (2048-dim): {self.collection_name}")

    def insert_chunks(
        self,
        chunks: List[EnrichedChunk],
        embeddings: List[List[float]],
        sparse_vectors: Optional[List[Dict]] = None,
    ) -> int:
        if not chunks or not embeddings:
            return 0
        assert len(chunks) == len(embeddings)

        rows = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            row = {
                "embedding": emb,
                "text": chunk.text[:65000],
                "content_type": str(chunk.content_type),
                "page_number": max(chunk.page_number, -1),
                "source_file": chunk.source_file[:512],
                "caption": chunk.caption[:4096],
                "bbox": json.dumps(chunk.bbox[:4] if chunk.bbox else []),
                "chunk_id": chunk.chunk_id[:32],
            }
            if sparse_vectors and i < len(sparse_vectors):
                row["sparse_vector"] = sparse_vectors[i]
            rows.append(row)

        try:
            connections.connect("default", uri=Config.MILVUS_URI)
            col = Collection(self.collection_name)
            col.load()
            result = col.insert(rows)
            col.flush()
            inserted = len(result.primary_keys) if hasattr(result, "primary_keys") else len(rows)
            logger.info(f"Inserted {inserted} chunks into {self.collection_name}")
            return inserted
        except Exception as exc:
            logger.error(f"Milvus insert failed: {exc}")
            try:
                simple_rows = [{k: v for k, v in r.items() if k != "sparse_vector"} for r in rows]
                self.reg.milvus.insert(self.collection_name, simple_rows)
                return len(simple_rows)
            except Exception as exc2:
                logger.error(f"Fallback insert failed: {exc2}")
                raise

    def search_dense(
        self,
        query_embedding: List[float],
        top_k: int = Config.DENSE_TOP_K,
        expr: Optional[str] = None,
    ) -> List[Dict]:
        try:
            params = {
                "collection_name": self.collection_name,
                "data": [query_embedding],
                "limit": top_k,
                "output_fields": ["text", "content_type", "page_number", "source_file", "caption", "chunk_id"],
            }
            if expr:
                params["filter"] = expr
            hits = self.reg.milvus.search(**params)[0]
            return [{
                "text": self._get_field(h, "text"),
                "content_type": self._get_field(h, "content_type"),
                "page_number": self._get_field(h, "page_number", -1),
                "source_file": self._get_field(h, "source_file"),
                "caption": self._get_field(h, "caption"),
                "vector_score": getattr(h, "distance", 0.0),
            } for h in hits]
        except Exception as exc:
            logger.error(f"Dense search failed: {exc}")
            return []

    def search_sparse(
        self,
        sparse_vector: Dict,
        top_k: int = Config.SPARSE_TOP_K,
        expr: Optional[str] = None,
    ) -> List[Dict]:
        try:
            connections.connect("default", uri=Config.MILVUS_URI)
            col = Collection(self.collection_name)
            col.load()
            results = col.search(
                data=[sparse_vector],
                anns_field="sparse_vector",
                param={"metric_type": "IP"},
                limit=top_k,
                expr=expr,
                output_fields=["text", "content_type", "page_number", "source_file", "caption"],
            )
            hits = results[0] if results else []
            return [{
                "text": h.entity.get("text", ""),
                "content_type": h.entity.get("content_type", "text"),
                "page_number": h.entity.get("page_number", -1),
                "source_file": h.entity.get("source_file", ""),
                "caption": h.entity.get("caption", ""),
                "bm25_score": h.distance,
            } for h in hits]
        except Exception as exc:
            logger.warning(f"Sparse search failed (dense only): {exc}")
            return []

    @staticmethod
    def _get_field(hit: Any, field: str, default: Any = "") -> Any:
        if isinstance(hit, dict):
            entity = hit.get("entity", hit)
            return entity.get(field, default) if isinstance(entity, dict) else default
        entity = getattr(hit, "entity", None)
        if entity is not None:
            return getattr(entity, field, default)
        return getattr(hit, field, default)

    def list_collections(self) -> List[str]:
        return self.reg.milvus.list_collections()

    def collection_stats(self, name: Optional[str] = None) -> Dict:
        name = name or self.collection_name
        try:
            return {"collection": name, "stats": self.reg.milvus.get_collection_stats(name)}
        except Exception as exc:
            return {"collection": name, "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════
# BM25 SPARSE ENCODER
# ═══════════════════════════════════════════════════════════════════

class BM25Encoder:
    _vectorizer = None
    _fitted: bool = False

    @classmethod
    def fit(cls, corpus: List[str]) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            cls._vectorizer = TfidfVectorizer(
                sublinear_tf=True, max_features=30000, ngram_range=(1, 2),
            )
            cls._vectorizer.fit(corpus)
            cls._fitted = True
            logger.info(f"BM25 encoder fitted on {len(corpus)} documents")
        except ImportError:
            logger.warning("sklearn not installed — sparse search disabled")

    @classmethod
    def encode(cls, text: str) -> Optional[Dict[int, float]]:
        if not cls._fitted or cls._vectorizer is None:
            return None
        try:
            vec = cls._vectorizer.transform([text])
            cx = vec.tocoo()
            return {int(col): float(val) for col, val in zip(cx.col, cx.data)}
        except Exception:
            return None

    @classmethod
    def encode_batch(cls, texts: List[str]) -> List[Optional[Dict]]:
        return [cls.encode(t) for t in texts]


# ═══════════════════════════════════════════════════════════════════
# HYBRID RETRIEVAL ENGINE
# ═══════════════════════════════════════════════════════════════════

class HybridRetrievalEngine:

    def __init__(self, registry: NIMRegistry, store: MilvusStore):
        self.reg = registry
        self.store = store

    @staticmethod
    def reciprocal_rank_fusion(
        dense_hits: List[Dict], sparse_hits: List[Dict], k: int = Config.RRF_K,
    ) -> List[Dict]:
        scores: Dict[str, float] = defaultdict(float)
        meta: Dict[str, Dict] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            key = hit["text"][:100]
            scores[key] += 1.0 / (k + rank)
            if key not in meta:
                meta[key] = {**hit, "dense_rank": rank}
        for rank, hit in enumerate(sparse_hits, start=1):
            key = hit["text"][:100]
            scores[key] += 1.0 / (k + rank)
            if key not in meta:
                meta[key] = {**hit, "sparse_rank": rank}
            else:
                meta[key]["sparse_rank"] = rank
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        fused = []
        for key in sorted_keys:
            entry = meta[key].copy()
            entry["rrf_score"] = scores[key]
            fused.append(entry)
        return fused

    @staticmethod
    def classify_confidence(logit: float) -> str:
        if logit >= Config.HIGH_CONFIDENCE:
            return ConfidenceLevel.HIGH
        elif logit >= Config.MEDIUM_CONFIDENCE:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    @staticmethod
    def build_expr(
        content_type_filter: Optional[str],
        page_filter: Optional[int],
        source_filter: Optional[str],
    ) -> Optional[str]:
        parts: List[str] = []
        if content_type_filter:
            parts.append(f'content_type == "{content_type_filter}"')
        if page_filter is not None:
            parts.append(f"page_number == {page_filter}")
        if source_filter:
            parts.append(f'source_file like "%{source_filter}%"')
        return " and ".join(parts) if parts else None

    def retrieve(
        self,
        query: str,
        collection: Optional[str] = None,
        top_k: Optional[int] = None,
        content_type_filter: Optional[str] = None,
        page_filter: Optional[int] = None,
        source_filter: Optional[str] = None,
    ) -> Tuple[List[RetrievedChunk], Dict]:
        stats: Dict[str, Any] = {}
        expr = self.build_expr(content_type_filter, page_filter, source_filter)

        t0 = time.time()
        q_emb = self.reg.embed_query(query)
        stats["embed_ms"] = round((time.time() - t0) * 1000, 1)

        t1 = time.time()
        dense_hits = self.store.search_dense(q_emb, top_k=top_k or Config.DENSE_TOP_K, expr=expr)
        stats["dense_ms"] = round((time.time() - t1) * 1000, 1)
        stats["dense_hits"] = len(dense_hits)

        t2 = time.time()
        sparse_vec = BM25Encoder.encode(query)
        sparse_hits: List[Dict] = []
        if sparse_vec:
            sparse_hits = self.store.search_sparse(sparse_vec, top_k=top_k or Config.SPARSE_TOP_K, expr=expr)
        stats["sparse_ms"] = round((time.time() - t2) * 1000, 1)
        stats["sparse_hits"] = len(sparse_hits)

        if not dense_hits and not sparse_hits:
            return [], stats

        t3 = time.time()
        fused = self.reciprocal_rank_fusion(dense_hits, sparse_hits)
        stats["rrf_ms"] = round((time.time() - t3) * 1000, 1)
        stats["fused_candidates"] = len(fused)

        t4 = time.time()
        rerank_candidates = fused[: Config.RERANK_TOP_K * 3]
        passages = [c["text"] for c in rerank_candidates]
        reranked_chunks: List[RetrievedChunk] = []

        try:
            rankings = self.reg.rerank(query, passages)
            for rank in rankings[: Config.RERANK_TOP_K]:
                idx = rank.get("index", 0)
                logit = rank.get("logit", 0.0)
                c = rerank_candidates[idx]
                reranked_chunks.append(RetrievedChunk(
                    text=c["text"],
                    content_type=c.get("content_type", "text"),
                    page_number=c.get("page_number", -1),
                    source_file=c.get("source_file", ""),
                    caption=c.get("caption", ""),
                    vector_score=c.get("vector_score", 0.0),
                    bm25_score=c.get("bm25_score", 0.0),
                    rrf_score=c.get("rrf_score", 0.0),
                    rerank_score=logit,
                    confidence=self.classify_confidence(logit),
                ))
        except Exception as exc:
            logger.warning(f"Reranker failed ({exc}) — using RRF order")
            for c in fused[: Config.RERANK_TOP_K]:
                reranked_chunks.append(RetrievedChunk(
                    text=c["text"],
                    content_type=c.get("content_type", "text"),
                    page_number=c.get("page_number", -1),
                    source_file=c.get("source_file", ""),
                    caption=c.get("caption", ""),
                    rrf_score=c.get("rrf_score", 0.0),
                    confidence=ConfidenceLevel.LOW,
                ))

        stats["rerank_ms"] = round((time.time() - t4) * 1000, 1)
        stats["final_chunks"] = len(reranked_chunks)
        return reranked_chunks, stats


# ═══════════════════════════════════════════════════════════════════
# INGEST ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

class IngestOrchestrator:

    def __init__(self, registry: NIMRegistry, store: MilvusStore):
        self.reg = registry
        self.store = store

    def ingest(self, file_paths: List[str]) -> Dict:
        t0 = time.time()
        chunks = NVIngestEngine.extract_only(file_paths, self.reg)
        if not chunks:
            return {"status": "no_chunks", "chunks": 0}

        BM25Encoder.fit([c.text for c in chunks])

        texts = [c.text for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks with {Config.EMBED_MODEL} (2048-dim)...")
        embeddings = self.reg.embed(texts, input_type="passage")

        sparse_vecs = BM25Encoder.encode_batch(texts)
        self.store.ensure_collection()
        inserted = self.store.insert_chunks(chunks, embeddings, sparse_vecs)

        elapsed = (time.time() - t0) * 1000
        return {
            "status": "success",
            "files": [os.path.basename(p) for p in file_paths],
            "chunks_extracted": len(chunks),
            "chunks_inserted": inserted,
            "elapsed_ms": round(elapsed, 1),
            "models_used": {
                "extraction": "nv-ingest-25.9.0",
                "page_detection": "nemoretriever-page-elements-v2",
                "table_structure": "nemoretriever-table-structure-v1",
                "ocr": "nemoretriever-ocr-v1",
                "vlm_caption": Config.CAPTION_MODEL,
                "embedding": Config.EMBED_MODEL,
                "embed_dim": Config.EMBED_DIM,
            },
            "breakdown": {
                ct: sum(1 for c in chunks if str(c.content_type) == ct)
                for ct in set(str(c.content_type) for c in chunks)
            },
        }


# ═══════════════════════════════════════════════════════════════════
# RAG ENGINE
# ═══════════════════════════════════════════════════════════════════

class RAGEngine:

    SYSTEM_PROMPT = (
        "You are a precise document analysis assistant. "
        "Answer questions using ONLY the provided context chunks below.\n\n"
        "Rules:\n"
        "- Use specific details (numbers, names, dates) from the context.\n"
        "- If the answer is NOT in the context, say exactly: "
        "'The provided documents do not contain this information.'\n"
        "- Never invent facts not present in the context.\n"
        "- When synthesizing multiple chunks, reference their types "
        "(e.g., 'According to the table on page 3...').\n"
        "- For table rows: reconstruct the full row data in your answer.\n"
        "- For images/charts: use the caption to describe what is shown."
    )

    def __init__(self, registry: NIMRegistry, retrieval: HybridRetrievalEngine):
        self.reg = registry
        self.retrieval = retrieval

    def query(
        self,
        query: str,
        collection: Optional[str] = None,
        top_k: Optional[int] = None,
        content_type_filter: Optional[str] = None,
        page_filter: Optional[int] = None,
        source_filter: Optional[str] = None,
    ) -> RAGResponse:
        t0 = time.time()

        flags = Guardrails.check_input(query)
        if "prompt_injection_detected" in flags:
            return RAGResponse(
                answer="This query has been flagged and cannot be processed.",
                guardrail_flags=flags,
                latency_ms=round((time.time() - t0) * 1000, 1),
            )
        if "empty_query" in flags:
            return RAGResponse(
                answer="Please provide a query.",
                guardrail_flags=flags,
                latency_ms=round((time.time() - t0) * 1000, 1),
            )

        chunks, retrieval_stats = self.retrieval.retrieve(
            query, collection=collection, top_k=top_k,
            content_type_filter=content_type_filter,
            page_filter=page_filter, source_filter=source_filter,
        )
        if not chunks:
            return RAGResponse(
                answer="No relevant documents found for this query.",
                confidence=ConfidenceLevel.LOW,
                retrieval_stats=retrieval_stats,
                latency_ms=round((time.time() - t0) * 1000, 1),
            )

        context_chunks = chunks[: Config.MAX_CONTEXT_CHUNKS]
        context_parts: List[str] = []
        sources: List[Dict] = []

        for i, c in enumerate(context_chunks):
            type_label = c.content_type.upper()
            page_label = f"page {c.page_number}" if c.page_number >= 0 else "unknown page"
            header = (
                f"[Chunk {i+1} | {type_label} | {page_label} | "
                f"source={c.source_file or 'unknown'} | confidence={c.confidence}]"
            )
            body = c.text
            if c.caption and c.caption not in body:
                body = f"Caption: {c.caption}\n{body}"
            context_parts.append(f"{header}\n{body}")
            sources.append({
                "chunk_index": i + 1,
                "content_type": c.content_type,
                "page_number": c.page_number,
                "source_file": c.source_file,
                "caption": c.caption[:200] if c.caption else "",
                "rerank_score": round(c.rerank_score, 4),
                "rrf_score": round(c.rrf_score, 6),
                "confidence": c.confidence,
            })

        context = "\n\n" + ("─" * 60) + "\n\n".join(context_parts)
        prompt = (
            f"{self.SYSTEM_PROMPT}\n\nContext:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )
        answer, model_used = self.reg.generate(prompt)
        fallback_used = model_used != Config.PRIMARY_LLM

        out_flags = Guardrails.check_output(answer, [c.text for c in context_chunks])
        all_flags = flags + out_flags

        high = sum(1 for c in context_chunks if c.confidence == ConfidenceLevel.HIGH)
        if high >= 2:
            overall = ConfidenceLevel.HIGH
        elif high >= 1 or any(c.confidence == ConfidenceLevel.MEDIUM for c in context_chunks):
            overall = ConfidenceLevel.MEDIUM
        else:
            overall = ConfidenceLevel.LOW

        elapsed = round((time.time() - t0) * 1000, 1)
        logger.info(
            f"RAG query | model={model_used.split('/')[-1]} | "
            f"chunks={len(context_chunks)} | confidence={overall} | {elapsed}ms"
        )

        return RAGResponse(
            answer=answer, model_used=model_used, fallback_used=fallback_used,
            confidence=overall, sources=sources, guardrail_flags=all_flags,
            latency_ms=elapsed, retrieval_stats=retrieval_stats,
        )


# ═══════════════════════════════════════════════════════════════════
# APP WIRING
# ═══════════════════════════════════════════════════════════════════

registry = NIMRegistry()
milvus_store = MilvusStore(registry)
hybrid_retrieval = HybridRetrievalEngine(registry, milvus_store)
rag_engine = RAGEngine(registry, hybrid_retrieval)
ingest_orchestrator = IngestOrchestrator(registry, milvus_store)


# ═══════════════════════════════════════════════════════════════════
# FASTAPI
# ═══════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Enterprise RAG API — NV-Ingest 25.9.0 + Best 2025 Models")
    os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
    milvus_store.ensure_collection()
    yield
    logger.info("Shutting down")
    registry.close()


app = FastAPI(
    title="Enterprise Multimodal RAG API",
    description="NV-Ingest 25.9.0 + llama-3.2-nv-embedqa-1b-v2 (2048-dim) + hybrid search",
    version="3.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "milvus": Config.MILVUS_URI,
        "collections": milvus_store.list_collections(),
        "models": {
            "page_detection": "nemoretriever-page-elements-v2",
            "graphic_detection": "nemoretriever-graphic-elements-v1",
            "table_structure": "nemoretriever-table-structure-v1",
            "ocr": "nemoretriever-ocr-v1",
            "vlm_captioner": Config.CAPTION_MODEL,
            "embedder": f"{Config.EMBED_MODEL} ({Config.EMBED_DIM}-dim)",
            "reranker": Config.RERANK_MODEL,
            "primary_llm": Config.PRIMARY_LLM,
            "fallback_llm": Config.FALLBACK_LLM,
        },
        "search_strategy": "dense(2048) + bm25_sparse + rrf + rerank",
    }


@app.post("/ingest")
async def ingest_endpoint(files: List[UploadFile] = File(...), collection: Optional[str] = None):
    saved: List[str] = []
    for f in files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in Config.SUPPORTED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type: {ext}")
        path = os.path.join(Config.UPLOAD_DIR, f.filename)
        data = await f.read()
        with open(path, "wb") as fh:
            fh.write(data)
        saved.append(path)

    if collection:
        milvus_store.collection_name = collection

    try:
        return ingest_orchestrator.ingest(saved)
    except Exception as exc:
        logger.error(f"Ingestion failed: {exc}")
        raise HTTPException(500, f"Ingestion failed: {str(exc)}")


@app.post("/query")
def query_endpoint(req: QueryRequest):
    return asdict(rag_engine.query(
        query=req.query, collection=req.collection, top_k=req.top_k,
        content_type_filter=req.content_type_filter,
        page_filter=req.page_filter, source_filter=req.source_filter,
    ))


@app.get("/collections")
def list_collections():
    names = milvus_store.list_collections()
    return {"collections": [milvus_store.collection_stats(n) for n in names]}


# ═══════════════════════════════════════════════════════════════════
# CLI MODE
# ═══════════════════════════════════════════════════════════════════

def cli_mode():
    import sys
    if not Config.NVIDIA_API_KEY:
        print("ERROR: Set NVIDIA_API_KEY environment variable")
        sys.exit(1)

    print("=" * 70)
    print("Enterprise Multimodal RAG — NV-Ingest 25.9.0 + Best 2025 Models")
    print(f"Embedding : {Config.EMBED_MODEL} ({Config.EMBED_DIM}-dim)")
    print(f"Reranker  : {Config.RERANK_MODEL}")
    print(f"LLM       : {Config.PRIMARY_LLM}")
    print("=" * 70)

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "./Docs/invoice-0-4.pdf"
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    print(f"\n[1/3] Ingesting: {pdf_path}")
    result = ingest_orchestrator.ingest([pdf_path])
    print(f"      Status         : {result['status']}")
    print(f"      Chunks found   : {result.get('chunks_extracted', 0)}")
    print(f"      Chunks stored  : {result.get('chunks_inserted', 0)}")
    print(f"      Breakdown      : {result.get('breakdown', {})}")
    print(f"      Time           : {result.get('elapsed_ms', 0):.0f}ms")
    print(f"      Models used    : {result.get('models_used', {})}")

    print(f"\n[2/3] Collection stats:")
    print(f"      {milvus_store.collection_stats()}")

    queries = [
        ("What is the total amount due?", None, None, None),
        ("Who is the salesperson?", None, None, None),
        ("List all items with quantity 25 or more", "table", None, None),
        ("Show charts or images from page 2", "image", 2, None),
    ]

    print(f"\n[3/3] Hybrid RAG queries (dense-2048 + BM25 + RRF + reranker)")
    print("=" * 70)

    for q, ct_filter, pg_filter, src_filter in queries:
        filters_desc = []
        if ct_filter:
            filters_desc.append(f"type={ct_filter}")
        if pg_filter:
            filters_desc.append(f"page={pg_filter}")
        filter_str = f" [filters: {', '.join(filters_desc)}]" if filters_desc else ""

        response = rag_engine.query(
            q, content_type_filter=ct_filter,
            page_filter=pg_filter, source_filter=src_filter,
        )
        print(f"\nQ: {q}{filter_str}")
        print(f"A: {response.answer[:600]}")
        print(f"   Model      : {response.model_used.split('/')[-1]} | Fallback: {response.fallback_used}")
        print(f"   Confidence : {response.confidence} | Latency: {response.latency_ms:.0f}ms")
        stats = response.retrieval_stats
        print(
            f"   Retrieval  : dense={stats.get('dense_hits', 0)} "
            f"sparse={stats.get('sparse_hits', 0)} "
            f"fused={stats.get('fused_candidates', 0)} "
            f"→ reranked={stats.get('final_chunks', 0)}"
        )
        if response.guardrail_flags:
            print(f"   Flags      : {response.guardrail_flags}")
        print(f"   Sources    : {len(response.sources)} chunks used")
        print("─" * 70)


if __name__ == "__main__":
    cli_mode()


NV-Ingest offers exactly two chunking strategies: text_depth in .extract() (document or page granularity), and the .split() task which is purely token-based using any HuggingFace tokenizer. nvidia That's it. The .split() stage is not pluggable per content-type. You cannot tell NV-Ingest "use layout-driven grouping for text, cell-level splitting for tables, VLM for vector images." The chunking layer has no concept of bbox, reading order, or element type — it only counts tokens.


NV-Ingest now supports custom Lambda stages — you can add user-defined stages to the pipeline. GitHub More critically:
extract_method="nemotron_parse" exists specifically for scanned documents or complex layouts — it provides higher-accuracy text extraction and is a direct swap in .extract(). NVIDIA
For PDF documents, NV-Ingest supports extraction via pdfium, nemoretriever-parse, Unstructured.io, and Adobe Content Extraction Services — multiple methods to balance throughput vs accuracy. GitHub
So you have four extraction backends, a dedicated advanced layout model, and custom Lambda stages. That's significant customization surface.

Your actual architectural question: "Can I route text/table/image to different parsers per element type?"
Short answer: No, not inside NV-Ingest's native chain — but yes via post-processing with user-defined stages.
Here's what you can build:
.extract(extract_method="nemotron_parse")   # replaces pdfium+YOLOX — layout-aware
# ↓ NV-Ingest now gives you typed elements: text, table, chart, image
# ↓ You intercept here with a custom Lambda stage
.user_defined_stage(layout_aware_chunker)   # your custom stage per content_type
.embed()
.vdb_upload()
The custom Lambda stage receives the enriched JSON with content_type, bbox, page_number, image_metadata already populated. You write Python that branches:

content_type == "text" → layout-driven paragraph grouper (reading order from bbox y-coords)
content_type == "table" → keep as atomic unit, no split (tables should never be token-split)
content_type == "image" → check if raster (has pixel data) → VLM caption; if vector SVG → fallback to structural description


The embedding dimension mismatch — this is a real bug in your code
Your Config has:
pythonVDB_DENSE_DIM: int = 2048  # NV-Ingest default embedder
EMBED_DIM: int = 1024      # nv-embedqa-e5-v5
NV-Ingest's default embed model is llama-3.2-nv-embedqa-1b-v2, not nv-embedqa-e5-v5. To use nv-embedqa-e5-v5 you must pass model_name and endpoint_url explicitly to the .embed() call. NVIDIA
You're uploading 2048-dim vectors from NV-Ingest's internal embedder, then querying with 1024-dim vectors from your own embed_query(). These live in different vector spaces. Queries will never match. Fix:
python.embed(
    model_name="nvidia/nv-embedqa-e5-v5",
    endpoint_url="https://integrate.api.nvidia.com/v1/embeddings",
    api_key=Config.NVIDIA_API_KEY,
    # also align input_type="passage" for storage
)
# and update VDB_DENSE_DIM = 1024

The vector image / no-caption problem
You're right this is a real gap. When YOLOX detects an image element but it's an SVG or diagram with no raster content, the VLM captioner gets an essentially blank image and returns a useless caption like "geometric shapes." Your options in priority order:

Structural caption from surrounding context — your post-processing Lambda stage can look at nearby_text (already in the metadata) and generate a synthetic caption: f"Figure on page {page}: surrounded by '{nearby_text[0][:100]}'". No VLM needed.
SVG text extraction — if the image blob is base64 SVG, you can parse its XML for <text> elements directly in Python.
Fallback LLM captioning — send the base64 image to your CAPTION_MODEL yourself. If the VLM returns fewer than 20 words, trigger a second pass with a structured prompt asking for axis labels, legend items, and chart title.

The base64 encoding you flagged as unused — that is the path to make this work. NV-Ingest stores image_content as base64 in the result JSON. Your MetadataEnricher currently ignores it.


PDF / DOCX / PPTX / Image
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 1 — .load()                                  │
│  Page splitting + rasterization                     │
│  Each page → image + raw bytes                      │
│  [CUSTOMIZABLE: file format routing]                │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 2 — .extract()                               │
│  YOLOX (nemoretriever-page-elements-v2)             │
│    → detects: text blocks, tables, charts,          │
│      images, infographics with bbox coordinates     │
│  PaddleOCR / nemoretriever-ocr-v1                   │
│    → reads text from each detected region           │
│  nemoretriever-table-structure-v1                   │
│    → detects rows/columns/cells inside tables       │
│  nemoretriever-graphic-elements-v1                  │
│    → detects elements inside charts                 │
│  Output: JSON with content_type, bbox, page_number  │
│  [CUSTOMIZABLE: which extractors to enable/disable] │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 3 — .caption()  ← YOUR CODE USES THIS       │
│  VLM (nemotron-nano-vl-8b-v1)                       │
│    → generates text descriptions of images/charts  │
│  [CUSTOMIZABLE: swap VLM model, endpoint, API key]  │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 4 — .split()  ← YOUR CODE SKIPS THIS        │
│  Token-based text chunking                          │
│  Uses Llama-3.2-1B tokenizer internally             │
│  [CUSTOMIZABLE: chunk_size, overlap, split_by]      │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 5 — .embed()  ← YOUR CODE SKIPS THIS        │
│  Generates dense vectors for all chunks             │
│  [CUSTOMIZABLE: model, endpoint, dim]               │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 6 — .vdb_upload()  ← YOUR CODE SKIPS THIS   │
│  Inserts embeddings into Milvus                     │
│  [CUSTOMIZABLE: collection, URI, sparse, dense_dim] │
└─────────────────────────────────────────────────────┘




1. SINGLE FILE AT A TIME
   NVIngestEngine.extract_only() passes file_paths[0] as source.
   Multi-file batching is broken — all chunks get source of
   first file only.
   Fix: loop per file or fix source attribution.

2. BM25 REFITS ON EVERY INGEST
   BM25Encoder.fit() rebuilds vocabulary from scratch every call.
   When you ingest a second document, old vocabulary is lost.
   Fix: persist and incrementally update the TF-IDF index.

3. NO DEDUP
   If you ingest the same PDF twice, chunks are inserted twice.
   NV-Ingest supports a dedup task natively — not wired here.
   Fix: add .dedup() to the ingestor chain or check chunk_id
   before insert.

4. IMAGE b64 STORED IN MEMORY ONLY
   image_b64 field in EnrichedChunk is never persisted to Milvus.
   So at query time you can't retrieve the original image.
   Fix: store image_b64 in a separate object store (S3/MinIO)
   and save the URL in Milvus instead.

5. NV-INGEST PIPELINE ONLY STARTS ONCE
   _pipeline_started flag means if the subprocess dies mid-run,
   the code never restarts it. No health check or recovery.
   Fix: add subprocess health monitoring.

6. MILVUS COLLECTION DIM IS HARDCODED AT CREATION
   If you change EMBED_DIM later and collection already exists,
   inserts silently fail or crash. No migration path.
   Fix: check dim on startup, drop+recreate if mismatch.

7. NO ASYNC INGESTION
   FastAPI /ingest endpoint is blocking — one file at a time,
   blocks the entire API server during ingestion.
   Fix: use BackgroundTasks or Celery for async ingestion.

8. .caption() MAY NOT EXIST IN 25.9.0 CLIENT
   The Ingestor().caption() chained call was added in 25.9.0
   but API shape may differ. If it fails, captioning silently
   falls back to VLM in ImageProcessor anyway — but you lose
   NV-Ingest's native caption integration.
