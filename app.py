from __future__ import annotations
import hashlib
import io
import json
import logging
import os
import re
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator


# ══════════════════════════════
# 1.  CONFIG 
# ══════════════════════════════

class CONFIG:
    NVIDIA_API_KEY: str       = os.getenv("NVIDIA_API_KEY", "")
    NVIDIA_BASE_URL: str      = "https://integrate.api.nvidia.com/v1"
    LLM_MODEL: str            = "meta/llama-3.3-70b-instruct"
    LLM_MAX_TOKENS: int       = 1024
    LLM_TEMPERATURE: float    = 0.1
    LLM_TOP_P: float          = 0.95

    # Asymmetric: "query: " prefix for queries, "passage: " for documents
    # All modalities (text / table / image-ocr) → SAME vector space
    EMBED_MODEL_ID: str       = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    EMBED_DIM: int            = 2048
    EMBED_DEVICE: str         = "cpu"       # "cuda" for GPU
    EMBED_BATCH_SIZE: int     = 32
    EMBED_QUERY_PREFIX: str   = "query: "
    EMBED_PASSAGE_PREFIX: str = "passage: "

    # ── Reranker (NVIDIA NIM — nv-rerankqa-mistral-4b-v3) ────────────────

    RERANKER_MODEL_ID: str    = "nvidia/nv-rerankqa-mistral-4b-v3"
    RERANKER_TOP_K_IN: int    = 20   # candidates fed to reranker
    RERANKER_TOP_K_OUT: int   = 5    # final results after reranking

    # ── Milvus Lite 
    MILVUS_URI: str           = "./milvus_pipeline.db"
    MILVUS_TEXT_COL: str      = "text_chunks"
    MILVUS_IMAGE_COL: str     = "image_chunks"
    MILVUS_TABLE_COL: str     = "table_chunks"
    MILVUS_METRIC: str        = "COSINE"
    HYBRID_DENSE_W: float     = 0.7    # weight for dense ANN score
    HYBRID_SPARSE_W: float    = 0.3    # weight for BM25 sparse score

    # ── OCR (batched) ─────────────────────────────────────────────────────
    PADDLE_OCR_LANG: str      = "en"
    OCR_BATCH_SIZE: int       = 8

    # ── BLIP caption model
    BLIP_MODEL_ID: str        = "Salesforce/blip-image-captioning-base"
    BLIP_MAX_TOKENS: int      = 64

    # ── Chunking ──────────────────────────────────────────────────────────
    CHUNK_MAX_TOKENS: int     = 512
    CHUNK_OVERLAP_TOKENS: int = 64
    MIN_CHUNK_TOKENS: int     = 30
    RECURSIVE_SEPS: list      = ["\n\n", "\n", ". ", " "]
    TABLE_PREVIEW_ROWS: int   = 3

    # ── Caption linking weights ───────────────────────────────────────────
    CAPTION_VERT_MULT: float  = 3.0
    CAPTION_W1: float         = 0.5   # vertical distance
    CAPTION_W2: float         = 0.3   # horizontal overlap penalty
    CAPTION_W3: float         = 0.2   # reading-order distance

    # ── Paths ─────────────────────────────────────────────────────────────
    IMAGE_SAVE_DIR: str       = "artifacts/images"
    PDF_IMAGES_SCALE: float   = 2.0


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LOGGER
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("doc_pipeline")


# ══════════════════════════════════
# 3.  PYDANTIC MODELS  —  metadata
# ══════════════════════════════════

def _now() -> str:  return datetime.now(timezone.utc).isoformat()
def _uid() -> str:  return str(uuid.uuid4())


class Modality(str, Enum):
    TEXT="text"; IMAGE="image"; TABLE="table"; EQUATION="equation"; CODE="code"

class ChunkType(str, Enum):
    PARAGRAPH="paragraph"; HEADING="heading"; CAPTION="caption"
    LIST="list"; FOOTNOTE="footnote"; ROW="row"

class ImageType(str, Enum):
    PHOTO="photo"; CHART="chart"; DIAGRAM="diagram"
    EQUATION_SNAP="equation_snap"; SCREENSHOT="screenshot"
    RASTER="raster"; VECTOR="vector"; UNKNOWN="unknown"


class BBox(BaseModel):
    x0: float; y0: float; x1: float; y1: float
    page_width:  Optional[float] = None
    page_height: Optional[float] = None

    @property
    def y_center(self) -> float: return (self.y0 + self.y1) / 2
    @property
    def x_center(self) -> float: return (self.x0 + self.x1) / 2
    @property
    def height(self)   -> float: return abs(self.y1 - self.y0)
    @property
    def width(self)    -> float: return abs(self.x1 - self.x0)
    @property
    def area(self)     -> float: return self.width * self.height


class TableSchema(BaseModel):
    headers:    list[str]      = []
    subheaders: list[str]      = []
    units:      dict[str, str] = {}   # col_name → unit string
    rows:       int            = 0
    cols:       int            = 0


class ChunkLinks(BaseModel):
    """All relationships navigable O(1) in both directions."""
    prev_id:       Optional[str] = None   # previous chunk in reading order
    next_id:       Optional[str] = None   # next chunk in reading order
    parent_id:     Optional[str] = None   # row subchunk → parent TableChunk
    caption_id:    Optional[str] = None   # image/table → its caption chunk
    image_id:      Optional[str] = None   # caption → image
    table_id:      Optional[str] = None   # caption → table
    figure_number: Optional[int] = None
    table_number:  Optional[int] = None
    section_title: Optional[str] = None
    row_ids:       list[str]     = []     # table → its row subchunk IDs


class RasterMeta(BaseModel):
    width_px:          int
    height_px:         int
    channels:          int
    dpi:               Optional[float]     = None
    aspect_ratio:      Optional[float]     = None
    file_size_bytes:   Optional[int]       = None
    pixel_array_shape: Optional[list[int]] = None   # [H, W, C]


class VectorMeta(BaseModel):
    centroid_x:        Optional[float] = None
    centroid_y:        Optional[float] = None
    area:              Optional[float] = None
    bbox_width:        Optional[float] = None
    bbox_height:       Optional[float] = None
    caption_generated: Optional[str]  = None


class BaseChunk(BaseModel):
    # ── Identity ──────────────────────────────────────────────────────────
    id:              str            = Field(default_factory=_uid)
    doc_id:          str
    source:          str
    modality:        Modality
    # ── Location ──────────────────────────────────────────────────────────
    page:            int
    bbox:            Optional[BBox] = None
    reading_order:   Optional[int]  = None
    # ── Structure ─────────────────────────────────────────────────────────
    section_path:    list[str]      = []   # ["H1", "H2", ...]
    links:           ChunkLinks     = Field(default_factory=ChunkLinks)
    # ── Timestamps (rich metadata) ────────────────────────────────────────
    ingested_at:     str            = Field(default_factory=_now)
    doc_created_at:  Optional[str]  = None
    # ── Embedding (single shared-space model) ─────────────────────────────
    embedding:       Optional[list[float]] = None
    embed_model:     Optional[str]         = None
    embed_at:        Optional[str]         = None


class TextChunk(BaseChunk):
    modality:      Modality  = Modality.TEXT
    chunk_type:    ChunkType = ChunkType.PARAGRAPH
    heading_level: Optional[int] = None
    content:       str       = ""
    token_count:   Optional[int] = None
    figure_number: Optional[int] = None
    table_number:  Optional[int] = None


class ImageChunk(BaseChunk):
    modality:       Modality  = Modality.IMAGE
    image_type:     ImageType = ImageType.UNKNOWN
    image_path:     str       = ""
    is_raster:      bool      = True
    # ── Raster ────────────────────────────────────────────────────────────
    raster_meta:    Optional[RasterMeta] = None
    dpi:            Optional[float]      = None
    aspect_ratio:   Optional[float]      = None
    # ── Vector ────────────────────────────────────────────────────────────
    vector_meta:    Optional[VectorMeta] = None
    # ── OCR ───────────────────────────────────────────────────────────────
    derived_text:   Optional[str]             = None
    ocr_confidence: Optional[float]           = None
    ocr_blocks:     list[dict[str, Any]]      = []
    # ── Text embedded for retrieval (OCR text or BLIP caption) ───────────
    search_text:    Optional[str]        = None
    chart_type:     Optional[str]        = None


class TableChunk(BaseChunk):
    modality:        Modality    = Modality.TABLE
    schema_:         TableSchema = Field(default_factory=TableSchema, alias="schema")
    html:            Optional[str] = None
    markdown:        Optional[str] = None
    csv:             Optional[str] = None
    table_number:    Optional[int] = None
    linearised_text: Optional[str] = None   # embedded as passage

    class Config:
        populate_by_name = True


class TableRowChunk(BaseChunk):
    modality:     Modality  = Modality.TABLE
    chunk_type:   ChunkType = ChunkType.ROW
    row_index:    int
    content:      str       = ""
    table_number: Optional[int] = None


class PipelineResult(BaseModel):
    doc_id:           str
    source:           str
    text_chunks:      list[TextChunk]     = []
    image_chunks:     list[ImageChunk]    = []
    table_chunks:     list[TableChunk]    = []
    table_row_chunks: list[TableRowChunk] = []
    total_chunks:     int                 = 0
    duration_s:       Optional[float]     = None
    completed_at:     str                 = Field(default_factory=_now)

    @model_validator(mode="after")
    def _count(self) -> "PipelineResult":
        self.total_chunks = (
            len(self.text_chunks) + len(self.image_chunks)
            + len(self.table_chunks) + len(self.table_row_chunks)
        )
        return self


class RetrievedChunk(BaseModel):
    """Returned by hybrid_search — full rich metadata for citation."""
    chunk_id:      str
    score:         float            # NVIDIA NIM reranker score
    modality:      str
    page:          int
    section_path:  list[str]        = []
    content:       str              = ""
    chunk_type:    Optional[str]    = None
    figure_number: Optional[int]    = None
    table_number:  Optional[int]    = None
    image_path:    Optional[str]    = None
    image_type:    Optional[str]    = None
    ocr_confidence: Optional[float] = None
    links:         dict             = {}
    doc_id:        str              = ""
    source:        str              = ""
    ingested_at:   str              = ""
    doc_created_at: Optional[str]   = None
    bbox:          Optional[dict]   = None
    reading_order: Optional[int]    = None
    token_count:   Optional[int]    = None


# ═══════════════════════════════
# 4.  REGEX
# ═══════════════════════════════

_FIG_RE  = re.compile(r"(?:Figure|Fig\.)\s*(\d+)", re.IGNORECASE)
_TAB_RE  = re.compile(r"Table\s*(\d+)",            re.IGNORECASE)
_UNIT_RE = re.compile(r"\(([^)]+)\)\s*$")

# ═══════════════════════════════════════════════════════
# 5.SINGLETONS  —  each model/client loaded exactly once
# ═══════════════════════════════════════════════════════

_EMBED_MODEL:    Any = None
_OCR_ENGINE:     Any = None
_BLIP_MODEL:     Any = None
_BLIP_PROCESSOR: Any = None
_MILVUS_CLIENT:  Any = None


def _get_embed_model():
    """
    No local model, no HuggingFace download.
    Returns a lightweight namespace with a single ._encode(texts, normalize)
    method that calls the NVIDIA NIM embedding API directly via requests.

    NVIDIA NIM endpoint:  POST /embeddings
    Auth:                 Bearer NVIDIA_API_KEY  (same key used for LLM + reranker)
    """
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        import requests
        import types

        def _encode(texts: list[str], normalize: bool = True) -> list[list[float]]:
            """
            Call NVIDIA NIM /embeddings for a list of already-prefixed texts.
            Batches according to CONFIG.EMBED_BATCH_SIZE to stay within API limits.

            Input:
                texts:     list[str]         — prefixed with "query: " or "passage: "
                normalize: bool              — ignored (NIM returns normalised vectors)
            Output:
                list[list[float]]            — shape (N, EMBED_DIM), L2-normalised
            """
            all_vecs: list[list[float]] = []

            for i in range(0, len(texts), CONFIG.EMBED_BATCH_SIZE):
                batch = texts[i: i + CONFIG.EMBED_BATCH_SIZE]
                payload = {
                    "model": CONFIG.EMBED_MODEL_ID,
                    "input": batch,
                    "input_type": "passage",   # overridden per call via prefix convention
                    "encoding_format": "float",
                    "truncate": "END",
                }
                headers = {
                    "Authorization": f"Bearer {CONFIG.NVIDIA_API_KEY}",
                    "Content-Type":  "application/json",
                }
                try:
                    resp = requests.post(
                        f"{CONFIG.NVIDIA_BASE_URL}/embeddings",
                        json=payload, headers=headers, timeout=60,
                    )
                    resp.raise_for_status()
                    # Response: {"data": [{"embedding": [...], "index": N}, ...]}
                    data = sorted(resp.json()["data"], key=lambda x: x["index"])
                    all_vecs.extend(item["embedding"] for item in data)
                except Exception as exc:
                    logger.error("[Embed] NVIDIA NIM call failed (batch %d): %s", i, exc)
                    raise

            return all_vecs

        # Thin namespace — no model weights, just the API caller
        proxy        = types.SimpleNamespace()
        proxy._encode = _encode
        _EMBED_MODEL  = proxy

        logger.info("[Embed] NVIDIA NIM embed ready | model=%s | endpoint=%s/embeddings",
                    CONFIG.EMBED_MODEL_ID, CONFIG.NVIDIA_BASE_URL)
    return _EMBED_MODEL


def _nvidia_rerank(query: str, passages: list[str]) -> list[float]:
    """
    Call NVIDIA NIM reranking endpoint (nv-rerankqa-mistral-4b-v3).
    No local model — runs on NVIDIA NIM, same API key as LLM.

    Input:
        query:    str         — user question
        passages: list[str]  — candidate passage texts
    Output:
        list[float]           — relevance score per passage (higher = better),
                                aligned with input index
    """
    import requests

    if not passages:
        return []

    payload = {
        "model":    CONFIG.RERANKER_MODEL_ID,
        "query":    {"text": query},
        "passages": [{"text": p[:2000]} for p in passages],   # NIM 2k char limit per passage
        "truncate": "END",
    }
    headers = {
        "Authorization": f"Bearer {CONFIG.NVIDIA_API_KEY}",
        "Content-Type":  "application/json",
    }
    try:
        resp = requests.post(
            f"{CONFIG.NVIDIA_BASE_URL}/ranking",
            json=payload, headers=headers, timeout=60,
        )
        resp.raise_for_status()
        # Response: {"rankings": [{"index": int, "logit": float}, ...]}
        rankings = resp.json().get("rankings", [])
        scores   = [0.0] * len(passages)
        for r in rankings:
            scores[int(r["index"])] = float(r["logit"])
        logger.info("[Reranker] %s | passages=%d", CONFIG.RERANKER_MODEL_ID, len(passages))
        return scores
    except Exception as exc:
        logger.warning("[Reranker] NVIDIA NIM call failed: %s — falling back to RRF scores", exc)
        return [0.0] * len(passages)


def _get_ocr():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from paddleocr import PaddleOCR
            _OCR_ENGINE = PaddleOCR(use_angle_cls=True, lang=CONFIG.PADDLE_OCR_LANG, show_log=False)
            logger.info("[OCR] PaddleOCR ready (lang=%s)", CONFIG.PADDLE_OCR_LANG)
        except ImportError:
            logger.warning("[OCR] PaddleOCR not installed — OCR disabled")
    return _OCR_ENGINE


def _get_blip() -> tuple[Any, Any]:
    """BLIP loaded once as singleton — never loaded per-image."""
    global _BLIP_MODEL, _BLIP_PROCESSOR
    if _BLIP_MODEL is None:
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            logger.info("[BLIP] Loading %s ...", CONFIG.BLIP_MODEL_ID)
            t0 = time.time()
            _BLIP_PROCESSOR = BlipProcessor.from_pretrained(CONFIG.BLIP_MODEL_ID)
            _BLIP_MODEL = BlipForConditionalGeneration.from_pretrained(CONFIG.BLIP_MODEL_ID)
            _BLIP_MODEL.eval()
            logger.info("[BLIP] Ready | %.2fs", time.time() - t0)
        except Exception as exc:
            logger.warning("[BLIP] Load failed: %s — captions disabled", exc)
    return _BLIP_MODEL, _BLIP_PROCESSOR


def _get_milvus():
    global _MILVUS_CLIENT
    if _MILVUS_CLIENT is None:
        from pymilvus import MilvusClient
        logger.info("[Milvus] Opening %s ...", CONFIG.MILVUS_URI)
        _MILVUS_CLIENT = MilvusClient(CONFIG.MILVUS_URI)
        logger.info("[Milvus] Client ready")
    return _MILVUS_CLIENT


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MILVUS COLLECTIONS SETUP  (idempotent)
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_collections() -> None:
    """
    Create text_chunks, image_chunks, table_chunks collections in Milvus Lite
    """
    from pymilvus import MilvusClient, DataType

    client = _get_milvus()

    def _create(name: str) -> None:
        if client.has_collection(name):
            logger.info("[Milvus] '%s' already exists — skip", name)
            return

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)

        # ── Primary / identity ──────────────────────────────────────────────
        schema.add_field("id",                DataType.VARCHAR, max_length=64,   is_primary=True)
        schema.add_field("doc_id",            DataType.VARCHAR, max_length=64)
        schema.add_field("source",            DataType.VARCHAR, max_length=512)
        schema.add_field("modality",          DataType.VARCHAR, max_length=32)
        schema.add_field("chunk_type",        DataType.VARCHAR, max_length=32)

        # ── Location ────────────────────────────────────────────────────────
        schema.add_field("page",              DataType.INT32)
        schema.add_field("reading_order",     DataType.INT32)
        schema.add_field("bbox_json",         DataType.VARCHAR, max_length=512)

        # ── Structure ───────────────────────────────────────────────────────
        schema.add_field("section_path_json", DataType.VARCHAR, max_length=1024)
        schema.add_field("links_json",        DataType.VARCHAR, max_length=2048)

        # ── Content (stored for reranker + LLM context) ─────────────────────
        schema.add_field("content_text",      DataType.VARCHAR, max_length=8192)
        schema.add_field("token_count",       DataType.INT32)

        # ── Rich metadata ───────────────────────────────────────────────────
        schema.add_field("figure_number",     DataType.INT32)
        schema.add_field("table_number",      DataType.INT32)
        schema.add_field("row_index",         DataType.INT32)
        schema.add_field("image_path",        DataType.VARCHAR, max_length=512)
        schema.add_field("image_type",        DataType.VARCHAR, max_length=32)
        schema.add_field("is_raster",         DataType.BOOL)
        schema.add_field("ocr_confidence",    DataType.FLOAT)

        # ── Timestamps ──────────────────────────────────────────────────────
        schema.add_field("ingested_at",       DataType.VARCHAR, max_length=64)
        schema.add_field("doc_created_at",    DataType.VARCHAR, max_length=64)

        # ── Embeddings ──────────────────────────────────────────────────────
        schema.add_field("embedding",         DataType.FLOAT_VECTOR,  dim=CONFIG.EMBED_DIM)
        schema.add_field("sparse_embedding",  DataType.SPARSE_FLOAT_VECTOR)

        # ── Indexes ─────────────────────────────────────────────────────────
        idx = client.prepare_index_params()
        idx.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type=CONFIG.MILVUS_METRIC,
            params={},
        )
        idx.add_index(
            field_name="sparse_embedding",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build": 0.2},
        )

        client.create_collection(collection_name=name, schema=schema, index_params=idx)
        logger.info("[Milvus] Created '%s' (dense HNSW + sparse BM25, dim=%d)", name, CONFIG.EMBED_DIM)

    _create(CONFIG.MILVUS_TEXT_COL)
    _create(CONFIG.MILVUS_IMAGE_COL)
    _create(CONFIG.MILVUS_TABLE_COL)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  STAGE 0 — DOCLING PARSER
# ══════════════════════════════════════════════════════════════════════════════

def _file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(65536), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def _norm_bbox(raw: Any, pw: float, ph: float) -> dict[str, float]:
    try:
        x0 = float(getattr(raw, "l", getattr(raw, "x0", 0)))
        y0 = float(getattr(raw, "t", getattr(raw, "y0", 0)))
        x1 = float(getattr(raw, "r", getattr(raw, "x1", 0)))
        y1 = float(getattr(raw, "b", getattr(raw, "y1", 0)))
    except Exception:
        x0, y0, x1, y1 = 0.0, 0.0, pw, ph
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "page_width": pw, "page_height": ph}


def _page_dims(raw_doc: Any) -> dict[int, tuple[float, float]]:
    dims: dict[int, tuple[float, float]] = {}
    try:
        for pg in raw_doc.pages.values():
            pno = int(pg.page_no)
            w   = float(getattr(pg, "width", None) or getattr(pg.size, "width", 595))
            h   = float(getattr(pg, "height", None) or getattr(pg.size, "height", 842))
            dims[pno] = (w, h)
    except Exception as exc:
        logger.warning("[Parser] page dims error: %s — A4 defaults used", exc)
    return dims


def _extract_table_data(item: Any) -> dict[str, Any]:
    """Extract table structure: html + markdown + csv + headers + rows_raw. Never flattened."""
    html = md = csv_s = ""
    headers:  list[str]       = []
    rows_raw: list[list[Any]] = []
    try:
        if hasattr(item, "export_to_html"):       html = item.export_to_html()
        if hasattr(item, "export_to_markdown"):   md   = item.export_to_markdown()
        if hasattr(item, "export_to_dataframe"):
            df       = item.export_to_dataframe()
            headers  = list(df.columns)
            rows_raw = df.values.tolist()
            csv_s    = df.to_csv(index=False)
        elif hasattr(item, "data") and hasattr(item.data, "grid"):
            grid     = item.data.grid
            headers  = [c.text for c in grid[0]] if grid else []
            rows_raw = [[c.text for c in r] for r in grid[1:]] if len(grid) > 1 else []
            csv_s    = "\n".join([",".join(headers)] + [",".join(str(v) for v in r) for r in rows_raw])
    except Exception as exc:
        logger.warning("[Parser] Table extract partial: %s", exc)
    return {"html": html, "markdown": md, "csv": csv_s, "headers": headers, "rows_raw": rows_raw}


def _extract_image_data(item: Any) -> dict[str, Any]:
    pil_image = None;  img_bytes = b"";  mime = "image/png";  is_vector = False
    try:
        if hasattr(item, "image") and item.image is not None:
            img_obj   = item.image
            pil_image = (img_obj.pil_image() if hasattr(img_obj, "pil_image")
                         else img_obj.as_pil() if hasattr(img_obj, "as_pil") else None)
            mime      = getattr(img_obj, "mimetype", "image/png") or "image/png"
            is_vector = mime in ("image/svg+xml", "application/pdf", "image/wmf", "image/emf")
            if pil_image is not None:
                buf = io.BytesIO();  pil_image.save(buf, format="PNG");  img_bytes = buf.getvalue()
    except Exception as exc:
        logger.warning("[Parser] Image extract partial: %s", exc)
    return {"pil_image": pil_image, "image_bytes": img_bytes, "mime": mime, "is_vector": is_vector}


def parse_document(file_path: str | Path) -> dict[str, Any]:
    """
    Stage 0: Docling parse with full structure preservation.

    Input:  file_path: str | Path
    Output: dict with keys — doc_id, source, blocks: list[dict], page_dims: dict
    """
    file_path = Path(file_path)
    doc_id    = _file_hash(file_path)
    source    = str(file_path)
    logger.info("[Parser] START | doc_id=%s | file=%s", doc_id, source)
    t0 = time.time()

    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
    except ImportError as exc:
        raise RuntimeError("docling not installed — run: pip install docling") from exc

    opts = PdfPipelineOptions()
    opts.do_ocr                  = False   # PaddleOCR runs separately in Stage 5
    opts.do_table_structure      = True    # CRITICAL — preserve cell structure
    opts.images_scale            = CONFIG.PDF_IMAGES_SCALE
    opts.generate_page_images    = False
    opts.generate_picture_images = True

    raw_doc   = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}).convert(source).document
    dims      = _page_dims(raw_doc)
    blocks:   list[dict[str, Any]] = []
    pg_ctrs:  dict[int, int]       = {}

    for item, _ in raw_doc.iterate_items():
        try:
            page_no = int(item.prov[0].page_no) if item.prov else 1
        except Exception:
            page_no = 1

        pw, ph  = dims.get(page_no, (595.0, 842.0))
        raw_bb  = item.prov[0].bbox if item.prov else None
        bbox    = _norm_bbox(raw_bb, pw, ph) if raw_bb else {"x0": 0.0, "y0": 0.0, "x1": pw, "y1": ph, "page_width": pw, "page_height": ph}
        ro      = pg_ctrs.get(page_no, 0);  pg_ctrs[page_no] = ro + 1
        itype   = type(item).__name__

        if itype in ("TextItem", "SectionHeaderItem", "ListItem", "FootnoteItem"):
            blocks.append({"block_type": "text", "page": page_no, "reading_order": ro, "bbox": bbox,
                           "text": getattr(item, "text", str(item)),
                           "heading_level": int(getattr(item, "level", 1)) if itype == "SectionHeaderItem" else None,
                           "item_type": itype, "table_data": None, "image_data": None})

        elif itype == "TableItem":
            tdata = _extract_table_data(item)
            blocks.append({"block_type": "table", "page": page_no, "reading_order": ro, "bbox": bbox,
                           "text": tdata.get("markdown", ""), "heading_level": None,
                           "item_type": itype, "table_data": tdata, "image_data": None})

        elif itype in ("PictureItem", "FigureItem"):
            idata = _extract_image_data(item)
            blocks.append({"block_type": "image", "page": page_no, "reading_order": ro, "bbox": bbox,
                           "text": "", "heading_level": None,
                           "item_type": itype, "table_data": None, "image_data": idata})

        else:
            txt = getattr(item, "text", "")
            if txt.strip():
                blocks.append({"block_type": "text", "page": page_no, "reading_order": ro, "bbox": bbox,
                               "text": txt, "heading_level": None,
                               "item_type": itype, "table_data": None, "image_data": None})

    logger.info("[Parser] %d blocks | %d pages | %.2fs", len(blocks), len(dims), time.time() - t0)
    return {"doc_id": doc_id, "source": source, "blocks": blocks, "page_dims": dims}