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