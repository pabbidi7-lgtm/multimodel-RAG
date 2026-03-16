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
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class CONFIG:
    NVIDIA_API_KEY:   str   = os.getenv("NVIDIA_API_KEY", "")
    NVIDIA_BASE_URL:  str   = "https://integrate.api.nvidia.com/v1"
    LLM_MODEL:        str   = "meta/llama-3.3-70b-instruct"
    LLM_MAX_TOKENS:   int   = 2048
    LLM_TEMPERATURE:  float = 0.1
    LLM_TOP_P:        float = 0.95

    EMBED_MODEL_ID:   str   = "nvidia/nv-embedqa-e5-v5"
    EMBED_DIM:        int   = 1024
    EMBED_BATCH_SIZE: int   = 8
    EMBED_MAX_CHARS:  int   = 2000

    RERANKER_MODEL_ID:  str = "nvidia/nv-rerankqa-mistral-4b-v3"
    RERANKER_TOP_K_IN:  int = 30
    RERANKER_TOP_K_OUT: int = 10

    MILVUS_URI:        str  = "./milvus_pipeline.db"
    MILVUS_COLLECTION: str  = "doc_chunks"
    MILVUS_METRIC:     str  = "COSINE"

    CHUNK_MAX_TOKENS:     int  = 400
    CHUNK_OVERLAP_TOKENS: int  = 50
    MIN_CHUNK_TOKENS:     int  = 25
    RECURSIVE_SEPS:       list = ["\n\n", "\n", ". ", " "]

    PADDLE_OCR_LANG:  str   = "en"
    OCR_BATCH_SIZE:   int   = 8
    BLIP_MODEL_ID:    str   = "Salesforce/blip-image-captioning-base"
    BLIP_MAX_TOKENS:  int   = 64
    IMAGE_SAVE_DIR:   str   = "artifacts/images"
    PDF_IMAGES_SCALE: float = 2.0

    CAPTION_VERT_MULT: float = 3.0
    CAPTION_W1:        float = 0.5
    CAPTION_W2:        float = 0.3
    CAPTION_W3:        float = 0.2


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


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ENUMS & MODELS — general purpose, no domain-specific types
# ══════════════════════════════════════════════════════════════════════════════

def _now() -> str: return datetime.now(timezone.utc).isoformat()
def _uid() -> str: return str(uuid.uuid4())


class Modality(str, Enum):
    TEXT  = "text"
    IMAGE = "image"
    TABLE = "table"


class BlockType(str, Enum):
    PARAGRAPH = "paragraph"
    HEADING   = "heading"
    CAPTION   = "caption"
    LIST_ITEM = "list_item"
    FOOTNOTE  = "footnote"
    TABLE     = "table"
    TABLE_ROW = "table_row"
    IMAGE     = "image"
    CODE      = "code"
    TITLE     = "title"
    UNKNOWN   = "unknown"


class ImageType(str, Enum):
    PHOTO      = "photo"
    CHART      = "chart"
    DIAGRAM    = "diagram"
    SCREENSHOT = "screenshot"
    RASTER     = "raster"
    VECTOR     = "vector"
    UNKNOWN    = "unknown"


class QueryIntent(str, Enum):
    SUMMARY      = "summary"
    COMPARISON   = "comparison"
    TABLE_LOOKUP = "table_lookup"
    GENERAL      = "general"


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


class ChunkLinks(BaseModel):
    prev_id:       Optional[str] = None
    next_id:       Optional[str] = None
    parent_id:     Optional[str] = None
    caption_id:    Optional[str] = None
    image_id:      Optional[str] = None
    table_id:      Optional[str] = None
    figure_number: Optional[int] = None
    table_number:  Optional[int] = None
    section_title: Optional[str] = None
    row_ids:       list[str]     = []


class RasterMeta(BaseModel):
    width_px: int; height_px: int; channels: int
    dpi:               Optional[float]     = None
    aspect_ratio:      Optional[float]     = None
    file_size_bytes:   Optional[int]       = None
    pixel_array_shape: Optional[list[int]] = None


class VectorMeta(BaseModel):
    centroid_x:        Optional[float] = None
    centroid_y:        Optional[float] = None
    area:              Optional[float] = None
    bbox_width:        Optional[float] = None
    bbox_height:       Optional[float] = None
    caption_generated: Optional[str]   = None


class TableSchema(BaseModel):
    headers: list[str] = []
    rows:    int        = 0
    cols:    int        = 0


class Chunk(BaseModel):
    id:             str            = Field(default_factory=_uid)
    doc_id:         str
    source:         str
    file_name:      str            = ""
    file_type:      str            = ""
    modality:       Modality
    block_type:     BlockType      = BlockType.UNKNOWN
    page:           int
    reading_order:  Optional[int]  = None
    bbox:           Optional[BBox] = None
    section_path:   list[str]      = []
    heading_level:  Optional[int]  = None
    content:        str            = ""
    token_count:    int            = 0
    table_schema:   Optional[TableSchema] = None
    table_html:     Optional[str]         = None
    table_markdown: Optional[str]         = None
    row_index:      int                   = -1
    image_path:     str                   = ""
    image_type:     ImageType             = ImageType.UNKNOWN
    is_raster:      bool                  = True
    raster_meta:    Optional[RasterMeta]  = None
    vector_meta:    Optional[VectorMeta]  = None
    derived_text:   Optional[str]         = None
    ocr_confidence: Optional[float]       = None
    search_text:    Optional[str]         = None
    figure_number:  Optional[int]         = None
    table_number:   Optional[int]         = None
    links:          ChunkLinks            = Field(default_factory=ChunkLinks)
    ingested_at:    str                   = Field(default_factory=_now)
    doc_created_at: Optional[str]         = None
    embed_model:    Optional[str]         = None
    embed_at:       Optional[str]         = None


class PipelineResult(BaseModel):
    doc_id: str; source: str; total_chunks: int = 0; text_count: int = 0
    image_count: int = 0; table_count: int = 0; row_count: int = 0
    duration_s: Optional[float] = None; completed_at: str = Field(default_factory=_now)


class RetrievedChunk(BaseModel):
    chunk_id: str; score: float; modality: str; block_type: str = ""
    page: int; section_path: list[str] = []; content: str = ""
    figure_number: Optional[int] = None; table_number: Optional[int] = None
    image_path: Optional[str] = None; image_type: Optional[str] = None
    ocr_confidence: Optional[float] = None; links: dict = {}
    doc_id: str = ""; source: str = ""; ingested_at: str = ""
    bbox: Optional[dict] = None; reading_order: Optional[int] = None
    token_count: int = 0; row_index: int = -1


# ══════════════════════════════════════════════════════════════════════════════
# 4.  REGEX — only structural (figure/table references for caption detection)
# ══════════════════════════════════════════════════════════════════════════════

_FIG_RE = re.compile(r"(?:Figure|Fig\.?)\s*(\d+)", re.IGNORECASE)
_TAB_RE = re.compile(r"Table\s*(\d+)",             re.IGNORECASE)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  LAZY SINGLETONS
# ══════════════════════════════════════════════════════════════════════════════

_EMBED_MODEL: Any = None; _OCR_ENGINE: Any = None
_BLIP_MODEL: Any = None; _BLIP_PROCESSOR: Any = None; _MILVUS_CLIENT: Any = None


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        import requests, types
        if not CONFIG.NVIDIA_API_KEY:
            raise RuntimeError("NVIDIA_API_KEY is not set.")
        def _encode(texts: list[str], input_type: str = "passage") -> list[list[float]]:
            safe = [t[:CONFIG.EMBED_MAX_CHARS] if t else " " for t in texts]
            all_vecs: list[list[float]] = []
            for i in range(0, len(safe), CONFIG.EMBED_BATCH_SIZE):
                batch = [t if t.strip() else " " for t in safe[i:i+CONFIG.EMBED_BATCH_SIZE]]
                payload = {"model": CONFIG.EMBED_MODEL_ID, "input": batch,
                           "input_type": input_type, "encoding_format": "float", "truncate": "END"}
                hdrs = {"Authorization": f"Bearer {CONFIG.NVIDIA_API_KEY}", "Content-Type": "application/json"}
                resp = requests.post(f"{CONFIG.NVIDIA_BASE_URL}/embeddings", json=payload, headers=hdrs, timeout=60)
                if not resp.ok:
                    logger.error("[Embed] HTTP %d | %s", resp.status_code, resp.text[:300])
                    if resp.status_code in (401, 403): raise RuntimeError(f"NVIDIA API auth failed ({resp.status_code}).")
                    resp.raise_for_status()
                data = sorted(resp.json()["data"], key=lambda x: x["index"])
                all_vecs.extend(item["embedding"] for item in data)
            return all_vecs
        proxy = types.SimpleNamespace(); proxy._encode = _encode; _EMBED_MODEL = proxy
        logger.info("[Embed] NVIDIA NIM ready | model=%s", CONFIG.EMBED_MODEL_ID)
    return _EMBED_MODEL


def _nvidia_rerank(query: str, passages: list[str]) -> Optional[list[float]]:
    import requests
    if not passages: return []
    payload = {"model": CONFIG.RERANKER_MODEL_ID, "query": {"text": query},
               "passages": [{"text": p[:2000]} for p in passages], "truncate": "END"}
    hdrs = {"Authorization": f"Bearer {CONFIG.NVIDIA_API_KEY}", "Content-Type": "application/json"}
    try:
        resp = requests.post(f"{CONFIG.NVIDIA_BASE_URL}/ranking", json=payload, headers=hdrs, timeout=60)
        if not resp.ok:
            logger.error("[Reranker] HTTP %d | %s", resp.status_code, resp.text[:300]); return None
        rankings = resp.json().get("rankings", [])
        scores = [0.0] * len(passages)
        for r in rankings: scores[int(r["index"])] = float(r["logit"])
        logger.info("[Reranker] OK | n=%d | top=%.4f", len(passages), max(scores) if scores else 0)
        return scores
    except Exception as exc:
        logger.warning("[Reranker] Failed: %s — RRF fallback", exc); return None


def _get_ocr():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from paddleocr import PaddleOCR
            _OCR_ENGINE = PaddleOCR(use_angle_cls=True, lang=CONFIG.PADDLE_OCR_LANG, show_log=False)
            logger.info("[OCR] PaddleOCR ready")
        except ImportError: logger.warning("[OCR] PaddleOCR not installed")
    return _OCR_ENGINE


def _get_blip() -> tuple[Any, Any]:
    global _BLIP_MODEL, _BLIP_PROCESSOR
    if _BLIP_MODEL is None:
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            _BLIP_PROCESSOR = BlipProcessor.from_pretrained(CONFIG.BLIP_MODEL_ID)
            _BLIP_MODEL = BlipForConditionalGeneration.from_pretrained(CONFIG.BLIP_MODEL_ID)
            _BLIP_MODEL.eval(); logger.info("[BLIP] Ready")
        except Exception as exc: logger.warning("[BLIP] Load failed: %s", exc)
    return _BLIP_MODEL, _BLIP_PROCESSOR


def _get_milvus():
    global _MILVUS_CLIENT
    if _MILVUS_CLIENT is None:
        from pymilvus import MilvusClient
        _MILVUS_CLIENT = MilvusClient(CONFIG.MILVUS_URI)
        logger.info("[Milvus] Client ready | uri=%s", CONFIG.MILVUS_URI)
    return _MILVUS_CLIENT


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MILVUS COLLECTION — clean general schema
# ══════════════════════════════════════════════════════════════════════════════

def _get_stored_embedding_dim(client: Any, col: str) -> Optional[int]:
    try:
        desc = client.describe_collection(col)
        fields = desc.get("fields", []) if isinstance(desc, dict) else getattr(desc, "fields", [])
        for f in fields:
            name = f.get("name") if isinstance(f, dict) else getattr(f, "name", "")
            if name == "embedding":
                params = f.get("params", {}) if isinstance(f, dict) else getattr(f, "params", {})
                dim = params.get("dim") if isinstance(params, dict) else getattr(params, "dim", None)
                if dim is not None: return int(dim)
    except Exception as exc: logger.warning("[Milvus] Schema inspect: %s", exc)
    return None


def _create_collection_schema(client: Any, col: str) -> None:
    from pymilvus import DataType
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id",                DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field("doc_id",            DataType.VARCHAR, max_length=64)
    schema.add_field("source",            DataType.VARCHAR, max_length=512)
    schema.add_field("file_name",         DataType.VARCHAR, max_length=256)
    schema.add_field("file_type",         DataType.VARCHAR, max_length=16)
    schema.add_field("modality",          DataType.VARCHAR, max_length=16)
    schema.add_field("block_type",        DataType.VARCHAR, max_length=32)
    schema.add_field("page",              DataType.INT32)
    schema.add_field("reading_order",     DataType.INT32)
    schema.add_field("bbox_json",         DataType.VARCHAR, max_length=512)
    schema.add_field("section_path_json", DataType.VARCHAR, max_length=1024)
    schema.add_field("section_depth",     DataType.INT32)
    schema.add_field("content_text",      DataType.VARCHAR, max_length=8192)
    schema.add_field("token_count",       DataType.INT32)
    schema.add_field("row_index",         DataType.INT32)
    schema.add_field("image_path",        DataType.VARCHAR, max_length=512)
    schema.add_field("image_type",        DataType.VARCHAR, max_length=32)
    schema.add_field("ocr_confidence",    DataType.FLOAT)
    schema.add_field("figure_number",     DataType.INT32)
    schema.add_field("table_number",      DataType.INT32)
    schema.add_field("links_json",        DataType.VARCHAR, max_length=2048)
    schema.add_field("ingested_at",       DataType.VARCHAR, max_length=64)
    schema.add_field("doc_created_at",    DataType.VARCHAR, max_length=64)
    schema.add_field("embedding",         DataType.FLOAT_VECTOR, dim=CONFIG.EMBED_DIM)
    schema.add_field("sparse_embedding",  DataType.SPARSE_FLOAT_VECTOR)
    idx = client.prepare_index_params()
    idx.add_index("embedding", index_type="AUTOINDEX", metric_type=CONFIG.MILVUS_METRIC, params={})
    idx.add_index("sparse_embedding", index_type="SPARSE_INVERTED_INDEX", metric_type="IP", params={"drop_ratio_build": 0.2})
    client.create_collection(collection_name=col, schema=schema, index_params=idx)
    logger.info("[Milvus] Created '%s' (dense+sparse, dim=%d)", col, CONFIG.EMBED_DIM)


def _ensure_collection() -> None:
    client = _get_milvus(); col = CONFIG.MILVUS_COLLECTION
    if client.has_collection(col):
        stored_dim = _get_stored_embedding_dim(client, col)
        if stored_dim is None: return
        if stored_dim == CONFIG.EMBED_DIM:
            logger.info("[Milvus] '%s' exists | dim=%d ✓", col, stored_dim); return
        logger.warning("[Milvus] Dim MISMATCH stored=%d config=%d — recreating.", stored_dim, CONFIG.EMBED_DIM)
        client.drop_collection(col)
    _create_collection_schema(client, col)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MULTI-FORMAT PARSER
# ══════════════════════════════════════════════════════════════════════════════

_DOCLING_FORMATS = {".pdf", ".docx", ".pptx", ".html", ".htm"}
_TEXT_FORMATS    = {".txt", ".md", ".rst", ".csv", ".tsv"}
_IMAGE_FORMATS   = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp"}
SUPPORTED_FORMATS = _DOCLING_FORMATS | _TEXT_FORMATS | _IMAGE_FORMATS


def _file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(65536), b""): h.update(blk)
    return h.hexdigest()[:16]

def _norm_bbox(raw: Any, pw: float, ph: float) -> dict:
    try:
        x0 = float(getattr(raw, "l", getattr(raw, "x0", 0))); y0 = float(getattr(raw, "t", getattr(raw, "y0", 0)))
        x1 = float(getattr(raw, "r", getattr(raw, "x1", 0))); y1 = float(getattr(raw, "b", getattr(raw, "y1", 0)))
    except Exception: x0, y0, x1, y1 = 0.0, 0.0, pw, ph
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "page_width": pw, "page_height": ph}

def _default_bbox() -> dict:
    return {"x0": 0.0, "y0": 0.0, "x1": 595.0, "y1": 842.0, "page_width": 595.0, "page_height": 842.0}

def _page_dims(raw_doc: Any) -> dict[int, tuple[float, float]]:
    dims: dict[int, tuple[float, float]] = {}
    try:
        for pg in raw_doc.pages.values():
            pno = int(pg.page_no)
            w = float(getattr(pg, "width", None) or getattr(pg.size, "width", 595))
            h = float(getattr(pg, "height", None) or getattr(pg.size, "height", 842))
            dims[pno] = (w, h)
    except Exception as exc: logger.warning("[Parser] page dims: %s", exc)
    return dims

def _extract_table_data(item: Any) -> dict[str, Any]:
    html = md = csv_s = ""; headers: list[str] = []; rows_raw: list[list[Any]] = []
    try:
        if hasattr(item, "export_to_html"):     html = item.export_to_html()
        if hasattr(item, "export_to_markdown"): md   = item.export_to_markdown()
        if hasattr(item, "export_to_dataframe"):
            df = item.export_to_dataframe()
            headers = list(df.columns); rows_raw = df.values.tolist(); csv_s = df.to_csv(index=False)
        elif hasattr(item, "data") and hasattr(item.data, "grid"):
            grid = item.data.grid
            headers = [c.text for c in grid[0]] if grid else []
            rows_raw = [[c.text for c in r] for r in grid[1:]] if len(grid) > 1 else []
    except Exception as exc: logger.warning("[Parser] Table extract: %s", exc)
    return {"html": html, "markdown": md, "csv": csv_s, "headers": headers, "rows_raw": rows_raw}

def _extract_image_data(item: Any) -> dict[str, Any]:
    pil_image = None; img_bytes = b""; mime = "image/png"; is_vector = False
    try:
        if hasattr(item, "image") and item.image is not None:
            img_obj = item.image
            pil_image = (img_obj.pil_image() if hasattr(img_obj, "pil_image")
                         else img_obj.as_pil() if hasattr(img_obj, "as_pil") else None)
            mime = getattr(img_obj, "mimetype", "image/png") or "image/png"
            is_vector = mime in ("image/svg+xml", "application/pdf", "image/wmf", "image/emf")
            if pil_image is not None:
                buf = io.BytesIO(); pil_image.save(buf, format="PNG"); img_bytes = buf.getvalue()
    except Exception as exc: logger.warning("[Parser] Image extract: %s", exc)
    return {"pil_image": pil_image, "image_bytes": img_bytes, "mime": mime, "is_vector": is_vector}


def _parse_with_docling(file_path: Path) -> dict[str, Any]:
    doc_id = _file_hash(file_path); source = str(file_path)
    logger.info("[Parser] Docling START | doc_id=%s | file=%s", doc_id, file_path.name); t0 = time.time()
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
    except ImportError as exc: raise RuntimeError("docling not installed") from exc

    opts = PdfPipelineOptions(); opts.do_ocr = False; opts.do_table_structure = True
    opts.images_scale = CONFIG.PDF_IMAGES_SCALE; opts.generate_page_images = False; opts.generate_picture_images = True
    format_options = {}
    if file_path.suffix.lower() == ".pdf":
        format_options[InputFormat.PDF] = PdfFormatOption(pipeline_options=opts)

    raw_doc = DocumentConverter(format_options=format_options).convert(source).document
    dims = _page_dims(raw_doc); blocks: list[dict] = []; pg_ctrs: dict[int, int] = {}

    for item, _ in raw_doc.iterate_items():
        try: page_no = int(item.prov[0].page_no) if item.prov else 1
        except Exception: page_no = 1
        pw, ph = dims.get(page_no, (595.0, 842.0))
        raw_bb = item.prov[0].bbox if item.prov else None
        bbox = _norm_bbox(raw_bb, pw, ph) if raw_bb else {"x0": 0, "y0": 0, "x1": pw, "y1": ph, "page_width": pw, "page_height": ph}
        ro = pg_ctrs.get(page_no, 0); pg_ctrs[page_no] = ro + 1; itype = type(item).__name__

        if itype in ("TextItem", "SectionHeaderItem", "ListItem", "FootnoteItem"):
            blocks.append({"block_type": itype, "page": page_no, "reading_order": ro, "bbox": bbox,
                           "text": getattr(item, "text", str(item)),
                           "heading_level": int(getattr(item, "level", 1)) if itype == "SectionHeaderItem" else None,
                           "table_data": None, "image_data": None})
        elif itype == "TableItem":
            tdata = _extract_table_data(item)
            blocks.append({"block_type": "TableItem", "page": page_no, "reading_order": ro, "bbox": bbox,
                           "text": tdata.get("markdown", ""), "heading_level": None, "table_data": tdata, "image_data": None})
        elif itype in ("PictureItem", "FigureItem"):
            idata = _extract_image_data(item)
            blocks.append({"block_type": itype, "page": page_no, "reading_order": ro, "bbox": bbox,
                           "text": "", "heading_level": None, "table_data": None, "image_data": idata})
        else:
            txt = getattr(item, "text", "")
            if txt.strip():
                blocks.append({"block_type": "TextItem", "page": page_no, "reading_order": ro, "bbox": bbox,
                               "text": txt, "heading_level": None, "table_data": None, "image_data": None})

    logger.info("[Parser] Docling DONE | %d blocks | %d pages | %.2fs", len(blocks), len(dims), time.time() - t0)
    return {"doc_id": doc_id, "source": source, "file_name": file_path.name,
            "file_type": file_path.suffix.lstrip(".").lower(), "blocks": blocks, "page_dims": dims}


def _parse_plain_text(file_path: Path) -> dict[str, Any]:
    doc_id = _file_hash(file_path); text = file_path.read_text(encoding="utf-8", errors="replace")
    suffix = file_path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        import csv as csv_mod
        delimiter = "\t" if suffix == ".tsv" else ","
        all_rows = list(csv_mod.reader(text.splitlines(), delimiter=delimiter))
        if len(all_rows) >= 2:
            headers = all_rows[0]; rows_raw = all_rows[1:]
            md = "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                           + ["| " + " | ".join(r) + " |" for r in rows_raw])
            return {"doc_id": doc_id, "source": str(file_path), "file_name": file_path.name,
                    "file_type": suffix.lstrip("."), "page_dims": {1: (595.0, 842.0)},
                    "blocks": [{"block_type": "TableItem", "page": 1, "reading_order": 0, "bbox": _default_bbox(),
                                "text": md, "heading_level": None,
                                "table_data": {"html": "", "markdown": md, "csv": text, "headers": headers, "rows_raw": rows_raw},
                                "image_data": None}]}
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    blocks: list[dict] = []
    for i, para in enumerate(paragraphs):
        hm = re.match(r"^(#{1,6})\s+(.+)$", para.split("\n")[0])
        if hm:
            blocks.append({"block_type": "SectionHeaderItem", "page": 1, "reading_order": i, "bbox": _default_bbox(),
                           "text": hm.group(2).strip(), "heading_level": len(hm.group(1)), "table_data": None, "image_data": None})
            rest = "\n".join(para.split("\n")[1:]).strip()
            if rest:
                blocks.append({"block_type": "TextItem", "page": 1, "reading_order": i, "bbox": _default_bbox(),
                               "text": rest, "heading_level": None, "table_data": None, "image_data": None})
        else:
            blocks.append({"block_type": "TextItem", "page": 1, "reading_order": i, "bbox": _default_bbox(),
                           "text": para, "heading_level": None, "table_data": None, "image_data": None})
    return {"doc_id": doc_id, "source": str(file_path), "file_name": file_path.name,
            "file_type": file_path.suffix.lstrip(".").lower(), "blocks": blocks, "page_dims": {1: (595.0, 842.0)}}


def _parse_image_file(file_path: Path) -> dict[str, Any]:
    doc_id = _file_hash(file_path); pil_image = None; img_bytes = b""; w = h = 0
    try:
        from PIL import Image
        pil_image = Image.open(str(file_path)); w, h = pil_image.size
        buf = io.BytesIO(); pil_image.save(buf, format="PNG"); img_bytes = buf.getvalue()
    except Exception as exc: logger.warning("[Parser] Image load: %s", exc)
    pw, ph = float(max(w, 1)), float(max(h, 1))
    return {"doc_id": doc_id, "source": str(file_path), "file_name": file_path.name,
            "file_type": file_path.suffix.lstrip(".").lower(), "page_dims": {1: (pw, ph)},
            "blocks": [{"block_type": "PictureItem", "page": 1, "reading_order": 0,
                        "bbox": {"x0": 0, "y0": 0, "x1": pw, "y1": ph, "page_width": pw, "page_height": ph},
                        "text": "", "heading_level": None, "table_data": None,
                        "image_data": {"pil_image": pil_image, "image_bytes": img_bytes, "mime": "image/png", "is_vector": False}}]}


def parse_document(file_path: str | Path) -> dict[str, Any]:
    file_path = Path(file_path); suffix = file_path.suffix.lower()
    if suffix in _DOCLING_FORMATS:  return _parse_with_docling(file_path)
    elif suffix in _TEXT_FORMATS:   return _parse_plain_text(file_path)
    elif suffix in _IMAGE_FORMATS:  return _parse_image_file(file_path)
    else: raise ValueError(f"Unsupported: {suffix}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  LAYOUT-AWARE CHUNKING — general purpose
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_block_type(itype: str, heading_level: Optional[int]) -> BlockType:
    if heading_level is not None: return BlockType.TITLE if heading_level == 1 else BlockType.HEADING
    return {"TextItem": BlockType.PARAGRAPH, "SectionHeaderItem": BlockType.HEADING,
            "ListItem": BlockType.LIST_ITEM, "FootnoteItem": BlockType.FOOTNOTE,
            "TableItem": BlockType.TABLE, "PictureItem": BlockType.IMAGE, "FigureItem": BlockType.IMAGE}.get(itype, BlockType.UNKNOWN)

def _approx_tokens(text: str) -> int: return int(len(text.split()) * 1.3)

def _recursive_split(text: str, max_tok: int, overlap_tok: int, seps: list[str]) -> list[str]:
    if _approx_tokens(text) <= max_tok: return [text.strip()] if text.strip() else []
    for sep in seps:
        parts = text.split(sep)
        if len(parts) < 2: continue
        chunks, current, obuf = [], "", ""
        for part in parts:
            cand = (obuf + sep + part) if obuf else part
            if _approx_tokens(current + sep + cand) <= max_tok:
                current = (current + sep + cand) if current else cand
            else:
                if current.strip():
                    chunks.append(current.strip()); words = current.split()
                    obuf = " ".join(words[-overlap_tok:]) if len(words) > overlap_tok else current
                current = cand
        if current.strip(): chunks.append(current.strip())
        result: list[str] = []
        for c in chunks: result.extend(_recursive_split(c, max_tok, overlap_tok, seps))
        return result
    words = text.split()
    return [" ".join(words[i:i+max_tok]) for i in range(0, len(words), max_tok)]


class _SectionTracker:
    def __init__(self): self._path: list[tuple[int, str]] = []
    def update(self, level: int, title: str) -> list[str]:
        self._path = [(l, t) for l, t in self._path if l < level]; self._path.append((level, title))
        return [t for _, t in self._path]
    @property
    def current(self) -> list[str]: return [t for _, t in self._path]


def chunk_document(parsed: dict[str, Any]) -> list[Chunk]:
    doc_id = parsed["doc_id"]; source = parsed["source"]
    file_name = parsed.get("file_name", ""); file_type = parsed.get("file_type", "")
    blocks = parsed["blocks"]
    logger.info("[Chunker] START | doc_id=%s | blocks=%d", doc_id, len(blocks))
    t0 = time.time(); tracker = _SectionTracker(); chunks: list[Chunk] = []

    for blk in blocks:
        itype = blk["block_type"]; page = blk["page"]; bbox = blk["bbox"]; ro = blk["reading_order"]
        heading_level = blk.get("heading_level"); block_type = _normalise_block_type(itype, heading_level)
        section_path = tracker.current[:]

        if block_type not in (BlockType.TABLE, BlockType.IMAGE):
            text = (blk.get("text") or "").strip()
            if not text: continue
            if block_type in (BlockType.HEADING, BlockType.TITLE):
                section_path = tracker.update(heading_level or 1, text)
            fm = _FIG_RE.search(text); tm = _TAB_RE.search(text)
            if fm or tm: block_type = BlockType.CAPTION
            sub_texts = (_recursive_split(text, CONFIG.CHUNK_MAX_TOKENS, CONFIG.CHUNK_OVERLAP_TOKENS, CONFIG.RECURSIVE_SEPS)
                if block_type == BlockType.PARAGRAPH and _approx_tokens(text) > CONFIG.CHUNK_MAX_TOKENS else [text])
            for st in sub_texts:
                if not st.strip(): continue
                chunks.append(Chunk(doc_id=doc_id, source=source, file_name=file_name, file_type=file_type,
                    modality=Modality.TEXT, block_type=block_type, page=page, reading_order=ro,
                    bbox=BBox(**bbox) if bbox else None, section_path=section_path[:], heading_level=heading_level,
                    content=st, token_count=_approx_tokens(st),
                    figure_number=int(fm.group(1)) if fm else None, table_number=int(tm.group(1)) if tm else None))

        elif block_type == BlockType.TABLE:
            tdata = blk.get("table_data") or {}; headers = [str(h) for h in tdata.get("headers", [])]
            rows_raw = tdata.get("rows_raw", []); section_label = " > ".join(section_path) if section_path else f"Page {page}"
            tschema = TableSchema(headers=headers, rows=len(rows_raw), cols=len(headers))
            lin = [f"{section_label} | Table ({len(rows_raw)} rows)", f"Columns: {', '.join(headers)}"]
            for i, row in enumerate(rows_raw):
                lin.append(f"Row {i}: " + " | ".join(f"{h}={v}" for h, v in zip(headers, row)))
            linearised = "\n".join(lin)
            tm_m = _TAB_RE.search(tdata.get("markdown", "") or ""); table_no = int(tm_m.group(1)) if tm_m else None

            table_chunk = Chunk(doc_id=doc_id, source=source, file_name=file_name, file_type=file_type,
                modality=Modality.TABLE, block_type=BlockType.TABLE, page=page, reading_order=ro,
                bbox=BBox(**bbox) if bbox else None, section_path=section_path[:],
                content=linearised, token_count=_approx_tokens(linearised), table_schema=tschema,
                table_html=tdata.get("html", ""), table_markdown=tdata.get("markdown", ""), table_number=table_no)
            chunks.append(table_chunk); row_ids: list[str] = []

            for i, rv in enumerate(rows_raw):
                cells = " | ".join(f"{h}: {v}" for h, v in zip(headers, rv) if str(v).strip())
                row_content = f"{section_label} | columns: [{', '.join(headers)}] | row {i+1}/{len(rows_raw)} | {cells}"
                rc = Chunk(doc_id=doc_id, source=source, file_name=file_name, file_type=file_type,
                    modality=Modality.TABLE, block_type=BlockType.TABLE_ROW, page=page, reading_order=ro,
                    section_path=section_path[:] if section_path else [f"Page {page}"],
                    content=row_content, token_count=_approx_tokens(row_content), row_index=i, table_number=table_no,
                    links=ChunkLinks(parent_id=table_chunk.id))
                chunks.append(rc); row_ids.append(rc.id)
            table_chunk.links.row_ids = row_ids

        elif block_type == BlockType.IMAGE:
            idata = blk.get("image_data") or {}
            ic = Chunk(doc_id=doc_id, source=source, file_name=file_name, file_type=file_type,
                modality=Modality.IMAGE, block_type=BlockType.IMAGE, page=page, reading_order=ro,
                bbox=BBox(**bbox) if bbox else None, section_path=section_path[:],
                is_raster=not idata.get("is_vector", False))
            ic.__dict__["_pil"] = idata.get("pil_image"); ic.__dict__["_bytes"] = idata.get("image_bytes", b"")
            chunks.append(ic)

    chunks = _adaptive_merge(chunks)
    top = [c for c in chunks if c.block_type != BlockType.TABLE_ROW]
    top.sort(key=lambda c: (c.page, c.reading_order or 0))
    for i, c in enumerate(top):
        c.links.prev_id = top[i-1].id if i > 0 else None
        c.links.next_id = top[i+1].id if i < len(top)-1 else None

    t_n = sum(1 for c in chunks if c.modality == Modality.TEXT)
    i_n = sum(1 for c in chunks if c.modality == Modality.IMAGE)
    b_n = sum(1 for c in chunks if c.block_type == BlockType.TABLE)
    r_n = sum(1 for c in chunks if c.block_type == BlockType.TABLE_ROW)
    logger.info("[Chunker] DONE | text=%d img=%d table=%d rows=%d | %.2fs", t_n, i_n, b_n, r_n, time.time() - t0)
    return chunks

def _adaptive_merge(chunks: list[Chunk]) -> list[Chunk]:
    if not chunks: return chunks
    merged, buf = [], chunks[0]
    for curr in chunks[1:]:
        if (buf.token_count < CONFIG.MIN_CHUNK_TOKENS and buf.section_path == curr.section_path
                and buf.page == curr.page and buf.block_type == BlockType.PARAGRAPH and curr.block_type == BlockType.PARAGRAPH):
            buf.content = buf.content.rstrip() + " " + curr.content.lstrip(); buf.token_count = _approx_tokens(buf.content)
        else: merged.append(buf); buf = curr
    merged.append(buf); return merged


# ══════════════════════════════════════════════════════════════════════════════
# 9.  IMAGE ENRICHMENT
# ══════════════════════════════════════════════════════════════════════════════

def _batch_ocr(paths: list[str]) -> list[tuple[Optional[str], Optional[float]]]:
    ocr = _get_ocr()
    if ocr is None: return [(None, None)] * len(paths)
    results = []
    for i in range(0, len(paths), CONFIG.OCR_BATCH_SIZE):
        for path in paths[i:i+CONFIG.OCR_BATCH_SIZE]:
            try:
                raw = ocr.ocr(path, cls=True); texts, confs = [], []
                if raw and raw[0]:
                    for line in raw[0]: _, (txt, conf) = line; texts.append(txt); confs.append(float(conf))
                results.append((" ".join(texts) or None, round(sum(confs)/len(confs), 4) if confs else None))
            except Exception as exc: logger.warning("[OCR] %s: %s", path, exc); results.append((None, None))
    return results

def _blip_batch(pil_images: list[Any]) -> list[Optional[str]]:
    model, processor = _get_blip()
    if model is None: return [None] * len(pil_images)
    import torch; captions = []
    for pil in pil_images:
        if pil is None: captions.append(None); continue
        try:
            inputs = processor(images=pil, return_tensors="pt")
            with torch.no_grad(): out = model.generate(**inputs, max_new_tokens=CONFIG.BLIP_MAX_TOKENS)
            captions.append(processor.decode(out[0], skip_special_tokens=True))
        except Exception as exc: logger.debug("[BLIP] %s", exc); captions.append(None)
    return captions

def _classify_image(text: str, aspect: Optional[float]) -> ImageType:
    t = (text or "").lower(); num_r = sum(c.isdigit() for c in t) / max(len(t), 1)
    if num_r > 0.15 and any(k in t for k in ["%", "axis", "legend", "total"]): return ImageType.CHART
    if aspect and aspect > 2.5: return ImageType.SCREENSHOT
    return ImageType.RASTER

def enrich_images(chunks: list[Chunk]) -> list[Chunk]:
    image_chunks = [c for c in chunks if c.modality == Modality.IMAGE]
    if not image_chunks: return chunks
    os.makedirs(CONFIG.IMAGE_SAVE_DIR, exist_ok=True); rasters, vectors = [], []
    for idx, c in enumerate(image_chunks):
        pil = c.__dict__.pop("_pil", None); byt = c.__dict__.pop("_bytes", b"")
        path = Path(CONFIG.IMAGE_SAVE_DIR) / f"{c.doc_id}_p{c.page}_{'r' if c.is_raster else 'v'}{idx}_{c.id[:8]}.png"
        if pil is not None:
            try: pil.save(str(path)); c.image_path = str(path)
            except Exception as exc: logger.warning("[ImgSvc] Save: %s", exc)
        if c.is_raster and pil is not None:
            w, h = pil.size; mode = getattr(pil, "mode", "RGB"); ch = len(mode) if mode not in ("L", "P") else 1
            c.raster_meta = RasterMeta(width_px=w, height_px=h, channels=ch,
                dpi=pil.info.get("dpi") and float(pil.info["dpi"][0]),
                aspect_ratio=round(w/h, 4) if h else None, file_size_bytes=len(byt), pixel_array_shape=[h, w, ch])
            c.__dict__["_pil_keep"] = pil; rasters.append(c)
        elif not c.is_raster:
            if c.bbox:
                b = c.bbox; c.vector_meta = VectorMeta(centroid_x=b.x_center, centroid_y=b.y_center,
                    area=b.area, bbox_width=b.width, bbox_height=b.height)
            c.__dict__["_pil_keep"] = pil; vectors.append(c)
    if rasters:
        for c, (derived, conf) in zip(rasters, _batch_ocr([c.image_path for c in rasters if c.image_path])):
            c.derived_text = derived; c.ocr_confidence = conf
            c.image_type = _classify_image(derived or "", getattr(c.raster_meta, "aspect_ratio", None)); c.search_text = derived
    if vectors:
        pils = [c.__dict__.pop("_pil_keep", None) for c in vectors]
        for c, cap in zip(vectors, _blip_batch(pils)):
            if c.vector_meta: c.vector_meta.caption_generated = cap
            c.derived_text = cap; c.search_text = cap; c.image_type = ImageType.VECTOR
    for c in image_chunks: c.__dict__.pop("_pil_keep", None)
    logger.info("[ImgSvc] enriched %d images", len(image_chunks)); return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 10. CAPTION LINKER
# ══════════════════════════════════════════════════════════════════════════════

def _score_cap(target: Chunk, cap: Chunk, max_ro: int) -> Optional[float]:
    tb, cb = target.bbox, cap.bbox; ph = (tb.page_height or 842.0) if tb else 842.0
    if tb and cb:
        vert = abs(tb.y_center - cb.y_center)
        if vert > CONFIG.CAPTION_VERT_MULT * max(cb.height, 10.0): return None
        vert_n = vert / ph; overlap = max(0.0, min(tb.x1, cb.x1) - max(tb.x0, cb.x0))
        horiz = 1.0 - overlap / max(tb.width, 1.0)
    else: vert_n = 1.0; horiz = 0.0
    ro_n = abs((target.reading_order or 0) - (cap.reading_order or 0)) / max(max_ro, 1)
    return CONFIG.CAPTION_W1 * vert_n + CONFIG.CAPTION_W2 * horiz + CONFIG.CAPTION_W3 * ro_n

def link_captions(chunks: list[Chunk]) -> list[Chunk]:
    captions = [c for c in chunks if c.block_type == BlockType.CAPTION]
    images = [c for c in chunks if c.modality == Modality.IMAGE]
    tables = [c for c in chunks if c.block_type == BlockType.TABLE]; claimed: set[str] = set()
    def _best(target):
        same = [c for c in captions if c.page == target.page and c.id not in claimed]
        if not same: return None
        max_ro = max((c.reading_order or 0) for c in same) or 1
        scored = [(s, c) for c in same if (s := _score_cap(target, c, max_ro)) is not None]
        if not scored: return None
        scored.sort(key=lambda x: x[0]); return scored[0][1]
    for img in images:
        cap = _best(img)
        if cap:
            img.links.caption_id = cap.id; cap.links.image_id = img.id
            fm = _FIG_RE.search(cap.content or "")
            if fm: img.links.figure_number = cap.links.figure_number = int(fm.group(1))
            claimed.add(cap.id)
    for tbl in tables:
        cap = _best(tbl)
        if cap:
            tbl.links.caption_id = cap.id; cap.links.table_id = tbl.id
            tm = _TAB_RE.search(cap.content or "")
            if tm: tbl.links.table_number = cap.links.table_number = int(tm.group(1))
            claimed.add(cap.id)
    logger.info("[CaptionLinker] linked %d captions", len(claimed)); return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 11. EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

def _build_embed_text(chunk: Chunk) -> str:
    section = " > ".join(chunk.section_path) if chunk.section_path else ""
    if chunk.block_type == BlockType.TABLE_ROW: return chunk.content
    if chunk.block_type == BlockType.TABLE:
        return f"TABLE | {section} | {chunk.content[:1500]}" if section else f"TABLE | {chunk.content[:1500]}"
    if chunk.modality == Modality.IMAGE:
        img_text = chunk.search_text or chunk.derived_text or f"image {chunk.image_type.value}"
        return f"IMAGE | {section} | {img_text}" if section else f"IMAGE | {img_text}"
    bt = chunk.block_type.value
    return f"{section} | {bt} | {chunk.content}" if section else f"{bt} | {chunk.content}"

def embed_passages(texts: list[str]) -> list[list[float]]:
    if not texts: return []; return _get_embed_model()._encode(texts, input_type="passage")
def embed_query(query: str) -> list[float]:
    return _get_embed_model()._encode([query], input_type="query")[0]

def _build_sparse_vectors(texts: list[str]) -> list[dict[int, float]]:
    result = []
    for text in texts:
        words = text.lower().split(); tf: dict[int, float] = {}
        for w in words:
            w = re.sub(r"[^\w]", "", w)
            if not w or len(w) < 2: continue
            tf[hash(w) % (2**20)] = tf.get(hash(w) % (2**20), 0.0) + 1.0
        for i in range(len(words)-1):
            w1 = re.sub(r"[^\w]", "", words[i]); w2 = re.sub(r"[^\w]", "", words[i+1])
            if w1 and w2: tid = hash(f"{w1}_{w2}") % (2**20); tf[tid] = tf.get(tid, 0.0) + 0.5
        total = max(sum(tf.values()), 1.0); result.append({k: v/total for k, v in tf.items()})
    return result
def _sparse_query(query: str) -> dict[int, float]: return _build_sparse_vectors([query])[0]


# ══════════════════════════════════════════════════════════════════════════════
# 12. MILVUS STORAGE
# ══════════════════════════════════════════════════════════════════════════════

def _to_milvus_row(chunk: Chunk, vec: list[float], sparse: dict[int, float]) -> dict[str, Any]:
    return {"id": chunk.id, "doc_id": chunk.doc_id, "source": chunk.source,
        "file_name": chunk.file_name, "file_type": chunk.file_type,
        "modality": chunk.modality.value, "block_type": chunk.block_type.value,
        "page": int(chunk.page), "reading_order": int(chunk.reading_order or 0),
        "bbox_json": json.dumps(chunk.bbox.model_dump()) if chunk.bbox else "{}",
        "section_path_json": json.dumps(chunk.section_path), "section_depth": len(chunk.section_path),
        "content_text": chunk.content[:8000], "token_count": int(chunk.token_count),
        "row_index": int(chunk.row_index), "image_path": chunk.image_path or "",
        "image_type": chunk.image_type.value, "ocr_confidence": float(chunk.ocr_confidence or 0.0),
        "figure_number": int(chunk.figure_number or 0), "table_number": int(chunk.table_number or 0),
        "links_json": json.dumps(chunk.links.model_dump()), "ingested_at": chunk.ingested_at,
        "doc_created_at": chunk.doc_created_at or "", "embedding": vec, "sparse_embedding": sparse}

def embed_and_store(chunks: list[Chunk]) -> None:
    client = _get_milvus(); t0 = time.time()
    raw_embed = [_build_embed_text(c) for c in chunks]
    valid_idx = [i for i, t in enumerate(raw_embed) if t and t.strip()]
    skipped = len(raw_embed) - len(valid_idx)
    if skipped: logger.warning("[Embed] Skipping %d chunks (no text)", skipped)
    if not valid_idx: return
    valid_texts = [raw_embed[i] for i in valid_idx]
    vecs = embed_passages(valid_texts); sparses = _build_sparse_vectors(valid_texts); ts = _now(); rows = []
    for list_pos, orig_idx in enumerate(valid_idx):
        c = chunks[orig_idx]; c.embed_model = CONFIG.EMBED_MODEL_ID; c.embed_at = ts
        rows.append(_to_milvus_row(c, vecs[list_pos], sparses[list_pos]))
    client.insert(collection_name=CONFIG.MILVUS_COLLECTION, data=rows)
    logger.info("[Embed+Store] n=%d | %.2fs", len(rows), time.time() - t0)


# ══════════════════════════════════════════════════════════════════════════════
# 13. QUERY INTENT
# ══════════════════════════════════════════════════════════════════════════════

_SUMMARY_PAT    = re.compile(r"\b(summary|summarise|summarize|overview|overall|list all|show all|full report)\b", re.I)
_COMPARISON_PAT = re.compile(r"\b(compare|versus|vs|difference|between|higher|lower|highest|lowest|most|least)\b", re.I)
_TABLE_PAT      = re.compile(r"\b(which row|how many rows|table|column|price of|cost of|quantity|find.*row|specific item)\b", re.I)

def detect_query_intent(query: str) -> QueryIntent:
    q = query.lower()
    if _SUMMARY_PAT.search(q):    return QueryIntent.SUMMARY
    if _COMPARISON_PAT.search(q): return QueryIntent.COMPARISON
    if _TABLE_PAT.search(q):      return QueryIntent.TABLE_LOOKUP
    return QueryIntent.GENERAL

_INTENT_TOP_K = {QueryIntent.SUMMARY.value: 20, QueryIntent.COMPARISON.value: 15,
                 QueryIntent.TABLE_LOOKUP.value: 15, QueryIntent.GENERAL.value: 10}

def _intent_to_filter(intent: QueryIntent) -> Optional[str]:
    tr = BlockType.TABLE_ROW.value; tb = BlockType.TABLE.value
    if intent in (QueryIntent.TABLE_LOOKUP, QueryIntent.COMPARISON):
        return f'block_type in ["{tr}", "{tb}"]'
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 14. HYBRID SEARCH + CONTEXT EXPANSION
# ══════════════════════════════════════════════════════════════════════════════

_OUTPUT_FIELDS = ["id", "doc_id", "source", "file_name", "modality", "block_type",
    "page", "reading_order", "bbox_json", "section_path_json", "content_text",
    "token_count", "figure_number", "table_number", "row_index",
    "image_path", "image_type", "ocr_confidence", "links_json", "ingested_at", "doc_created_at"]

def _expand_with_parent_tables(retrieved: list[dict], client: Any) -> list[dict]:
    seen_ids = {c.get("id", "") for c in retrieved}; parent_ids: set[str] = set()
    for cand in retrieved:
        if cand.get("block_type") == BlockType.TABLE_ROW.value:
            links = json.loads(cand.get("links_json", "{}")); pid = links.get("parent_id")
            if pid and pid not in seen_ids: parent_ids.add(pid)
    if not parent_ids: return retrieved
    try:
        id_filter = " or ".join(f'id == "{pid}"' for pid in parent_ids)
        parents = client.query(collection_name=CONFIG.MILVUS_COLLECTION, filter=id_filter,
                               output_fields=_OUTPUT_FIELDS, limit=len(parent_ids))
        for p in parents:
            p = dict(p) if not isinstance(p, dict) else p; p["_rerank_score"] = 999.0; retrieved.append(p)
    except Exception as exc: logger.warning("[ContextExpand] %s", exc)
    return retrieved

def hybrid_search(query: str, doc_id: Optional[str] = None, page: Optional[int] = None,
    modality: Optional[str] = None, block_type: Optional[str] = None,
    figure_number: Optional[int] = None, table_number: Optional[int] = None,
    top_k_retrieve: int = None, top_k_return: int = None) -> list[RetrievedChunk]:
    from pymilvus import AnnSearchRequest, RRFRanker
    intent = detect_query_intent(query)
    top_k_retrieve = top_k_retrieve or CONFIG.RERANKER_TOP_K_IN
    top_k_return = top_k_return or _INTENT_TOP_K.get(intent.value, CONFIG.RERANKER_TOP_K_OUT)
    client = _get_milvus(); intent_fexpr = _intent_to_filter(intent)
    logger.info("[Search] intent=%s | top_k=%d | query='%s'", intent.value, top_k_return, query[:60])

    filters: list[str] = []
    if doc_id:        filters.append(f'doc_id == "{doc_id}"')
    if page:          filters.append(f'page == {page}')
    if modality:      filters.append(f'modality == "{modality}"')
    if block_type:    filters.append(f'block_type == "{block_type}"')
    if figure_number: filters.append(f'figure_number == {figure_number}')
    if table_number:  filters.append(f'table_number == {table_number}')
    if intent_fexpr and not block_type: filters.append(f"({intent_fexpr})")
    fexpr = " and ".join(filters) if filters else None

    try: q_dense = embed_query(query)
    except RuntimeError as e:
        if "auth" in str(e).lower(): raise HTTPException(503, str(e)) from e
        raise
    q_sparse = _sparse_query(query)

    candidates: list[dict] = []
    try:
        dense_req = AnnSearchRequest(data=[q_dense], anns_field="embedding",
            param={"metric_type": CONFIG.MILVUS_METRIC, "params": {"ef": 200}}, limit=top_k_retrieve, expr=fexpr)
        sparse_req = AnnSearchRequest(data=[q_sparse], anns_field="sparse_embedding",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}}, limit=top_k_retrieve, expr=fexpr)
        res = client.hybrid_search(collection_name=CONFIG.MILVUS_COLLECTION,
            reqs=[dense_req, sparse_req], ranker=RRFRanker(k=60), limit=top_k_retrieve, output_fields=_OUTPUT_FIELDS)
        for hit in res[0]:
            entity = hit.entity if hasattr(hit, "entity") else hit
            d = dict(entity) if not isinstance(entity, dict) else entity; d["_rrf_score"] = getattr(hit, "distance", 0.0)
            candidates.append(d)
    except Exception as exc: logger.warning("[Search] ANN failed: %s", exc)
    if not candidates: return []

    seen: set[str] = set(); unique: list[dict] = []
    for c in candidates:
        cid = c.get("id", "")
        if cid not in seen: seen.add(cid); unique.append(c)
    passages = [c.get("content_text", "") for c in unique]
    scores = _nvidia_rerank(query, passages)
    if scores is None:
        for c in unique: c["_rerank_score"] = c.get("_rrf_score", 0.0)
    else:
        for c, s in zip(unique, scores): c["_rerank_score"] = s
    unique.sort(key=lambda x: x["_rerank_score"], reverse=True)
    top_results = unique[:top_k_return]

    if intent in (QueryIntent.TABLE_LOOKUP, QueryIntent.COMPARISON):
        top_results = _expand_with_parent_tables(top_results, client)

    retrieved: list[RetrievedChunk] = []
    for cand in top_results:
        try:
            retrieved.append(RetrievedChunk(chunk_id=cand.get("id", ""),
                score=cand.get("_rerank_score", cand.get("_rrf_score", 0.0)),
                modality=cand.get("modality", ""), block_type=cand.get("block_type", ""),
                page=int(cand.get("page", 0)), section_path=json.loads(cand.get("section_path_json", "[]")),
                content=cand.get("content_text", ""), figure_number=cand.get("figure_number") or None,
                table_number=cand.get("table_number") or None, image_path=cand.get("image_path") or None,
                image_type=cand.get("image_type") or None, ocr_confidence=cand.get("ocr_confidence") or None,
                links=json.loads(cand.get("links_json", "{}")), doc_id=cand.get("doc_id", ""),
                source=cand.get("source", ""), ingested_at=cand.get("ingested_at", ""),
                bbox=json.loads(cand.get("bbox_json", "{}")) or None,
                reading_order=cand.get("reading_order"), token_count=cand.get("token_count", 0),
                row_index=cand.get("row_index", -1)))
        except Exception as exc: logger.warning("[Search] Build error: %s", exc)
    retrieved.sort(key=lambda r: (0 if r.block_type == "table" else 1, -r.score))
    logger.info("[Search] intent=%s | cands=%d → returned=%d", intent.value, len(unique), len(retrieved))
    return retrieved


# ══════════════════════════════════════════════════════════════════════════════
# 15. LLM — general purpose grounded QA
# ══════════════════════════════════════════════════════════════════════════════

_GROUNDED_SYSTEM = """You are a precise document question-answering assistant.

RULES:
1. Answer ONLY from the provided context chunks. Never use outside knowledge.
2. Cite every claim with [page=N | id=CHUNK_ID].
3. You receive two types of table context:
   - TABLE chunks: full table structure with all rows.
   - TABLE_ROW chunks: individual rows with column headers embedded.
   Use TABLE for structure/aggregation. Use TABLE_ROW for specific lookups.
4. When multiple rows match, list ALL of them.
5. For images, describe what the OCR text or caption reveals.
6. If context is insufficient: "Insufficient evidence in the provided documents."
7. Be precise — use exact values from context, do not round or approximate.
"""

def llm_answer(retrieved: list[RetrievedChunk], query: str) -> str:
    import requests
    if not retrieved: return "Insufficient evidence in the provided documents."
    ctx_parts = []
    for r in retrieved:
        sec = " > ".join(r.section_path) if r.section_path else "—"
        ctx_parts.append(f"[{r.block_type.upper()} | page={r.page} | section={sec} | id={r.chunk_id[:8]}]\n{r.content}")
    payload = {"model": CONFIG.LLM_MODEL, "messages": [
        {"role": "system", "content": _GROUNDED_SYSTEM},
        {"role": "user", "content": "CONTEXT:\n" + "\n\n---\n\n".join(ctx_parts) + f"\n\nQUESTION: {query}"}],
        "max_tokens": CONFIG.LLM_MAX_TOKENS, "temperature": CONFIG.LLM_TEMPERATURE, "top_p": CONFIG.LLM_TOP_P}
    hdrs = {"Authorization": f"Bearer {CONFIG.NVIDIA_API_KEY}", "Content-Type": "application/json"}
    t0 = time.time()
    try:
        resp = requests.post(f"{CONFIG.NVIDIA_BASE_URL}/chat/completions", json=payload, headers=hdrs, timeout=120)
        resp.raise_for_status(); answer = resp.json()["choices"][0]["message"]["content"]
        logger.info("[LLM] Done | %.2fs", time.time() - t0); return answer
    except Exception as exc: logger.error("[LLM] Failed: %s", exc); raise


# ══════════════════════════════════════════════════════════════════════════════
# 16. PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(file_path: str | Path) -> PipelineResult:
    t0 = time.time(); logger.info("══ PIPELINE START | file=%s ══", file_path)
    parsed = parse_document(file_path); chunks = chunk_document(parsed)
    chunks = enrich_images(chunks); chunks = link_captions(chunks); embed_and_store(chunks)
    t_n = sum(1 for c in chunks if c.modality == Modality.TEXT)
    i_n = sum(1 for c in chunks if c.modality == Modality.IMAGE)
    b_n = sum(1 for c in chunks if c.block_type == BlockType.TABLE)
    r_n = sum(1 for c in chunks if c.block_type == BlockType.TABLE_ROW)
    result = PipelineResult(doc_id=parsed["doc_id"], source=parsed["source"],
        total_chunks=len(chunks), text_count=t_n, image_count=i_n, table_count=b_n,
        row_count=r_n, duration_s=round(time.time() - t0, 2))
    logger.info("══ PIPELINE DONE | chunks=%d | %.2fs ══", len(chunks), result.duration_s); return result


# ══════════════════════════════════════════════════════════════════════════════
# 17. FASTAPI
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def _lifespan(app: FastAPI):
    logger.info("Starting pipeline ..."); _ensure_collection(); yield; logger.info("Shutdown.")

app = FastAPI(title="Document Intelligence Pipeline", version="6.0.0",
    description="General-purpose layout-aware multimodal RAG. Any document, any domain.",
    lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class IngestResponse(BaseModel):
    doc_id: str; source: str; total_chunks: int; text_count: int
    image_count: int; table_count: int; row_count: int; duration_s: Optional[float]; completed_at: str

class QueryRequest(BaseModel):
    doc_id: Optional[str] = None; query: str; top_k_retrieve: int = CONFIG.RERANKER_TOP_K_IN
    top_k_return: int = 0; page: Optional[int] = None; modality: Optional[str] = None
    block_type: Optional[str] = None; figure_number: Optional[int] = None; table_number: Optional[int] = None

class QueryResponse(BaseModel):
    doc_id: Optional[str]; query: str; detected_intent: str; answer: str; retrieved: list[RetrievedChunk]

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "version": "6.0.0", "embed_model": CONFIG.EMBED_MODEL_ID,
            "reranker": CONFIG.RERANKER_MODEL_ID, "llm": CONFIG.LLM_MODEL,
            "supported_formats": sorted(SUPPORTED_FORMATS)}

@app.get("/formats", tags=["System"])
async def formats():
    return {"docling": sorted(_DOCLING_FORMATS), "text": sorted(_TEXT_FORMATS),
            "image": sorted(_IMAGE_FORMATS), "all": sorted(SUPPORTED_FORMATS)}

@app.post("/ingest", response_model=IngestResponse, tags=["Pipeline"])
async def ingest(file: UploadFile = File(...)):
    suffix = Path(file.filename or "doc.pdf").suffix.lower()
    if suffix not in SUPPORTED_FORMATS: raise HTTPException(415, f"Unsupported: {suffix}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read()); tmp_path = tmp.name
    doc_id_preview = _file_hash(tmp_path); client = _get_milvus()
    try:
        existing = client.query(collection_name=CONFIG.MILVUS_COLLECTION,
            filter=f'doc_id == "{doc_id_preview}"', output_fields=["id"], limit=1)
        if existing:
            os.unlink(tmp_path)
            try:
                ac = client.query(collection_name=CONFIG.MILVUS_COLLECTION,
                    filter=f'doc_id == "{doc_id_preview}"', output_fields=["block_type", "modality"], limit=10000)
                return IngestResponse(doc_id=doc_id_preview, source=file.filename or "",
                    total_chunks=len(ac), text_count=sum(1 for c in ac if c.get("modality") == "text"),
                    image_count=sum(1 for c in ac if c.get("modality") == "image"),
                    table_count=sum(1 for c in ac if c.get("block_type") == "table"),
                    row_count=sum(1 for c in ac if c.get("block_type") == "table_row"),
                    duration_s=0.0, completed_at=_now())
            except Exception:
                return IngestResponse(doc_id=doc_id_preview, source=file.filename or "",
                    total_chunks=-1, text_count=-1, image_count=-1, table_count=-1, row_count=-1,
                    duration_s=0.0, completed_at=_now())
    except Exception: pass
    try: result = run_pipeline(tmp_path)
    except Exception as exc: logger.error("[API] %s", exc, exc_info=True); raise HTTPException(500, str(exc))
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)
    return IngestResponse(doc_id=result.doc_id, source=result.source, total_chunks=result.total_chunks,
        text_count=result.text_count, image_count=result.image_count, table_count=result.table_count,
        row_count=result.row_count, duration_s=result.duration_s, completed_at=result.completed_at)

@app.post("/query", response_model=QueryResponse, tags=["Retrieval"])
async def query_endpoint(req: QueryRequest):
    doc_id = req.doc_id if req.doc_id not in (None, "", "string") else None
    page = req.page if req.page not in (None, 0) else None
    modality = req.modality if req.modality not in (None, "", "string") else None
    block_type = req.block_type if req.block_type not in (None, "", "string") else None
    intent = detect_query_intent(req.query)
    retrieved = hybrid_search(query=req.query, doc_id=doc_id, page=page, modality=modality,
        block_type=block_type, figure_number=req.figure_number or None, table_number=req.table_number or None,
        top_k_retrieve=req.top_k_retrieve, top_k_return=req.top_k_return or None)
    try: answer = llm_answer(retrieved, req.query)
    except Exception as exc: raise HTTPException(500, str(exc))
    return QueryResponse(doc_id=doc_id, query=req.query, detected_intent=intent.value, answer=answer, retrieved=retrieved)

@app.get("/search", tags=["Retrieval"])
async def search_only(query: str, doc_id: Optional[str] = None, modality: Optional[str] = None,
    block_type: Optional[str] = None, page: Optional[int] = None, top_k: int = 0):
    intent = detect_query_intent(query)
    results = hybrid_search(query=query, doc_id=doc_id, modality=modality, block_type=block_type,
        page=page, top_k_return=top_k or _INTENT_TOP_K.get(intent.value, CONFIG.RERANKER_TOP_K_OUT))
    return {"query": query, "detected_intent": intent.value, "results": [r.model_dump() for r in results]}

@app.delete("/doc/{doc_id}", tags=["Management"])
async def delete_doc(doc_id: str):
    try: _get_milvus().delete(collection_name=CONFIG.MILVUS_COLLECTION, filter=f'doc_id == "{doc_id}"')
    except Exception as exc: raise HTTPException(500, str(exc))
    return {"doc_id": doc_id, "status": "deleted"}

@app.get("/collections", tags=["Management"])
async def collection_stats():
    try: return {CONFIG.MILVUS_COLLECTION: _get_milvus().get_collection_stats(CONFIG.MILVUS_COLLECTION)}
    except Exception: return {CONFIG.MILVUS_COLLECTION: {"error": "not found"}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pipeline:app", host="0.0.0.0", port=8000, reload=True, log_level="info")