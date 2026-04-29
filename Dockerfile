from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import math
import os
import re
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TypedDict

import requests
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.FileHandler("enterprise_rag.log", mode="a")],
)
logger = logging.getLogger("enterprise_rag")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
logger.addHandler(console_handler)


# ── Terminal colors ───────────────────────────────────────────────────────────

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    GRAY    = "\033[90m"
    MAGENTA = "\033[95m"


def pstatus(msg: str, color: str = C.CYAN):  print(f"  {color}>{C.RESET} {msg}")
def pok(msg: str):                            print(f"  {C.GREEN}OK{C.RESET}  {msg}")
def perr(msg: str):                           print(f"  {C.RED}ERR{C.RESET} {msg}")
def psec(title: str):
    print(f"\n{C.BOLD}{C.CYAN}── {title} {'─' * max(0, 55 - len(title))}{C.RESET}")


def banner():
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════════╗
║  Enterprise Multimodal RAG Agent                             ║
║  NV-Ingest + LangGraph + VLM Reread + Whisper Audio          ║
║                                                              ║
║  Ingests: PDF · DOCX · PPTX · HTML · TXT                     ║
║           JPEG/PNG/WEBP (direct, no conversion)              ║
║           MP3/WAV/M4A (Whisper transcription)                ║
║                                                              ║
║  Nodes: guardrail → intent_router → query_expander           ║
║         → dual_retriever → cross_reranker → layout_rescue    ║
║         → vlm_reread → evidence_builder → generator          ║
║                                                              ║
║  Fixes: VLM reread (legend/color/spatial queries work)       ║
║         Cross-encoder reranker preserved                     ║
║         Quality gate preserved                               ║
║         Dual Milvus collections (text + image)               ║
╚══════════════════════════════════════════════════════════════╝{C.RESET}
""")


# ── Environment / constants ───────────────────────────────────────────────────

NVIDIA_API_KEY   = os.environ.get("NVIDIA_API_KEY", "")
CHAT_API_BASE    = os.environ.get("NVIDIA_CHAT_API_BASE", "https://integrate.api.nvidia.com")

EMBED_URL        = os.environ.get("EMBED_URL",    f"{CHAT_API_BASE}/v1/embeddings")
EMBED_MODEL      = os.environ.get("EMBED_MODEL",  "nvidia/nv-embedqa-e5-v5")
DIM              = int(os.environ.get("EMBED_DIM", "1024"))

LLM_URL          = os.environ.get("LLM_URL",       f"{CHAT_API_BASE}/v1/chat/completions")
PRIMARY_LLM      = os.environ.get("PRIMARY_LLM",   "meta/llama-3.3-70b-instruct")
FALLBACK_LLM     = os.environ.get("FALLBACK_LLM",  "nvidia/llama-3.1-nemotron-70b-instruct")

CAPTION_URL      = os.environ.get("CAPTION_URL",   f"{CHAT_API_BASE}/v1/chat/completions")
CAPTION_MODEL    = os.environ.get("CAPTION_MODEL", "nvidia/llama-3.1-nemotron-nano-vl-8b-v1")

MILVUS_DB        = os.environ.get("MILVUS_DB",     "./enterprise_rag_milvus.db")
TEXT_COLLECTION  = os.environ.get("TEXT_COLLECTION",  "rag_text_chunks")
IMAGE_COLLECTION = os.environ.get("IMAGE_COLLECTION", "rag_image_patches")

BROKER_HOST      = os.environ.get("BROKER_HOST", "localhost")
BROKER_PORT      = int(os.environ.get("BROKER_PORT", "7671"))

ARTIFACT_ROOT    = Path(os.environ.get("ARTIFACT_ROOT", "./rag_artifacts"))
IMAGE_STORE      = ARTIFACT_ROOT / "images"       # persisted image bytes for VLM reread
AUDIO_STORE      = ARTIFACT_ROOT / "audio"        # persisted audio for reference
CHUNK_MANIFEST   = ARTIFACT_ROOT / "manifest.jsonl"

RETRIEVAL_TOP_K      = int(os.environ.get("RETRIEVAL_TOP_K", "60"))
RERANK_TOP_K         = int(os.environ.get("RERANK_TOP_K",    "20"))
MAX_CONTEXT          = int(os.environ.get("MAX_CONTEXT",     "8"))
VLM_REREAD_TOP_N     = int(os.environ.get("VLM_REREAD_TOP_N", "3"))   # images to re-read at query time
MIN_GENERATION_SCORE = float(os.environ.get("MIN_GENERATION_SCORE", "-10.0"))
CONFIDENCE_HIGH      = float(os.environ.get("CONFIDENCE_HIGH",  "-3.0"))
CONFIDENCE_MED       = float(os.environ.get("CONFIDENCE_MED",   "-8.0"))
MAX_RETRIES          = 1

WHISPER_MODEL    = os.environ.get("WHISPER_MODEL", "base")   # base/small/medium/large
RERANK_MODEL     = os.environ.get("RERANK_MODEL",  "cross-encoder/ms-marco-MiniLM-L-12-v2")

VISUAL_INTENTS = {"image", "chart", "diagram", "infographic", "mixed"}

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4"}
SUPPORTED_DOC_EXTS   = {".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".md"}

BLOCKED_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"forget\s+(all\s+)?previous",
    r"you\s+are\s+now",
    r"system\s*prompt",
    r"<\s*script",
    r"jailbreak",
]

HALLUCINATION_PHRASES = [
    "based on my knowledge",
    "as an ai",
    "i don't have access",
    "based on my training",
    "i cannot access",
    "my knowledge cutoff",
]


INGEST_FILES: List[str] = [
    "Docs/Singapore_NID_F 1.jpeg",
]

# ── Type definitions ──────────────────────────────────────────────────────────

class ChunkRecord(TypedDict, total=False):
    chunk_id:       str
    file_path:      str
    file_name:      str
    modality:       str       # text|table|chart|diagram|image|infographic|caption|audio_transcript
    source_type:    str
    page_num:       int
    parent_id:      str
    figure_id:      str
    title:          str
    text:           str
    ocr_text:       str
    caption_text:   str
    table_text:     str
    chart_text:     str
    layout_text:    str
    color_info:     str       # NEW: color descriptions extracted at ingestion
    spatial_info:   str       # NEW: spatial layout descriptions
    audio_start_s:  float     # NEW: audio timestamp start (seconds)
    audio_end_s:    float     # NEW: audio timestamp end (seconds)
    image_path:     str       # absolute path to stored image file
    bbox:           str
    search_text:    str
    vector:         List[float]


class AgentState(TypedDict, total=False):
    original_query:      str
    current_query:       str
    query_variants:      List[str]
    detected_intent:     str
    has_visual_intent:   bool
    conversation_history: List[Dict[str, str]]
    raw_chunks:          List[Dict[str, Any]]
    ranked_chunks:       List[Dict[str, Any]]
    rescued_chunks:      List[Dict[str, Any]]
    vlm_evidence:        List[Dict[str, Any]]   # NEW: from VLM reread node
    evidence_chunks:     List[Dict[str, Any]]
    overall_confidence:  str
    quality_gate_failed: bool
    answer:              str
    model_used:          str
    fallback_used:       bool
    guardrail_flags:     List[str]
    retry_count:         int
    node_latencies:      Dict[str, float]
    sources:             List[Dict[str, Any]]


# ── Utilities ─────────────────────────────────────────────────────────────────

def _ensure_dirs():
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    IMAGE_STORE.mkdir(parents=True, exist_ok=True)
    AUDIO_STORE.mkdir(parents=True, exist_ok=True)


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type":  "application/json",
    }


def _api_call(url: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    response: Optional[requests.Response] = None
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, headers=_headers(), timeout=timeout)
            if response.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            logger.warning("API attempt %s failed: %s | URL: %s", attempt + 1, exc, url)
            if response is not None:
                logger.warning("Body: %s", response.text[:500])
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
        except (requests.ConnectionError, requests.Timeout) as exc:
            logger.warning("API attempt %s connection error: %s", attempt + 1, exc)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return {}


def _extract_message_text(data: Dict[str, Any]) -> str:
    choice  = (data.get("choices") or [{}])[0]
    message = choice.get("message", {}) or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [item["text"] for item in content if isinstance(item, dict) and isinstance(item.get("text"), str)]
        return "\n".join(parts).strip()
    return ""


def _normalize_vector(v: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _bbox_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(x) for x in value)
    return json.dumps(value, ensure_ascii=True)


def _pick(*values: Any) -> str:
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _sha1(*parts: Any) -> str:
    return hashlib.sha1(
        json.dumps(list(parts), sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()


def _image_to_base64(path: str) -> Optional[str]:
    """Load image from disk and return base64-encoded string."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as exc:
        logger.warning("Failed to load image %s: %s", path, exc)
        return None


def _image_mime(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".webp": "image/webp",
        ".bmp": "image/bmp",  ".tiff": "image/tiff",
    }.get(ext, "image/jpeg")


# ── Embeddings ────────────────────────────────────────────────────────────────

def embed_texts(texts: List[str], input_type: str = "query") -> List[List[float]]:
    """
    Embed texts using NVIDIA API.
    CRITICAL: API has 512-token limit (~400 words).
    Truncates any oversized texts automatically.
    """
    # ── Truncate to ~400 words (≈512 tokens) to stay under API limit ────────
    max_words = 400
    cleaned = []
    
    for t in texts:
        if not t or not t.strip():
            cleaned.append("<empty>")
        else:
            words = t.strip().split()
            if len(words) > max_words:
                logger.warning(f"Text truncated from {len(words)} to {max_words} words for embedding")
                t = " ".join(words[:max_words])
            cleaned.append(t)
    
    data = _api_call(EMBED_URL, {
        "model":           EMBED_MODEL,
        "input":           cleaned,
        "input_type":      input_type,
        "encoding_format": "float",
    })
    return [_normalize_vector(item["embedding"]) for item in data.get("data", [])]


# ── LLM ───────────────────────────────────────────────────────────────────────

def llm_generate(
    prompt: str,
    model: str         = PRIMARY_LLM,
    system_prompt: str = "",
    max_tokens: int    = 1024,
    temperature: float = 0.2,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    data = _api_call(LLM_URL, {
        "model":       model,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
    })
    return _extract_message_text(data)


# ── VLM (Vision Language Model) ───────────────────────────────────────────────

def vlm_describe_image(
    image_path: str,
    question: str = "",
    detail_prompt: str = "",
) -> Dict[str, str]:
    """
    Call the VLM with a specific question about an image.
    Returns dict with: caption, ocr_text, color_info, spatial_info, targeted_answer.
    Works for BOTH ingestion-time rich captioning AND query-time targeted reread.
    """
    b64 = _image_to_base64(image_path)
    if not b64:
        return {"caption": "", "ocr_text": "", "color_info": "", "spatial_info": "", "targeted_answer": ""}

    mime = _image_mime(image_path)

    # Build the prompt depending on context
    if question:
        # Query-time: targeted question about the image
        user_text = (
            f"Look at this image carefully and answer the following question:\n"
            f"Question: {question}\n\n"
            f"Instructions:\n"
            f"- Read all text visible in the image (OCR)\n"
            f"- Identify all colors, labels, legends, arrows, and spatial elements\n"
            f"- Describe what each color/legend represents if applicable\n"
            f"- Answer the question as precisely as possible using visual evidence\n"
            f"- If the image contains a chart, describe what each line/bar/slice represents\n"
            f"- If the image contains an ID card or form, extract all visible field values exactly"
        )
    else:
        # Ingestion-time: rich comprehensive description
        user_text = (
            detail_prompt or
            "Provide a comprehensive analysis of this image. Include:\n"
            "1. FULL OCR: transcribe ALL visible text exactly as it appears\n"
            "2. LAYOUT: describe spatial positions (top-left, center, etc.)\n"
            "3. COLORS: list all significant colors and what they represent (legends, labels)\n"
            "4. STRUCTURE: describe charts/tables/diagrams including axes, legends, data points\n"
            "5. ARROWS/INDICATORS: describe all arrows, pointers, and what they point to\n"
            "6. CONTENT SUMMARY: overall meaning and key information\n"
            "Format: use section headers OCR:, LAYOUT:, COLORS:, STRUCTURE:, ARROWS:, SUMMARY:"
        )

    payload = {
        "model":      CAPTION_MODEL,
        "max_tokens": 1024,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                },
                {"type": "text", "text": user_text},
            ],
        }],
    }

    data = _api_call(CAPTION_URL, payload, timeout=180)
    raw = _extract_message_text(data)

    if question:
        return {
            "caption": "",
            "ocr_text": "",
            "color_info": "",
            "spatial_info": "",
            "targeted_answer": raw,
        }

    # Parse sections from ingestion-time output
    def _extract_section(text: str, tag: str) -> str:
        pattern = rf"{tag}:\s*(.*?)(?=\n[A-Z]+:|$)"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    return {
        "caption":        _extract_section(raw, "SUMMARY") or raw[:500],
        "ocr_text":       _extract_section(raw, "OCR"),
        "color_info":     _extract_section(raw, "COLORS"),
        "spatial_info":   _extract_section(raw, "LAYOUT") + "\n" + _extract_section(raw, "ARROWS"),
        "targeted_answer": "",
        "raw_description": raw,
    }


# ── Cross-encoder reranker ────────────────────────────────────────────────────

_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(RERANK_MODEL)
    return _cross_encoder


def rerank_passages(query: str, passages: List[str]) -> List[Dict[str, Any]]:
    if not passages:
        return []
    ce     = _get_cross_encoder()
    pairs  = [(query, p[:2500]) for p in passages]
    scores = ce.predict(pairs).tolist()
    rows   = [{"index": i, "score": float(s)} for i, s in enumerate(scores)]
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


# ── Whisper audio transcription ───────────────────────────────────────────────

_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            pstatus(f"Loading Whisper model '{WHISPER_MODEL}'...")
            _whisper_model = whisper.load_model(WHISPER_MODEL)
            pok(f"Whisper '{WHISPER_MODEL}' loaded")
        except ImportError:
            raise RuntimeError(
                "openai-whisper not installed. Run: pip install openai-whisper"
            )
    return _whisper_model


def transcribe_audio(audio_path: str) -> List[Dict[str, Any]]:
    """
    Transcribe audio file using Whisper.
    Returns list of segment dicts with: text, start, end.
    """
    model  = _get_whisper()
    result = model.transcribe(audio_path, verbose=False, word_timestamps=False)
    segments = result.get("segments", [])
    if not segments:
        # Fallback: treat entire transcript as one segment
        return [{"text": result.get("text", ""), "start": 0.0, "end": 0.0}]
    return [
        {"text": seg["text"].strip(), "start": seg["start"], "end": seg["end"]}
        for seg in segments
        if seg.get("text", "").strip()
    ]


def _chunk_audio_segments(
    segments:     List[Dict[str, Any]],
    file_path:    str,
    chunk_tokens: int = 512,
    overlap:      int = 1,
) -> List[ChunkRecord]:
    """
    Group Whisper segments into overlapping text chunks for embedding.
    chunk_tokens is approximate (words × 1.3 ≈ tokens).
    """
    file_name = Path(file_path).name
    records: List[ChunkRecord] = []
    i = 0
    while i < len(segments):
        chunk_segs: List[Dict] = []
        word_count = 0
        j = i
        while j < len(segments):
            seg_words = len(segments[j]["text"].split())
            if word_count + seg_words > (chunk_tokens * 0.7) and chunk_segs:
                break
            chunk_segs.append(segments[j])
            word_count += seg_words
            j += 1

        text = " ".join(s["text"] for s in chunk_segs).strip()
        if not text:
            i = j
            continue

        start_s = chunk_segs[0]["start"]
        end_s   = chunk_segs[-1]["end"]

        record: ChunkRecord = {
            "chunk_id":      _sha1(file_path, start_s, text[:80]),
            "file_path":     file_path,
            "file_name":     file_name,
            "modality":      "audio_transcript",
            "source_type":   "audio_transcript",
            "page_num":      0,
            "parent_id":     f"{file_name}:audio",
            "figure_id":     "",
            "title":         f"{file_name} [{start_s:.0f}s-{end_s:.0f}s]",
            "text":          text,
            "ocr_text":      "",
            "caption_text":  "",
            "table_text":    "",
            "chart_text":    "",
            "layout_text":   "",
            "color_info":    "",
            "spatial_info":  "",
            "audio_start_s": start_s,
            "audio_end_s":   end_s,
            "image_path":    "",
            "bbox":          "",
        }
        record["search_text"] = _build_search_text(record)
        records.append(record)

        # overlap: step back by `overlap` segments
        i = max(i + 1, j - overlap)

    return records


# ── HTML / DOCX / PPTX fallback extractors ───────────────────────────────────

def _extract_html(file_path: str) -> List[ChunkRecord]:
    try:
        from bs4 import BeautifulSoup
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "head"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    except ImportError:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = re.sub(r"<[^>]+>", " ", f.read())

    return _text_to_chunks(text, file_path, modality="text")


def _extract_txt(file_path: str) -> List[ChunkRecord]:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return _text_to_chunks(text, file_path, modality="text")


def _extract_docx(file_path: str) -> List[ChunkRecord]:
    try:
        from docx import Document as DocxDocument
        doc    = DocxDocument(file_path)
        chunks = []

        # Paragraphs
        para_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        chunks.extend(_text_to_chunks(para_text, file_path, modality="text"))

        # Tables
        for tbl_idx, table in enumerate(doc.tables):
            rows = []
            for row in table.rows:
                rows.append(" | ".join(c.text.strip() for c in row.cells))
            table_md = "\n".join(rows)
            if table_md.strip():
                rec: ChunkRecord = {
                    "chunk_id":    _sha1(file_path, "table", tbl_idx),
                    "file_path":   file_path,
                    "file_name":   Path(file_path).name,
                    "modality":    "table",
                    "source_type": "table",
                    "page_num":    0,
                    "parent_id":   f"{Path(file_path).name}:table:{tbl_idx}",
                    "figure_id":   "",
                    "title":       f"Table {tbl_idx + 1}",
                    "text":        "",
                    "ocr_text":    "",
                    "caption_text": "",
                    "table_text":  table_md,
                    "chart_text":  "",
                    "layout_text": "",
                    "color_info":  "",
                    "spatial_info": "",
                    "audio_start_s": 0.0,
                    "audio_end_s":   0.0,
                    "image_path":  "",
                    "bbox":        "",
                }
                rec["search_text"] = _build_search_text(rec)
                chunks.append(rec)

        return chunks
    except ImportError:
        pstatus("python-docx not installed, treating DOCX as binary text", C.YELLOW)
        return []


def _extract_pptx(file_path: str) -> List[ChunkRecord]:
    try:
        from pptx import Presentation
        prs    = Presentation(file_path)
        chunks = []
        for slide_idx, slide in enumerate(prs.slides):
            parts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        t = para.text.strip()
                        if t:
                            parts.append(t)
            slide_text = "\n".join(parts)
            if not slide_text.strip():
                continue
            rec: ChunkRecord = {
                "chunk_id":    _sha1(file_path, "slide", slide_idx),
                "file_path":   file_path,
                "file_name":   Path(file_path).name,
                "modality":    "text",
                "source_type": "pptx_slide",
                "page_num":    slide_idx + 1,
                "parent_id":   f"{Path(file_path).name}:slide:{slide_idx}",
                "figure_id":   "",
                "title":       f"Slide {slide_idx + 1}",
                "text":        slide_text,
                "ocr_text":    "",
                "caption_text": "",
                "table_text":  "",
                "chart_text":  "",
                "layout_text": "",
                "color_info":  "",
                "spatial_info": "",
                "audio_start_s": 0.0,
                "audio_end_s":   0.0,
                "image_path":  "",
                "bbox":        "",
            }
            rec["search_text"] = _build_search_text(rec)
            chunks.append(rec)
        return chunks
    except ImportError:
        pstatus("python-pptx not installed, skipping PPTX", C.YELLOW)
        return []


def _text_to_chunks(
    text: str,
    file_path: str,
    modality:  str = "text",
    chunk_size: int = 512,
    overlap:    int = 80,
) -> List[ChunkRecord]:
    """Simple word-based chunker for raw text."""
    words = text.split()
    step  = max(1, chunk_size - overlap)
    file_name = Path(file_path).name
    records: List[ChunkRecord] = []
    for i in range(0, len(words), step):
        chunk_words = words[i: i + chunk_size]
        chunk_text  = " ".join(chunk_words).strip()
        if len(chunk_text) < 20:
            continue
        rec: ChunkRecord = {
            "chunk_id":    _sha1(file_path, i, chunk_text[:60]),
            "file_path":   file_path,
            "file_name":   file_name,
            "modality":    modality,
            "source_type": modality,
            "page_num":    0,
            "parent_id":   f"{file_name}:text",
            "figure_id":   "",
            "title":       "",
            "text":        chunk_text,
            "ocr_text":    "",
            "caption_text": "",
            "table_text":  "",
            "chart_text":  "",
            "layout_text": "",
            "color_info":  "",
            "spatial_info": "",
            "audio_start_s": 0.0,
            "audio_end_s":   0.0,
            "image_path":  "",
            "bbox":        "",
        }
        rec["search_text"] = _build_search_text(rec)
        records.append(rec)
    return records


# ── Chunk record helpers ──────────────────────────────────────────────────────

def _build_search_text(r: ChunkRecord) -> str:
    """
    Build a rich searchable string from all modality fields.
    This is what gets embedded into the text vector index.
    
    CRITICAL: Keep total length under ~400 words (≈512 tokens).
    The NVIDIA embedding API has a 512-token limit.
    For images: prioritize OCR > caption > spatial/color info.
    """
    modality = r.get("modality", "text")
    
    # ── For images: prioritize OCR and caption, skip verbose fields ──────────
    if modality == "image":
        parts = [
            f"type: image",
            f"file: {r.get('title', '')[:60]}",  # Keep filename short
        ]
        
        # OCR is most important for images - include if present
        ocr = r.get("ocr_text", "").strip()
        if ocr:
            # Truncate to 200 words (~260 tokens) to leave room for other fields
            ocr_words = ocr.split()[:200]
            parts.append(f"text: {' '.join(ocr_words)}")
        
        # Caption provides context
        caption = r.get("caption_text", "").strip()
        if caption:
            caption_words = caption.split()[:150]
            parts.append(f"caption: {' '.join(caption_words)}")
        
        # Color and spatial for visual queries (more concise)
        color = r.get("color_info", "").strip()
        if color and len(color) < 200:
            parts.append(f"colors: {color[:150]}")
        
        spatial = r.get("spatial_info", "").strip()
        if spatial and len(spatial) < 200:
            parts.append(f"layout: {spatial[:150]}")
        
        result = "\n".join(parts).strip()
        
    # ── For documents: include all fields but truncate each ──────────────────
    else:
        parts = [
            f"modality: {modality}",
            f"title: {r.get('title', '')[:60]}",  # Short title
        ]
        
        # Main text content (limit to 300 words ≈ 390 tokens)
        text = r.get("text", "").strip()
        if text:
            text_words = text.split()[:300]
            parts.append(f"content: {' '.join(text_words)}")
        
        # OCR (if present, e.g., from scanned docs)
        ocr = r.get("ocr_text", "").strip()
        if ocr:
            ocr_words = ocr.split()[:100]
            parts.append(f"ocr: {' '.join(ocr_words)}")
        
        # Table data (concise)
        table = r.get("table_text", "").strip()
        if table:
            table_words = table.split()[:80]
            parts.append(f"table: {' '.join(table_words)}")
        
        # Chart/diagram (concise)
        chart = r.get("chart_text", "").strip()
        if chart:
            chart_words = chart.split()[:80]
            parts.append(f"chart: {' '.join(chart_words)}")
        
        # Skip caption_text, layout_text, color_info, spatial_info for documents
        # to keep overall length manageable
        
        result = "\n".join(parts).strip()
    
    # ── Final safety check: limit to ~400 words (≈520 tokens) ────────────────
    words = result.split()
    if len(words) > 400:
        pstatus(f"Truncating search_text from {len(words)} words to 400", C.YELLOW)
        result = " ".join(words[:400])
    
    return result


def _normalize_modality(raw: str) -> str:
    v = (raw or "").lower()
    if any(t in v for t in ["table", "markdown_table"]):          return "table"
    if any(t in v for t in ["chart", "graph", "plot"]):           return "chart"
    if any(t in v for t in ["diagram", "figure"]):                return "diagram"
    if any(t in v for t in ["infographic"]):                      return "infographic"
    if any(t in v for t in ["image", "photo", "picture"]):        return "image"
    if any(t in v for t in ["caption"]):                          return "caption"
    if any(t in v for t in ["audio", "transcript", "speech"]):    return "audio_transcript"
    return "text"


def _normalize_nvingest_record(item: Dict[str, Any]) -> Optional[ChunkRecord]:
    """Convert raw NV-Ingest output dict into a canonical ChunkRecord."""
    meta          = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    raw_src       = _pick(item.get("source_type"), item.get("content_type"),
                          meta.get("source_type"), meta.get("content_type"))
    modality      = _normalize_modality(raw_src)
    text          = _pick(item.get("text"), item.get("content"), meta.get("text"))
    caption_text  = _pick(item.get("caption"), item.get("caption_text"), meta.get("caption"))
    ocr_text      = _pick(item.get("ocr_text"), meta.get("ocr_text"))
    table_text    = _pick(item.get("table_text"), item.get("markdown"), meta.get("markdown"))
    chart_text    = _pick(item.get("chart_text"), meta.get("chart_text"))
    layout_text   = _pick(item.get("layout_text"), item.get("surrounding_text"),
                          meta.get("surrounding_text"), meta.get("layout_text"))
    # Color/spatial info may be embedded in caption or layout text from NV-Ingest
    color_info    = _pick(meta.get("color_info"), "")
    spatial_info  = _pick(meta.get("spatial_info"), "")
    title         = _pick(item.get("title"), meta.get("title"), meta.get("section_title"))
    file_path     = _pick(item.get("file_path"), meta.get("file_path"))
    file_name     = Path(file_path).name if file_path else _pick(item.get("file_name"), meta.get("file_name"))
    page_num      = int(item.get("page_num") or meta.get("page_num") or meta.get("page") or 0)
    figure_id     = _pick(item.get("figure_id"), meta.get("figure_id"))
    parent_id     = _pick(item.get("parent_id"), meta.get("parent_id"))
    image_path    = _pick(item.get("image_path"), meta.get("image_path"))
    bbox          = _bbox_to_str(item.get("bbox", meta.get("bbox")))
    vector        = item.get("embedding") or item.get("vector") or meta.get("embedding")

    # If NV-Ingest extracted an image, copy it to our IMAGE_STORE for VLM reread
    if image_path and Path(image_path).exists():
        dest = IMAGE_STORE / Path(image_path).name
        if not dest.exists():
            shutil.copy2(image_path, dest)
        image_path = str(dest)

    rec: ChunkRecord = {
        "chunk_id":    _pick(item.get("chunk_id")) or _sha1(
            file_path, page_num, modality, text[:80], ocr_text[:40], caption_text[:40], bbox),
        "file_path":   file_path,
        "file_name":   file_name,
        "modality":    modality,
        "source_type": raw_src or modality,
        "page_num":    page_num,
        "parent_id":   parent_id or f"{file_name}:page:{page_num}",
        "figure_id":   figure_id,
        "title":       title,
        "text":        text,
        "ocr_text":    ocr_text,
        "caption_text": caption_text,
        "table_text":  table_text,
        "chart_text":  chart_text,
        "layout_text": layout_text,
        "color_info":  color_info,
        "spatial_info": spatial_info,
        "audio_start_s": 0.0,
        "audio_end_s":   0.0,
        "image_path":  image_path,
        "bbox":        bbox,
    }
    rec["search_text"] = _build_search_text(rec)
    if not rec["search_text"]:
        return None
    if isinstance(vector, list) and vector:
        rec["vector"] = _normalize_vector([float(x) for x in vector])
    return rec


# ── Milvus ────────────────────────────────────────────────────────────────────

_milvus_client = None


def get_milvus():
    global _milvus_client
    if _milvus_client is None:
        from pymilvus import MilvusClient
        pstatus(f"Connecting to Milvus: {MILVUS_DB}")
        _milvus_client = MilvusClient(uri=MILVUS_DB)

        # Text collection
        if not _milvus_client.has_collection(TEXT_COLLECTION):
            _milvus_client.create_collection(
                collection_name=TEXT_COLLECTION,
                dimension=DIM,
                metric_type="IP",
                auto_id=True,
                enable_dynamic_field=True,
            )
            pok(f"Created text collection: {TEXT_COLLECTION}")
        else:
            pok(f"Text collection exists: {TEXT_COLLECTION}")

        # Image collection (same embedding dim — we use CLIP-style text-image embedding
        # via the same nvidia embed model with visual captions as the index key)
        if not _milvus_client.has_collection(IMAGE_COLLECTION):
            _milvus_client.create_collection(
                collection_name=IMAGE_COLLECTION,
                dimension=DIM,
                metric_type="IP",
                auto_id=True,
                enable_dynamic_field=True,
            )
            pok(f"Created image collection: {IMAGE_COLLECTION}")
        else:
            pok(f"Image collection exists: {IMAGE_COLLECTION}")

    return _milvus_client


def _close_milvus():
    global _milvus_client
    if _milvus_client is not None:
        try:
            _milvus_client.close()
        except Exception:
            pass
        _milvus_client = None


def reset_collections():
    milvus = get_milvus()
    for col in [TEXT_COLLECTION, IMAGE_COLLECTION]:
        if milvus.has_collection(col):
            milvus.drop_collection(col)
    _ensure_dirs()
    if CHUNK_MANIFEST.exists():
        CHUNK_MANIFEST.unlink()
    get_milvus()  # recreates
    pok("Collections reset")


# ── Manifest (local JSONL cache) ──────────────────────────────────────────────

def _manifest_rows() -> List[ChunkRecord]:
    if not CHUNK_MANIFEST.exists():
        return []
    rows: List[ChunkRecord] = []
    with CHUNK_MANIFEST.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _manifest_map() -> Dict[str, ChunkRecord]:
    return {r["chunk_id"]: r for r in _manifest_rows()}


def _persist_manifest(records: List[ChunkRecord], replace: bool = False):
    existing = {} if replace else _manifest_map()
    for r in records:
        existing[r["chunk_id"]] = r
    with CHUNK_MANIFEST.open("w", encoding="utf-8") as f:
        for row in existing.values():
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


# ── Upsert chunks into Milvus ─────────────────────────────────────────────────

def _upsert_chunks(records: List[ChunkRecord], replace_manifest: bool = False):
    if not records:
        return
    milvus    = get_milvus()
    text_rows: List[Dict[str, Any]] = []
    img_rows:  List[Dict[str, Any]] = []

    for rec in records:
        vector = rec.get("vector")
        if not vector:
            vector = embed_texts([rec["search_text"]], input_type="passage")[0]
            rec["vector"] = vector

        base_payload = {
            "vector":      vector,
            "chunk_id":    rec["chunk_id"],
            "search_text": rec["search_text"],
            "modality":    rec.get("modality", "text"),
            "source_type": rec.get("source_type", "text"),
            "file_path":   rec.get("file_path", ""),
            "file_name":   rec.get("file_name", ""),
            "page_num":    int(rec.get("page_num", 0)),
            "parent_id":   rec.get("parent_id", ""),
            "figure_id":   rec.get("figure_id", ""),
            "title":       rec.get("title", ""),
            "bbox":        rec.get("bbox", ""),
            "image_path":  rec.get("image_path", ""),
            "color_info":  rec.get("color_info", ""),
            "spatial_info": rec.get("spatial_info", ""),
            "audio_start_s": float(rec.get("audio_start_s", 0.0)),
            "audio_end_s":   float(rec.get("audio_end_s",   0.0)),
        }

        text_rows.append(base_payload)

        # Also add to image collection if this is a visual chunk with a path
        if rec.get("image_path") and rec.get("modality") in VISUAL_INTENTS | {"caption"}:
            img_rows.append({**base_payload})

    milvus.insert(collection_name=TEXT_COLLECTION,  data=text_rows)
    if img_rows:
        milvus.insert(collection_name=IMAGE_COLLECTION, data=img_rows)
        pstatus(f"Stored {len(img_rows)} visual chunks in image collection", C.MAGENTA)

    _persist_manifest(records, replace=replace_manifest)


# ── NV-Ingest pipeline ────────────────────────────────────────────────────────

_pipeline_started = False


def _wait_for_broker(host: str = BROKER_HOST, port: int = BROKER_PORT, timeout: int = 120):
    pstatus(f"Waiting for broker {host}:{port}...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        if s.connect_ex((host, port)) == 0:
            s.close()
            pok("Broker ready")
            return
        s.close()
        time.sleep(0.5)
    raise RuntimeError(f"Broker not reachable after {timeout}s")


def _start_pipeline_once():
    global _pipeline_started
    if _pipeline_started:
        return
    pstatus("Importing NV-Ingest...")
    t0 = time.time()
    from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import (
        PipelineCreationSchema,
        run_pipeline,
    )
    pok(f"NV-Ingest imported ({time.time() - t0:.1f}s)")
    cfg = PipelineCreationSchema()
    run_pipeline(cfg, block=False, disable_dynamic_scaling=True, run_in_subprocess=True)
    _wait_for_broker()
    _pipeline_started = True
    pok(f"Pipeline ready ({time.time() - t0:.1f}s)")


def _run_nvingest(file_paths: List[str]) -> List[ChunkRecord]:
    """Run NV-Ingest on document files and return normalized ChunkRecords."""
    from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
    from nv_ingest_client.client import Ingestor, NvIngestClient

    _close_milvus()

    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=BROKER_PORT,
        message_client_hostname=BROKER_HOST,
    )

    ingestor = (
        Ingestor(client=client)
        .files(file_paths)
        .load()
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=True,           # images extracted + image_path set
            extract_infographics=True,
            table_output_format="markdown",
            text_depth="block",            # block-level = preserves spatial layout
        )
        .split(
            tokenizer="meta-llama/Llama-3.2-1B",
            chunk_size=512,
            chunk_overlap=80,
            params={
                "split_source_types": [
                    "text", "table", "chart", "image",
                    "infographic", "caption"
                ],
            },
        )
        .caption(
            endpoint_url=CAPTION_URL,
            model_name=CAPTION_MODEL,
            api_key=NVIDIA_API_KEY,
        )
        # NOTE: we do NOT use .embed() or .vdb_upload() here.
        # We manually embed+store so we can dual-index into text AND image collections,
        # and preserve all metadata fields that vdb_upload silently drops.
    )

    results, failures = ingestor.ingest(show_progress=True, return_failures=True)
    raw = list(results)

    records: List[ChunkRecord] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        rec = _normalize_nvingest_record(item)
        if rec is not None:
            records.append(rec)

    if not records:
        raise RuntimeError(
            "NV-Ingest returned no usable records. "
            "Check pipeline config and _normalize_nvingest_record()."
        )

    n_fail = len(failures) if failures else 0
    if n_fail:
        pstatus(f"NV-Ingest failures: {n_fail}", C.YELLOW)

    return records


# ── Image ingestion (direct — no PDF conversion) ──────────────────────────────

def _ingest_image_direct(file_path: str) -> List[ChunkRecord]:
    """
    Ingest a JPEG/PNG/WEBP directly via the VLM.
    Produces a rich ChunkRecord with: full_caption, ocr_text, color_info, spatial_info.
    The actual image is copied to IMAGE_STORE for VLM reread at query time.
    NO PDF conversion — this is the critical fix.
    """
    _ensure_dirs()
    file_name = Path(file_path).name

    try:
        # Verify file exists and is readable
        if not os.path.isfile(file_path):
            perr(f"Image file not found: {file_path}")
            return []
        
        file_size = os.path.getsize(file_path)
        pstatus(f"VLM captioning image: {file_name} ({file_size/1024:.1f} KB)", C.MAGENTA)

        # Copy image to persistent store
        dest = IMAGE_STORE / file_name
        if not dest.exists():
            shutil.copy2(file_path, dest)
            pstatus(f"  Copied to: {dest}", C.GRAY)
        stored_path = str(dest)

        info = vlm_describe_image(stored_path)
        pok(f"VLM response received for {file_name}")

        if not any([info.get("caption"), info.get("ocr_text"), info.get("raw_description")]):
            pstatus(f"VLM returned empty for {file_name}, skipping", C.YELLOW)
            logger.warning(f"VLM empty response for {file_name}")
            return []

        # Build a comprehensive search text from all VLM output fields
        full_description = info.get("raw_description", "")
        caption          = info.get("caption", "")
        ocr_text         = info.get("ocr_text", "")
        color_info       = info.get("color_info", "")
        spatial_info     = info.get("spatial_info", "")

        rec: ChunkRecord = {
            "chunk_id":    _sha1(file_path, "direct_image", file_name),
            "file_path":   file_path,
            "file_name":   file_name,
            "modality":    "image",
            "source_type": "direct_image",
            "page_num":    0,
            "parent_id":   f"{file_name}:image",
            "figure_id":   "",
            "title":       file_name,
            "text":        full_description,
            "ocr_text":    ocr_text,
            "caption_text": caption,
            "table_text":  "",
            "chart_text":  "",
            "layout_text": spatial_info,
            "color_info":  color_info,
            "spatial_info": spatial_info,
            "audio_start_s": 0.0,
            "audio_end_s":   0.0,
            "image_path":  stored_path,   # CRITICAL: preserved for VLM reread at query time
            "bbox":        "",
        }
        rec["search_text"] = _build_search_text(rec)
        pok(f"Image chunk created: {rec['chunk_id'][:16]}...")
        return [rec]
    
    except Exception as e:
        perr(f"Error ingesting image {file_path}: {e}")
        logger.exception(f"Image ingestion error for {file_path}")
        return []


# ── Audio ingestion ───────────────────────────────────────────────────────────

def _ingest_audio(file_path: str) -> List[ChunkRecord]:
    """Transcribe audio via Whisper and chunk the transcript."""
    _ensure_dirs()
    file_name = Path(file_path).name

    # Copy to audio store
    dest = AUDIO_STORE / file_name
    if not dest.exists():
        shutil.copy2(file_path, dest)

    pstatus(f"Transcribing audio: {file_name}", C.MAGENTA)
    segments = transcribe_audio(file_path)
    pok(f"Transcribed {len(segments)} segments from {file_name}")

    return _chunk_audio_segments(segments, file_path)


# ── Main ingest dispatcher ────────────────────────────────────────────────────

def run_ingest(file_paths: List[str], reset: bool = False) -> Dict[str, Any]:
    _ensure_dirs()

    if reset:
        reset_collections()

    # ── Validate all files exist before starting ──────────────────────────────
    pstatus(f"Validating {len(file_paths)} file(s)...")
    missing = []
    for p in file_paths:
        if not os.path.isfile(p):
            missing.append(p)
            perr(f"Not found: {p}")
        else:
            size_kb = os.path.getsize(p) / 1024
            pok(f"Found: {Path(p).name} ({size_kb:.1f} KB)")
    
    if missing:
        perr(f"{len(missing)} file(s) not found. Cannot proceed.")
        return {
            "files": [],
            "chunks_ingested": 0,
            "elapsed_ms": 0,
            "modalities": {},
            "image_files": 0,
            "audio_files": 0,
            "doc_files": 0,
        }

    # Separate files by type
    image_files = [p for p in file_paths if Path(p).suffix.lower() in SUPPORTED_IMAGE_EXTS]
    audio_files = [p for p in file_paths if Path(p).suffix.lower() in SUPPORTED_AUDIO_EXTS]
    doc_files   = [p for p in file_paths
                   if Path(p).suffix.lower() in SUPPORTED_DOC_EXTS
                   and Path(p).suffix.lower() not in {".html", ".htm", ".txt", ".md"}]
    html_files  = [p for p in file_paths if Path(p).suffix.lower() in {".html", ".htm"}]
    txt_files   = [p for p in file_paths if Path(p).suffix.lower() in {".txt", ".md"}]
    docx_files  = [p for p in file_paths if Path(p).suffix.lower() == ".docx"]
    pptx_files  = [p for p in file_paths if Path(p).suffix.lower() == ".pptx"]

    pstatus(f"File types: {len(doc_files)} docs, {len(image_files)} images, {len(audio_files)} audio, {len(html_files)} html, {len(txt_files)} text", C.GRAY)

    # Determine which docs go to NV-Ingest vs fallback extractors
    nvingest_files = [p for p in doc_files
                      if Path(p).suffix.lower() in {".pdf"}]
    # DOCX/PPTX: try NV-Ingest first, fall back to python-docx/pptx
    nvingest_files += [p for p in doc_files
                       if Path(p).suffix.lower() in {".docx", ".pptx"}]

    all_records: List[ChunkRecord] = []
    t0 = time.time()

    # NV-Ingest path
    if nvingest_files:
        psec("NV-Ingest (PDF/DOCX/PPTX)")
        _start_pipeline_once()
        for p in nvingest_files:
            pstatus(f"  -> {Path(p).name} ({os.path.getsize(p)/1024:.0f} KB)", C.GRAY)
        records = _run_nvingest(nvingest_files)
        all_records.extend(records)
        pok(f"NV-Ingest: {len(records)} chunks")

    # Fallback DOCX (if not via NV-Ingest or NV-Ingest not available)
    for p in [x for x in docx_files if x not in nvingest_files]:
        psec(f"DOCX fallback: {Path(p).name}")
        all_records.extend(_extract_docx(p))

    # Fallback PPTX
    for p in [x for x in pptx_files if x not in nvingest_files]:
        psec(f"PPTX fallback: {Path(p).name}")
        all_records.extend(_extract_pptx(p))

    # HTML
    for p in html_files:
        psec(f"HTML: {Path(p).name}")
        all_records.extend(_extract_html(p))

    # TXT/MD
    for p in txt_files:
        psec(f"TXT: {Path(p).name}")
        all_records.extend(_extract_txt(p))

    # Images (CRITICAL: direct VLM captioning, no PDF conversion)
    if image_files:
        psec("Direct Image VLM Captioning")
        for p in image_files:
            pstatus(f"Processing image: {Path(p).name}", C.CYAN)
            try:
                records = _ingest_image_direct(p)
                all_records.extend(records)
                pok(f"Image processed: {len(records)} chunk(s)")
            except Exception as e:
                perr(f"Failed to ingest image {Path(p).name}: {e}")
                logger.exception(f"Image ingestion error: {p}")

    # Audio
    if audio_files:
        psec("Audio Transcription (Whisper)")
        for p in audio_files:
            pstatus(f"Processing audio: {Path(p).name}", C.CYAN)
            try:
                records = _ingest_audio(p)
                all_records.extend(records)
                pok(f"Audio processed: {len(records)} chunk(s)")
            except Exception as e:
                perr(f"Failed to ingest audio {Path(p).name}: {e}")
                logger.exception(f"Audio ingestion error: {p}")

    # Embed and store everything
    if all_records:
        psec("Embedding + storing in Milvus")
        _upsert_chunks(all_records, replace_manifest=reset)

    elapsed = round((time.time() - t0) * 1000)
    modality_counts: Dict[str, int] = {}
    for r in all_records:
        modality_counts[r["modality"]] = modality_counts.get(r["modality"], 0) + 1

    info = {
        "files":            [Path(p).name for p in file_paths],
        "chunks_ingested":  len(all_records),
        "elapsed_ms":       elapsed,
        "modalities":       modality_counts,
        "image_files":      len(image_files),
        "audio_files":      len(audio_files),
        "doc_files":        len(nvingest_files) + len(html_files) + len(txt_files),
    }
    pok(f"Total: {len(all_records)} chunks | {elapsed:,}ms | modalities: {modality_counts}")
    return info


# ── Intent detection ──────────────────────────────────────────────────────────

def _detect_intent(query: str) -> tuple[str, bool]:
    """Returns (intent_label, has_visual_intent)."""
    q = query.lower()
    if any(w in q for w in ["table", "row", "column", "cell", "spreadsheet", "grid", "list of"]):
        return "table", False
    if any(w in q for w in ["chart", "graph", "legend", "axis", "bar", "line", "plot", "slice", "series"]):
        return "chart", True
    if any(w in q for w in ["diagram", "flow", "arrow", "box", "layout", "region", "process"]):
        return "diagram", True
    if any(w in q for w in ["infographic", "poster", "visual summary"]):
        return "infographic", True
    if any(w in q for w in ["image", "photo", "picture", "logo", "seal", "barcode",
                             "fingerprint", "card", "id", "document", "scan",
                             "color", "colour", "red", "blue", "green", "yellow",
                             "what does", "what is shown", "what can you see",
                             "identify", "describe the"]):
        return "image", True
    if any(w in q for w in ["audio", "transcript", "said", "recorded", "spoken", "voice", "speech"]):
        return "audio", False
    return "text", False


# ── Agent nodes ───────────────────────────────────────────────────────────────

def node_guardrail(state: AgentState) -> Dict[str, Any]:
    t0    = time.time()
    query = state["current_query"]
    flags = list(state.get("guardrail_flags", []))

    if not query.strip():
        flags.append("empty_query")
        return {"guardrail_flags": flags,
                "node_latencies": {**state.get("node_latencies", {}),
                                   "guardrail": round((time.time()-t0)*1000, 1)}}

    if len(query) > 2000:
        flags.append("query_too_long")
        query = query[:2000]

    for pat in BLOCKED_PATTERNS:
        if re.search(pat, query, re.IGNORECASE):
            flags.append("prompt_injection_detected")
            break

    filler_patterns = [
        r"^(please\s+)?(can\s+you\s+)?(tell\s+me|show\s+me|explain|find|give\s+me|"
        r"what\s+is|what\s+are|what\s+does\s+it\s+say\s+about)\s+",
        r"^(in\s+the\s+(document|file|image|pdf|chart|table)[,]?\s+)?",
        r"\s+(from\s+the\s+(document|file|image|pdf))\s*$",
    ]
    cleaned = query
    for pat in filler_patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
    if len(cleaned) < 4:
        cleaned = query

    elapsed = round((time.time()-t0)*1000, 1)
    pstatus(f"guardrail: clean='{cleaned[:80]}' ({elapsed:.0f}ms)")
    return {
        "current_query":  cleaned,
        "guardrail_flags": flags,
        "node_latencies": {**state.get("node_latencies", {}), "guardrail": elapsed},
    }


def node_intent_router(state: AgentState) -> Dict[str, Any]:
    t0 = time.time()
    intent, has_visual = _detect_intent(state["current_query"])
    elapsed = round((time.time()-t0)*1000, 1)
    pstatus(f"intent_router: intent={intent} visual={has_visual} ({elapsed:.0f}ms)")
    return {
        "detected_intent":   intent,
        "has_visual_intent": has_visual,
        "node_latencies":    {**state.get("node_latencies", {}), "intent_router": elapsed},
    }


def node_query_expander(state: AgentState) -> Dict[str, Any]:
    t0     = time.time()
    query  = state["current_query"]
    flags  = state.get("guardrail_flags", [])
    intent = state.get("detected_intent", "text")

    if "prompt_injection_detected" in flags or "empty_query" in flags:
        return {"query_variants": [query],
                "node_latencies": {**state.get("node_latencies", {}),
                                   "expander": round((time.time()-t0)*1000, 1)}}

    variants = [query, f"{intent} {query}".strip()]
    try:
        raw = llm_generate(
            f"Rewrite this query in exactly 2 alternative forms.\n"
            f"Keep meaning identical. Use retrieval-friendly vocabulary.\n"
            f"Return ONLY 2 lines.\n\nQuery: {query}",
            max_tokens=120, temperature=0.4,
        )
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        variants.extend(lines[:2])
    except Exception as exc:
        logger.warning("Expander failed: %s", exc)

    # Dedup
    deduped: List[str] = []
    seen: set = set()
    for v in variants:
        k = v.lower()
        if k not in seen:
            seen.add(k)
            deduped.append(v)

    elapsed = round((time.time()-t0)*1000, 1)
    pstatus(f"query_expander: {len(deduped)} variants ({elapsed:.0f}ms)", C.GRAY)
    return {
        "query_variants": deduped,
        "node_latencies": {**state.get("node_latencies", {}), "expander": elapsed},
    }


def node_dual_retriever(state: AgentState) -> Dict[str, Any]:
    """
    Searches BOTH text collection and image collection (if visual intent).
    Merges and deduplicates results by chunk_id.
    """
    t0        = time.time()
    query     = state["current_query"]
    variants  = state.get("query_variants", [query])
    flags     = state.get("guardrail_flags", [])
    has_vis   = state.get("has_visual_intent", False)

    if "prompt_injection_detected" in flags:
        return {"raw_chunks": [],
                "node_latencies": {**state.get("node_latencies", {}),
                                   "dual_retriever": round((time.time()-t0)*1000, 1)}}

    milvus   = get_milvus()
    manifest = _manifest_map()
    seen:    set = set()
    raw_chunks: List[Dict[str, Any]] = []

    OUTPUT_FIELDS = [
        "chunk_id", "search_text", "modality", "source_type",
        "file_path", "file_name", "page_num", "parent_id",
        "figure_id", "title", "bbox", "image_path",
        "color_info", "spatial_info", "audio_start_s", "audio_end_s",
    ]

    def _search(collection: str, emb: List[float], limit: int):
        return milvus.search(
            collection_name=collection,
            data=[emb],
            limit=limit,
            output_fields=OUTPUT_FIELDS,
        )[0]

    for variant in variants:
        embs = embed_texts([variant], input_type="query")
        if not embs:
            continue
        emb = embs[0]

        # Text collection (always)
        for hit in _search(TEXT_COLLECTION, emb, RETRIEVAL_TOP_K):
            entity   = hit.get("entity", hit) if isinstance(hit, dict) else hit.entity
            chunk_id = (entity.get("chunk_id") if isinstance(entity, dict)
                        else getattr(entity, "chunk_id", ""))
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            row    = manifest.get(chunk_id, {})
            stext  = row.get("search_text") or (
                entity.get("search_text", "") if isinstance(entity, dict)
                else getattr(entity, "search_text", ""))
            score  = (hit.get("distance", 0.0) if isinstance(hit, dict)
                      else getattr(hit, "distance", 0.0))
            if not stext:
                continue

            def _field(key: str, default: Any = "") -> Any:
                return row.get(key) or (
                    entity.get(key, default) if isinstance(entity, dict)
                    else getattr(entity, key, default))

            raw_chunks.append({
                "chunk_id":    chunk_id,
                "text":        stext,
                "vector_score": float(score),
                "modality":    _field("modality", "text"),
                "file_path":   _field("file_path"),
                "file_name":   _field("file_name"),
                "page_num":    _field("page_num", 0),
                "parent_id":   _field("parent_id"),
                "figure_id":   _field("figure_id"),
                "title":       _field("title"),
                "bbox":        _field("bbox"),
                "image_path":  _field("image_path"),
                "color_info":  _field("color_info"),
                "spatial_info": _field("spatial_info"),
                "audio_start_s": float(_field("audio_start_s", 0.0)),
                "audio_end_s":   float(_field("audio_end_s",   0.0)),
                "row":         row,
                "from_collection": "text",
            })

        # Image collection (additional search when visual intent)
        if has_vis:
            for hit in _search(IMAGE_COLLECTION, emb, RETRIEVAL_TOP_K // 2):
                entity   = hit.get("entity", hit) if isinstance(hit, dict) else hit.entity
                chunk_id = (entity.get("chunk_id") if isinstance(entity, dict)
                            else getattr(entity, "chunk_id", ""))
                if not chunk_id or chunk_id in seen:
                    continue
                seen.add(chunk_id)
                row   = manifest.get(chunk_id, {})
                stext = row.get("search_text") or (
                    entity.get("search_text", "") if isinstance(entity, dict)
                    else getattr(entity, "search_text", ""))
                score = (hit.get("distance", 0.0) if isinstance(hit, dict)
                         else getattr(hit, "distance", 0.0))
                if not stext:
                    continue

                def _f(key: str, default: Any = "") -> Any:
                    return row.get(key) or (
                        entity.get(key, default) if isinstance(entity, dict)
                        else getattr(entity, key, default))

                raw_chunks.append({
                    "chunk_id":    chunk_id,
                    "text":        stext,
                    "vector_score": float(score) + 0.1,  # slight boost for image collection hits
                    "modality":    _f("modality", "image"),
                    "file_path":   _f("file_path"),
                    "file_name":   _f("file_name"),
                    "page_num":    _f("page_num", 0),
                    "parent_id":   _f("parent_id"),
                    "figure_id":   _f("figure_id"),
                    "title":       _f("title"),
                    "bbox":        _f("bbox"),
                    "image_path":  _f("image_path"),
                    "color_info":  _f("color_info"),
                    "spatial_info": _f("spatial_info"),
                    "audio_start_s": 0.0,
                    "audio_end_s":   0.0,
                    "row":         row,
                    "from_collection": "image",
                })

    elapsed = round((time.time()-t0)*1000, 1)
    n_img = sum(1 for c in raw_chunks if c.get("from_collection") == "image")
    pstatus(f"dual_retriever: {len(raw_chunks)} chunks (image_col={n_img}) ({elapsed:.0f}ms)")
    return {
        "raw_chunks":   raw_chunks,
        "node_latencies": {**state.get("node_latencies", {}), "dual_retriever": elapsed},
    }


def node_cross_reranker(state: AgentState) -> Dict[str, Any]:
    """
    Cross-encoder reranker — preserved from v2.
    Also applies the quality gate: if top chunk score < MIN_GENERATION_SCORE, skip LLM.
    """
    t0         = time.time()
    query      = state["current_query"]
    raw_chunks = state.get("raw_chunks", [])
    intent     = state.get("detected_intent", "text")

    if not raw_chunks:
        return {
            "ranked_chunks":       [],
            "overall_confidence":  "low",
            "quality_gate_failed": False,
            "node_latencies":      {**state.get("node_latencies", {}),
                                    "cross_reranker": round((time.time()-t0)*1000, 1)},
        }

    try:
        rankings = rerank_passages(query, [c["text"] for c in raw_chunks])
        ranked: List[Dict[str, Any]] = []
        for row in rankings[:RERANK_TOP_K]:
            idx = row["index"]
            if 0 <= idx < len(raw_chunks):
                chunk = raw_chunks[idx].copy()
                score = row["score"]
                chunk["rerank_score"] = score
                if score >= CONFIDENCE_HIGH:
                    chunk["confidence"] = "high"
                elif score >= CONFIDENCE_MED:
                    chunk["confidence"] = "medium"
                else:
                    chunk["confidence"] = "low"
                ranked.append(chunk)
    except Exception as exc:
        logger.warning("Reranker failed, using vector order: %s", exc)
        ranked = [{**c, "rerank_score": c["vector_score"], "confidence": "medium"}
                  for c in raw_chunks[:RERANK_TOP_K]]

    quality_gate_failed = False
    confidence = "low"
    if ranked:
        top = ranked[0]["rerank_score"]
        if top < MIN_GENERATION_SCORE:
            quality_gate_failed = True
            pstatus(f"cross_reranker: quality gate FAILED (top={top:.2f})", C.YELLOW)
        elif top >= CONFIDENCE_HIGH:
            confidence = "high"
        elif top >= CONFIDENCE_MED:
            confidence = "medium"

    elapsed = round((time.time()-t0)*1000, 1)
    pstatus(f"cross_reranker: {len(ranked)} chunks, {confidence} ({elapsed:.0f}ms)")
    return {
        "ranked_chunks":       ranked,
        "overall_confidence":  confidence,
        "quality_gate_failed": quality_gate_failed,
        "node_latencies":      {**state.get("node_latencies", {}), "cross_reranker": elapsed},
    }


def node_layout_rescue(state: AgentState) -> Dict[str, Any]:
    """
    Pull sibling chunks from the same page/parent when:
    - intent is visual (image/chart/diagram)
    - overall confidence is low
    Ensures that text around a chart/figure is always included.
    """
    t0       = time.time()
    ranked   = state.get("ranked_chunks", [])
    intent   = state.get("detected_intent", "text")
    manifest = _manifest_map()
    has_vis  = state.get("has_visual_intent", False)

    if not ranked:
        return {"rescued_chunks": [],
                "node_latencies": {**state.get("node_latencies", {}),
                                   "layout_rescue": round((time.time()-t0)*1000, 1)}}

    selected    = list(ranked[:RERANK_TOP_K])
    selected_ids = {c["chunk_id"] for c in selected}

    if has_vis or state.get("overall_confidence") == "low":
        anchor_parents = {c.get("parent_id", "") for c in ranked[:5] if c.get("parent_id")}
        for row in manifest.values():
            if row.get("parent_id") not in anchor_parents:
                continue
            if row["chunk_id"] in selected_ids:
                continue
            if has_vis and row.get("modality") == "text" and not row.get("layout_text"):
                continue
            selected_ids.add(row["chunk_id"])
            selected.append({
                "chunk_id":    row["chunk_id"],
                "text":        row["search_text"],
                "vector_score": 0.0,
                "rerank_score": ranked[-1].get("rerank_score", 0.0) if ranked else 0.0,
                "confidence":  "low",
                "modality":    row.get("modality", "text"),
                "file_path":   row.get("file_path", ""),
                "file_name":   row.get("file_name", ""),
                "page_num":    row.get("page_num", 0),
                "parent_id":   row.get("parent_id", ""),
                "figure_id":   row.get("figure_id", ""),
                "title":       row.get("title", ""),
                "bbox":        row.get("bbox", ""),
                "image_path":  row.get("image_path", ""),
                "color_info":  row.get("color_info", ""),
                "spatial_info": row.get("spatial_info", ""),
                "audio_start_s": float(row.get("audio_start_s", 0.0)),
                "audio_end_s":   float(row.get("audio_end_s",   0.0)),
                "row":         row,
                "rescued":     True,
            })

    added = len(selected) - len(ranked[:RERANK_TOP_K])
    elapsed = round((time.time()-t0)*1000, 1)
    pstatus(f"layout_rescue: +{max(0, added)} sibling chunks ({elapsed:.0f}ms)")
    return {
        "rescued_chunks": selected,
        "node_latencies": {**state.get("node_latencies", {}), "layout_rescue": elapsed},
    }


def node_vlm_reread(state: AgentState) -> Dict[str, Any]:
    """
    THE KEY NODE — re-feeds actual image bytes to the VLM with the specific user question.

    Why this matters:
    - Ingestion-time captions are generic summaries; they miss query-specific details
    - "What does the red legend indicate?" requires re-reading the chart with that question
    - "What is the ID card number?" requires re-reading the image with that field in mind
    - This node runs ONLY for visual queries (image/chart/diagram/infographic)
    - Takes the top VLM_REREAD_TOP_N chunks that have image_path set
    - Calls VLM with original_query + image -> appends as additional evidence
    """
    t0       = time.time()
    intent   = state.get("detected_intent", "text")
    has_vis  = state.get("has_visual_intent", False)
    query    = state.get("original_query", state.get("current_query", ""))
    rescued  = state.get("rescued_chunks", [])
    ranked   = state.get("ranked_chunks", [])
    candidates = rescued or ranked

    # Only activate for visual queries
    if not has_vis or not candidates:
        return {
            "vlm_evidence": [],
            "node_latencies": {**state.get("node_latencies", {}),
                               "vlm_reread": 0.0},
        }

    # Collect unique image paths from top chunks
    image_chunks = [
        c for c in candidates
        if c.get("image_path") and Path(c["image_path"]).exists()
    ]

    # Deduplicate by image_path
    seen_paths: set = set()
    unique_image_chunks: List[Dict] = []
    for c in image_chunks:
        p = c["image_path"]
        if p not in seen_paths:
            seen_paths.add(p)
            unique_image_chunks.append(c)
        if len(unique_image_chunks) >= VLM_REREAD_TOP_N:
            break

    if not unique_image_chunks:
        pstatus("vlm_reread: no image files available for reread", C.GRAY)
        return {
            "vlm_evidence": [],
            "node_latencies": {**state.get("node_latencies", {}),
                               "vlm_reread": 0.0},
        }

    pstatus(f"vlm_reread: re-reading {len(unique_image_chunks)} image(s) with query", C.MAGENTA)
    vlm_evidence: List[Dict[str, Any]] = []

    for chunk in unique_image_chunks:
        img_path = chunk["image_path"]
        pstatus(f"  -> VLM reread: {Path(img_path).name}", C.GRAY)
        try:
            result = vlm_describe_image(img_path, question=query)
            targeted = result.get("targeted_answer", "").strip()
            if targeted:
                vlm_evidence.append({
                    "chunk_id":      f"vlm_reread_{chunk['chunk_id']}",
                    "text":          f"[VLM direct image analysis]\n{targeted}",
                    "modality":      "vlm_reread",
                    "image_path":    img_path,
                    "file_name":     chunk.get("file_name", Path(img_path).name),
                    "page_num":      chunk.get("page_num", 0),
                    "rerank_score":  999.0,   # always include VLM reread evidence
                    "confidence":    "high",
                    "is_vlm_reread": True,
                })
        except Exception as exc:
            logger.warning("VLM reread failed for %s: %s", img_path, exc)

    elapsed = round((time.time()-t0)*1000, 1)
    pstatus(f"vlm_reread: {len(vlm_evidence)} targeted answers ({elapsed:.0f}ms)", C.MAGENTA)
    return {
        "vlm_evidence": vlm_evidence,
        "node_latencies": {**state.get("node_latencies", {}), "vlm_reread": elapsed},
    }


def node_evidence_builder(state: AgentState) -> Dict[str, Any]:
    """
    Assemble final evidence set.
    VLM reread results are prepended (highest priority).
    Then ranked/rescued chunks, grouped by parent for context continuity.
    """
    t0          = time.time()
    intent      = state.get("detected_intent", "text")
    vlm_evid    = state.get("vlm_evidence", [])
    candidates  = state.get("rescued_chunks") or state.get("ranked_chunks", [])

    evidence:   List[Dict[str, Any]] = []
    sources:    List[Dict[str, Any]] = []
    used_ids:   set = set()

    # VLM reread always goes first
    for ev in vlm_evid:
        evidence.append(ev)
        sources.append({
            "index":        len(sources) + 1,
            "chunk_id":     ev["chunk_id"],
            "modality":     ev["modality"],
            "confidence":   ev["confidence"],
            "rerank_score": 999.0,
            "file_name":    ev.get("file_name", ""),
            "page_num":     ev.get("page_num", 0),
            "bbox":         "",
            "rescued":      False,
            "is_vlm_reread": True,
            "text_preview": ev["text"][:180],
        })
        used_ids.add(ev["chunk_id"])

    # Group remaining candidates by parent_id for layout continuity
    grouped: Dict[str, List[Dict]] = {}
    for c in candidates:
        grouped.setdefault(c.get("parent_id", c["chunk_id"]), []).append(c)

    for _, siblings in grouped.items():
        siblings.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        preferred = list(siblings[:2])
        if intent in VISUAL_INTENTS:
            preferred += [s for s in siblings if s.get("modality") in VISUAL_INTENTS][:2]
        elif intent == "table":
            preferred += [s for s in siblings if s.get("modality") == "table"][:2]
        elif intent == "audio":
            preferred += [s for s in siblings if s.get("modality") == "audio_transcript"][:2]

        for chunk in preferred:
            if chunk["chunk_id"] in used_ids:
                continue
            used_ids.add(chunk["chunk_id"])
            evidence.append(chunk)
            preview = re.sub(r"\s+", " ", chunk.get("text", "")).strip()
            sources.append({
                "index":        len(sources) + 1,
                "chunk_id":     chunk["chunk_id"],
                "modality":     chunk.get("modality", "text"),
                "confidence":   chunk.get("confidence", "low"),
                "rerank_score": round(float(chunk.get("rerank_score", 0.0)), 4),
                "file_name":    chunk.get("file_name", ""),
                "page_num":     chunk.get("page_num", 0),
                "bbox":         chunk.get("bbox", ""),
                "rescued":      bool(chunk.get("rescued")),
                "is_vlm_reread": False,
                "text_preview": preview[:180] + ("..." if len(preview) > 180 else ""),
            })
            if len(evidence) >= MAX_CONTEXT:
                break
        if len(evidence) >= MAX_CONTEXT:
            break

    elapsed = round((time.time()-t0)*1000, 1)
    pstatus(f"evidence_builder: {len(evidence)} pieces ({elapsed:.0f}ms)")
    return {
        "evidence_chunks": evidence,
        "sources":         sources,
        "node_latencies":  {**state.get("node_latencies", {}), "evidence_builder": elapsed},
    }


# ── Generator system prompt ───────────────────────────────────────────────────

_SYS = """You are a precise enterprise multimodal document assistant.

Use ONLY the supplied evidence to answer. Evidence may contain:
- Plain text (any document type)
- OCR text from images, ID cards, scanned docs
- Tables in markdown — read row/column intersections carefully
- Captions, color descriptions, spatial descriptions from charts/diagrams
- VLM direct image analysis (most reliable for visual queries)
- Audio transcripts with timestamps
- Layout-aware snippets with page numbers and bounding boxes

Rules:
1. VLM direct image analysis evidence is the most authoritative source for visual queries.
2. Preserve numbers, IDs, names, dates, addresses exactly as they appear.
3. For tables: identify exact row+column intersection before answering.
4. For colors/legends/arrows: use the color_info, spatial_info, and VLM reread evidence.
5. For audio transcripts: reference the timestamp range when citing spoken content.
6. For identity documents: extract each field (name, DOB, ID number) as a structured list.
7. If evidence is insufficient, say exactly: "The provided documents do not contain this information."
8. Never use outside knowledge. Never guess.
9. Mention supporting file name and page/timestamp when possible."""


def node_generator(state: AgentState) -> Dict[str, Any]:
    t0     = time.time()
    flags  = list(state.get("guardrail_flags", []))
    query  = state.get("original_query", "")

    if "prompt_injection_detected" in flags:
        return {
            "answer":    "This query has been flagged and cannot be processed.",
            "model_used": "none", "fallback_used": False,
            "guardrail_flags": flags, "sources": [],
            "node_latencies": {**state.get("node_latencies", {}), "generator": 0.0},
        }

    if state.get("quality_gate_failed", False):
        return {
            "answer":    "The provided documents do not contain relevant information for this query.",
            "model_used": "none", "fallback_used": False,
            "guardrail_flags": flags, "sources": [],
            "node_latencies": {**state.get("node_latencies", {}), "generator": 0.0},
        }

    evidence = state.get("evidence_chunks", [])
    if not evidence:
        return {
            "answer":    "The provided documents do not contain this information.",
            "model_used": "none", "fallback_used": False,
            "guardrail_flags": flags, "sources": [],
            "node_latencies": {**state.get("node_latencies", {}), "generator": 0.0},
        }

    # Build conversation history prefix
    history_text = ""
    for turn in state.get("conversation_history", [])[-5:]:
        history_text += f"User: {turn.get('query', '')}\nAssistant: {turn.get('answer', '')}\n\n"

    # Build context blocks
    context_parts: List[str] = []
    for idx, chunk in enumerate(evidence, 1):
        lines = [
            f"[Evidence {idx}]",
            f"type: {chunk.get('modality', 'text')}",
            f"file: {chunk.get('file_name', '')}",
        ]
        if chunk.get("page_num"):
            lines.append(f"page: {chunk['page_num']}")
        if chunk.get("audio_start_s") or chunk.get("audio_end_s"):
            lines.append(f"timestamp: {chunk.get('audio_start_s', 0):.1f}s – "
                         f"{chunk.get('audio_end_s', 0):.1f}s")
        if chunk.get("bbox"):
            lines.append(f"bbox: {chunk['bbox']}")
        if chunk.get("color_info"):
            lines.append(f"colors: {chunk['color_info']}")
        if chunk.get("spatial_info"):
            lines.append(f"spatial: {chunk['spatial_info']}")
        if chunk.get("is_vlm_reread"):
            lines.append("*** VLM DIRECT IMAGE ANALYSIS (high reliability) ***")
        lines.append(f"content:\n{chunk.get('text', '')}")
        context_parts.append("\n".join(lines).strip())

    prompt = (
        f"Previous conversation:\n{history_text}\n"
        if history_text else ""
    ) + (
        f"Question: {query}\n\n"
        f"Evidence:\n{'─'*40}\n"
        f"\n{'─'*40}\n".join(context_parts) +
        f"\n{'─'*40}\n\nAnswer:"
    )

    answer = ""
    model_used    = "none"
    fallback_used = False

    for model in [PRIMARY_LLM, FALLBACK_LLM]:
        try:
            candidate = llm_generate(prompt, model=model, system_prompt=_SYS,
                                     max_tokens=1024, temperature=0.2)
            if candidate.strip():
                answer        = candidate.strip()
                model_used    = model
                fallback_used = (model == FALLBACK_LLM)
                break
        except Exception as exc:
            logger.warning("LLM %s failed: %s", model, exc)

    if not answer:
        answer = "Both LLMs failed to generate a response."
        flags.append("empty_answer")

    if any(ph in answer.lower() for ph in HALLUCINATION_PHRASES):
        flags.append("possible_hallucination")

    elapsed = round((time.time()-t0)*1000, 1)
    pstatus(f"generator: {model_used.split('/')[-1]} ({elapsed:.0f}ms)")
    return {
        "answer":          answer,
        "model_used":      model_used,
        "fallback_used":   fallback_used,
        "guardrail_flags": flags,
        "sources":         state.get("sources", []),
        "node_latencies":  {**state.get("node_latencies", {}), "generator": elapsed},
    }


# ── Graph wiring ──────────────────────────────────────────────────────────────

_compiled_graph = None


def route_after_guardrail(state: AgentState) -> str:
    flags = state.get("guardrail_flags", [])
    if "prompt_injection_detected" in flags or "empty_query" in flags:
        return "generator"
    return "intent_router"


def get_graph():
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    graph = StateGraph(AgentState)
    graph.add_node("guardrail",        node_guardrail)
    graph.add_node("intent_router",    node_intent_router)
    graph.add_node("query_expander",   node_query_expander)
    graph.add_node("dual_retriever",   node_dual_retriever)
    graph.add_node("cross_reranker",   node_cross_reranker)
    graph.add_node("layout_rescue",    node_layout_rescue)
    graph.add_node("vlm_reread",       node_vlm_reread)
    graph.add_node("evidence_builder", node_evidence_builder)
    graph.add_node("generator",        node_generator)

    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"intent_router": "intent_router", "generator": "generator"},
    )
    graph.add_edge("intent_router",    "query_expander")
    graph.add_edge("query_expander",   "dual_retriever")
    graph.add_edge("dual_retriever",   "cross_reranker")
    graph.add_edge("cross_reranker",   "layout_rescue")
    graph.add_edge("layout_rescue",    "vlm_reread")
    graph.add_edge("vlm_reread",       "evidence_builder")
    graph.add_edge("evidence_builder", "generator")
    graph.add_edge("generator",        END)

    _compiled_graph = graph.compile()
    return _compiled_graph


# ── Agent runner ──────────────────────────────────────────────────────────────

def _initial_state(query: str, history: Optional[List] = None) -> AgentState:
    return {
        "original_query":      query,
        "current_query":       query,
        "query_variants":      [query],
        "detected_intent":     "text",
        "has_visual_intent":   False,
        "conversation_history": history or [],
        "raw_chunks":          [],
        "ranked_chunks":       [],
        "rescued_chunks":      [],
        "vlm_evidence":        [],
        "evidence_chunks":     [],
        "overall_confidence":  "low",
        "quality_gate_failed": False,
        "answer":              "",
        "model_used":          "",
        "fallback_used":       False,
        "guardrail_flags":     [],
        "retry_count":         0,
        "node_latencies":      {},
        "sources":             [],
    }


def _prepare_retry_state(state: AgentState) -> Optional[AgentState]:
    retry_count = state.get("retry_count", 0)
    if retry_count >= MAX_RETRIES:
        return None
    if "possible_hallucination" not in state.get("guardrail_flags", []):
        return None
    return {
        **state,
        "current_query":  f"{state['original_query']} answer only with exact facts from the evidence.",
        "query_variants": [state["original_query"]],
        "raw_chunks": [], "ranked_chunks": [], "rescued_chunks": [],
        "vlm_evidence": [], "evidence_chunks": [],
        "answer": "", "model_used": "", "fallback_used": False,
        "retry_count":         retry_count + 1,
        "quality_gate_failed": False,
        "guardrail_flags":     [f for f in state.get("guardrail_flags", [])
                                 if f != "possible_hallucination"],
        "node_latencies": {}, "sources": [],
    }


def run_agent(query: str, history: Optional[List] = None) -> Dict[str, Any]:
    graph  = get_graph()
    state  = _initial_state(query, history)
    t0     = time.time()
    while True:
        result      = graph.invoke(state)
        retry_state = _prepare_retry_state(result)
        if retry_state is None:
            result["wall_ms"] = round((time.time() - t0) * 1000)
            return result
        state = retry_state


# ── Output formatting ─────────────────────────────────────────────────────────

def print_answer(
    answer:      str,
    confidence:  str,
    wall_ms:     int,
    model:       str,
    retry_count: int,
    sources:     List[Dict[str, Any]],
    latencies:   Dict[str, float],
    flags:       List[str],
):
    cc          = {"high": C.GREEN, "medium": C.YELLOW, "low": C.RED}.get(confidence, C.GRAY)
    model_label = model.split("/")[-1] if model and model != "none" else "none"

    print(f"\n{C.BOLD}╔═══ ANSWER ═══════════════════════════════════════════════════╗{C.RESET}")
    print(f"{C.BOLD}║{C.RESET} {cc}{confidence.upper()} CONFIDENCE{C.RESET}  |  "
          f"{C.GRAY}{model_label}{C.RESET}  |  {C.GRAY}{wall_ms:,}ms{C.RESET}")
    if retry_count:
        print(f"{C.BOLD}║{C.RESET} {C.YELLOW}retried {retry_count}x{C.RESET}")
    print(f"{C.BOLD}╠══════════════════════════════════════════════════════════════╣{C.RESET}")
    for line in (answer or "No answer generated.").splitlines():
        print(f"{C.BOLD}║{C.RESET} {line}")
    print(f"{C.BOLD}╚══════════════════════════════════════════════════════════════╝{C.RESET}")

    if latencies:
        print(f"\n  {C.GRAY}Latency:{C.RESET}")
        max_ms = max(latencies.values()) or 1
        nodes  = ["guardrail", "intent_router", "query_expander", "dual_retriever",
                  "cross_reranker", "layout_rescue", "vlm_reread", "evidence_builder", "generator"]
        for node in nodes:
            if node in latencies:
                ms      = latencies[node]
                bar_len = int(ms / max_ms * 28)
                bar     = "#" * bar_len + "." * (28 - bar_len)
                tag     = " [VLM]" if node == "vlm_reread" else ""
                print(f"    {C.GRAY}{node:>18}{C.RESET}  {C.BLUE}{bar}{C.RESET}  {ms:.0f}ms{tag}")

    if sources:
        print(f"\n  {C.GRAY}Sources:{C.RESET}")
        for s in sources[:8]:
            vlm_tag  = f" {C.MAGENTA}[VLM-REREAD]{C.RESET}" if s.get("is_vlm_reread") else ""
            res_tag  = " [rescued]" if s.get("rescued") else ""
            score    = s.get("rerank_score", 0)
            score_s  = "  (VLM)" if score == 999.0 else f"  score={score:.3f}"
            print(f"    {C.BLUE}#{s['index']}{C.RESET} [{s.get('modality','text')}]"
                  f"{vlm_tag}{res_tag}  {s.get('file_name','')}:p{s.get('page_num',0)}{score_s}")
            print(f"    {C.DIM}{s.get('text_preview','')}{C.RESET}")

    if flags:
        print(f"\n  {C.YELLOW}Flags: {', '.join(flags)}{C.RESET}")


# ── Interactive loop ──────────────────────────────────────────────────────────

def interactive_loop():
    print(f"""
{C.BOLD}Commands:{C.RESET}
  {C.CYAN}ingest <path> [paths...]     {C.RESET} Ingest files (PDF/DOCX/PPTX/HTML/TXT/JPEG/PNG/MP3/WAV)
  {C.CYAN}ingest --reset <path> [...]  {C.RESET} Reset collections then ingest
  {C.CYAN}stats                        {C.RESET} Show collection stats
  {C.CYAN}reset                        {C.RESET} Clear all collections + manifest
  {C.CYAN}history                      {C.RESET} Show conversation memory
  {C.CYAN}clear                        {C.RESET} Clear conversation memory
  {C.CYAN}quit                         {C.RESET} Exit
""")
    conversation_history: List[Dict[str, str]] = []

    while True:
        try:
            user_input = input(f"\n{C.GREEN}{C.BOLD}Q: {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.GRAY}Session ended.{C.RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print(f"{C.GRAY}Session ended.{C.RESET}")
            break

        if user_input.lower() == "reset":
            reset_collections()
            continue

        if user_input.lower() == "clear":
            conversation_history = []
            pok("Conversation memory cleared.")
            continue

        if user_input.lower() == "history":
            for i, turn in enumerate(conversation_history, 1):
                print(f"  {i}. Q: {turn['query'][:80]}")
                print(f"     A: {turn['answer'][:80]}")
            continue

        if user_input.lower().startswith("ingest"):
            parts    = user_input.split()
            do_reset = "--reset" in parts
            paths    = [p for p in parts[1:] if p != "--reset"]
            valid    = [p for p in paths if os.path.isfile(p)]
            for p in paths:
                if not os.path.isfile(p):
                    perr(f"File not found: {p}")
            if valid:
                run_ingest(valid, reset=do_reset)
            continue

        if user_input.lower() == "stats":
            milvus = get_milvus()
            for col in [TEXT_COLLECTION, IMAGE_COLLECTION]:
                stats = milvus.get_collection_stats(col)
                pok(f"{col}: {json.dumps(stats)}")
            counts: Dict[str, int] = {}
            for row in _manifest_rows():
                counts[row["modality"]] = counts.get(row["modality"], 0) + 1
            pstatus(f"Manifest modalities: {counts}", C.GRAY)
            continue

        result = run_agent(user_input, history=conversation_history)

        if result.get("answer") and result.get("model_used") != "none":
            conversation_history.append({"query": user_input, "answer": result["answer"]})
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]

        print_answer(
            result.get("answer", ""),
            result.get("overall_confidence", "low"),
            result.get("wall_ms", 0),
            result.get("model_used", "none"),
            result.get("retry_count", 0),
            result.get("sources", []),
            result.get("node_latencies", {}),
            result.get("guardrail_flags", []),
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Enterprise Multimodal RAG Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported file types:
  Documents : PDF, DOCX, PPTX (via NV-Ingest)
  Text      : HTML, TXT, MD (direct extraction)
  Images    : JPEG, PNG, WEBP, BMP, TIFF (direct VLM — no PDF conversion)
  Audio     : MP3, WAV, M4A, OGG, FLAC (Whisper transcription)

Examples:
  python enterprise_rag.py
  python enterprise_rag.py --ingest report.pdf id_card.jpg meeting.mp3
  python enterprise_rag.py --reset --ingest data.pdf
        """,
    )
    parser.add_argument("--reset",  action="store_true", help="Reset collections before ingest")
    parser.add_argument("--ingest", nargs="+", metavar="FILE", help="Files to ingest on startup")
    args = parser.parse_args()

    banner()

    if not NVIDIA_API_KEY:
        perr("NVIDIA_API_KEY not set.")
        perr("export NVIDIA_API_KEY='nvapi-...'")
        sys.exit(1)

    pok(f"Milvus DB         : {MILVUS_DB}")
    pok(f"Text collection   : {TEXT_COLLECTION}")
    pok(f"Image collection  : {IMAGE_COLLECTION}")
    pok(f"Embed model       : {EMBED_MODEL} (dim={DIM})")
    pok(f"Primary LLM       : {PRIMARY_LLM}")
    pok(f"Fallback LLM      : {FALLBACK_LLM}")
    pok(f"Caption/VLM model : {CAPTION_MODEL}")
    pok(f"Reranker          : {RERANK_MODEL}")
    pok(f"Whisper model     : {WHISPER_MODEL}")
    pok(f"VLM reread top-N  : {VLM_REREAD_TOP_N}")
    pok(f"Quality gate      : skip if top rerank < {MIN_GENERATION_SCORE}")
    pok(f"Artifact root     : {ARTIFACT_ROOT}")

    # ── Auto-ingest files from INGEST_FILES if defined ──────────────────────
    files_to_ingest = []
    if INGEST_FILES:
        files_to_ingest = [f.strip() for f in INGEST_FILES if f.strip()]
        if files_to_ingest:
            pstatus(f"Found {len(files_to_ingest)} file(s) in INGEST_FILES")

    # ── Handle command-line --ingest argument (overrides INGEST_FILES) ───────
    if args.ingest:
        files_to_ingest = args.ingest
        pstatus(f"Using {len(files_to_ingest)} file(s) from --ingest argument")

    # ── Perform ingestion if any files specified ────────────────────────────
    if files_to_ingest:
        valid   = [p for p in files_to_ingest if os.path.isfile(p)]
        missing = [p for p in files_to_ingest if not os.path.isfile(p)]
        
        for p in missing:
            perr(f"File not found: {p}")
        
        if valid:
            psec(f"Auto-ingesting {len(valid)} file(s)...")
            try:
                result = run_ingest(valid, reset=args.reset)
                pok(f"Ingestion complete: {result.get('chunks_ingested', 0)} chunks ingested in {result.get('elapsed_ms', 0):,}ms")
            except Exception as exc:
                perr(f"Ingestion failed: {exc}")
                logger.exception("Ingestion failed")
                sys.exit(1)
    elif args.reset:
        reset_collections()
        pok("Collections reset.")

    interactive_loop()


if __name__ == "__main__":
    main()
