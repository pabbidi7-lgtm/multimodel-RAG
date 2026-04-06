"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   Enterprise Multimodal RAG Agent v3                                         ║
║   NV-Ingest 25.9.0 + LangGraph + NVIDIA Mistral Reranker                     ║
║                                                                              ║
║   Supports: PDF, DOCX, PPTX, XLSX, images, handwritten, invoices,           ║
║   identity docs, tables, charts, diagrams, captions, infographics            ║
║                                                                              ║
║   AGENT GRAPH (Adaptive, NOT linear):                                        ║
║                                                                              ║
║   START                                                                      ║
║     │                                                                        ║
║     ▼                                                                        ║
║   [N1: guardrail]  ──injection──► [N6: generator] ──► END                   ║
║     │                                                                        ║
║     ▼                                                                        ║
║   [N2: intent_router]  ──────────────────────────────────┐                  ║
║     │                                                     │                  ║
║     ▼                                                     │                  ║
║   [N3: multi_expander]  (query × modal variants)         │                  ║
║     │                                                     │                  ║
║     ▼                                                     │                  ║
║   [N4: hybrid_retriever]  (dense + metadata filter)      │                  ║
║     │                                                     │                  ║
║     ▼                                                     │                  ║
║   [N5: nvidia_reranker]  (mistral-4b rerank API)         │                  ║
║     │                                                     │                  ║
║     ├── score OK ──────────────────────────────────────► [N6: generator]    ║
║     │                                                     │                  ║
║     ├── score LOW  ──► [N7: adaptive_rescue]              │                  ║
║     │                     │                               │                  ║
║     │                     ├── rescue OK ──────────────► [N6: generator]     ║
║     │                     │                               │                  ║
║     │                     └── rescue fail ─────────────► [N6: generator]    ║
║     │                          (best-effort fallback)     │                  ║
║     └────────────────────────────────────────────────────►│                 ║
║                                                           ▼                  ║
║                                                          END                 ║
║                                                                              ║
║   Key Fixes vs v2:                                                           ║
║   • NVIDIA nv-rerankqa-mistral-4b-v3 via API (no local cross-encoder)        ║
║   • Quality gate NEVER silences — always generates best-effort answer        ║
║   • N7 adaptive_rescue: lexical BM25 fallback + caption-first re-retrieval   ║
║   • Modal-aware prompt: text/table/caption/chart/diagram handled separately  ║
║   • Score thresholds calibrated for Mistral reranker (0.0–1.0 scale)         ║
║   • Conversation memory: last 5 turns                                        ║
║   • Retry on hallucination with strictness escalation                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import socket
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, TypedDict

import requests
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.FileHandler("rag_agent_v3.log", mode="a")],
)
logger = logging.getLogger("rag_agent_v3")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
logger.addHandler(console_handler)


# ── Console colors ────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m";  BOLD  = "\033[1m";  DIM   = "\033[2m"
    RED    = "\033[91m"; GREEN = "\033[92m";  YELLOW= "\033[93m"
    BLUE   = "\033[94m"; CYAN  = "\033[96m";  WHITE = "\033[97m"
    GRAY   = "\033[90m"; MAGENTA = "\033[95m"


def banner():
    print(f"""
{C.CYAN}{C.BOLD}╔═══════════════════════════════════════════════════════════╗
║  Enterprise Multimodal RAG Agent v3                       ║
║  NV-Ingest 25.9.0 + LangGraph + NVIDIA Mistral Reranker   ║
║                                                           ║
║  Modalities: text · table · caption · chart · diagram     ║
║              infographic · OCR · handwritten · invoice    ║
║                                                           ║
║  Graph: guardrail → intent_router → multi_expander →      ║
║         hybrid_retriever → nvidia_reranker →              ║
║         [adaptive_rescue?] → generator                    ║
║                                                           ║
║  NEVER silences — always generates best-effort answer     ║
║  Reranker: nvidia/nv-rerankqa-mistral-4b-v3 (API)         ║
╚═══════════════════════════════════════════════════════════╝{C.RESET}
""")


def pstatus(msg: str, color: str = C.CYAN):
    print(f"  {color}▶{C.RESET} {msg}")


def pok(msg: str):
    print(f"  {C.GREEN}✓{C.RESET} {msg}")


def perr(msg: str):
    print(f"  {C.RED}✗{C.RESET} {msg}")


def pwarn(msg: str):
    print(f"  {C.YELLOW}⚠{C.RESET} {msg}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

INGEST_FILES: List[str] = [
    "/home/clouduser01/jaswanth/Docs/Ascent_of_Open.pdf",
]

NVIDIA_API_KEY       = os.environ.get("NVIDIA_API_KEY", "")
MILVUS_DB            = os.environ.get("MILVUS_DB", "./milvus_rag_v3.db")
COLLECTION           = os.environ.get("COLLECTION", "rag_documents_v3")
DIM                  = int(os.environ.get("EMBED_DIM", "1024"))

CHAT_API_BASE        = os.environ.get("NVIDIA_CHAT_API_BASE", "https://integrate.api.nvidia.com")
RETRIEVAL_API_BASE   = os.environ.get("NVIDIA_RETRIEVAL_API_BASE", "https://integrate.api.nvidia.com")

EMBED_URL            = os.environ.get("EMBED_URL",  f"{CHAT_API_BASE}/v1/embeddings")
EMBED_MODEL          = os.environ.get("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")

# ── NVIDIA Mistral reranker (replaces local ms-marco cross-encoder) ───────────
RERANK_URL           = os.environ.get("RERANK_URL", f"{RETRIEVAL_API_BASE}/v1/ranking")
RERANK_MODEL         = os.environ.get("RERANK_MODEL", "nvidia/nv-rerankqa-mistral-4b-v3")

LLM_URL              = os.environ.get("LLM_URL",    f"{CHAT_API_BASE}/v1/chat/completions")
PRIMARY_LLM          = os.environ.get("PRIMARY_LLM",  "meta/llama-3.3-70b-instruct")
FALLBACK_LLM         = os.environ.get("FALLBACK_LLM", "nvidia/llama-3.1-nemotron-70b-instruct")

CAPTION_URL          = os.environ.get("CAPTION_URL",   f"{CHAT_API_BASE}/v1/chat/completions")
CAPTION_MODEL        = os.environ.get("CAPTION_MODEL",  "nvidia/llama-3.1-nemotron-nano-vl-8b-v1")

BROKER_HOST          = os.environ.get("BROKER_HOST", "localhost")
BROKER_PORT          = int(os.environ.get("BROKER_PORT", 7671))

# ── Retrieval tuning ──────────────────────────────────────────────────────────
RETRIEVAL_TOP_K      = int(os.environ.get("RETRIEVAL_TOP_K", "60"))   # wider net for visual docs
RERANK_TOP_K         = int(os.environ.get("RERANK_TOP_K", "20"))
MAX_CONTEXT          = int(os.environ.get("MAX_CONTEXT", "8"))
MAX_RETRIES          = 1

# ── NVIDIA Mistral reranker score thresholds (0.0 – 1.0 logit space) ─────────
# Mistral reranker returns scores 0–1; calibrated from empirical runs on
# mixed-modality docs (visual-heavy PDFs, tables, captions, diagrams)
CONFIDENCE_HIGH      = float(os.environ.get("CONF_HIGH",   "0.45"))
CONFIDENCE_MEDIUM    = float(os.environ.get("CONF_MEDIUM",  "0.15"))
# Adaptive rescue triggers below this threshold
RESCUE_THRESHOLD     = float(os.environ.get("RESCUE_THRESHOLD", "0.10"))
# Minimum score to pass generator — set very low; we NEVER want silence
MIN_GENERATION_SCORE = float(os.environ.get("MIN_GEN_SCORE", "0.0"))   # always generate

# ── Safety patterns ───────────────────────────────────────────────────────────
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
    "i don't have information",
]

# ── Modality types ─────────────────────────────────────────────────────────────
MODALITY_TYPES = ["text", "table", "caption", "chart", "diagram", "infographic",
                  "mixed", "unknown"]


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP / API UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }


def _api_call(url: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    response: Optional[requests.Response] = None
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload, headers=_headers(), timeout=timeout)
            if response.status_code == 429:
                wait = 2 ** (attempt + 1)
                logger.warning("Rate limited — sleeping %ss", wait)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            logger.warning("API attempt %s HTTP error: %s | URL: %s", attempt + 1, exc, url)
            if response is not None:
                try:
                    logger.warning("Body: %s", response.text[:500])
                except Exception:
                    pass
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
        except (requests.ConnectionError, requests.Timeout) as exc:
            logger.warning("API attempt %s connection error: %s | URL: %s", attempt + 1, exc, url)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
        except Exception as exc:
            logger.warning("API attempt %s unexpected: %s | URL: %s", attempt + 1, exc, url)
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
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(p for p in parts if p).strip()
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════════

def embed_texts(texts: List[str], input_type: str = "query") -> List[List[float]]:
    cleaned = [t.strip() if t and t.strip() else "<empty>" for t in texts]
    data = _api_call(EMBED_URL, {
        "model":           EMBED_MODEL,
        "input":           cleaned,
        "input_type":      input_type,
        "encoding_format": "float",
    })
    return [item["embedding"] for item in data.get("data", [])]


# ═══════════════════════════════════════════════════════════════════════════════
# NVIDIA MISTRAL RERANKER  (API-based — no local model)
# ═══════════════════════════════════════════════════════════════════════════════

def rerank_passages_nvidia(query: str, passages: List[str]) -> List[Dict[str, Any]]:
    """
    Calls nvidia/nv-rerankqa-mistral-4b-v3 via the NVIDIA Retrieval API.
    Returns list of {index, score} sorted descending by score.
    Score range: 0.0 – 1.0 (higher = more relevant).
    Falls back to uniform scores on any API failure so the pipeline never breaks.
    """
    if not passages:
        return []

    # API accepts max 50 passages at a time; chunk if needed
    MAX_PER_CALL = 50
    all_results: List[Dict[str, Any]] = []

    for batch_start in range(0, len(passages), MAX_PER_CALL):
        batch = passages[batch_start: batch_start + MAX_PER_CALL]
        payload = {
            "model": RERANK_MODEL,
            "query": {"text": query},
            "passages": [{"text": p[:2000]} for p in batch],
            "truncate": "END",
        }
        try:
            data = _api_call(RERANK_URL, payload, timeout=60)
            rankings = data.get("rankings", [])
            for r in rankings:
                idx   = int(r.get("index", 0))
                score = float(r.get("logit", r.get("score", 0.0)))
                all_results.append({
                    "index": batch_start + idx,
                    "score": score,
                })
        except Exception as exc:
            logger.warning("NVIDIA reranker batch failed: %s — using fallback uniform scores", exc)
            # Fallback: assign uniform mid-range score so retrieval still works
            for i in range(len(batch)):
                all_results.append({
                    "index": batch_start + i,
                    "score": CONFIDENCE_MEDIUM + 0.01,  # just above medium threshold
                })

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# LLM GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def llm_generate(
    prompt: str,
    model: str = PRIMARY_LLM,
    system_prompt: str = "",
    max_tokens: int = 1024,
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


# ═══════════════════════════════════════════════════════════════════════════════
# MILVUS
# ═══════════════════════════════════════════════════════════════════════════════

_milvus_client = None


def get_milvus():
    global _milvus_client
    if _milvus_client is None:
        from pymilvus import MilvusClient
        pstatus(f"Connecting to Milvus: {MILVUS_DB}")
        _milvus_client = MilvusClient(uri=MILVUS_DB)
        if not _milvus_client.has_collection(COLLECTION):
            _milvus_client.create_collection(
                collection_name=COLLECTION,
                dimension=DIM,
                metric_type="IP",    # inner-product (cosine after L2 norm)
                auto_id=True,
            )
            pok(f"Created collection: {COLLECTION}")
        else:
            pok(f"Collection '{COLLECTION}' exists")
    return _milvus_client


def reset_collection():
    milvus = get_milvus()
    if milvus.has_collection(COLLECTION):
        milvus.drop_collection(COLLECTION)
    milvus.create_collection(
        collection_name=COLLECTION,
        dimension=DIM,
        metric_type="IP",
        auto_id=True,
    )
    pok(f"Reset collection: {COLLECTION}")


def _close_milvus():
    global _milvus_client
    if _milvus_client is not None:
        try:
            _milvus_client.close()
        except Exception:
            pass
        _milvus_client = None


# ═══════════════════════════════════════════════════════════════════════════════
# NV-INGEST PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

_pipeline_started = False


def _wait_for_broker(host: str = BROKER_HOST, port: int = BROKER_PORT, timeout: int = 120):
    pstatus(f"Waiting for broker {host}:{port}...")
    deadline = time.time() + timeout
    dots = 0
    while time.time() < deadline:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        if s.connect_ex((host, port)) == 0:
            s.close()
            pok("Broker ready")
            return
        s.close()
        dots += 1
        if dots % 20 == 0:
            pstatus(f"Still waiting… ({int(time.time()-(deadline-timeout))}s)", C.GRAY)
        time.sleep(0.5)
    raise RuntimeError(f"Broker not reachable after {timeout}s")


def _start_pipeline_once():
    global _pipeline_started
    if _pipeline_started:
        return
    pstatus("Importing NV-Ingest (loads Ray internally)…")
    t0 = time.time()
    from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import (
        PipelineCreationSchema, run_pipeline,
    )
    pok(f"NV-Ingest imported ({time.time()-t0:.1f}s)")
    pstatus("Launching pipeline subprocess…")
    pwarn("First run takes 2–5 min. Please wait.")
    cfg = PipelineCreationSchema()
    run_pipeline(cfg, block=False, disable_dynamic_scaling=True, run_in_subprocess=True)
    _wait_for_broker()
    _pipeline_started = True
    pok(f"Pipeline ready ({time.time()-t0:.1f}s)")


def run_ingest(file_paths: List[str], reset: bool = False) -> Dict[str, Any]:
    from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
    from nv_ingest_client.client import Ingestor, NvIngestClient

    _start_pipeline_once()
    if reset:
        reset_collection()
    _close_milvus()

    pstatus(f"Ingesting {len(file_paths)} file(s)…")
    for fp in file_paths:
        pstatus(f"  → {os.path.basename(fp)} ({os.path.getsize(fp)/1024:.0f} KB)", C.GRAY)

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
            extract_images=True,
            extract_infographics=True,
            table_output_format="markdown",
            text_depth="page",
        )
        .split(
            tokenizer="bert-base-uncased",
            chunk_size=512,
            chunk_overlap=100,
            params={"split_source_types": ["text", "structured", "chart"]},
        )
        .caption(
            endpoint_url=CAPTION_URL,
            model_name=CAPTION_MODEL,
            api_key=NVIDIA_API_KEY,
        )
        .embed(
            endpoint_url=CHAT_API_BASE + "/v1",
            model_name=EMBED_MODEL,
            api_key=NVIDIA_API_KEY,
        )
        .vdb_upload(
            collection_name=COLLECTION,
            milvus_uri=MILVUS_DB,
            dense_dim=DIM,
        )
    )

    t0 = time.time()
    pstatus("Running: load → extract → split → caption → embed → vdb_upload")
    results, failures = ingestor.ingest(show_progress=True, return_failures=True)
    results  = list(results)
    elapsed  = round((time.time() - t0) * 1000)
    n_fail   = len(failures) if failures else 0

    pok(f"{len(results)} chunks ingested in {elapsed:,}ms ({n_fail} failures)")
    return {
        "files":           [os.path.basename(p) for p in file_paths],
        "chunks_ingested": len(results),
        "failures":        n_fail,
        "elapsed_ms":      elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CHUNK TYPE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_chunk_modality(text: str) -> str:
    """Classify chunk by its content signals."""
    t = text.strip()
    t_lower = t.lower()

    # NV-Ingest caption output always starts with [Caption
    if t.startswith("[Caption") or t.startswith("[caption"):
        return "caption"
    # Markdown table: multiple pipes
    if "|" in t and t.count("|") > 4:
        return "table"
    # Chart signals
    if any(w in t_lower for w in [
        "chart shows", "graph depicts", "plot of", "x-axis", "y-axis",
        "bar chart", "line graph", "scatter", "trend line",
    ]):
        return "chart"
    # Diagram / figure signals
    if any(w in t_lower for w in [
        "figure", "diagram", "illustration", "schematic",
        "arrow", "node", "flow", "zone", "region",
        "annotated", "labelled", "label",
    ]):
        return "diagram"
    # Infographic
    if any(w in t_lower for w in ["infographic", "visual summary", "icon"]):
        return "infographic"
    return "text"


# ═══════════════════════════════════════════════════════════════════════════════
# BM25-STYLE LEXICAL RESCUE  (no external dependency — pure Python)
# ═══════════════════════════════════════════════════════════════════════════════

def bm25_score(query_tokens: List[str], doc_text: str, k1: float = 1.5, b: float = 0.75,
               avg_dl: float = 200.0) -> float:
    doc_lower  = doc_text.lower()
    doc_tokens = re.findall(r"\w+", doc_lower)
    dl         = len(doc_tokens)
    tf_map     = Counter(doc_tokens)
    score = 0.0
    for tok in query_tokens:
        tf = tf_map.get(tok.lower(), 0)
        idf_proxy = 1.0  # simplified (no corpus DF here)
        num   = tf * (k1 + 1)
        denom = tf + k1 * (1 - b + b * dl / max(avg_dl, 1))
        score += idf_proxy * (num / max(denom, 1e-9))
    return score


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS — modality-aware
# ═══════════════════════════════════════════════════════════════════════════════

_SYS_BASE = """You are a precise, expert document assistant. Answer questions using ONLY the context chunks provided.

The context may contain these modalities — handle each correctly:
• TEXT     — plain prose from any document type
• TABLE    — markdown tables; read row/column intersections carefully
• CAPTION  — auto-generated descriptions of images, diagrams, charts, or figures
              (treat caption text AS the content of that visual)
• CHART    — data extracted or described from charts/graphs
• DIAGRAM  — flow diagrams, anatomical diagrams, process maps; use text labels

STRICT RULES:
1. Read ALL chunks before answering — the answer may span multiple chunks.
2. Quote numbers, dates, names, codes, and percentages EXACTLY as they appear.
3. For tables: identify the correct row AND column before extracting a value.
4. For captions: the caption IS the content — answer directly from it.
5. For diagrams: use the text labels and structural descriptions.
6. If a chunk looks like OCR noise, reason around it to find the intended value.
7. NEVER use outside knowledge. NEVER invent facts.
8. If the context genuinely does not contain the answer, say:
   "The provided documents do not contain this information."
   Then summarise the CLOSEST relevant information you did find."""

_SYS_BESTEFF = _SYS_BASE + """

IMPORTANT — BEST-EFFORT MODE:
Retrieval confidence was LOW for this query. The chunks below are the CLOSEST
matches found. Attempt to answer using available evidence. Be explicit about
your confidence level. If you can only partially answer, do so and state what
is missing."""

_SYS_STRICT = _SYS_BASE + """

STRICTNESS MODE — A previous answer was flagged for possible hallucination.
Answer ONLY with exact facts stated verbatim in the context. Do NOT infer,
interpolate, or extrapolate. If uncertain, say so."""


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT STATE
# ═══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    # Query
    original_query:       str
    current_query:        str
    query_variants:       List[str]
    detected_modality:    str          # dominant modality of the query

    # Retrieval
    raw_chunks:           List[Dict[str, Any]]
    ranked_chunks:        List[Dict[str, Any]]
    rescue_attempted:     bool
    rescue_chunks:        List[Dict[str, Any]]

    # Generation
    overall_confidence:   str
    top_score:            float
    answer:               str
    model_used:           str
    fallback_used:        bool
    generation_mode:      str          # "normal" | "best_effort" | "strict"

    # Infra
    guardrail_flags:      List[str]
    retry_count:          int
    node_latencies:       Dict[str, float]
    sources:              List[Dict[str, Any]]
    conversation_history: List[Dict]


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 1 — GUARDRAIL + QUERY CLEANER
# ═══════════════════════════════════════════════════════════════════════════════

def node_guardrail(state: AgentState) -> Dict[str, Any]:
    """
    • Validates query (length, injection patterns)
    • Strips filler phrases that add embedding noise
    • Does NOT block any modality — only injection attempts
    """
    t0    = time.time()
    query = state["current_query"]
    flags = list(state.get("guardrail_flags", []))

    if not query.strip():
        flags.append("empty_query")
        return {
            "guardrail_flags": flags,
            "node_latencies":  {**state.get("node_latencies", {}),
                                 "guardrail": round((time.time()-t0)*1000, 1)},
        }

    if len(query) > 2000:
        flags.append("query_too_long")
        query = query[:2000]

    for pat in BLOCKED_PATTERNS:
        if re.search(pat, query, re.IGNORECASE):
            flags.append("prompt_injection_detected")
            break

    # Strip filler phrases
    filler_patterns = [
        r"^(please\s+)?(can\s+you\s+)?(tell\s+me|explain|describe|show\s+me|find|"
        r"give\s+me|what\s+is|what\s+are|what\s+does\s+it\s+say\s+about|"
        r"what\s+do\s+you\s+know\s+about|i\s+want\s+to\s+know|i\s+need\s+to\s+know)\s+",
        r"^(in\s+the\s+(document[s]?|pdf|file)[,]?\s+)?(according\s+to\s+the\s+"
        r"(document[s]?|file|pdf|text)[,]?\s+)?",
        r"\s+(from\s+the\s+(document[s]?|file|pdf|text))\s*$",
    ]
    cleaned = query
    for pat in filler_patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
    if len(cleaned) < 4:
        cleaned = query  # safety

    elapsed = round((time.time()-t0)*1000, 1)
    pstatus(f"[N1] guardrail clean='{cleaned[:60]}' ({elapsed:.0f}ms)")
    return {
        "current_query": cleaned,
        "guardrail_flags": flags,
        "node_latencies": {**state.get("node_latencies", {}), "guardrail": elapsed},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 2 — INTENT / MODALITY ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

def node_intent_router(state: AgentState) -> Dict[str, Any]:
    """
    Detects the dominant modality of the query so that:
    1. The expander can generate modality-specific paraphrases
    2. The generator uses the right system prompt emphasis
    3. Rescue node knows which retrieval strategy to try
    """
    t0    = time.time()
    query = state["current_query"].lower()

    if any(w in query for w in [
        "table", "row", "column", "spreadsheet", "grid", "list of values",
        "entry", "record", "field",
    ]):
        modality = "table"
    elif any(w in query for w in [
        "image", "photo", "picture", "caption", "alt text", "figure",
        "illustration", "visual",
    ]):
        modality = "caption"
    elif any(w in query for w in [
        "chart", "graph", "plot", "trend", "bar", "pie", "line chart",
        "scatter", "x-axis", "y-axis",
    ]):
        modality = "chart"
    elif any(w in query for w in [
        "diagram", "flow", "zone", "region", "anatomy", "process map",
        "schematic", "labelled", "arrow",
    ]):
        modality = "diagram"
    elif any(w in query for w in ["colour", "color", "contrast", "palette", "hue",
                                   "accessibility", "wcag", "font", "text size"]):
        modality = "mixed"   # visual + text — this PDF specifically
    else:
        modality = "text"

    elapsed = round((time.time()-t0)*1000, 1)
    pstatus(f"[N2] intent_router modality={modality} ({elapsed:.0f}ms)")
    return {
        "detected_modality": modality,
        "node_latencies": {**state.get("node_latencies", {}), "intent_router": elapsed},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 3 — MULTI-MODAL QUERY EXPANDER
# ═══════════════════════════════════════════════════════════════════════════════

def node_multi_expander(state: AgentState) -> Dict[str, Any]:
    """
    Generates semantically diverse query variants:
    • 2 paraphrases via LLM (vocabulary gap bridging)
    • 1 modality-specific variant (e.g. "what does the caption say about X?")
    • 1 keyword extraction variant (stripped noun phrases)

    This gives 4–5 total queries for retrieval, dramatically improving recall
    on visual-heavy documents where phrasing mismatch is the #1 failure mode.
    """
    t0       = time.time()
    query    = state["current_query"]
    modality = state.get("detected_modality", "text")
    flags    = state.get("guardrail_flags", [])

    if "prompt_injection_detected" in flags or "empty_query" in flags:
        return {
            "query_variants":  [query],
            "node_latencies":  {**state.get("node_latencies", {}),
                                 "expander": round((time.time()-t0)*1000, 1)},
        }

    variants = [query]

    # ── LLM paraphrases ───────────────────────────────────────────────────────
    try:
        expand_prompt = (
            f"Generate exactly 2 alternative phrasings of this query.\n"
            f"Use different vocabulary and sentence structure but keep the same meaning.\n"
            f"Return ONLY the 2 alternatives, one per line, no numbering, no explanation.\n\n"
            f"Query: {query}"
        )
        raw   = llm_generate(expand_prompt, model=PRIMARY_LLM,
                             max_tokens=150, temperature=0.6)
        lines = [l.strip() for l in raw.strip().split("\n")
                 if l.strip() and len(l.strip()) > 4 and l.strip() != query]
        variants += lines[:2]
    except Exception as exc:
        logger.warning("LLM expander failed: %s", exc)

    # ── Modality-specific variant ─────────────────────────────────────────────
    modal_prefix_map = {
        "table":       "In the table, what is the value for",
        "caption":     "What does the figure caption describe about",
        "chart":       "In the chart or graph, what does the data show about",
        "diagram":     "In the diagram or figure, what does the label indicate for",
        "infographic": "In the visual or infographic, what information is shown about",
        "mixed":       "In the accessibility guide figures and text, what is described about",
        "text":        None,
    }
    prefix = modal_prefix_map.get(modality)
    if prefix:
        # Extract core noun phrase (drop question words)
        core = re.sub(
            r"^(what|how|where|when|why|who|which|does|is|are|can|will|"
            r"show me|tell me|find|describe)\s+",
            "", query, flags=re.IGNORECASE
        ).strip().rstrip("?")
        if core:
            variants.append(f"{prefix} {core}?")

    # ── Keyword extraction variant ────────────────────────────────────────────
    stop = {"the", "a", "an", "in", "of", "for", "to", "is", "are", "what",
            "how", "where", "when", "why", "who", "which", "does", "do",
            "can", "could", "would", "with", "and", "or", "not", "any"}
    kws = [w for w in re.findall(r"\b\w{3,}\b", query.lower()) if w not in stop]
    if len(kws) >= 2:
        variants.append(" ".join(kws))

    # Deduplicate while preserving order
    seen: set = set()
    unique_variants: List[str] = []
    for v in variants:
        key = v.lower().strip()
        if key not in seen:
            seen.add(key)
            unique_variants.append(v)

    elapsed = round((time.time()-t0)*1000, 1)
    pstatus(f"[N3] expander {len(unique_variants)} variants ({elapsed:.0f}ms)", C.GRAY)
    for i, v in enumerate(unique_variants):
        logger.info("  Variant %d: %s", i+1, v)

    return {
        "query_variants":  unique_variants,
        "node_latencies":  {**state.get("node_latencies", {}), "expander": elapsed},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 4 — HYBRID RETRIEVER  (dense ANN × N variants + dedup)
# ═══════════════════════════════════════════════════════════════════════════════

def node_hybrid_retriever(state: AgentState) -> Dict[str, Any]:
    """
    Runs independent Milvus ANN search per variant → merges → deduplicates.
    Also boosts caption/table chunks slightly so visual-heavy docs don't get
    buried under plain text chunks that happen to score marginally better.
    """
    t0       = time.time()
    query    = state["current_query"]
    variants = state.get("query_variants", [query])
    flags    = state.get("guardrail_flags", [])
    modality = state.get("detected_modality", "text")

    if "prompt_injection_detected" in flags:
        return {
            "raw_chunks": [],
            "node_latencies": {**state.get("node_latencies", {}),
                                "retriever": round((time.time()-t0)*1000, 1)},
        }

    raw_chunks: List[Dict[str, Any]] = []
    seen_hashes: set = set()

    try:
        milvus = get_milvus()

        for v_idx, variant in enumerate(variants):
            try:
                embeddings = embed_texts([variant], input_type="query")
            except Exception as exc:
                logger.warning("Embedding failed for variant %d: %s", v_idx, exc)
                continue

            if not embeddings:
                continue

            q_emb = embeddings[0]
            try:
                hits = milvus.search(
                    collection_name=COLLECTION,
                    data=[q_emb],
                    limit=RETRIEVAL_TOP_K,
                    output_fields=["text"],
                )[0]
            except Exception as exc:
                logger.warning("Milvus search failed for variant %d: %s", v_idx, exc)
                continue

            for hit in hits:
                entity = (hit.get("entity", hit) if isinstance(hit, dict)
                          else hit.entity)
                text   = (entity.get("text", "") if isinstance(entity, dict)
                          else getattr(entity, "text", ""))
                score  = (hit.get("distance", 0.0) if isinstance(hit, dict)
                          else getattr(hit, "distance", 0.0))

                if not text or not text.strip():
                    continue

                text_hash = hashlib.sha256(text.strip().encode()).hexdigest()
                if text_hash in seen_hashes:
                    continue
                seen_hashes.add(text_hash)

                chunk_modality = detect_chunk_modality(text)

                # ── Modality boost: if user asked about visual content,
                #    prioritise visual chunks so reranker sees them
                boost = 0.0
                if modality in ("caption", "chart", "diagram", "infographic", "mixed"):
                    if chunk_modality in ("caption", "chart", "diagram"):
                        boost = 0.05   # small boost to push into reranker window

                raw_chunks.append({
                    "text":           text,
                    "vector_score":   float(score) + boost,
                    "chunk_modality": chunk_modality,
                    "variant_idx":    v_idx,
                })

    except Exception as exc:
        perr(f"Retrieval failed: {exc}")
        logger.exception("Retrieval failed")

    # Sort by vector_score so reranker gets the best candidates first
    raw_chunks.sort(key=lambda x: x["vector_score"], reverse=True)

    elapsed = round((time.time()-t0)*1000, 1)
    modality_dist = Counter(c["chunk_modality"] for c in raw_chunks)
    pstatus(
        f"[N4] retriever {C.CYAN}{len(raw_chunks)} unique chunks{C.RESET} "
        f"from {len(variants)} variants | "
        f"modalities={dict(modality_dist)} ({elapsed:.0f}ms)"
    )
    return {
        "raw_chunks":    raw_chunks,
        "node_latencies": {**state.get("node_latencies", {}), "retriever": elapsed},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 5 — NVIDIA RERANKER
# ═══════════════════════════════════════════════════════════════════════════════

def node_nvidia_reranker(state: AgentState) -> Dict[str, Any]:
    """
    Reranks retrieved chunks using nvidia/nv-rerankqa-mistral-4b-v3 via API.
    Score range 0.0–1.0.

    Thresholds (calibrated for Mistral reranker on multimodal content):
      ≥ 0.45  → HIGH confidence
      ≥ 0.15  → MEDIUM confidence
      < 0.15  → LOW confidence  →  triggers adaptive_rescue
      ≥ 0.0   →  never silences  (MIN_GENERATION_SCORE = 0.0)

    CRITICAL: We never return empty — worst case we pass low-confidence
    chunks and generate a best-effort answer with explicit uncertainty.
    """
    t0         = time.time()
    query      = state["current_query"]
    raw_chunks = state.get("raw_chunks", [])
    ranked_chunks: List[Dict[str, Any]] = []
    overall_confidence = "low"
    top_score          = 0.0

    if not raw_chunks:
        elapsed = round((time.time()-t0)*1000, 1)
        pwarn("[N5] reranker: no chunks to rerank")
        return {
            "ranked_chunks":      [],
            "overall_confidence": "low",
            "top_score":          0.0,
            "node_latencies":     {**state.get("node_latencies", {}),
                                   "reranker": elapsed},
        }

    try:
        passages = [c["text"] for c in raw_chunks]
        rankings = rerank_passages_nvidia(query, passages)

        for rank in rankings[:RERANK_TOP_K]:
            idx   = int(rank.get("index", 0))
            score = float(rank.get("score", 0.0))
            if 0 <= idx < len(raw_chunks):
                chunk = raw_chunks[idx].copy()
                chunk["rerank_score"] = score

                if score >= CONFIDENCE_HIGH:
                    chunk["confidence"] = "high"
                elif score >= CONFIDENCE_MEDIUM:
                    chunk["confidence"] = "medium"
                else:
                    chunk["confidence"] = "low"

                ranked_chunks.append(chunk)

        if not ranked_chunks:
            # Fallback: pass all raw in vector order with medium confidence
            ranked_chunks = [
                {**c, "rerank_score": 0.2, "confidence": "medium"}
                for c in raw_chunks[:RERANK_TOP_K]
            ]

        top_score = ranked_chunks[0]["rerank_score"] if ranked_chunks else 0.0

        if top_score >= CONFIDENCE_HIGH:
            overall_confidence = "high"
        elif top_score >= CONFIDENCE_MEDIUM:
            overall_confidence = "medium"
        # else: "low" — rescue node will handle

    except Exception as exc:
        perr(f"Reranker failed: {exc} — falling back to vector order")
        ranked_chunks = [
            {**c, "rerank_score": CONFIDENCE_MEDIUM + 0.01, "confidence": "medium"}
            for c in raw_chunks[:RERANK_TOP_K]
        ]
        overall_confidence = "medium"
        top_score          = CONFIDENCE_MEDIUM + 0.01

    elapsed = round((time.time()-t0)*1000, 1)
    cc = {"high": C.GREEN, "medium": C.YELLOW, "low": C.RED}.get(
        overall_confidence, C.GRAY)
    pstatus(
        f"[N5] reranker {len(ranked_chunks)} chunks | "
        f"top_score={top_score:.4f} | "
        f"{cc}{overall_confidence.upper()}{C.RESET} ({elapsed:.0f}ms)"
    )

    return {
        "ranked_chunks":      ranked_chunks,
        "overall_confidence": overall_confidence,
        "top_score":          top_score,
        "node_latencies":     {**state.get("node_latencies", {}), "reranker": elapsed},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 7 — ADAPTIVE RESCUE  (triggered when top_score < RESCUE_THRESHOLD)
# ═══════════════════════════════════════════════════════════════════════════════

def node_adaptive_rescue(state: AgentState) -> Dict[str, Any]:
    """
    Three-strategy rescue for low-confidence retrievals:

    Strategy A — Caption-first re-retrieval:
      Reformulate query as "what does the figure/caption/image say about X?"
      and run a fresh ANN search. This catches visual chunks that were missed
      because the original query was phrased as a text/factual question.

    Strategy B — BM25 lexical fallback:
      Score ALL already-retrieved raw_chunks lexically against query keywords.
      Sometimes the dense embedding space clusters visual chunks far from
      the query embedding even when lexical match is strong.

    Strategy C — Expand to broader topic:
      Generate a broader version of the query and retrieve again.
      Handles cases where the query is too specific/narrow.

    Rescue chunks are merged with (and may replace) low-confidence ranked_chunks.
    Generation is ALWAYS triggered after rescue — worst case = best-effort answer.
    """
    t0       = time.time()
    query    = state["current_query"]
    raw      = state.get("raw_chunks", [])
    ranked   = state.get("ranked_chunks", [])
    modality = state.get("detected_modality", "text")

    pstatus(f"[N7] adaptive_rescue triggered (top_score={state.get('top_score',0):.4f})", C.YELLOW)

    rescue_chunks: List[Dict[str, Any]] = []
    seen_hashes: set = {
        hashlib.sha256(c["text"].strip().encode()).hexdigest()
        for c in ranked
    }

    # ─── Strategy A: Caption-first re-retrieval ───────────────────────────────
    try:
        core = re.sub(
            r"^(what|how|where|when|does|is|are|tell me|show me|describe|find)\s+",
            "", query, flags=re.IGNORECASE
        ).strip().rstrip("?") or query

        caption_query = f"What does the figure, diagram, caption, or chart show about {core}?"
        emb = embed_texts([caption_query], input_type="query")
        if emb:
            milvus = get_milvus()
            hits = milvus.search(
                collection_name=COLLECTION,
                data=[emb[0]],
                limit=30,
                output_fields=["text"],
            )[0]
            added_A = 0
            for hit in hits:
                entity = (hit.get("entity", hit) if isinstance(hit, dict)
                          else hit.entity)
                text   = (entity.get("text", "") if isinstance(entity, dict)
                          else getattr(entity, "text", ""))
                if not text or not text.strip():
                    continue
                h = hashlib.sha256(text.strip().encode()).hexdigest()
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    rescue_chunks.append({
                        "text":            text,
                        "vector_score":    0.0,
                        "chunk_modality":  detect_chunk_modality(text),
                        "rescue_strategy": "A_caption_requery",
                        "rerank_score":    RESCUE_THRESHOLD + 0.01,
                        "confidence":      "low",
                    })
                    added_A += 1
            pstatus(f"  [Rescue-A] caption re-query added {added_A} new chunks", C.GRAY)
    except Exception as exc:
        logger.warning("Rescue Strategy A failed: %s", exc)

    # ─── Strategy B: BM25 lexical re-scoring of already-retrieved chunks ──────
    try:
        query_tokens = re.findall(r"\w+", query.lower())
        stop = {"the", "a", "an", "in", "of", "for", "to", "is", "are",
                "what", "how", "where", "when", "why", "does", "do"}
        query_tokens = [t for t in query_tokens if t not in stop and len(t) > 2]

        if query_tokens and raw:
            # Compute avg doc length
            avg_dl = sum(len(re.findall(r"\w+", c["text"])) for c in raw) / max(len(raw), 1)
            bm25_scored: List[tuple] = []
            for chunk in raw:
                sc = bm25_score(query_tokens, chunk["text"], avg_dl=avg_dl)
                bm25_scored.append((sc, chunk))

            bm25_scored.sort(key=lambda x: x[0], reverse=True)
            added_B = 0
            for bsc, chunk in bm25_scored[:15]:
                if bsc < 0.5:
                    break
                h = hashlib.sha256(chunk["text"].strip().encode()).hexdigest()
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    rescue_chunks.append({
                        **chunk,
                        "rescue_strategy": "B_bm25",
                        "rerank_score":    RESCUE_THRESHOLD + 0.01,
                        "confidence":      "low",
                    })
                    added_B += 1
            pstatus(f"  [Rescue-B] BM25 re-score added {added_B} new chunks", C.GRAY)
    except Exception as exc:
        logger.warning("Rescue Strategy B failed: %s", exc)

    # ─── Strategy C: Broader topic re-retrieval ───────────────────────────────
    try:
        broad_prompt = (
            f"Rewrite this query as a broader, more general version that covers the same topic.\n"
            f"Return ONLY the rewritten query, no explanation.\n\nQuery: {query}"
        )
        broad_q = llm_generate(broad_prompt, model=PRIMARY_LLM,
                               max_tokens=80, temperature=0.3)
        if broad_q and broad_q.strip() and broad_q.strip() != query:
            emb = embed_texts([broad_q.strip()], input_type="query")
            if emb:
                milvus = get_milvus()
                hits = milvus.search(
                    collection_name=COLLECTION,
                    data=[emb[0]],
                    limit=25,
                    output_fields=["text"],
                )[0]
                added_C = 0
                for hit in hits:
                    entity = (hit.get("entity", hit) if isinstance(hit, dict)
                              else hit.entity)
                    text   = (entity.get("text", "") if isinstance(entity, dict)
                              else getattr(entity, "text", ""))
                    if not text or not text.strip():
                        continue
                    h = hashlib.sha256(text.strip().encode()).hexdigest()
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        rescue_chunks.append({
                            "text":            text,
                            "vector_score":    0.0,
                            "chunk_modality":  detect_chunk_modality(text),
                            "rescue_strategy": "C_broader",
                            "rerank_score":    RESCUE_THRESHOLD + 0.01,
                            "confidence":      "low",
                        })
                        added_C += 1
                pstatus(f"  [Rescue-C] broad re-query added {added_C} new chunks", C.GRAY)
    except Exception as exc:
        logger.warning("Rescue Strategy C failed: %s", exc)

    # ─── Re-rank rescue chunks via NVIDIA reranker ────────────────────────────
    all_rescue = ranked + rescue_chunks   # merge existing + new rescue chunks
    if rescue_chunks:
        try:
            passages = [c["text"] for c in all_rescue]
            re_rankings = rerank_passages_nvidia(query, passages)
            reranked: List[Dict[str, Any]] = []
            for r in re_rankings[:RERANK_TOP_K]:
                idx   = int(r.get("index", 0))
                score = float(r.get("score", 0.0))
                if 0 <= idx < len(all_rescue):
                    chunk = all_rescue[idx].copy()
                    chunk["rerank_score"] = score
                    chunk["confidence"]   = (
                        "high"   if score >= CONFIDENCE_HIGH   else
                        "medium" if score >= CONFIDENCE_MEDIUM  else
                        "low"
                    )
                    reranked.append(chunk)
            if reranked:
                all_rescue = reranked
        except Exception as exc:
            logger.warning("Rescue re-rank failed: %s", exc)

    # Ensure we always have something
    if not all_rescue:
        all_rescue = ranked  # last resort: use original ranked

    top_rescue_score = (all_rescue[0]["rerank_score"]
                        if all_rescue else RESCUE_THRESHOLD)
    new_confidence   = (
        "high"   if top_rescue_score >= CONFIDENCE_HIGH   else
        "medium" if top_rescue_score >= CONFIDENCE_MEDIUM  else
        "low"
    )

    elapsed = round((time.time()-t0)*1000, 1)
    pstatus(
        f"[N7] rescue done: {len(rescue_chunks)} new chunks | "
        f"total={len(all_rescue)} | top={top_rescue_score:.4f} | "
        f"conf={new_confidence} ({elapsed:.0f}ms)",
        C.YELLOW
    )

    return {
        "ranked_chunks":      all_rescue,
        "rescue_attempted":   True,
        "rescue_chunks":      rescue_chunks,
        "overall_confidence": new_confidence,
        "top_score":          top_rescue_score,
        "generation_mode":    "best_effort",
        "node_latencies":     {**state.get("node_latencies", {}), "rescue": elapsed},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NODE 6 — GENERATOR  (modal-aware, with conversation memory)
# ═══════════════════════════════════════════════════════════════════════════════

def node_generator(state: AgentState) -> Dict[str, Any]:
    """
    • Selects system prompt based on generation_mode:
        "normal"      → _SYS_BASE
        "best_effort" → _SYS_BESTEFF  (rescued or low-confidence path)
        "strict"      → _SYS_STRICT   (after hallucination retry)
    • Prepends last 5 turns of conversation history
    • Builds modality-annotated context
    • Tries PRIMARY_LLM → FALLBACK_LLM
    • NEVER returns empty — always gives client something usable
    """
    t0            = time.time()
    query         = state["original_query"]
    ranked        = state.get("ranked_chunks", [])
    flags         = list(state.get("guardrail_flags", []))
    history       = state.get("conversation_history", [])
    gen_mode      = state.get("generation_mode", "normal")
    confidence    = state.get("overall_confidence", "low")
    rescue_used   = state.get("rescue_attempted", False)

    # ── Injection / empty block ───────────────────────────────────────────────
    if "prompt_injection_detected" in flags:
        return {
            "answer":     "This query has been flagged and cannot be processed.",
            "model_used": "none", "fallback_used": False,
            "guardrail_flags": flags, "sources": [],
            "node_latencies": {**state.get("node_latencies", {}), "generator": 0.0},
        }

    if "empty_query" in flags:
        return {
            "answer":     "Please provide a question or query.",
            "model_used": "none", "fallback_used": False,
            "guardrail_flags": flags, "sources": [],
            "node_latencies": {**state.get("node_latencies", {}), "generator": 0.0},
        }

    # ── Build context ─────────────────────────────────────────────────────────
    # Always generate — even with 0 ranked chunks (edge case: empty index)
    if not ranked:
        return {
            "answer": (
                "No documents have been ingested yet, or no chunks matched your query.\n"
                "Please ingest a file first using: ingest <path>"
            ),
            "model_used": "none", "fallback_used": False,
            "guardrail_flags": flags, "sources": [],
            "node_latencies": {**state.get("node_latencies", {}), "generator": 0.0},
        }

    ctx_chunks = ranked[:MAX_CONTEXT]
    parts:   List[str] = []
    sources: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(ctx_chunks, 1):
        conf       = chunk.get("confidence", "?")
        score      = round(float(chunk.get("rerank_score", 0.0)), 4)
        cmod       = chunk.get("chunk_modality", detect_chunk_modality(chunk["text"]))
        rescue_tag = f" [rescue:{chunk['rescue_strategy']}]" \
                     if chunk.get("rescue_strategy") else ""
        parts.append(
            f"[Chunk {idx} | modality={cmod} | confidence={conf} | "
            f"score={score}{rescue_tag}]\n{chunk['text']}"
        )
        preview = chunk["text"][:200] + ("…" if len(chunk["text"]) > 200 else "")
        sources.append({
            "index":       idx,
            "text_preview":preview,
            "confidence":  conf,
            "rerank_score":score,
            "modality":    cmod,
            "rescue":      bool(chunk.get("rescue_strategy")),
        })

    context = "\n\n---\n\n".join(parts)

    # ── Conversation memory ───────────────────────────────────────────────────
    history_text = ""
    if history:
        history_text = "Previous conversation (most recent 5 turns):\n"
        for turn in history[-5:]:
            history_text += (
                f"User: {turn.get('query','')}\n"
                f"Assistant: {turn.get('answer','')}\n\n"
            )

    # ── System prompt selection ───────────────────────────────────────────────
    sys_map = {
        "normal":      _SYS_BASE,
        "best_effort": _SYS_BESTEFF,
        "strict":      _SYS_STRICT,
    }
    sys_prompt = sys_map.get(gen_mode, _SYS_BASE)

    # Add mode note to prompt
    mode_note = ""
    if gen_mode == "best_effort":
        mode_note = (
            "\n[SYSTEM NOTE: Retrieval confidence was low for this query. "
            "The chunks below are the closest available. "
            "Attempt to answer using available evidence and be transparent about confidence.]\n\n"
        )
    elif rescue_used:
        mode_note = (
            "\n[SYSTEM NOTE: Adaptive rescue was used to broaden the retrieved context.]\n\n"
        )

    prompt = (
        f"{history_text}"
        f"{mode_note}"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )

    # ── LLM call with fallback ────────────────────────────────────────────────
    answer       = ""
    model_used   = "none"
    fallback_used = False

    for model in [PRIMARY_LLM, FALLBACK_LLM]:
        try:
            pstatus(f"  Generating with {model.split('/')[-1]}…", C.GRAY)
            candidate = llm_generate(
                prompt,
                model=model,
                system_prompt=sys_prompt,
                max_tokens=1024,
                temperature=0.2,
            )
            logger.info("LLM %s → length=%d", model, len(candidate))
            if candidate.strip():
                answer       = candidate.strip()
                model_used   = model
                fallback_used = (model == FALLBACK_LLM)
                break
            logger.warning("LLM %s returned empty, trying fallback", model)
        except Exception as exc:
            perr(f"LLM {model.split('/')[-1]} failed: {exc}")
            logger.exception("LLM %s failed", model)

    if not answer:
        # Absolute last resort — synthesise from chunk previews
        answer = (
            "Both LLMs failed to respond. "
            "Here are the most relevant passages found in the document:\n\n"
        )
        for s in sources[:3]:
            answer += f"• [{s['modality'].upper()}] {s['text_preview']}\n"
        flags.append("empty_answer")

    # ── Hallucination detection ───────────────────────────────────────────────
    lowered = answer.lower()
    for phrase in HALLUCINATION_PHRASES:
        if phrase in lowered:
            flags.append("possible_hallucination")
            break

    elapsed = round((time.time()-t0)*1000, 1)
    model_label = model_used.split("/")[-1] if model_used != "none" else "none"
    pstatus(
        f"[N6] generator {C.CYAN}{model_label}{C.RESET} | "
        f"mode={gen_mode} | rescue={rescue_used} ({elapsed:.0f}ms)"
    )

    return {
        "answer":         answer,
        "model_used":     model_used,
        "fallback_used":  fallback_used,
        "guardrail_flags":flags,
        "sources":        sources,
        "node_latencies": {**state.get("node_latencies", {}), "generator": elapsed},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def route_after_guardrail(state: AgentState) -> str:
    flags = state.get("guardrail_flags", [])
    if "prompt_injection_detected" in flags or "empty_query" in flags:
        return "generator"
    return "intent_router"


def route_after_reranker(state: AgentState) -> str:
    """
    Key routing decision:
    - If top_score < RESCUE_THRESHOLD → adaptive_rescue
    - Otherwise → generator
    This replaces the old quality gate that silenced the pipeline.
    We ALWAYS proceed to generation.
    """
    top_score = state.get("top_score", 0.0)
    ranked    = state.get("ranked_chunks", [])

    if not ranked:
        # No chunks at all — go to generator which will explain this
        return "generator"

    if top_score < RESCUE_THRESHOLD:
        pwarn(f"[ROUTE] top_score={top_score:.4f} < {RESCUE_THRESHOLD} → adaptive_rescue")
        return "adaptive_rescue"

    pstatus(f"[ROUTE] top_score={top_score:.4f} → generator", C.GREEN)
    return "generator"


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("guardrail",        node_guardrail)
    graph.add_node("intent_router",    node_intent_router)
    graph.add_node("multi_expander",   node_multi_expander)
    graph.add_node("hybrid_retriever", node_hybrid_retriever)
    graph.add_node("nvidia_reranker",  node_nvidia_reranker)
    graph.add_node("adaptive_rescue",  node_adaptive_rescue)
    graph.add_node("generator",        node_generator)

    # Edges
    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"intent_router": "intent_router", "generator": "generator"},
    )
    graph.add_edge("intent_router",    "multi_expander")
    graph.add_edge("multi_expander",   "hybrid_retriever")
    graph.add_edge("hybrid_retriever", "nvidia_reranker")
    graph.add_conditional_edges(
        "nvidia_reranker",
        route_after_reranker,
        {"adaptive_rescue": "adaptive_rescue", "generator": "generator"},
    )
    graph.add_edge("adaptive_rescue",  "generator")
    graph.add_edge("generator",        END)

    _compiled_graph = graph.compile()
    return _compiled_graph


# ═══════════════════════════════════════════════════════════════════════════════
# INITIAL STATE & RETRY
# ═══════════════════════════════════════════════════════════════════════════════

def _initial_state(query: str,
                   history: Optional[List] = None) -> AgentState:
    return {
        "original_query":      query,
        "current_query":       query,
        "query_variants":      [query],
        "detected_modality":   "unknown",
        "conversation_history":history or [],
        "raw_chunks":          [],
        "ranked_chunks":       [],
        "rescue_attempted":    False,
        "rescue_chunks":       [],
        "overall_confidence":  "low",
        "top_score":           0.0,
        "answer":              "",
        "model_used":          "",
        "fallback_used":       False,
        "generation_mode":     "normal",
        "guardrail_flags":     [],
        "retry_count":         0,
        "node_latencies":      {},
        "sources":             [],
    }


def _prepare_retry_state(state: AgentState) -> Optional[AgentState]:
    retry_count = state.get("retry_count", 0)
    if retry_count >= MAX_RETRIES:
        return None
    flags = state.get("guardrail_flags", [])
    if "possible_hallucination" not in flags:
        return None

    pwarn(f"Hallucination detected → retry {retry_count + 1} (strict mode)")
    new_query = (
        f"{state['original_query']} "
        f"-- answer ONLY with exact verbatim facts from the document. Do not infer."
    )
    return {
        **state,
        "current_query":    new_query,
        "query_variants":   [new_query],
        "raw_chunks":       [],
        "ranked_chunks":    [],
        "rescue_attempted": False,
        "rescue_chunks":    [],
        "sources":          [],
        "answer":           "",
        "model_used":       "",
        "fallback_used":    False,
        "top_score":        0.0,
        "generation_mode":  "strict",
        "retry_count":      retry_count + 1,
        "guardrail_flags":  [f for f in flags
                             if f not in {"possible_hallucination", "empty_answer"}],
        "node_latencies":   {},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RUN AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_agent(query: str,
              history: Optional[List] = None) -> Dict[str, Any]:
    graph  = get_graph()
    state  = _initial_state(query, history=history)
    t0     = time.time()

    while True:
        result      = graph.invoke(state)
        retry_state = _prepare_retry_state(result)
        if retry_state is None:
            result["wall_ms"] = round((time.time() - t0) * 1000)
            return result
        state = retry_state


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

def print_answer(answer, confidence, wall_ms, model, retry_count,
                 sources, latencies, flags, gen_mode, rescue_used):
    cc = {"high": C.GREEN, "medium": C.YELLOW, "low": C.RED}.get(confidence, C.GRAY)
    model_label = model.split("/")[-1] if model and model != "none" else "none"
    mode_badge  = {
        "normal":      "",
        "best_effort": f" | {C.YELLOW}BEST-EFFORT{C.RESET}",
        "strict":      f" | {C.MAGENTA}STRICT{C.RESET}",
    }.get(gen_mode, "")
    rescue_badge = f" | {C.YELLOW}RESCUED{C.RESET}" if rescue_used else ""

    print(f"\n{C.BOLD}╔── ANSWER ────────────────────────────────────────────────╗{C.RESET}")
    print(
        f"{C.BOLD}║{C.RESET} {cc}{confidence.upper()} CONFIDENCE{C.RESET}"
        f"{mode_badge}{rescue_badge}"
        f"  │  {C.GRAY}{model_label}{C.RESET}  │  {C.GRAY}{wall_ms:,}ms{C.RESET}"
    )
    if retry_count > 0:
        print(f"{C.BOLD}║{C.RESET} {C.YELLOW}retried {retry_count}x (hallucination guard){C.RESET}")
    print(f"{C.BOLD}╠──────────────────────────────────────────────────────────╣{C.RESET}")

    rendered = answer or "No answer generated."
    for line in rendered.split("\n"):
        while len(line) > 56:
            idx = line[:56].rfind(" ")
            if idx == -1:
                idx = 56
            print(f"{C.BOLD}║{C.RESET} {line[:idx]}")
            line = line[idx:].lstrip()
        print(f"{C.BOLD}║{C.RESET} {line}")

    print(f"{C.BOLD}╠──────────────────────────────────────────────────────────╣{C.RESET}")

    if latencies:
        print(f"\n  {C.GRAY}Latency breakdown:{C.RESET}")
        all_nodes = ["guardrail", "intent_router", "expander", "retriever",
                     "reranker", "rescue", "generator"]
        max_ms = max(latencies.values()) if latencies.values() else 1
        for node in all_nodes:
            if node in latencies:
                ms = latencies[node]
                bar_len = int(ms / max(max_ms, 1) * 28)
                bar = "█" * bar_len + "░" * (28 - bar_len)
                print(f"    {C.GRAY}{node:>14}{C.RESET}  {C.BLUE}{bar}{C.RESET}  {ms:.0f}ms")

    if sources:
        print(f"\n  {C.GRAY}Sources ({len(sources)} chunks passed to LLM):{C.RESET}")
        for s in sources[:5]:
            sc = {"high": C.GREEN, "medium": C.YELLOW, "low": C.RED}.get(
                s.get("confidence", "low"), C.GRAY)
            rescue_tag = f" {C.YELLOW}[RESCUED]{C.RESET}" if s.get("rescue") else ""
            print(
                f"    {C.BLUE}#{s['index']}{C.RESET} "
                f"{sc}{s.get('confidence','?')}{C.RESET}  "
                f"[{s.get('modality','?').upper()}]  "
                f"score={s.get('rerank_score',0):.4f}{rescue_tag}"
            )
            print(f"    {C.DIM}{s.get('text_preview','')[:100]}…{C.RESET}")

    if flags:
        print(f"\n  {C.YELLOW}⚑ Flags: {', '.join(flags)}{C.RESET}")

    print(f"{C.BOLD}╚══════════════════════════════════════════════════════════╝{C.RESET}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_loop():
    print(f"""
{C.BOLD}Commands:{C.RESET}
  {C.CYAN}ingest <path> [path2 …]{C.RESET}   Ingest file(s)
  {C.CYAN}ingest --reset <path>{C.RESET}     Reset collection then ingest
  {C.CYAN}stats{C.RESET}                     Show chunk count
  {C.CYAN}reset{C.RESET}                     Clear collection
  {C.CYAN}history{C.RESET}                   Show conversation memory
  {C.CYAN}clear{C.RESET}                     Clear conversation memory
  {C.CYAN}quit{C.RESET}                      Exit
""")

    conversation_history: List[Dict] = []

    while True:
        try:
            user_input = input(f"\n{C.GREEN}{C.BOLD}Q: {C.RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.GRAY}Session ended.{C.RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print(f"{C.GRAY}Session ended.{C.RESET}")
            break

        if user_input.lower() == "reset":
            try:
                reset_collection()
            except Exception as exc:
                perr(f"Reset failed: {exc}")
            continue

        if user_input.lower() == "clear":
            conversation_history = []
            pok("Conversation memory cleared.")
            continue

        if user_input.lower() == "history":
            if not conversation_history:
                pstatus("No conversation history yet.", C.GRAY)
            for i, turn in enumerate(conversation_history, 1):
                print(f"  {C.BLUE}Turn {i}{C.RESET}")
                print(f"    Q: {turn.get('query','')[:80]}")
                print(f"    A: {turn.get('answer','')[:80]}…")
            continue

        if user_input.lower().startswith("ingest"):
            parts   = user_input.split()
            do_reset = "--reset" in parts
            paths   = [p for p in parts[1:] if p != "--reset"]
            valid   = [p for p in paths if os.path.isfile(p)]
            for p in paths:
                if not os.path.isfile(p):
                    perr(f"File not found: {p}")
            if valid:
                try:
                    run_ingest(valid, reset=do_reset)
                except Exception as exc:
                    perr(f"Ingest failed: {exc}")
                    logger.exception("Ingest failed")
            continue

        if user_input.lower() == "stats":
            try:
                milvus = get_milvus()
                stats  = milvus.get_collection_stats(COLLECTION)
                pok(f"Collection: {COLLECTION}")
                pstatus(f"Stats: {json.dumps(stats, indent=2)}", C.GRAY)
            except Exception as exc:
                perr(f"Stats failed: {exc}")
            continue

        pstatus(f"Query: {C.WHITE}{user_input}{C.RESET}")
        if conversation_history:
            pstatus(f"Memory: {len(conversation_history)} turn(s) in context", C.GRAY)
        print()

        try:
            result = run_agent(user_input, history=conversation_history)

            # Update memory (only when model actually answered)
            if result.get("answer") and result.get("model_used") != "none":
                conversation_history.append({
                    "query":  user_input,
                    "answer": result["answer"],
                })
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]

            print_answer(
                answer       = result.get("answer", ""),
                confidence   = result.get("overall_confidence", "low"),
                wall_ms      = result.get("wall_ms", 0),
                model        = result.get("model_used", "?"),
                retry_count  = result.get("retry_count", 0),
                sources      = result.get("sources", []),
                latencies    = result.get("node_latencies", {}),
                flags        = result.get("guardrail_flags", []),
                gen_mode     = result.get("generation_mode", "normal"),
                rescue_used  = result.get("rescue_attempted", False),
            )
        except Exception as exc:
            perr(f"Agent failed: {exc}")
            logger.exception("Agent failed")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Enterprise Multimodal RAG Agent v3 — NV-Ingest + LangGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rag_agent_v3.py                         # Auto-ingest INGEST_FILES then query
  python rag_agent_v3.py --reset                 # Drop collection, re-ingest, query
  python rag_agent_v3.py --ingest Docs/Oxford.pdf
        """,
    )
    parser.add_argument("--reset",  action="store_true",
                        help="Drop and recreate the Milvus collection")
    parser.add_argument("--ingest", nargs="+", metavar="FILE",
                        help="Files to ingest on startup")
    args = parser.parse_args()

    banner()

    if not NVIDIA_API_KEY:
        perr("NVIDIA_API_KEY not set.")
        pstatus("export NVIDIA_API_KEY='nvapi-...'", C.GRAY)
        sys.exit(1)

    pok(f"NVIDIA_API_KEY  : {NVIDIA_API_KEY[:15]}…")
    pok(f"Milvus DB       : {MILVUS_DB}")
    pok(f"Collection      : {COLLECTION}")
    pok(f"Embed model     : {EMBED_MODEL}")
    pok(f"Reranker        : {RERANK_MODEL}  (NVIDIA API)")
    pok(f"LLM primary     : {PRIMARY_LLM}")
    pok(f"LLM fallback    : {FALLBACK_LLM}")
    pok(f"Confidence HIGH : ≥{CONFIDENCE_HIGH}  MEDIUM: ≥{CONFIDENCE_MEDIUM}  RESCUE: <{RESCUE_THRESHOLD}")
    pok(f"Retrieval top-k : {RETRIEVAL_TOP_K}   Rerank top-k: {RERANK_TOP_K}   Context: {MAX_CONTEXT}")
    pok(f"Min gen score   : {MIN_GENERATION_SCORE} (NEVER silences)")

    # ── Files to ingest ────────────────────────────────────────────────────────
    files_to_ingest = args.ingest or [f for f in INGEST_FILES if f.strip()]
    if files_to_ingest:
        valid   = [f for f in files_to_ingest if os.path.isfile(f)]
        missing = [f for f in files_to_ingest if not os.path.isfile(f)]
        for f in missing:
            perr(f"File not found: {f}")
        if valid:
            pstatus(f"Auto-ingesting {len(valid)} file(s)…")
            try:
                run_ingest(valid, reset=args.reset)
            except Exception as exc:
                perr(f"Ingest failed: {exc}")
                logger.exception("Ingest failed")
                sys.exit(1)
    elif args.reset:
        reset_collection()
        pok("Collection reset.")

    interactive_loop()


if __name__ == "__main__":
    main()
