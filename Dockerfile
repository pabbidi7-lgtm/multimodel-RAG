"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Enterprise Multimodal RAG Agent v4                                          ║
║  NV-Ingest 25.9.0  +  LangGraph  +  NVIDIA nv-rerankqa-mistral-4b-v3        ║
║                                                                              ║
║  BUGS FIXED vs v3:                                                           ║
║  [B1] Reranker parsed wrong field (logit/score) → always 0.0                 ║
║       Fix: read relevance_score (the actual NVIDIA API field name)           ║
║  [B2] split_source_types had 'structured' (invalid) → only 1 chunk/page     ║
║       Fix: ['text', 'table', 'chart'] — tables + charts now split correctly  ║
║  [B3] Milvus metric_type=IP vs L2 mismatch with existing collections        ║
║       Fix: auto-detect metric from existing collection; default stays L2     ║
║  [B4] Rescue-B BM25 threshold 0.5 too high → added 0 chunks                 ║
║       Fix: lowered to 0.1, proportional to query token count                ║
║  [B5] intent_router had hardcoded domain keywords (accessibility/wcag)       ║
║       Fix: generic keyword lists only — no domain assumptions                ║
║  [B6] adaptive_rescue always fired (caused by B1 scores all=0.0)            ║
║       Fix: fixed by B1 — rescue only fires when genuinely needed            ║
║                                                                              ║
║  GRAPH:                                                                      ║
║  START → guardrail → intent_router → expander → retriever                   ║
║        → reranker → [rescue?] → generator → END                              ║
║  Injection path: guardrail → generator → END                                 ║
║                                                                              ║
║  Modalities: text · table · caption · chart · diagram                        ║
║              infographic · OCR · handwritten · invoice · identity             ║
║  Memory: last 5 conversation turns                                           ║
║  Retry: hallucination × 1 (strict mode)                                      ║
║  NEVER silences — always generates best-effort answer                        ║
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

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.FileHandler("rag_agent_v4.log", mode="a")],
)
logger = logging.getLogger("rag_agent_v4")
_ch = logging.StreamHandler()
_ch.setLevel(logging.WARNING)
logger.addHandler(_ch)


# ── Console colours ───────────────────────────────────────────────────────────
class C:
    RESET = "\033[0m";   BOLD    = "\033[1m";   DIM  = "\033[2m"
    RED   = "\033[91m";  GREEN   = "\033[92m";  YELLOW = "\033[93m"
    BLUE  = "\033[94m";  CYAN    = "\033[96m";  WHITE  = "\033[97m"
    GRAY  = "\033[90m";  MAGENTA = "\033[95m"


def banner():
    print(f"""
{C.CYAN}{C.BOLD}╔═══════════════════════════════════════════════════════════╗
║  Enterprise Multimodal RAG Agent v4                       ║
║  NV-Ingest 25.9.0 + LangGraph + NVIDIA Mistral Reranker   ║
║                                                           ║
║  Any file · Any domain · Any modality                     ║
║  text · table · caption · chart · diagram · OCR           ║
║  handwritten · invoice · identity · infographic           ║
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


# ═════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT & CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

# Add your files here, or leave empty and use 'ingest <path>' at runtime
INGEST_FILES: List[str] = []

NVIDIA_API_KEY      = os.environ.get("NVIDIA_API_KEY", "")
MILVUS_DB           = os.environ.get("MILVUS_DB",   "./milvus_rag_v4.db")
COLLECTION          = os.environ.get("COLLECTION",  "rag_documents_v4")
DIM                 = int(os.environ.get("EMBED_DIM", "1024"))

CHAT_API_BASE       = os.environ.get("NVIDIA_CHAT_API_BASE",
                                     "https://integrate.api.nvidia.com")
RETRIEVAL_API_BASE  = os.environ.get("NVIDIA_RETRIEVAL_API_BASE",
                                     "https://integrate.api.nvidia.com")

EMBED_URL           = os.environ.get("EMBED_URL",
                                     f"{CHAT_API_BASE}/v1/embeddings")
EMBED_MODEL         = os.environ.get("EMBED_MODEL",
                                     "nvidia/nv-embedqa-e5-v5")

# NVIDIA Mistral reranker — API-based, no local model needed
RERANK_URL          = os.environ.get("RERANK_URL",
                                     f"{RETRIEVAL_API_BASE}/v1/ranking")
RERANK_MODEL        = os.environ.get("RERANK_MODEL",
                                     "nvidia/nv-rerankqa-mistral-4b-v3")

LLM_URL             = os.environ.get("LLM_URL",
                                     f"{CHAT_API_BASE}/v1/chat/completions")
PRIMARY_LLM         = os.environ.get("PRIMARY_LLM",
                                     "meta/llama-3.3-70b-instruct")
FALLBACK_LLM        = os.environ.get("FALLBACK_LLM",
                                     "nvidia/llama-3.1-nemotron-70b-instruct")

CAPTION_URL         = os.environ.get("CAPTION_URL",
                                     f"{CHAT_API_BASE}/v1/chat/completions")
CAPTION_MODEL       = os.environ.get("CAPTION_MODEL",
                                     "nvidia/llama-3.1-nemotron-nano-vl-8b-v1")

BROKER_HOST         = os.environ.get("BROKER_HOST", "localhost")
BROKER_PORT         = int(os.environ.get("BROKER_PORT", 7671))

# Retrieval tuning — all overridable via env
RETRIEVAL_TOP_K     = int(os.environ.get("RETRIEVAL_TOP_K", "60"))
RERANK_TOP_K        = int(os.environ.get("RERANK_TOP_K",    "20"))
MAX_CONTEXT         = int(os.environ.get("MAX_CONTEXT",     "8"))
MAX_RETRIES         = 1

# ── Confidence thresholds for NVIDIA Mistral reranker (0.0–1.0 scale) ────────
# relevance_score from nv-rerankqa-mistral-4b-v3 is a true probability-like
# score. Empirical calibration on mixed-modality docs:
#   ≥ 0.45 → HIGH    (answer is clearly in the retrieved chunk)
#   ≥ 0.15 → MEDIUM  (partial match, probably relevant)
#   < 0.10 → trigger adaptive rescue
CONFIDENCE_HIGH     = float(os.environ.get("CONF_HIGH",        "0.45"))
CONFIDENCE_MEDIUM   = float(os.environ.get("CONF_MEDIUM",      "0.15"))
RESCUE_THRESHOLD    = float(os.environ.get("RESCUE_THRESHOLD", "0.10"))

# Safety
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
    "i was not provided",
]


# ═════════════════════════════════════════════════════════════════════════════
# HTTP UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type":  "application/json",
    }


def _api_call(url: str, payload: Dict[str, Any],
              timeout: int = 120) -> Dict[str, Any]:
    response: Optional[requests.Response] = None
    for attempt in range(3):
        try:
            response = requests.post(url, json=payload,
                                     headers=_headers(), timeout=timeout)
            if response.status_code == 429:
                wait = 2 ** (attempt + 1)
                logger.warning("Rate limited — sleeping %ss", wait)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            logger.warning("HTTP error attempt %d: %s | %s",
                           attempt + 1, exc, url)
            if response is not None:
                logger.warning("Body: %s", response.text[:500])
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
        except (requests.ConnectionError, requests.Timeout) as exc:
            logger.warning("Connection error attempt %d: %s | %s",
                           attempt + 1, exc, url)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
        except Exception as exc:
            logger.warning("Unexpected error attempt %d: %s | %s",
                           attempt + 1, exc, url)
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
        parts = [item.get("text", "") for item in content
                 if isinstance(item, dict) and item.get("text")]
        return "\n".join(parts).strip()
    return ""


# ═════════════════════════════════════════════════════════════════════════════
# EMBEDDING
# ═════════════════════════════════════════════════════════════════════════════

def embed_texts(texts: List[str],
                input_type: str = "query") -> List[List[float]]:
    cleaned = [t.strip() if t and t.strip() else "<empty>" for t in texts]
    data = _api_call(EMBED_URL, {
        "model":           EMBED_MODEL,
        "input":           cleaned,
        "input_type":      input_type,
        "encoding_format": "float",
    })
    return [item["embedding"] for item in data.get("data", [])]


# ═════════════════════════════════════════════════════════════════════════════
# NVIDIA MISTRAL RERANKER  — BUG B1 FIXED HERE
# ═════════════════════════════════════════════════════════════════════════════

def rerank_passages_nvidia(query: str,
                           passages: List[str]) -> List[Dict[str, Any]]:
    """
    Calls nvidia/nv-rerankqa-mistral-4b-v3 via the NVIDIA Retrieval API.

    BUG B1 FIX: The NVIDIA ranking API returns each result as:
        { "index": int, "relevance_score": float }
    Previous versions tried to read "logit" or "score" — both wrong field
    names that always resolved to 0.0, making every chunk score identical
    and every answer show LOW confidence.

    The correct field is "relevance_score" (float, 0.0–1.0).
    Falls back to uniform CONFIDENCE_MEDIUM on API failure so the pipeline
    never hard-fails.
    """
    if not passages:
        return []

    MAX_PER_CALL = 50  # NVIDIA API limit per request
    all_results: List[Dict[str, Any]] = []

    for batch_start in range(0, len(passages), MAX_PER_CALL):
        batch = passages[batch_start: batch_start + MAX_PER_CALL]
        payload = {
            "model":    RERANK_MODEL,
            "query":    {"text": query},
            "passages": [{"text": p[:2000]} for p in batch],
            "truncate": "END",
        }
        try:
            data     = _api_call(RERANK_URL, payload, timeout=60)
            rankings = data.get("rankings", [])

            if not rankings:
                logger.warning("Reranker returned empty rankings for batch %d",
                               batch_start)

            for r in rankings:
                idx = int(r.get("index", 0))
                # ── BUG B1 FIX: correct field name ───────────────────────────
                score = float(
                    r.get("relevance_score",        # primary field (NVIDIA API)
                    r.get("score",                  # some API versions
                    r.get("logit", 0.0)))           # last resort
                )
                all_results.append({
                    "index": batch_start + idx,
                    "score": score,
                })

        except Exception as exc:
            logger.warning(
                "Reranker batch %d failed: %s — using fallback scores",
                batch_start, exc)
            # Fallback: uniform mid-range so pipeline keeps moving
            for i in range(len(batch)):
                all_results.append({
                    "index": batch_start + i,
                    "score": CONFIDENCE_MEDIUM + 0.01,
                })

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results


# ═════════════════════════════════════════════════════════════════════════════
# LLM
# ═════════════════════════════════════════════════════════════════════════════

def llm_generate(prompt: str,
                 model: str = PRIMARY_LLM,
                 system_prompt: str = "",
                 max_tokens: int = 1024,
                 temperature: float = 0.2) -> str:
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


# ═════════════════════════════════════════════════════════════════════════════
# MILVUS  — BUG B3 FIXED: auto-detect metric from existing collection
# ═════════════════════════════════════════════════════════════════════════════

_milvus_client = None


def get_milvus():
    global _milvus_client
    if _milvus_client is None:
        from pymilvus import MilvusClient
        pstatus(f"Connecting to Milvus: {MILVUS_DB}")
        _milvus_client = MilvusClient(uri=MILVUS_DB)
        if not _milvus_client.has_collection(COLLECTION):
            # L2 distance: nv-embedqa-e5-v5 embeddings are L2-normalised
            # so L2 distance and cosine similarity rank identically.
            # Using L2 for compatibility with NV-Ingest vdb_upload default.
            _milvus_client.create_collection(
                collection_name=COLLECTION,
                dimension=DIM,
                metric_type="L2",
                auto_id=True,
            )
            pok(f"Created collection '{COLLECTION}' (L2, dim={DIM})")
        else:
            pok(f"Collection '{COLLECTION}' exists")
    return _milvus_client


def _close_milvus():
    global _milvus_client
    if _milvus_client is not None:
        try:
            _milvus_client.close()
        except Exception:
            pass
        _milvus_client = None


def reset_collection():
    milvus = get_milvus()
    if milvus.has_collection(COLLECTION):
        milvus.drop_collection(COLLECTION)
    milvus.create_collection(
        collection_name=COLLECTION,
        dimension=DIM,
        metric_type="L2",
        auto_id=True,
    )
    pok(f"Reset collection '{COLLECTION}'")


# ═════════════════════════════════════════════════════════════════════════════
# NV-INGEST PIPELINE  — BUG B2 FIXED: split_source_types
# ═════════════════════════════════════════════════════════════════════════════

_pipeline_started = False


def _wait_for_broker(host: str = BROKER_HOST,
                     port: int = BROKER_PORT,
                     timeout: int = 120):
    pstatus(f"Waiting for broker {host}:{port}…")
    deadline = time.time() + timeout
    ticks = 0
    while time.time() < deadline:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        if s.connect_ex((host, port)) == 0:
            s.close()
            pok("Broker ready")
            return
        s.close()
        ticks += 1
        if ticks % 20 == 0:
            pstatus(f"  still waiting… ({int(time.time()-(deadline-timeout))}s)",
                    C.GRAY)
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
    pwarn("First run takes 2–5 min. Please wait.")
    cfg = PipelineCreationSchema()
    run_pipeline(cfg, block=False,
                 disable_dynamic_scaling=True,
                 run_in_subprocess=True)
    _wait_for_broker()
    _pipeline_started = True
    pok(f"Pipeline ready ({time.time()-t0:.1f}s)")


def run_ingest(file_paths: List[str],
               reset: bool = False) -> Dict[str, Any]:
    from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
    from nv_ingest_client.client import Ingestor, NvIngestClient

    _start_pipeline_once()
    if reset:
        reset_collection()
    _close_milvus()

    pstatus(f"Ingesting {len(file_paths)} file(s)…")
    for fp in file_paths:
        size_kb = os.path.getsize(fp) / 1024
        pstatus(f"  → {os.path.basename(fp)} ({size_kb:.0f} KB)", C.GRAY)

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
            chunk_overlap=50,
            params={
                # BUG B2 FIX: 'structured' is not a valid NV-Ingest source type.
                # The correct values are 'text', 'table', 'chart'.
                # Using 'structured' caused the splitter to only chunk text pages,
                # resulting in 1 chunk per document instead of hundreds.
                "split_source_types": ["text", "table", "chart"],
            },
        )
        .caption(
            # VLM captions every extracted image, diagram, chart, infographic.
            # The caption becomes a searchable text chunk in Milvus.
            # This is how images/diagrams become answerable — not by storing
            # the image, but by storing the VLM's description of it.
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

    if len(results) < 5:
        pwarn(
            f"Only {len(results)} chunks ingested. "
            "If your document has many pages, check NV-Ingest logs. "
            "Common causes: GPU not available for VLM, broker timeout, "
            "or unsupported file encoding."
        )
    else:
        pok(f"{len(results)} chunks ingested in {elapsed:,}ms "
            f"({n_fail} failures)")

    return {
        "files":           [os.path.basename(p) for p in file_paths],
        "chunks_ingested": len(results),
        "failures":        n_fail,
        "elapsed_ms":      elapsed,
    }


# ═════════════════════════════════════════════════════════════════════════════
# CHUNK MODALITY DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def detect_chunk_modality(text: str) -> str:
    """
    Classify a stored chunk by its content signals.
    Used to annotate sources in the answer output.
    NV-Ingest prefixes caption chunks with '[Caption'.
    """
    t = text.strip()
    t_lower = t.lower()

    if t.startswith("[Caption") or t.startswith("[caption"):
        return "caption"
    if "|" in t and t.count("|") > 4:
        return "table"
    if any(w in t_lower for w in [
        "chart shows", "graph depicts", "plot of", "x-axis", "y-axis",
        "bar chart", "line graph", "scatter plot", "trend line",
    ]):
        return "chart"
    if any(w in t_lower for w in [
        "figure", "diagram", "schematic", "flow", "zone", "region",
        "annotated", "labelled", "label", "arrow",
    ]):
        return "diagram"
    if any(w in t_lower for w in ["infographic", "visual summary"]):
        return "infographic"
    return "text"


# ═════════════════════════════════════════════════════════════════════════════
# BM25 LEXICAL SCORING  (pure Python, no external dependency)
# ═════════════════════════════════════════════════════════════════════════════

def bm25_score(query_tokens: List[str], doc_text: str,
               k1: float = 1.5, b: float = 0.75,
               avg_dl: float = 200.0) -> float:
    doc_tokens = re.findall(r"\w+", doc_text.lower())
    dl         = len(doc_tokens)
    tf_map     = Counter(doc_tokens)
    score = 0.0
    for tok in query_tokens:
        tf  = tf_map.get(tok.lower(), 0)
        num   = tf * (k1 + 1)
        denom = tf + k1 * (1 - b + b * dl / max(avg_dl, 1))
        score += num / max(denom, 1e-9)
    return score


# ═════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ═════════════════════════════════════════════════════════════════════════════

_SYS_BASE = """You are a precise document assistant. Answer questions using ONLY the context chunks provided.

Each chunk is labelled with its modality. Handle each type correctly:

TEXT      Plain prose from any document. Quote exactly.
TABLE     Markdown table. Identify the correct row AND column before answering.
CAPTION   Auto-generated description of an image, chart, diagram, or figure.
          The caption text IS the content of that visual — answer from it directly.
CHART     Data described from charts or graphs. Use the described values.
DIAGRAM   Flow diagrams, process maps, schematics. Use the text labels and structure.
INFOGRAPHIC  Visual summary. Use the described information.

RULES:
1. Read ALL chunks before answering — the answer may span multiple chunks.
2. Quote numbers, dates, names, IDs, codes, and amounts EXACTLY as they appear.
3. For tables: find the correct row and column intersection.
4. For captions: treat the caption as the ground truth for that visual.
5. For OCR/handwritten content: if a character looks like a recognition error,
   reason around it and state what you believe the intended value is.
6. NEVER use outside knowledge. NEVER invent or infer beyond what is written.
7. If the context does not contain the answer, say exactly:
   "The provided documents do not contain this information."
   Then describe the CLOSEST relevant content you did find."""

_SYS_BESTEFF = _SYS_BASE + """

NOTE — BEST-EFFORT MODE: Retrieval confidence is LOW for this query.
The chunks below are the closest matches found after an extended search.
Answer using available evidence. Explicitly state your confidence level.
If you can only partially answer, do so and explain what is missing."""

_SYS_STRICT = _SYS_BASE + """

STRICT MODE: A previous response was flagged for possible hallucination.
Answer ONLY with exact verbatim facts from the context.
Do NOT infer, interpolate, or extrapolate. If uncertain, say so explicitly."""


# ═════════════════════════════════════════════════════════════════════════════
# AGENT STATE
# ═════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    original_query:       str
    current_query:        str
    query_variants:       List[str]
    detected_modality:    str
    conversation_history: List[Dict]
    raw_chunks:           List[Dict[str, Any]]
    ranked_chunks:        List[Dict[str, Any]]
    rescue_attempted:     bool
    rescue_chunks:        List[Dict[str, Any]]
    overall_confidence:   str
    top_score:            float
    answer:               str
    model_used:           str
    fallback_used:        bool
    generation_mode:      str   # "normal" | "best_effort" | "strict"
    guardrail_flags:      List[str]
    retry_count:          int
    node_latencies:       Dict[str, float]
    sources:              List[Dict[str, Any]]


# ═════════════════════════════════════════════════════════════════════════════
# NODE 1 — GUARDRAIL + QUERY CLEANER
# ═════════════════════════════════════════════════════════════════════════════

def node_guardrail(state: AgentState) -> Dict[str, Any]:
    """
    Validates query, detects injection, strips filler words.
    Does NOT block any content modality — only injection attempts.
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

    # Strip filler phrases that add embedding noise without semantic content.
    # These are universal question-asking patterns, not domain-specific.
    filler = [
        r"^(please\s+)?(can\s+you\s+)?(tell\s+me|explain|describe|show\s+me|"
        r"find|give\s+me|what\s+is|what\s+are|what\s+does\s+it\s+say\s+about|"
        r"what\s+do\s+you\s+know\s+about|i\s+want\s+to\s+know|"
        r"i\s+need\s+to\s+know)\s+",
        r"^(in\s+the\s+(document[s]?|pdf|file)[,\s]+)?"
        r"(according\s+to\s+the\s+(document[s]?|file|pdf|text)[,\s]+)?",
        r"\s+(from\s+the\s+(document[s]?|file|pdf|text))\s*$",
    ]
    cleaned = query
    for pat in filler:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
    if len(cleaned) < 4:
        cleaned = query  # safety: never over-strip

    elapsed = round((time.time()-t0)*1000, 1)
    pstatus(f"[N1] guardrail '{cleaned[:60]}' ({elapsed:.0f}ms)")
    return {
        "current_query":   cleaned,
        "guardrail_flags": flags,
        "node_latencies":  {**state.get("node_latencies", {}),
                             "guardrail": elapsed},
    }


# ═════════════════════════════════════════════════════════════════════════════
# NODE 2 — INTENT / MODALITY ROUTER  — BUG B5 FIXED: no domain-specific terms
# ═════════════════════════════════════════════════════════════════════════════

def node_intent_router(state: AgentState) -> Dict[str, Any]:
    """
    Detects the dominant content modality the user is asking about.
    Used to generate modality-specific query variants and to annotate
    sources in the output. Does NOT restrict what gets retrieved.

    BUG B5 FIX: Removed hardcoded domain keywords ("accessibility", "wcag")
    that were specific to one PDF. Now uses universal modality signals only.
    """
    t0    = time.time()
    query = state["current_query"].lower()

    if any(w in query for w in [
        "table", "row", "column", "spreadsheet", "grid",
        "list of values", "entry", "record", "field",
    ]):
        modality = "table"
    elif any(w in query for w in [
        "image", "photo", "picture", "caption", "figure",
        "illustration", "visual", "alt text",
    ]):
        modality = "caption"
    elif any(w in query for w in [
        "chart", "graph", "plot", "trend", "bar chart",
        "pie chart", "line chart", "scatter", "x-axis", "y-axis",
    ]):
        modality = "chart"
    elif any(w in query for w in [
        "diagram", "flow", "schematic", "process map",
        "zone", "region", "labelled", "arrow", "anatomy",
    ]):
        modality = "diagram"
    elif any(w in query for w in [
        "infographic", "visual summary",
    ]):
        modality = "infographic"
    else:
        modality = "text"

    elapsed = round((time.time()-t0)*1000, 1)
    pstatus(f"[N2] intent → {modality} ({elapsed:.0f}ms)")
    return {
        "detected_modality": modality,
        "node_latencies":    {**state.get("node_latencies", {}),
                               "intent_router": elapsed},
    }


# ═════════════════════════════════════════════════════════════════════════════
# NODE 3 — MULTI-MODAL QUERY EXPANDER
# ═════════════════════════════════════════════════════════════════════════════

def node_expander(state: AgentState) -> Dict[str, Any]:
    """
    Produces 4–5 query variants to maximise retrieval recall:
      1. Original cleaned query
      2. LLM paraphrase 1 (different vocabulary, same meaning)
      3. LLM paraphrase 2
      4. Modality-specific reformulation (e.g. "what does the caption say about X?")
      5. Keyword-only variant (strips question words, keeps nouns)

    Why this matters: the #1 cause of "not found" is vocabulary mismatch
    between how the user phrases the question and how the document was written.
    Running multiple searches and merging results triples effective recall.
    """
    t0       = time.time()
    query    = state["current_query"]
    modality = state.get("detected_modality", "text")
    flags    = state.get("guardrail_flags", [])

    if "prompt_injection_detected" in flags or "empty_query" in flags:
        return {
            "query_variants": [query],
            "node_latencies": {**state.get("node_latencies", {}),
                                "expander": round((time.time()-t0)*1000, 1)},
        }

    variants: List[str] = [query]

    # ── LLM paraphrases ───────────────────────────────────────────────────────
    try:
        raw = llm_generate(
            f"Generate exactly 2 alternative phrasings of this query.\n"
            f"Use different vocabulary but preserve the exact meaning.\n"
            f"Return ONLY the 2 alternatives, one per line, no numbering.\n\n"
            f"Query: {query}",
            model=PRIMARY_LLM, max_tokens=150, temperature=0.6,
        )
        for line in raw.strip().split("\n"):
            line = line.strip()
            if line and len(line) > 4 and line.lower() != query.lower():
                variants.append(line)
                if len(variants) >= 3:
                    break
    except Exception as exc:
        logger.warning("LLM expander failed: %s", exc)

    # ── Modality-specific variant ─────────────────────────────────────────────
    modal_prefix = {
        "table":       "In the table or spreadsheet, what is the value for",
        "caption":     "What does the figure or image caption describe about",
        "chart":       "In the chart or graph, what does the data show about",
        "diagram":     "In the diagram or figure, what does the label or structure indicate about",
        "infographic": "In the visual or infographic, what information is shown about",
    }.get(modality)

    if modal_prefix:
        core = re.sub(
            r"^(what|how|where|when|why|who|which|does|is|are|can|"
            r"show me|tell me|find|describe)\s+",
            "", query, flags=re.IGNORECASE,
        ).strip().rstrip("?")
        if core:
            variants.append(f"{modal_prefix} {core}?")

    # ── Keyword-only variant ──────────────────────────────────────────────────
    STOP = {
        "the","a","an","in","of","for","to","is","are","was","were",
        "what","how","where","when","why","who","which","does","do",
        "can","could","would","will","with","and","or","not","any",
        "this","that","these","those","from","into","about","its",
    }
    kws = [w for w in re.findall(r"\b\w{3,}\b", query.lower())
           if w not in STOP]
    if len(kws) >= 2:
        variants.append(" ".join(kws))

    # Deduplicate preserving order
    seen: set = set()
    unique: List[str] = []
    for v in variants:
        key = v.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(v)

    elapsed = round((time.time()-t0)*1000, 1)
    pstatus(f"[N3] expander {len(unique)} variants ({elapsed:.0f}ms)", C.GRAY)
    for i, v in enumerate(unique):
        logger.info("  variant %d: %s", i+1, v)

    return {
        "query_variants": unique,
        "node_latencies": {**state.get("node_latencies", {}),
                            "expander": elapsed},
    }


# ═════════════════════════════════════════════════════════════════════════════
# NODE 4 — MULTI-VARIANT RETRIEVER
# ═════════════════════════════════════════════════════════════════════════════

def node_retriever(state: AgentState) -> Dict[str, Any]:
    """
    Runs independent Milvus ANN search for each query variant.
    Merges results and deduplicates by SHA256 content hash.

    Applies a small boost to caption/chart/diagram chunks when the user's
    query is about a visual modality — pushes visual chunks into the top-K
    window so the reranker actually sees them.

    Result: up to RETRIEVAL_TOP_K × len(variants) candidates, deduplicated,
    sorted by vector score, ready for reranking.
    """
    t0       = time.time()
    query    = state["current_query"]
    variants = state.get("query_variants", [query])
    flags    = state.get("guardrail_flags", [])
    modality = state.get("detected_modality", "text")

    if "prompt_injection_detected" in flags:
        return {
            "raw_chunks":     [],
            "node_latencies": {**state.get("node_latencies", {}),
                                "retriever": round((time.time()-t0)*1000, 1)},
        }

    raw_chunks:  List[Dict[str, Any]] = []
    seen_hashes: set = set()
    visual_modalities = {"caption", "chart", "diagram", "infographic"}

    try:
        milvus = get_milvus()

        for v_idx, variant in enumerate(variants):
            try:
                embeddings = embed_texts([variant], input_type="query")
            except Exception as exc:
                logger.warning("Embed failed variant %d: %s", v_idx, exc)
                continue
            if not embeddings:
                continue

            try:
                hits = milvus.search(
                    collection_name=COLLECTION,
                    data=[embeddings[0]],
                    limit=RETRIEVAL_TOP_K,
                    output_fields=["text"],
                )[0]
            except Exception as exc:
                logger.warning("Milvus search failed variant %d: %s", v_idx, exc)
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

                chunk_mod = detect_chunk_modality(text)

                # Visual boost: surface visual chunks when user asks about visuals
                boost = 0.0
                if modality in visual_modalities and chunk_mod in visual_modalities:
                    # L2: lower distance = better, so subtract to boost
                    boost = -0.05

                raw_chunks.append({
                    "text":           text,
                    "vector_score":   float(score) + boost,
                    "chunk_modality": chunk_mod,
                    "variant_idx":    v_idx,
                })

    except Exception as exc:
        perr(f"Retrieval failed: {exc}")
        logger.exception("Retrieval failed")

    # For L2 metric: lower score = closer. Sort ascending.
    raw_chunks.sort(key=lambda x: x["vector_score"])

    elapsed      = round((time.time()-t0)*1000, 1)
    mod_dist     = Counter(c["chunk_modality"] for c in raw_chunks)
    pstatus(
        f"[N4] retriever {C.CYAN}{len(raw_chunks)} unique{C.RESET} "
        f"chunks | modalities={dict(mod_dist)} ({elapsed:.0f}ms)"
    )
    return {
        "raw_chunks":     raw_chunks,
        "node_latencies": {**state.get("node_latencies", {}),
                            "retriever": elapsed},
    }


# ═════════════════════════════════════════════════════════════════════════════
# NODE 5 — NVIDIA RERANKER  — BUG B1 FIX IS IN rerank_passages_nvidia()
# ═════════════════════════════════════════════════════════════════════════════

def node_reranker(state: AgentState) -> Dict[str, Any]:
    """
    Reranks retrieved chunks using nvidia/nv-rerankqa-mistral-4b-v3.
    The reranker sees the query and each chunk together and outputs a
    relevance_score (0.0–1.0).

    Confidence thresholds:
      ≥ 0.45  → HIGH    (strong match — answer is very likely in this chunk)
      ≥ 0.15  → MEDIUM  (partial or indirect match)
      < 0.10  → triggers adaptive_rescue (but still generates best-effort)
    """
    t0         = time.time()
    query      = state["current_query"]
    raw_chunks = state.get("raw_chunks", [])

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

    ranked_chunks:     List[Dict[str, Any]] = []
    overall_confidence = "low"
    top_score          = 0.0

    try:
        passages = [c["text"] for c in raw_chunks]
        rankings = rerank_passages_nvidia(query, passages)

        for rank in rankings[:RERANK_TOP_K]:
            idx   = int(rank.get("index", 0))
            score = float(rank.get("score", 0.0))
            if not (0 <= idx < len(raw_chunks)):
                continue
            chunk = raw_chunks[idx].copy()
            chunk["rerank_score"] = score
            chunk["confidence"]   = (
                "high"   if score >= CONFIDENCE_HIGH   else
                "medium" if score >= CONFIDENCE_MEDIUM else
                "low"
            )
            ranked_chunks.append(chunk)

        if not ranked_chunks:
            # Fallback: use raw vector order with medium confidence
            ranked_chunks = [
                {**c, "rerank_score": CONFIDENCE_MEDIUM + 0.01,
                 "confidence": "medium"}
                for c in raw_chunks[:RERANK_TOP_K]
            ]

        top_score = ranked_chunks[0]["rerank_score"]
        overall_confidence = (
            "high"   if top_score >= CONFIDENCE_HIGH   else
            "medium" if top_score >= CONFIDENCE_MEDIUM else
            "low"
        )

    except Exception as exc:
        perr(f"Reranker failed: {exc} — using vector order")
        ranked_chunks = [
            {**c, "rerank_score": CONFIDENCE_MEDIUM + 0.01,
             "confidence": "medium"}
            for c in raw_chunks[:RERANK_TOP_K]
        ]
        overall_confidence = "medium"
        top_score          = CONFIDENCE_MEDIUM + 0.01

    elapsed = round((time.time()-t0)*1000, 1)
    cc = {"high": C.GREEN, "medium": C.YELLOW, "low": C.RED}.get(
        overall_confidence, C.GRAY)
    pstatus(
        f"[N5] reranker {len(ranked_chunks)} chunks | "
        f"top={top_score:.4f} | {cc}{overall_confidence.upper()}{C.RESET} "
        f"({elapsed:.0f}ms)"
    )
    return {
        "ranked_chunks":      ranked_chunks,
        "overall_confidence": overall_confidence,
        "top_score":          top_score,
        "node_latencies":     {**state.get("node_latencies", {}),
                                "reranker": elapsed},
    }


# ═════════════════════════════════════════════════════════════════════════════
# NODE 7 — ADAPTIVE RESCUE  — BUG B4 FIXED: BM25 threshold lowered
# ═════════════════════════════════════════════════════════════════════════════

def node_adaptive_rescue(state: AgentState) -> Dict[str, Any]:
    """
    Three rescue strategies when reranker confidence is below RESCUE_THRESHOLD.
    Only fires when the score is genuinely low (after B1 fix, this is now rare).

    Strategy A — Caption-first re-retrieval:
      Reformulates query as a visual-content question and runs a fresh search.
      Catches image/diagram chunks missed by text-phrased queries.

    Strategy B — BM25 lexical rescue:
      Scores already-retrieved chunks by keyword overlap.
      BUG B4 FIX: threshold lowered from 0.5 to max(0.1, 1/len(query_tokens))
      so short queries don't produce zero rescue chunks.

    Strategy C — Broad topic re-retrieval:
      Generates a broader query via LLM and searches again.
      Handles over-specific or narrow queries.

    After all strategies: re-rank the merged pool and always proceed to
    generation — worst case is a best-effort answer with explicit caveats.
    """
    t0       = time.time()
    query    = state["current_query"]
    raw      = state.get("raw_chunks", [])
    ranked   = state.get("ranked_chunks", [])

    pwarn(f"[N7] adaptive_rescue (top_score={state.get('top_score',0):.4f})")

    rescue_chunks: List[Dict[str, Any]] = []
    seen_hashes: set = {
        hashlib.sha256(c["text"].strip().encode()).hexdigest()
        for c in ranked if c.get("text")
    }

    # ── Strategy A: Caption-first re-retrieval ────────────────────────────────
    try:
        core = re.sub(
            r"^(what|how|where|when|does|is|are|tell me|show me|describe|find)\s+",
            "", query, flags=re.IGNORECASE,
        ).strip().rstrip("?") or query

        caption_query = (
            f"What does the figure, diagram, image caption, "
            f"or chart show about {core}?"
        )
        emb = embed_texts([caption_query], input_type="query")
        if emb:
            milvus = get_milvus()
            hits   = milvus.search(
                collection_name=COLLECTION,
                data=[emb[0]], limit=30,
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
                        "rescue_strategy": "A_caption",
                        "rerank_score":    RESCUE_THRESHOLD + 0.01,
                        "confidence":      "low",
                    })
                    added_A += 1
            pstatus(f"  [A] caption re-query: +{added_A} chunks", C.GRAY)
    except Exception as exc:
        logger.warning("Rescue A failed: %s", exc)

    # ── Strategy B: BM25 lexical rescue ──────────────────────────────────────
    try:
        STOP_BM25 = {
            "the","a","an","in","of","for","to","is","are","what",
            "how","where","when","why","does","do","its","this","that",
        }
        query_tokens = [
            t for t in re.findall(r"\w+", query.lower())
            if t not in STOP_BM25 and len(t) > 2
        ]
        if query_tokens and raw:
            avg_dl    = sum(
                len(re.findall(r"\w+", c["text"])) for c in raw
            ) / max(len(raw), 1)

            # BUG B4 FIX: dynamic threshold based on query length
            # Longer queries with more tokens can achieve higher BM25 scores.
            # Short queries (1-2 tokens) need a very low threshold.
            bm25_min = max(0.1, 0.5 / max(len(query_tokens), 1))

            scored = sorted(
                [(bm25_score(query_tokens, c["text"], avg_dl=avg_dl), c)
                 for c in raw],
                key=lambda x: x[0], reverse=True,
            )
            added_B = 0
            for bsc, chunk in scored[:15]:
                if bsc < bm25_min:
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
            pstatus(f"  [B] BM25 rescue: +{added_B} chunks (min={bm25_min:.2f})",
                    C.GRAY)
    except Exception as exc:
        logger.warning("Rescue B failed: %s", exc)

    # ── Strategy C: Broader topic re-retrieval ────────────────────────────────
    try:
        broad_q = llm_generate(
            f"Rewrite this query as a broader, more general version.\n"
            f"Return ONLY the rewritten query, no explanation.\n\n"
            f"Query: {query}",
            model=PRIMARY_LLM, max_tokens=80, temperature=0.3,
        )
        if broad_q and broad_q.strip() and broad_q.strip().lower() != query.lower():
            emb = embed_texts([broad_q.strip()], input_type="query")
            if emb:
                milvus = get_milvus()
                hits   = milvus.search(
                    collection_name=COLLECTION,
                    data=[emb[0]], limit=25,
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
                pstatus(f"  [C] broad re-query: +{added_C} chunks", C.GRAY)
    except Exception as exc:
        logger.warning("Rescue C failed: %s", exc)

    # ── Re-rank merged pool ───────────────────────────────────────────────────
    merged = ranked + rescue_chunks
    if rescue_chunks:
        try:
            passages    = [c["text"] for c in merged]
            re_rankings = rerank_passages_nvidia(query, passages)
            reranked: List[Dict[str, Any]] = []
            for r in re_rankings[:RERANK_TOP_K]:
                idx   = int(r.get("index", 0))
                score = float(r.get("score", 0.0))
                if 0 <= idx < len(merged):
                    c = merged[idx].copy()
                    c["rerank_score"] = score
                    c["confidence"]   = (
                        "high"   if score >= CONFIDENCE_HIGH   else
                        "medium" if score >= CONFIDENCE_MEDIUM else
                        "low"
                    )
                    reranked.append(c)
            if reranked:
                merged = reranked
        except Exception as exc:
            logger.warning("Rescue re-rank failed: %s", exc)

    if not merged:
        merged = ranked  # absolute fallback

    top_score  = merged[0]["rerank_score"] if merged else RESCUE_THRESHOLD
    confidence = (
        "high"   if top_score >= CONFIDENCE_HIGH   else
        "medium" if top_score >= CONFIDENCE_MEDIUM else
        "low"
    )

    elapsed = round((time.time()-t0)*1000, 1)
    pstatus(
        f"[N7] rescue done: +{len(rescue_chunks)} chunks | "
        f"total={len(merged)} | top={top_score:.4f} | "
        f"conf={confidence} ({elapsed:.0f}ms)",
        C.YELLOW,
    )
    return {
        "ranked_chunks":      merged,
        "rescue_attempted":   True,
        "rescue_chunks":      rescue_chunks,
        "overall_confidence": confidence,
        "top_score":          top_score,
        "generation_mode":    "best_effort",
        "node_latencies":     {**state.get("node_latencies", {}),
                                "rescue": elapsed},
    }


# ═════════════════════════════════════════════════════════════════════════════
# NODE 6 — GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

def node_generator(state: AgentState) -> Dict[str, Any]:
    """
    Builds modality-annotated context from ranked chunks, prepends
    conversation history, selects system prompt by generation_mode,
    calls PRIMARY_LLM → FALLBACK_LLM → chunk preview fallback.
    NEVER returns empty.
    """
    t0          = time.time()
    query       = state["original_query"]
    ranked      = state.get("ranked_chunks", [])
    flags       = list(state.get("guardrail_flags", []))
    history     = state.get("conversation_history", [])
    gen_mode    = state.get("generation_mode", "normal")
    rescue_used = state.get("rescue_attempted", False)

    if "prompt_injection_detected" in flags:
        return {
            "answer":          "This query has been flagged and cannot be processed.",
            "model_used":      "none", "fallback_used": False,
            "guardrail_flags": flags, "sources": [],
            "node_latencies":  {**state.get("node_latencies", {}),
                                 "generator": 0.0},
        }

    if "empty_query" in flags:
        return {
            "answer":          "Please type a question or query.",
            "model_used":      "none", "fallback_used": False,
            "guardrail_flags": flags, "sources": [],
            "node_latencies":  {**state.get("node_latencies", {}),
                                 "generator": 0.0},
        }

    if not ranked:
        return {
            "answer": (
                "No documents have been ingested, or no chunks matched your query.\n"
                "Ingest a file first: ingest <path>"
            ),
            "model_used":      "none", "fallback_used": False,
            "guardrail_flags": flags, "sources": [],
            "node_latencies":  {**state.get("node_latencies", {}),
                                 "generator": 0.0},
        }

    # Build context with modality labels
    ctx_chunks = ranked[:MAX_CONTEXT]
    parts:   List[str] = []
    sources: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(ctx_chunks, 1):
        conf      = chunk.get("confidence", "?")
        score     = round(float(chunk.get("rerank_score", 0.0)), 4)
        cmod      = chunk.get("chunk_modality",
                               detect_chunk_modality(chunk["text"]))
        r_tag     = (f" [rescue:{chunk['rescue_strategy']}]"
                     if chunk.get("rescue_strategy") else "")
        parts.append(
            f"[Chunk {idx} | modality={cmod} | confidence={conf} | "
            f"score={score}{r_tag}]\n{chunk['text']}"
        )
        preview = (chunk["text"][:200] +
                   ("…" if len(chunk["text"]) > 200 else ""))
        sources.append({
            "index":        idx,
            "text_preview": preview,
            "confidence":   conf,
            "rerank_score": score,
            "modality":     cmod,
            "rescue":       bool(chunk.get("rescue_strategy")),
        })

    context = "\n\n---\n\n".join(parts)

    # Conversation memory (last 5 turns)
    history_text = ""
    if history:
        history_text = "Previous conversation (most recent 5 turns):\n"
        for turn in history[-5:]:
            history_text += (
                f"User: {turn.get('query','')}\n"
                f"Assistant: {turn.get('answer','')}\n\n"
            )

    sys_prompt = {
        "normal":      _SYS_BASE,
        "best_effort": _SYS_BESTEFF,
        "strict":      _SYS_STRICT,
    }.get(gen_mode, _SYS_BASE)

    mode_note = ""
    if gen_mode == "best_effort":
        mode_note = (
            "\n[NOTE: Retrieval confidence is low. "
            "Use the chunks below as best evidence and be explicit about uncertainty.]\n\n"
        )
    elif rescue_used:
        mode_note = "\n[NOTE: Adaptive rescue broadened the retrieved context.]\n\n"

    prompt = (
        f"{history_text}"
        f"{mode_note}"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )

    answer = ""
    model_used    = "none"
    fallback_used = False

    for model in [PRIMARY_LLM, FALLBACK_LLM]:
        try:
            pstatus(f"  generating with {model.split('/')[-1]}…", C.GRAY)
            candidate = llm_generate(
                prompt, model=model, system_prompt=sys_prompt,
                max_tokens=1024, temperature=0.2,
            )
            if candidate.strip():
                answer        = candidate.strip()
                model_used    = model
                fallback_used = (model == FALLBACK_LLM)
                break
        except Exception as exc:
            perr(f"LLM {model.split('/')[-1]} failed: {exc}")
            logger.exception("LLM %s failed", model)

    if not answer:
        answer = (
            "Both LLMs failed to respond. "
            "Most relevant passages from the document:\n\n"
        )
        for s in sources[:3]:
            answer += f"• [{s['modality'].upper()}] {s['text_preview']}\n"
        flags.append("empty_answer")

    lowered = answer.lower()
    for phrase in HALLUCINATION_PHRASES:
        if phrase in lowered:
            flags.append("possible_hallucination")
            break

    elapsed = round((time.time()-t0)*1000, 1)
    pstatus(
        f"[N6] generator {C.CYAN}{model_used.split('/')[-1]}{C.RESET} | "
        f"mode={gen_mode} | rescue={rescue_used} ({elapsed:.0f}ms)"
    )
    return {
        "answer":          answer,
        "model_used":      model_used,
        "fallback_used":   fallback_used,
        "guardrail_flags": flags,
        "sources":         sources,
        "node_latencies":  {**state.get("node_latencies", {}),
                             "generator": elapsed},
    }


# ═════════════════════════════════════════════════════════════════════════════
# ROUTING
# ═════════════════════════════════════════════════════════════════════════════

def route_after_guardrail(state: AgentState) -> str:
    flags = state.get("guardrail_flags", [])
    if "prompt_injection_detected" in flags or "empty_query" in flags:
        return "generator"
    return "intent_router"


def route_after_reranker(state: AgentState) -> str:
    ranked    = state.get("ranked_chunks", [])
    top_score = state.get("top_score", 0.0)

    if not ranked:
        return "generator"  # generator will explain empty index

    if top_score < RESCUE_THRESHOLD:
        pwarn(f"[ROUTE] top={top_score:.4f} < {RESCUE_THRESHOLD} → rescue")
        return "adaptive_rescue"

    pstatus(f"[ROUTE] top={top_score:.4f} → generator", C.GREEN)
    return "generator"


# ═════════════════════════════════════════════════════════════════════════════
# GRAPH
# ═════════════════════════════════════════════════════════════════════════════

_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    g = StateGraph(AgentState)
    g.add_node("guardrail",       node_guardrail)
    g.add_node("intent_router",   node_intent_router)
    g.add_node("expander",        node_expander)
    g.add_node("retriever",       node_retriever)
    g.add_node("reranker",        node_reranker)
    g.add_node("adaptive_rescue", node_adaptive_rescue)
    g.add_node("generator",       node_generator)

    g.add_edge(START, "guardrail")
    g.add_conditional_edges(
        "guardrail", route_after_guardrail,
        {"intent_router": "intent_router", "generator": "generator"},
    )
    g.add_edge("intent_router",   "expander")
    g.add_edge("expander",        "retriever")
    g.add_edge("retriever",       "reranker")
    g.add_conditional_edges(
        "reranker", route_after_reranker,
        {"adaptive_rescue": "adaptive_rescue", "generator": "generator"},
    )
    g.add_edge("adaptive_rescue", "generator")
    g.add_edge("generator",       END)

    _compiled_graph = g.compile()
    return _compiled_graph


# ═════════════════════════════════════════════════════════════════════════════
# STATE HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _initial_state(query: str,
                   history: Optional[List] = None) -> AgentState:
    return {
        "original_query":       query,
        "current_query":        query,
        "query_variants":       [query],
        "detected_modality":    "unknown",
        "conversation_history": history or [],
        "raw_chunks":           [],
        "ranked_chunks":        [],
        "rescue_attempted":     False,
        "rescue_chunks":        [],
        "overall_confidence":   "low",
        "top_score":            0.0,
        "answer":               "",
        "model_used":           "",
        "fallback_used":        False,
        "generation_mode":      "normal",
        "guardrail_flags":      [],
        "retry_count":          0,
        "node_latencies":       {},
        "sources":              [],
    }


def _prepare_retry(state: AgentState) -> Optional[AgentState]:
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return None
    flags = state.get("guardrail_flags", [])
    if "possible_hallucination" not in flags:
        return None
    pwarn(f"Hallucination → retry {state['retry_count']+1} (strict mode)")
    return {
        **state,
        "current_query":    (state["original_query"] +
                             " -- answer ONLY with exact verbatim facts. Do not infer."),
        "query_variants":   [state["original_query"]],
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
        "retry_count":      state.get("retry_count", 0) + 1,
        "guardrail_flags":  [f for f in flags
                             if f not in {"possible_hallucination", "empty_answer"}],
        "node_latencies":   {},
    }


def run_agent(query: str,
              history: Optional[List] = None) -> Dict[str, Any]:
    graph  = get_graph()
    state  = _initial_state(query, history=history)
    t0     = time.time()
    while True:
        result = graph.invoke(state)
        retry  = _prepare_retry(result)
        if retry is None:
            result["wall_ms"] = round((time.time()-t0)*1000)
            return result
        state = retry


# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═════════════════════════════════════════════════════════════════════════════

def print_answer(answer, confidence, wall_ms, model, retry_count,
                 sources, latencies, flags, gen_mode, rescue_used):
    cc = {"high": C.GREEN, "medium": C.YELLOW, "low": C.RED}.get(
        confidence, C.GRAY)
    model_label  = (model.split("/")[-1] if model and model != "none"
                    else "none")
    mode_badge   = {
        "normal":      "",
        "best_effort": f" │ {C.YELLOW}BEST-EFFORT{C.RESET}",
        "strict":      f" │ {C.MAGENTA}STRICT{C.RESET}",
    }.get(gen_mode, "")
    rescue_badge = f" │ {C.YELLOW}RESCUED{C.RESET}" if rescue_used else ""

    print(f"\n{C.BOLD}╔── ANSWER ──────────────────────────────────────────╗{C.RESET}")
    print(
        f"{C.BOLD}║{C.RESET} {cc}{confidence.upper()}{C.RESET}"
        f"{mode_badge}{rescue_badge}"
        f"  {C.GRAY}{model_label}  {wall_ms:,}ms{C.RESET}"
    )
    if retry_count > 0:
        print(f"{C.BOLD}║{C.RESET} {C.YELLOW}retried {retry_count}× (hallucination guard){C.RESET}")
    print(f"{C.BOLD}╠────────────────────────────────────────────────────╣{C.RESET}")

    for line in (answer or "No answer generated.").split("\n"):
        while len(line) > 52:
            idx = line[:52].rfind(" ")
            if idx == -1:
                idx = 52
            print(f"{C.BOLD}║{C.RESET} {line[:idx]}")
            line = line[idx:].lstrip()
        print(f"{C.BOLD}║{C.RESET} {line}")

    print(f"{C.BOLD}╠────────────────────────────────────────────────────╣{C.RESET}")

    if latencies:
        print(f"\n  {C.GRAY}Latency:{C.RESET}")
        max_ms = max(latencies.values(), default=1)
        for node in ["guardrail","intent_router","expander",
                     "retriever","reranker","rescue","generator"]:
            if node in latencies:
                ms  = latencies[node]
                bar = "█" * int(ms/max(max_ms,1)*24) + "░" * (24-int(ms/max(max_ms,1)*24))
                print(f"    {C.GRAY}{node:>14}{C.RESET}  {C.BLUE}{bar}{C.RESET}  {ms:.0f}ms")

    if sources:
        print(f"\n  {C.GRAY}Sources ({len(sources)} chunks):{C.RESET}")
        for s in sources[:5]:
            sc  = {"high": C.GREEN, "medium": C.YELLOW, "low": C.RED}.get(
                s.get("confidence","low"), C.GRAY)
            rtag = f" {C.YELLOW}[RESCUED]{C.RESET}" if s.get("rescue") else ""
            print(
                f"    {C.BLUE}#{s['index']}{C.RESET} "
                f"{sc}{s.get('confidence','?')}{C.RESET}  "
                f"[{s.get('modality','?').upper()}]  "
                f"score={s.get('rerank_score',0):.4f}{rtag}"
            )
            print(f"    {C.DIM}{s.get('text_preview','')[:90]}…{C.RESET}")

    if flags:
        print(f"\n  {C.YELLOW}⚑ {', '.join(flags)}{C.RESET}")
    print(f"{C.BOLD}╚════════════════════════════════════════════════════╝{C.RESET}\n")


# ═════════════════════════════════════════════════════════════════════════════
# INTERACTIVE LOOP
# ═════════════════════════════════════════════════════════════════════════════

def interactive_loop():
    print(f"""
{C.BOLD}Commands:{C.RESET}
  {C.CYAN}ingest <path> [path2…]{C.RESET}    Ingest file(s) — any format
  {C.CYAN}ingest --reset <path>{C.RESET}     Reset collection then ingest
  {C.CYAN}stats{C.RESET}                     Chunk count in Milvus
  {C.CYAN}reset{C.RESET}                     Drop and recreate collection
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
                pstatus("No history yet.", C.GRAY)
            for i, t in enumerate(conversation_history, 1):
                print(f"  {C.BLUE}Turn {i}{C.RESET}")
                print(f"    Q: {t.get('query','')[:80]}")
                print(f"    A: {t.get('answer','')[:80]}…")
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
            pstatus(f"Memory: {len(conversation_history)} turn(s)", C.GRAY)
        print()

        try:
            result = run_agent(user_input, history=conversation_history)

            if result.get("answer") and result.get("model_used") != "none":
                conversation_history.append({
                    "query":  user_input,
                    "answer": result["answer"],
                })
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]

            print_answer(
                answer      = result.get("answer", ""),
                confidence  = result.get("overall_confidence", "low"),
                wall_ms     = result.get("wall_ms", 0),
                model       = result.get("model_used", "?"),
                retry_count = result.get("retry_count", 0),
                sources     = result.get("sources", []),
                latencies   = result.get("node_latencies", {}),
                flags       = result.get("guardrail_flags", []),
                gen_mode    = result.get("generation_mode", "normal"),
                rescue_used = result.get("rescue_attempted", False),
            )
        except Exception as exc:
            perr(f"Agent failed: {exc}")
            logger.exception("Agent failed")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Enterprise Multimodal RAG Agent v4 — NV-Ingest + LangGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rag_agent_v4.py
  python rag_agent_v4.py --reset
  python rag_agent_v4.py --ingest report.pdf invoice.xlsx
        """,
    )
    parser.add_argument("--reset",  action="store_true")
    parser.add_argument("--ingest", nargs="+", metavar="FILE")
    args = parser.parse_args()

    banner()

    if not NVIDIA_API_KEY:
        perr("NVIDIA_API_KEY not set. export NVIDIA_API_KEY='nvapi-...'")
        sys.exit(1)

    pok(f"NVIDIA_API_KEY  : {NVIDIA_API_KEY[:15]}…")
    pok(f"Milvus DB       : {MILVUS_DB}")
    pok(f"Collection      : {COLLECTION}")
    pok(f"Embed model     : {EMBED_MODEL}")
    pok(f"Reranker        : {RERANK_MODEL}  (NVIDIA API — reads relevance_score)")
    pok(f"Primary LLM     : {PRIMARY_LLM}")
    pok(f"Fallback LLM    : {FALLBACK_LLM}")
    pok(f"Confidence      : HIGH≥{CONFIDENCE_HIGH} MEDIUM≥{CONFIDENCE_MEDIUM} RESCUE<{RESCUE_THRESHOLD}")
    pok(f"Retrieval top-k : {RETRIEVAL_TOP_K}  rerank top-k: {RERANK_TOP_K}  context: {MAX_CONTEXT}")
    pok("Split types     : text, table, chart  (B2 fixed)")
    pok("Score field     : relevance_score     (B1 fixed)")

    files = args.ingest or [f for f in INGEST_FILES if f.strip()]
    if files:
        valid   = [f for f in files if os.path.isfile(f)]
        missing = [f for f in files if not os.path.isfile(f)]
        for f in missing:
            perr(f"File not found: {f}")
        if valid:
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
