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
from typing import Any, Dict, List, Optional, TypedDict

import requests
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()

ingest_files: list[str] = [
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.FileHandler("rag_agent.log", mode="a")],
)
logger = logging.getLogger("rag_agent")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
logger.addHandler(console_handler)

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

def banner():
    print(f"""
{C.CYAN}{C.BOLD}+===========================================================+
|                    |
|  NV-Ingest 25.9.0 + LangGraph                             |
|                                                           |
|  Supports: PDF, DOCX, PPTX, XLSX, images, handwritten,   |
|  invoices, identity docs, tables, charts, captions        |
                                                |
|  Nodes: guardrail → expander → retriever → reranker →     |
|         quality_gate → generator                          |
|  Memory: last 3 conversation turns                        |
|                                 |
+===========================================================+{C.RESET}
""")

def pstatus(msg: str, color: str = C.CYAN):
    print(f"  {color}>{C.RESET} {msg}")

def pok(msg: str):
    print(f"  {C.GREEN}OK{C.RESET} {msg}")

def perr(msg: str):
    print(f"  {C.RED}ERR{C.RESET} {msg}")

def print_answer(answer, confidence, wall_ms, model, retry_count, sources, latencies, flags):
    cc = {"high": C.GREEN, "medium": C.YELLOW, "low": C.RED}.get(confidence, C.GRAY)
    model_label = model.split("/")[-1] if model and model != "none" else "none"

    print(f"\n{C.BOLD}+-- ANSWER --------------------------------------------------+{C.RESET}")
    print(
        f"{C.BOLD}|{C.RESET} {cc}{confidence.upper()} CONFIDENCE{C.RESET}  |  "
        f"{C.GRAY}{model_label}{C.RESET}  |  {C.GRAY}{wall_ms:,}ms{C.RESET}"
    )
    if retry_count > 0:
        print(f"{C.BOLD}|{C.RESET} {C.YELLOW}retried {retry_count}x{C.RESET}")
    print(f"{C.BOLD}+------------------------------------------------------------+{C.RESET}")

    rendered = answer or "No answer generated."
    for line in rendered.split("\n"):
        while len(line) > 56:
            idx = line[:56].rfind(" ")
            if idx == -1:
                idx = 56
            print(f"{C.BOLD}|{C.RESET} {line[:idx]}")
            line = line[idx:].lstrip()
        print(f"{C.BOLD}|{C.RESET} {line}")

    print(f"{C.BOLD}+------------------------------------------------------------+{C.RESET}")

    if latencies:
        print(f"\n  {C.GRAY}Latency:{C.RESET}")
        max_ms = max(latencies.values()) if latencies.values() else 1
        for node in ["guardrail", "expander", "retriever", "reranker", "generator"]:
            if node in latencies:
                ms = latencies[node]
                bar_len = int(ms / max(max_ms, 1) * 30)
                bar = "#" * bar_len + "." * (30 - bar_len)
                print(f"    {C.GRAY}{node:>12}{C.RESET}  {C.BLUE}{bar}{C.RESET}  {ms:.0f}ms")

    if sources:
        print(f"\n  {C.GRAY}Sources ({len(sources)} chunks):{C.RESET}")
        for s in sources[:4]:
            sc = {"high": C.GREEN, "medium": C.YELLOW, "low": C.RED}.get(s.get("confidence", "low"), C.GRAY)
            preview = s.get("text_preview", "")[:80]
            src_type = s.get("source_type", "text")
            print(
                f"    {C.BLUE}Chunk {s['index']}{C.RESET}  "
                f"{sc}{s.get('confidence', '?')}{C.RESET}  "
                f"[{src_type}]  score={s.get('rerank_score', 0):.3f}"
            )
            print(f"    {C.DIM}{preview}...{C.RESET}")

    if flags:
        print(f"\n  {C.YELLOW}Flags: {', '.join(flags)}{C.RESET}")

# ── Files to ingest on startup ──────────────────────────────────────────────
INGEST_FILES: List[str] = [
    "Docs/Infinity-Ensure-Brochure.pdf"
]

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
MILVUS_DB = os.environ.get("MILVUS_DB", "./milvus_rag.db")
COLLECTION = os.environ.get("COLLECTION", "rag_documents")
DIM = int(os.environ.get("EMBED_DIM", "1024"))

CHAT_API_BASE = os.environ.get("NVIDIA_CHAT_API_BASE", "https://integrate.api.nvidia.com")
RETRIEVAL_API_BASE = os.environ.get("NVIDIA_RETRIEVAL_API_BASE", "https://ai.api.nvidia.com")

EMBED_URL = os.environ.get("EMBED_URL", f"{CHAT_API_BASE}/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")

RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")

LLM_URL = os.environ.get("LLM_URL", f"{CHAT_API_BASE}/v1/chat/completions")
PRIMARY_LLM = os.environ.get("PRIMARY_LLM", "meta/llama-3.3-70b-instruct")
FALLBACK_LLM = os.environ.get("FALLBACK_LLM", "nvidia/llama-3.1-nemotron-70b-instruct")

CAPTION_URL = os.environ.get("CAPTION_URL", f"{CHAT_API_BASE}/v1/chat/completions")
CAPTION_MODEL = os.environ.get("CAPTION_MODEL", "nvidia/llama-3.1-nemotron-nano-vl-8b-v1")

BROKER_HOST = os.environ.get("BROKER_HOST", "localhost")
BROKER_PORT = int(os.environ.get("BROKER_PORT", 7671))

# ── Tuning constants ─────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 40       

# FIX: reranker sees more candidates after wider retrieval
RERANK_TOP_K = 15       

# FIX: fewer but higher-quality chunks to LLM improves answer precision
MAX_CONTEXT = 6            # was 8 — quality over quantity

# FIX: skip generation entirely if best chunk is noise
MIN_GENERATION_SCORE = -10.0

MAX_RETRIES = 1
CONFIDENCE_HIGH_THRESHOLD = -3.0
CONFIDENCE_MEDIUM_THRESHOLD = -8.0

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

# ── Query intent types — detected dynamically, not hardcoded ─────────────────
# These are used to annotate state only, not to gate any processing
INTENT_TYPES = ["text", "table", "image_caption", "chart", "mixed", "unknown"]

def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

def _extract_message_text(data: Dict[str, Any]) -> str:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message", {}) or {}
    content = message.get("content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()

    return ""

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
                try:
                    logger.warning("Response body: %s", response.text[:500])
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
            logger.warning("API attempt %s unexpected error: %s | URL: %s", attempt + 1, exc, url)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return {}

def embed_texts(texts: List[str], input_type: str = "query") -> List[List[float]]:
    cleaned = [text.strip() if text and text.strip() else "<empty>" for text in texts]
    data = _api_call(
        EMBED_URL,
        {
            "model": EMBED_MODEL,
            "input": cleaned,
            "input_type": input_type,
            "encoding_format": "float",
        },
    )
    return [item["embedding"] for item in data.get("data", [])]

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

    ce = _get_cross_encoder()
    pairs = [(query, passage[:2000]) for passage in passages]
    scores = ce.predict(pairs).tolist()
    results = [{"index": i, "logit": float(score)} for i, score in enumerate(scores)]
    return sorted(results, key=lambda x: x["logit"], reverse=True)

def llm_generate(
    prompt: str,
    model: str = PRIMARY_LLM,
    system_prompt: str = "",
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    data = _api_call(
        LLM_URL,
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    return _extract_message_text(data)

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
                metric_type="L2",
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
        metric_type="L2",
        auto_id=True,
    )
    pok(f"Reset collection: {COLLECTION}")

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
            elapsed = int(time.time() - (deadline - timeout))
            pstatus(f"Still waiting... ({elapsed}s)", C.GRAY)
        time.sleep(0.5)
    raise RuntimeError(f"Broker not reachable after {timeout}s")

def _start_pipeline_once():
    global _pipeline_started
    if _pipeline_started:
        return

    pstatus("Importing NV-Ingest (loads Ray internally)...")
    t0 = time.time()
    from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import (
        PipelineCreationSchema,
        run_pipeline,
    )

    pok(f"NV-Ingest imported ({time.time() - t0:.1f}s)")
    pstatus("Launching pipeline subprocess...")
    pstatus(f"{C.YELLOW}First run takes 2-5 min. Please wait.{C.RESET}", C.YELLOW)
    cfg = PipelineCreationSchema()
    run_pipeline(cfg, block=False, disable_dynamic_scaling=True, run_in_subprocess=True)
    _wait_for_broker()
    _pipeline_started = True
    pok(f"Pipeline ready ({time.time() - t0:.1f}s)")

def _close_milvus():
    """Release the local Milvus Lite file lock before NV-Ingest opens its own connection."""
    global _milvus_client
    if _milvus_client is not None:
        try:
            _milvus_client.close()
        except Exception:
            pass
        _milvus_client = None

def run_ingest(file_paths: List[str], reset: bool = False):
    from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
    from nv_ingest_client.client import Ingestor, NvIngestClient

    _start_pipeline_once()

    if reset:
        reset_collection()

    # Milvus Lite (local .db) allows only one connection at a time.
    # Release our handle so NV-Ingest's vdb_upload can open its own connection.
    _close_milvus()

    pstatus(f"Ingesting {len(file_paths)} file(s)...")
    for file_path in file_paths:
        pstatus(f"  -> {os.path.basename(file_path)} ({os.path.getsize(file_path) / 1024:.0f} KB)", C.GRAY)

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
    results = list(results)
    elapsed = round((time.time() - t0) * 1000)
    n_fail = len(failures) if failures else 0

    info = {
        "files": [os.path.basename(path) for path in file_paths],
        "chunks_ingested": len(results),
        "failures": n_fail,
        "elapsed_ms": elapsed,
    }
    pok(f"{len(results)} chunks ingested in {elapsed:,}ms ({n_fail} failures)")
    return info

# ── Agent state ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    original_query: str
    current_query: str
    query_variants: List[str]          # NEW: expander produces multiple search queries
    detected_intent: str               # NEW: text / table / image_caption / chart / mixed
    conversation_history: List[Dict]   # NEW: last N turns for memory
    raw_chunks: List[Dict[str, Any]]
    ranked_chunks: List[Dict[str, Any]]
    overall_confidence: str
    answer: str
    model_used: str
    fallback_used: bool
    guardrail_flags: List[str]
    retry_count: int
    node_latencies: Dict[str, float]
    sources: List[Dict[str, Any]]

# ── Node 1: Guardrail + query cleaner ────────────────────────────────────────

def node_guardrail(state: AgentState):
    """
    Validates the query, detects injection attempts, strips filler words
    that add noise to embeddings, and detects the likely content intent
    (text / table / image / chart) so the retriever and generator can be
    aware — without gating or restricting any query type.
    """
    t0 = time.time()
    query = state["current_query"]
    flags = list(state.get("guardrail_flags", []))

    if not query.strip():
        flags.append("empty_query")
        elapsed = round((time.time() - t0) * 1000, 1)
        return {
            "guardrail_flags": flags,
            "node_latencies": {**state.get("node_latencies", {}), "guardrail": elapsed},
        }

    if len(query) > 2000:
        flags.append("query_too_long")
        query = query[:2000]

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            flags.append("prompt_injection_detected")
            break

    # FIX: strip filler phrases that add vector noise without semantic value
    # This is dynamic — not domain-specific — and improves embedding quality
    # for any query type across any document domain
    filler_patterns = [
        r"^(please\s+)?(can\s+you\s+)?(tell\s+me|explain|describe|show\s+me|find|give\s+me|what\s+is|what\s+are|what\s+does\s+it\s+say\s+about|what\s+do\s+you\s+know\s+about|i\s+want\s+to\s+know|i\s+need\s+to\s+know)\s+",
        r"^(in\s+the\s+document[s]?,?\s+)?(according\s+to\s+the\s+(document[s]?|file|pdf|text)[,]?\s+)?",
        r"\s+(from\s+the\s+(document[s]?|file|pdf|text))\s*$",
    ]
    cleaned_query = query
    for pattern in filler_patterns:
        cleaned_query = re.sub(pattern, "", cleaned_query, flags=re.IGNORECASE).strip()
    if len(cleaned_query) < 4:
        cleaned_query = query  # safety: never over-strip

    # Detect likely content type the user is asking about
    # Not used to restrict — used to annotate chunks in sources output
    q_lower = query.lower()
    if any(w in q_lower for w in ["table", "row", "column", "spreadsheet", "grid", "list of"]):
        detected_intent = "table"
    elif any(w in q_lower for w in ["image", "photo", "picture", "diagram", "figure", "illustration", "caption"]):
        detected_intent = "image_caption"
    elif any(w in q_lower for w in ["chart", "graph", "plot", "trend", "bar", "pie", "line chart"]):
        detected_intent = "chart"
    else:
        detected_intent = "text"

    elapsed = round((time.time() - t0) * 1000, 1)
    pstatus(f"Node 1 guardrail: intent={detected_intent} ({elapsed:.0f}ms)")

    return {
        "current_query": cleaned_query,
        "detected_intent": detected_intent,
        "guardrail_flags": flags,
        "node_latencies": {**state.get("node_latencies", {}), "guardrail": elapsed},
    }

# ── Node 2: Query expander (NEW) ─────────────────────────────────────────────

def node_query_expander(state: AgentState):
    """
    FIX for retrieval misses:
    Takes the cleaned query and generates 2 semantic paraphrases using the
    primary LLM. All 3 variants (original + 2 paraphrases) are then passed
    to the retriever for independent Milvus searches. Results are merged
    and deduplicated. This bridges the vocabulary gap between how users
    phrase questions and how the document content is written — for ANY domain,
    ANY file type, ANY language style.

    If the LLM call fails, falls back gracefully to single-query retrieval.
    """
    t0 = time.time()
    query = state["current_query"]
    flags = state.get("guardrail_flags", [])

    if "prompt_injection_detected" in flags or "empty_query" in flags:
        elapsed = round((time.time() - t0) * 1000, 1)
        return {
            "query_variants": [query],
            "node_latencies": {**state.get("node_latencies", {}), "expander": elapsed},
        }

    variants = [query]
    try:
        expand_prompt = (
            f"Rephrase the following query in exactly 2 alternative ways. "
            f"Use different vocabulary and sentence structure but preserve the same meaning. "
            f"Return only the 2 alternatives, one per line, no numbering, no explanation.\n\n"
            f"Query: {query}"
        )
        raw = llm_generate(expand_prompt, model=PRIMARY_LLM, max_tokens=120, temperature=0.5)
        lines = [l.strip() for l in raw.strip().split("\n") if l.strip() and len(l.strip()) > 4]
        variants = [query] + lines[:2]
        pstatus(f"Node 2 expander: {len(variants)} variants ({round((time.time()-t0)*1000):.0f}ms)", C.GRAY)
    except Exception as exc:
        logger.warning("Query expander failed, using original query: %s", exc)
        pstatus(f"Node 2 expander: fallback to single query", C.YELLOW)

    elapsed = round((time.time() - t0) * 1000, 1)
    return {
        "query_variants": variants,
        "node_latencies": {**state.get("node_latencies", {}), "expander": elapsed},
    }

# ── Node 3: Multi-variant retriever ──────────────────────────────────────────

def node_retriever(state: AgentState):
    """
    FIX for retrieval misses:
    Runs an independent Milvus ANN search for each query variant produced by
    the expander. Results from all variants are merged and deduplicated by
    text hash so the reranker sees the widest possible candidate set.

    RETRIEVAL_TOP_K is now 40 (was 25) — this alone catches most chunks that
    were previously just outside the retrieval window.

    Deduplication is by SHA256 hash of the text content — domain-agnostic.
    """
    t0 = time.time()
    query = state["current_query"]
    variants = state.get("query_variants", [query])
    flags = state.get("guardrail_flags", [])
    raw_chunks: List[Dict[str, Any]] = []

    if "prompt_injection_detected" in flags:
        elapsed = round((time.time() - t0) * 1000, 1)
        return {"raw_chunks": [], "node_latencies": {**state.get("node_latencies", {}), "retriever": elapsed}}

    try:
        milvus = get_milvus()
        seen_hashes: set = set()

        for variant in variants:
            embeddings = embed_texts([variant], input_type="query")
            if not embeddings:
                logger.warning("Embedding returned empty for variant: %s", variant)
                continue

            q_emb = embeddings[0]
            hits = milvus.search(
                collection_name=COLLECTION,
                data=[q_emb],
                limit=RETRIEVAL_TOP_K,
                output_fields=["text"],
            )[0]

            for hit in hits:
                entity = hit.get("entity", hit) if isinstance(hit, dict) else hit.entity
                text = entity.get("text", "") if isinstance(entity, dict) else getattr(entity, "text", "")
                score = hit.get("distance", 0.0) if isinstance(hit, dict) else getattr(hit, "distance", 0.0)

                if not text or not text.strip():
                    continue

                # Deduplicate across variants by content hash
                text_hash = hashlib.sha256(text.strip().encode()).hexdigest()
                if text_hash in seen_hashes:
                    continue
                seen_hashes.add(text_hash)

                raw_chunks.append({"text": text, "vector_score": float(score)})

    except Exception as exc:
        perr(f"Retrieval failed: {exc}")
        logger.exception("Retrieval failed")

    elapsed = round((time.time() - t0) * 1000, 1)
    pstatus(
        f"Node 3 retriever: {C.CYAN}{len(raw_chunks)} unique chunks{C.RESET} "
        f"from {len(variants)} variants ({elapsed:.0f}ms)"
    )
    return {
        "raw_chunks": raw_chunks,
        "node_latencies": {**state.get("node_latencies", {}), "retriever": elapsed},
    }

# ── Node 4: Reranker + quality gate ──────────────────────────────────────────

def node_reranker(state: AgentState):
    """
    FIX for low confidence scores:
    Uses the same cross-encoder but with corrected thresholds that match
    what ms-marco-MiniLM-L-12-v2 actually outputs on structured documents
    of any type:

        Old: >= 0.0 high, >= -5.0 medium
        New: >= -3.0 high, >= -8.0 medium  (CONFIDENCE_HIGH/MEDIUM_THRESHOLD)

    FIX for wasted generation on irrelevant chunks:
    Adds a quality gate — if the top-ranked chunk scores below
    MIN_GENERATION_SCORE (-10.0), the pipeline short-circuits and returns
    "no relevant content found" without calling the LLM. This eliminates
    the case where 8 irrelevant chunks get passed to the generator and it
    hallucinates or produces a generic "not found" after a full LLM call.
    """
    t0 = time.time()
    query = state["current_query"]
    raw_chunks = state.get("raw_chunks", [])
    ranked_chunks: List[Dict[str, Any]] = []
    overall_confidence = "low"
    quality_gate_failed = False

    if not raw_chunks:
        elapsed = round((time.time() - t0) * 1000, 1)
        return {
            "ranked_chunks": [],
            "overall_confidence": "low",
            "node_latencies": {**state.get("node_latencies", {}), "reranker": elapsed},
        }

    try:
        passages = [chunk["text"] for chunk in raw_chunks]
        rankings = rerank_passages(query, passages)

        for rank in rankings[:RERANK_TOP_K]:
            idx = int(rank.get("index", 0))
            logit = float(rank.get("logit", rank.get("score", 0.0)))
            if 0 <= idx < len(raw_chunks):
                chunk = raw_chunks[idx].copy()
                chunk["rerank_score"] = logit

                # FIX: corrected thresholds — calibrated for structured docs of any type
                if logit >= CONFIDENCE_HIGH_THRESHOLD:
                    chunk["confidence"] = "high"
                elif logit >= CONFIDENCE_MEDIUM_THRESHOLD:
                    chunk["confidence"] = "medium"
                else:
                    chunk["confidence"] = "low"

                ranked_chunks.append(chunk)

        if not ranked_chunks:
            raise RuntimeError("Reranker returned no ranked passages.")

        top_score = ranked_chunks[0]["rerank_score"]

        # FIX: quality gate — don't call LLM if best chunk is irrelevant noise
        if top_score < MIN_GENERATION_SCORE:
            quality_gate_failed = True
            pstatus(f"Node 4 reranker: quality gate failed (top={top_score:.2f}) — skipping generation", C.YELLOW)
        else:
            if top_score >= CONFIDENCE_HIGH_THRESHOLD:
                overall_confidence = "high"
            elif top_score >= CONFIDENCE_MEDIUM_THRESHOLD:
                overall_confidence = "medium"

    except Exception as exc:
        perr(f"Reranker failed ({exc}) -- using vector order")
        ranked_chunks = [
            {**chunk, "rerank_score": 0.0, "confidence": "medium"}
            for chunk in raw_chunks[:RERANK_TOP_K]
        ]
        overall_confidence = "medium" if ranked_chunks else "low"

    elapsed = round((time.time() - t0) * 1000, 1)
    if not quality_gate_failed:
        cc = {"high": C.GREEN, "medium": C.YELLOW, "low": C.RED}.get(overall_confidence, C.GRAY)
        pstatus(f"Node 4 reranker: {len(ranked_chunks)} chunks, {cc}{overall_confidence}{C.RESET} ({elapsed:.0f}ms)")

    return {
        "ranked_chunks": ranked_chunks,
        "overall_confidence": overall_confidence,
        "quality_gate_failed": quality_gate_failed,
        "node_latencies": {**state.get("node_latencies", {}), "reranker": elapsed},
    }

# ── System prompt — universal, content-type aware ────────────────────────────

_SYS = """You are a precise document assistant. Your job is to answer questions using ONLY the context chunks provided.

The context may contain any of the following content types — handle each correctly:
- Plain text from any document type (reports, manuals, policies, study material, legal documents)
- Tables formatted in markdown — read rows and columns carefully before answering
- Captions describing images, diagrams, photos, charts, or infographics — treat caption text as the content of that visual
- Data extracted from spreadsheets or slides
- OCR output from scanned documents or handwritten text — minor OCR errors may be present; reason around them
- Structured data from identity documents, invoices, or forms — extract specific field values exactly as they appear

Instructions:
1. Read ALL context chunks before answering. The answer may span multiple chunks.
2. Answer any type of question directly and completely.
3. For numbers, dates, names, IDs, amounts — quote them exactly as they appear in the context.
4. For tables: identify the correct row and column intersection before answering.
5. For image captions: the caption text IS the content — answer from it directly.
6. For OCR/handwritten: if a value looks like a recognition error but the intent is clear, state what you see and note the uncertainty.
7. If the context genuinely does not contain the answer, say exactly: "The provided documents do not contain this information."
8. Never use outside knowledge. Never guess or infer beyond what is written in the context."""

# ── Node 5: Generator with memory ────────────────────────────────────────────

def node_generator(state: AgentState):
    """
    FIX for stateless queries (no follow-up support):
    Prepends the last 3 turns of conversation history to the prompt so
    follow-up questions ("what about clause 3?" / "how does that compare
    to X?") work correctly without the user repeating context.

    FIX for quality gate:
    Checks quality_gate_failed from the reranker — if True, returns a
    clean "no relevant content" response without an LLM call.

    The system prompt (_SYS) covers all content types universally —
    text, tables, image captions, charts, OCR, identity docs, invoices.
    """
    t0 = time.time()
    query = state["original_query"]
    ranked_chunks = state.get("ranked_chunks", [])
    flags = list(state.get("guardrail_flags", []))
    history = state.get("conversation_history", [])
    quality_gate_failed = state.get("quality_gate_failed", False)

    if "prompt_injection_detected" in flags:
        return {
            "answer": "This query has been flagged and cannot be processed.",
            "model_used": "none",
            "fallback_used": False,
            "guardrail_flags": flags,
            "sources": [],
            "node_latencies": {**state.get("node_latencies", {}), "generator": 0.0},
        }

    if quality_gate_failed or not ranked_chunks:
        return {
            "answer": "The provided documents do not contain relevant information for this query. "
                      "Please ensure the relevant files have been ingested.",
            "model_used": "none",
            "fallback_used": False,
            "guardrail_flags": flags,
            "sources": [],
            "node_latencies": {**state.get("node_latencies", {}), "generator": 0.0},
        }

    ctx_chunks = ranked_chunks[:MAX_CONTEXT]
    parts: List[str] = []
    sources: List[Dict[str, Any]] = []

    for index, chunk in enumerate(ctx_chunks, 1):
        conf = chunk.get("confidence", "?")
        score = round(float(chunk.get("rerank_score", 0.0)), 4)
        parts.append(f"[Chunk {index} | confidence={conf} | score={score}]\n{chunk['text']}")
        preview = chunk["text"][:150]
        if len(chunk["text"]) > 150:
            preview += "..."
        sources.append({
            "index": index,
            "text_preview": preview,
            "confidence": conf,
            "rerank_score": score,
            "source_type": _detect_chunk_type(chunk["text"]),
        })

    context = "\n\n---\n\n".join(parts)

    # FIX: prepend conversation history for multi-turn memory
    history_text = ""
    if history:
        history_text = "Previous conversation:\n"
        for turn in history[-3:]:
            history_text += f"User: {turn.get('query', '')}\nAssistant: {turn.get('answer', '')}\n\n"

    prompt = f"{history_text}Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    answer = ""
    model_used = "none"
    fallback_used = False

    for model in [PRIMARY_LLM, FALLBACK_LLM]:
        try:
            pstatus(f"Generating with {model.split('/')[-1]}...", C.GRAY)
            candidate = llm_generate(
                prompt,
                model=model,
                system_prompt=_SYS,
                max_tokens=1024,
                temperature=0.3,
            )
            logger.info("LLM %s returned answer length: %s", model, len(candidate))
            if candidate.strip():
                answer = candidate.strip()
                model_used = model
                fallback_used = model == FALLBACK_LLM
                break
            logger.warning("LLM %s returned empty answer, trying fallback", model)
        except Exception as exc:
            perr(f"LLM {model.split('/')[-1]} failed: {exc}")
            logger.exception("LLM %s failed", model)

    if not answer:
        answer = "Both LLMs failed to generate a response."
        flags.append("empty_answer")

    lowered = answer.lower()
    for phrase in HALLUCINATION_PHRASES:
        if phrase in lowered:
            flags.append("possible_hallucination")
            break

    elapsed = round((time.time() - t0) * 1000, 1)
    generator_label = model_used.split("/")[-1] if model_used != "none" else "none"
    pstatus(f"Node 5 generator: {C.CYAN}{generator_label}{C.RESET} ({elapsed:.0f}ms)")

    return {
        "answer": answer,
        "model_used": model_used,
        "fallback_used": fallback_used,
        "guardrail_flags": flags,
        "sources": sources,
        "node_latencies": {**state.get("node_latencies", {}), "generator": elapsed},
    }

def _detect_chunk_type(text: str) -> str:
    """Detect the likely content type of a chunk for source annotation."""
    if "|" in text and text.count("|") > 3:
        return "table"
    if text.strip().startswith("[Caption"):
        return "image_caption"
    if any(w in text.lower() for w in ["figure", "chart shows", "graph depicts", "plot of"]):
        return "chart"
    return "text"

# ── Graph construction ────────────────────────────────────────────────────────

_compiled_graph = None

def route_after_guardrail(state: AgentState):
    flags = state.get("guardrail_flags", [])
    if "prompt_injection_detected" in flags or "empty_query" in flags:
        return "generator"
    return "expander"

def get_graph():
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    graph = StateGraph(AgentState)
    graph.add_node("guardrail", node_guardrail)
    graph.add_node("expander", node_query_expander)
    graph.add_node("retriever", node_retriever)
    graph.add_node("reranker", node_reranker)
    graph.add_node("generator", node_generator)

    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"expander": "expander", "generator": "generator"},
    )
    graph.add_edge("expander", "retriever")
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "generator")
    graph.add_edge("generator", END)

    _compiled_graph = graph.compile()
    return _compiled_graph

def _initial_state(query: str, history: Optional[List] = None) -> AgentState:
    return {
        "original_query": query,
        "current_query": query,
        "query_variants": [query],
        "detected_intent": "unknown",
        "conversation_history": history or [],
        "raw_chunks": [],
        "ranked_chunks": [],
        "overall_confidence": "low",
        "answer": "",
        "model_used": "",
        "fallback_used": False,
        "guardrail_flags": [],
        "retry_count": 0,
        "node_latencies": {},
        "sources": [],
        "quality_gate_failed": False,
    }

def _prepare_retry_state(state: AgentState) -> Optional[AgentState]:
    retry_count = state.get("retry_count", 0)
    if retry_count >= MAX_RETRIES:
        return None

    flags = state.get("guardrail_flags", [])
    if "possible_hallucination" not in flags:
        return None

    pstatus(f"{C.YELLOW}Hallucination detected -> retry {retry_count + 1}{C.RESET}", C.YELLOW)
    new_query = (
        f"{state['original_query']} -- answer using ONLY exact facts stated in the document. Do not infer."
    )

    next_state: AgentState = {
        **state,
        "current_query": new_query,
        "query_variants": [new_query],
        "raw_chunks": [],
        "ranked_chunks": [],
        "sources": [],
        "answer": "",
        "model_used": "",
        "fallback_used": False,
        "quality_gate_failed": False,
        "retry_count": retry_count + 1,
        "guardrail_flags": [f for f in flags if f not in {"possible_hallucination", "empty_answer"}],
        "node_latencies": {},
    }
    return next_state

def run_agent(query: str, history: Optional[List] = None):
    graph = get_graph()
    state: AgentState = _initial_state(query, history=history)
    t0 = time.time()

    while True:
        result = graph.invoke(state)
        retry_state = _prepare_retry_state(result)
        if retry_state is None:
            result["wall_ms"] = round((time.time() - t0) * 1000)
            return result
        state = retry_state

# ── Interactive loop with conversation memory ─────────────────────────────────

def interactive_loop():
    print(f"""
{C.BOLD}Commands:{C.RESET}
  {C.CYAN}ingest <path>{C.RESET}        Ingest a file (any format)
  {C.CYAN}ingest <p1> <p2> ...{C.RESET} Ingest multiple files
  {C.CYAN}stats{C.RESET}                Show chunk count in Milvus
  {C.CYAN}reset{C.RESET}                Clear the collection
  {C.CYAN}history{C.RESET}              Show current conversation memory
  {C.CYAN}clear{C.RESET}                Clear conversation memory
  {C.CYAN}quit{C.RESET}                 Exit
""")

    # FIX: conversation history persisted across turns in the session
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
                print(f"    Q: {turn.get('query', '')[:80]}")
                print(f"    A: {turn.get('answer', '')[:80]}...")
            continue

        if user_input.lower().startswith("ingest "):
            paths = user_input[7:].strip().split()
            valid = [path for path in paths if os.path.isfile(path)]
            for path in paths:
                if not os.path.isfile(path):
                    perr(f"File not found: {path}")
            if valid:
                try:
                    run_ingest(valid)
                except Exception as exc:
                    perr(f"Ingest failed: {exc}")
                    logger.exception("Ingest failed")
            continue

        if user_input.lower() == "stats":
            try:
                milvus = get_milvus()
                stats = milvus.get_collection_stats(COLLECTION)
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

            # Update conversation memory
            if result.get("answer") and result.get("model_used") != "none":
                conversation_history.append({
                    "query": user_input,
                    "answer": result["answer"],
                })
                # Keep last 10 turns to avoid unbounded growth
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]

            print_answer(
                result.get("answer", ""),
                result.get("overall_confidence", "low"),
                result.get("wall_ms", 0),
                result.get("model_used", "?"),
                result.get("retry_count", 0),
                result.get("sources", []),
                result.get("node_latencies", {}),
                result.get("guardrail_flags", []),
            )
        except Exception as exc:
            perr(f"Agent failed: {exc}")
            logger.exception("Agent failed")

def main():
    parser = argparse.ArgumentParser(
        description="Enterprise Multimodal RAG Agent v2 — NV-Ingest + LangGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rag_agent_v2.py                   # Auto-ingest INGEST_FILES then query loop
  python rag_agent_v2.py --reset           # Drop collection, re-ingest, then query
        """,
    )
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the Milvus collection")
    args = parser.parse_args()

    banner()

    if not NVIDIA_API_KEY:
        perr("NVIDIA_API_KEY not set.")
        pstatus("export NVIDIA_API_KEY='nvapi-...'", C.GRAY)
        sys.exit(1)

    pok(f"NVIDIA_API_KEY: {NVIDIA_API_KEY[:15]}...")
    pok(f"Milvus DB: {MILVUS_DB}")
    pok(f"Collection: {COLLECTION}")
    pok(f"Embed URL: {EMBED_URL}")
    pok(f"Reranker: {RERANK_MODEL}")
    pok(f"Confidence thresholds: high>={CONFIDENCE_HIGH_THRESHOLD}, medium>={CONFIDENCE_MEDIUM_THRESHOLD}")
    pok(f"Quality gate: skip generation if top score < {MIN_GENERATION_SCORE}")
    pok(f"Retrieval top-k: {RETRIEVAL_TOP_K}, Rerank top-k: {RERANK_TOP_K}, Context: {MAX_CONTEXT}")
    pok(f"LLM: {PRIMARY_LLM} → fallback: {FALLBACK_LLM}")

    files_to_ingest = [f for f in INGEST_FILES if f.strip()]
    if files_to_ingest:
        valid = [f for f in files_to_ingest if os.path.isfile(f)]
        missing = [f for f in files_to_ingest if not os.path.isfile(f)]
        for f in missing:
            perr(f"File not found: {f}")
        if valid:
            pstatus(f"Auto-ingesting {len(valid)} file(s)...")
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

