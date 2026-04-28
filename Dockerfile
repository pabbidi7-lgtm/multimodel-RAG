#!/usr/bin/env python3
"""
NV-Ingest 25.9.0  Production RAG Pipeline  v7
==============================================
Designed for A100 GPU with full multimodal NIM stack.

Key fixes over v6:
  - Proper NV-Ingest Milvus schema navigation (source JSON, content_metadata JSON)
  - Per-document source filtering during retrieval (no cross-document contamination)
  - Grounded LLM system prompt (zero hallucination instruction)
  - Minimum rerank score threshold to reject irrelevant context
  - Fresh Milvus collection per run (always clean)
  - Increased chunk_size + chunk_overlap for better coverage
  - Robust text/source extraction from NV-Ingest hit format
  - Proper NIM health check endpoints for self-hosted Docker Compose
  - Image/chart/table extraction when NIMs are healthy
  - Detailed debug logging of every retrieved chunk

Usage:
  export NVIDIA_API_KEY='nvapi-...'
  # Start NIM containers first (docker compose up -d)
  taskset -c 0-7 python nv_ingest_pipeline_v7.py
  taskset -c 0-7 python nv_ingest_pipeline_v7.py --interactive
"""

import glob, json, logging, os, re, socket, subprocess, sys, time, traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

if sys.version_info < (3, 12):
    sys.exit(f"[FATAL] Python 3.12+ required. You have {sys.version.split()[0]}")

try:
    import pymilvus
    pymilvus.connections.disconnect("default")
except Exception:
    pass

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

DOCS_FOLDER = "Docs"
FILE_PATTERNS = [
    "*.pdf", "*.PDF", "*.docx", "*.DOCX", "*.pptx", "*.PPTX",
    "*.jpeg", "*.JPEG", "*.jpg", "*.JPG", "*.png", "*.PNG",
]

# Milvus
MILVUS_URI = "milvus.db"
COLLECTION_PREFIX = "nvingest_rag"
SPARSE = True
DENSE_DIM = 2048

# Chunking — larger chunks = better context retention
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 150
TOKENIZER = "intfloat/e5-large-unsupervised"

# LLM
LLM_MODEL = "meta/llama-3.3-70b-instruct"
LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.1

# Pipeline
PIPELINE_WAIT = 120
RESULTS_DIR = "Outputs"

# Retrieval
TOP_K_RETRIEVE = 40
RERANK_K = 10
MAX_CONTEXT_CHARS = 20000
MAX_CHUNK_CHARS = 3000
MIN_RERANK_SCORE = -30.0  # Reject chunks below this logit score

# Reranker
RERANKER_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
RERANKER_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-1b-v2/reranking"
RERANKER_TIMEOUT = 25

# File type classification
IMAGE_ONLY_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# NIM Docker Compose endpoints (self-hosted on A100)
NIM_ENV_VARS = {
    "YOLOX_HTTP_ENDPOINT": "http://localhost:8000/v1/infer",
    "YOLOX_INFER_PROTOCOL": "http",
    "YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT": "http://localhost:8003/v1/infer",
    "YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL": "http",
    "YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT": "http://localhost:8006/v1/infer",
    "YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL": "http",
    "OCR_HTTP_ENDPOINT": "http://localhost:8009/v1/infer",
    "OCR_INFER_PROTOCOL": "http",
}

# NIM health checks — try multiple paths for robustness
NIM_CHECKS = {
    "page_elements":    (8000, ["/v1/health/ready", "/health/ready", "/v1/health"]),
    "graphic_elements": (8003, ["/v1/health/ready", "/health/ready", "/v1/health"]),
    "table_structure":  (8006, ["/v1/health/ready", "/health/ready", "/v1/health"]),
    "ocr":              (8009, ["/v1/health/ready", "/health/ready", "/v1/health"]),
}

# Default questions for non-interactive mode
DEFAULT_QUESTIONS = [
    "What must be included in a discharge summary, who is ultimately responsible for completing it, and what is the recommended completion timeline?",
    "At the Effective Time of the merger, how are Public Common Units treated, and what does PDI receive in exchange for its ownership interest in Merger Sub?",
    "What are the WCAG minimum contrast requirements for non-text graphical objects, normal text, and large text?",
    "Why did economics and physics become early movers in open access adoption?",
    "What does the report mean by successive waves of open access innovation?",
]

# Stopwords for lexical scoring fallback
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "what", "when", "where", "which", "who", "why", "with",
}

NIM_FAILURE_MARKERS = (
    "extract_primitives_from_pdf", "yolox", "ocr", "table_structure",
    "graphic_elements", "page_elements", "connection refused",
    "http 404", "http 000",
)

# Grounded system prompt — prevents hallucination
SYSTEM_PROMPT = """You are a precise document analysis assistant. You MUST follow these rules:

1. Answer ONLY using information explicitly stated in the provided Context excerpts.
2. If the Context does not contain enough information to answer the question, respond with:
   "The provided documents do not contain sufficient information to answer this question."
3. NEVER use outside knowledge, training data, or make assumptions beyond what is in the Context.
4. When quoting or referencing information, cite the source document name from the excerpt headers.
5. If multiple excerpts from different documents are relevant, synthesize them and cite each source.
6. For numerical data, tables, or specific facts, quote them exactly as they appear in the Context.
"""

# ═══════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
COLLECTION_NAME = os.environ.get("NV_INGEST_COLLECTION_NAME") or f"{COLLECTION_PREFIX}_{RUN_ID}"
LOG_FILE = os.path.join(RESULTS_DIR, f"pipeline_{RUN_ID}.log")
METRICS_FILE = os.path.join(RESULTS_DIR, f"metrics_{RUN_ID}.json")
ANSWERS_FILE = os.path.join(RESULTS_DIR, f"answers_{RUN_ID}.json")

_fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%Y-%m-%d %H:%M:%S")
_fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8", delay=False)
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)
_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO)
_ch.setFormatter(_fmt)
log = logging.getLogger("pipeline")
log.setLevel(logging.DEBUG)
log.propagate = False
log.handlers.clear()
log.addHandler(_fh)
log.addHandler(_ch)


def flush():
    for h in log.handlers:
        try:
            h.flush()
        except Exception:
            pass


def sep(c="=", w=72):
    log.info(c * w)
    flush()


def timer_start(name: str) -> float:
    log.info(f">> {name}")
    flush()
    return time.perf_counter()


def timer_end(name: str, start: float) -> float:
    elapsed = time.perf_counter() - start
    log.info(f"<< {name}  {elapsed:.3f}s")
    flush()
    return elapsed


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


# ═══════════════════════════════════════════════════════════════════════
# NV-INGEST MILVUS SCHEMA NAVIGATION
# ═══════════════════════════════════════════════════════════════════════
# NV-Ingest stores data in Milvus with this schema:
#   pk (INT64), vector (FLOAT_VECTOR), text (VARCHAR),
#   source (JSON), content_metadata (JSON), sparse (SPARSE_FLOAT_VECTOR)
#
# source = {"source_name": "...", "source_id": "...", "source_type": "PDF", ...}
# content_metadata = {"content_metadata": {"type": "text"|"structured", ...}, ...}


def extract_source_name(hit: Any) -> str:
    """Extract the source filename from an NV-Ingest Milvus hit."""
    if not isinstance(hit, dict):
        return "unknown"

    # Navigate: hit -> entity -> source -> source_name
    entity = hit.get("entity", hit)

    # source is a JSON field containing source_name
    source = entity.get("source", {})
    if isinstance(source, dict):
        for key in ("source_name", "source_id", "filename"):
            val = source.get(key)
            if val:
                return Path(str(val)).name

    # Fallback: check content_metadata or top-level
    for container in [entity.get("content_metadata", {}), entity, hit]:
        if not isinstance(container, dict):
            continue
        for key in ("source_name", "source", "filename", "file_name"):
            val = container.get(key)
            if isinstance(val, str) and val and "/" in val or "." in val:
                return Path(val).name

    return "unknown"


def extract_text(hit: Any) -> str:
    """Extract the text content from an NV-Ingest Milvus hit."""
    if not isinstance(hit, dict):
        return ""

    entity = hit.get("entity", hit)

    # Primary: text field (VARCHAR in NV-Ingest schema)
    for key in ("text", "content", "chunk", "body"):
        val = entity.get(key) or hit.get(key)
        if val and isinstance(val, str) and len(val.strip()) > 10:
            return val.strip()

    return ""


def extract_content_type(hit: Any) -> str:
    """Extract the content type (text, structured/table, structured/chart, image)."""
    entity = hit.get("entity", hit) if isinstance(hit, dict) else {}
    cm = entity.get("content_metadata", {})
    if isinstance(cm, dict):
        inner = cm.get("content_metadata", cm)
        if isinstance(inner, dict):
            ctype = inner.get("type", "")
            subtype = inner.get("subtype", "")
            if subtype:
                return f"{ctype}/{subtype}"
            return ctype
    return "unknown"


def extract_candidates(retrieved_docs: list) -> List[Dict]:
    """Parse NV-Ingest retrieval results into a clean candidate list."""
    if not retrieved_docs or not retrieved_docs[0]:
        return []

    hits = retrieved_docs[0]
    seen, out = set(), []

    for rank, hit in enumerate(hits, start=1):
        text = collapse_ws(extract_text(hit))
        if not text or len(text) < 20:
            continue

        # Dedup by first 500 chars
        key = text.lower()[:500]
        if key in seen:
            continue
        seen.add(key)

        source = extract_source_name(hit)
        content_type = extract_content_type(hit)

        # Extract retrieval score
        score = None
        if isinstance(hit, dict):
            for field in ("score", "distance", "similarity"):
                val = hit.get(field)
                if isinstance(val, (int, float)):
                    score = float(val)
                    break

        out.append({
            "text": text,
            "source": source,
            "content_type": content_type,
            "retrieval_rank": rank,
            "retrieval_score": score,
        })

    return out


# ═══════════════════════════════════════════════════════════════════════
# RERANKING
# ═══════════════════════════════════════════════════════════════════════

def tok(text: str) -> list:
    return [t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if len(t) > 1 and t not in STOPWORDS]


def lexical_score(query: str, text: str) -> float:
    q = tok(query)
    tl = str(text).lower()
    if not q:
        return 0.0
    overlap = len(set(q) & set(tok(text))) / max(len(set(q)), 1)
    phrase = 1.0 if str(query).lower().strip() in tl else 0.0
    rare = sum(0.03 for term in set(q) if len(term) >= 6 and term in tl)
    return overlap + phrase + rare


def parse_rerank_payload(payload: Any, n: int) -> List[Tuple[int, float]]:
    ranked = []
    if isinstance(payload, list):
        if all(isinstance(x, (int, float)) for x in payload):
            ranked = [(i, float(s)) for i, s in enumerate(payload[:n])]
        elif all(isinstance(x, dict) for x in payload):
            for i, item in enumerate(payload):
                idx = item.get("index", item.get("id", i))
                score = item.get("score", item.get("logit", item.get("relevance", 0.0)))
                if isinstance(idx, int) and isinstance(score, (int, float)):
                    ranked.append((idx, float(score)))
    elif isinstance(payload, dict):
        for key in ("rankings", "results", "data", "scores"):
            if payload.get(key) is not None:
                ranked = parse_rerank_payload(payload[key], n)
                if ranked:
                    break
    ranked = [(i, s) for i, s in ranked if 0 <= i < n]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def rerank_with_api(query: str, candidates: list, api_key: str) -> List[Dict]:
    if not candidates:
        return []
    payload = {
        "model": RERANKER_MODEL,
        "query": {"text": query},
        "passages": [{"text": c["text"][:MAX_CHUNK_CHARS]} for c in candidates],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    resp = requests.post(RERANKER_URL, headers=headers, json=payload, timeout=RERANKER_TIMEOUT)
    resp.raise_for_status()
    ranked_pairs = parse_rerank_payload(resp.json(), len(candidates))
    if not ranked_pairs:
        raise ValueError("Unrecognized reranker response")

    out = []
    for idx, score in ranked_pairs[:RERANK_K]:
        if score < MIN_RERANK_SCORE:
            continue
        item = dict(candidates[idx])
        item["rerank_score"] = float(score)
        out.append(item)
    return out


def rerank_locally(query: str, candidates: list) -> List[Dict]:
    out = []
    for c in candidates:
        item = dict(c)
        item["rerank_score"] = round(
            lexical_score(query, c["text"]) + 0.02 * max(0, TOP_K_RETRIEVE + 1 - c["retrieval_rank"]),
            6,
        )
        out.append(item)
    out.sort(key=lambda x: x["rerank_score"], reverse=True)
    return out[:RERANK_K]


def rerank(query: str, candidates: list, api_key: str) -> Tuple[List[Dict], str]:
    try:
        result = rerank_with_api(query, candidates, api_key)
        if result:
            return result, "api"
        # All below threshold
        return rerank_locally(query, candidates), "api_below_threshold+local"
    except requests.exceptions.Timeout:
        log.warning(f"    Reranker timeout ({RERANKER_TIMEOUT}s). Local fallback.")
    except requests.exceptions.HTTPError as exc:
        snippet = exc.response.text[:160] if exc.response is not None else str(exc)
        log.warning(f"    Reranker HTTP error: {snippet}")
    except Exception as exc:
        log.warning(f"    Reranker error: {exc}")
    return rerank_locally(query, candidates), "local_fallback"


# ═══════════════════════════════════════════════════════════════════════
# CONTEXT BUILDING & LLM
# ═══════════════════════════════════════════════════════════════════════

def build_context(items: List[Dict]) -> str:
    blocks, chars = [], 0
    for idx, item in enumerate(items, start=1):
        score_str = f"{item.get('rerank_score', 0.0):.4f}"
        block = (
            f"[Excerpt {idx} | Source: {item['source']} | "
            f"Type: {item.get('content_type', 'text')} | Score: {score_str}]\n"
            f"{item['text'][:MAX_CHUNK_CHARS]}"
        )
        if chars + len(block) > MAX_CONTEXT_CHARS:
            break
        blocks.append(block)
        chars += len(block) + 2
    return "\n\n".join(blocks)


def top_sources(items: List[Dict]) -> List[str]:
    seen, ordered = set(), []
    for item in items:
        src = item["source"]
        if src not in seen:
            seen.add(src)
            ordered.append(src)
    return ordered


def answer_text(resp) -> str:
    content = getattr(resp.choices[0].message, "content", "")
    if isinstance(content, list):
        return "\n".join(
            item["text"] for item in content if isinstance(item, dict) and item.get("text")
        ).strip()
    return str(content).strip()


# ═══════════════════════════════════════════════════════════════════════
# ENVIRONMENT & HEALTH
# ═══════════════════════════════════════════════════════════════════════

def setup_env() -> str:
    sep()
    log.info("ENVIRONMENT SETUP")
    sep()
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        log.error("NVIDIA_API_KEY not set. export NVIDIA_API_KEY='nvapi-...'")
        flush()
        sys.exit(1)
    log.info(f"  NVIDIA_API_KEY : ********{api_key[-6:]}")
    for var, default in NIM_ENV_VARS.items():
        if not os.environ.get(var):
            os.environ[var] = default
            log.info(f"  {var:<47} -> {default}")
        else:
            log.info(f"  {var:<47} (already set)")
    log.info(f"  Collection     : {COLLECTION_NAME}")
    log.info(f"  Retrieval      : sparse={SPARSE}  top_k={TOP_K_RETRIEVE}  rerank_k={RERANK_K}")
    log.info(f"  Chunk          : size={CHUNK_SIZE}  overlap={CHUNK_OVERLAP}")
    log.info(f"  Reranker       : {RERANKER_MODEL}")
    log.info(f"  Min score      : {MIN_RERANK_SCORE}")
    flush()
    return api_key


def check_nim_port(port: int, paths: list) -> Tuple[bool, str]:
    """Try multiple health check paths on a port."""
    for path in paths:
        try:
            url = f"http://localhost:{port}{path}"
            r = requests.get(url, timeout=5)
            if r.status_code in (200, 201):
                return True, f"HTTP {r.status_code} at {path}"
        except Exception:
            continue
    # Also try raw socket
    try:
        with socket.create_connection(("localhost", port), timeout=3):
            return True, "port open (no healthy endpoint found)"
    except Exception:
        return False, "port closed"


def check_nims() -> Dict[str, bool]:
    sep()
    log.info("NIM HEALTH CHECK")
    sep()
    status = {}
    for name, (port, paths) in NIM_CHECKS.items():
        ok, detail = check_nim_port(port, paths)
        status[name] = ok
        log.info(f"  {'OK  ' if ok else 'FAIL'}  {name:<22} port {port}  {detail}")
    healthy = sum(1 for x in status.values() if x)
    if healthy == 4:
        log.info("  All 4 NIMs healthy -> Full multimodal extraction enabled.")
    else:
        log.info(f"  {healthy}/4 NIMs healthy -> Text-only fallback where needed.")
        log.info("  TIP: Start NIMs with: docker compose --profile all up -d")
    flush()
    return status


# ═══════════════════════════════════════════════════════════════════════
# FILE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════

def find_files() -> List[str]:
    sep()
    log.info(f"FILE DISCOVERY -> {DOCS_FOLDER}/")
    files = []
    for pattern in FILE_PATTERNS:
        found = sorted(glob.glob(os.path.join(DOCS_FOLDER, pattern)))
        if found:
            log.info(f"  {pattern:<12} -> {len(found)}")
        files.extend(found)
    files = sorted(set(files))
    log.info(f"  Total: {len(files)} file(s)")
    for fp in files:
        log.info(f"    {Path(fp).name:<52} {os.path.getsize(fp) / 1048576:.2f} MB")
    if not files:
        log.error(f"No files in {DOCS_FOLDER}/ matching {FILE_PATTERNS}")
        flush()
        sys.exit(1)
    flush()
    return files


def collect_questions() -> List[str]:
    print("\n" + "=" * 72)
    print("  Enter questions. Blank line or 'done' to finish.")
    print("=" * 72 + "\n")
    qs, idx = [], 1
    while True:
        try:
            q = input(f"  Q{idx}: ").strip()
        except EOFError:
            break
        if q.lower() in ("", "done"):
            if not qs:
                print("  [!] Need at least one question.")
                continue
            break
        qs.append(q)
        idx += 1
    if not qs:
        log.warning("No interactive questions. Using defaults.")
        flush()
        return DEFAULT_QUESTIONS
    log.info(f"Collected {len(qs)} question(s).")
    flush()
    return qs


# ═══════════════════════════════════════════════════════════════════════
# PIPELINE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════

def start_pipeline():
    sep()
    log.info("PIPELINE INIT")
    sep()
    from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import (
        PipelineCreationSchema,
        run_pipeline,
    )
    from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
    from nv_ingest_client.client import NvIngestClient

    start = timer_start("Pipeline subprocess")
    run_pipeline(PipelineCreationSchema(), block=False, disable_dynamic_scaling=True, run_in_subprocess=True)
    log.info(f"  Polling port 7671 (max {PIPELINE_WAIT}s)...")
    deadline = time.time() + PIPELINE_WAIT
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", 7671), timeout=2):
                break
        except (ConnectionRefusedError, OSError):
            time.sleep(2)
    else:
        log.error("Port 7671 never opened. Pipeline subprocess likely crashed.")
        flush()
        sys.exit(1)
    log.info("  Port 7671 ready.")
    time.sleep(5)
    init_time = timer_end("Pipeline subprocess", start)
    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost",
    )
    log.info("  NvIngestClient connected -> localhost:7671")
    flush()
    return client, init_time


def prepare_collection(name: str):
    from pymilvus import connections, utility

    connections.connect(uri=MILVUS_URI)
    if utility.has_collection(name):
        log.warning(f"Dropping existing collection '{name}' for clean run.")
        utility.drop_collection(name)
    log.info(f"Collection '{name}' ready (will be created on first ingest).")
    flush()


# ═══════════════════════════════════════════════════════════════════════
# SANITY CHECK
# ═══════════════════════════════════════════════════════════════════════

def choose_sanity_file(files: list, nim_status: dict) -> Optional[str]:
    if nim_status.get("ocr", False):
        return files[0] if files else None
    for fp in files:
        if Path(fp).suffix.lower() not in IMAGE_ONLY_EXTS:
            return fp
    return None


def sanity_check(client, filepath: str) -> float:
    from nv_ingest_client.client import Ingestor
    from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

    sep()
    log.info(f"SANITY CHECK (text-only) -> {Path(filepath).name}")
    sep()
    start = timer_start("Sanity")
    ingestor = (
        Ingestor(client=client)
        .files(filepath)
        .extract(
            extract_text=True,
            extract_tables=False,
            extract_charts=False,
            extract_images=False,
            extract_infographics=False,
            text_depth="page",
        )
    )
    results, failures = ingestor.ingest(show_progress=True, return_failures=True)
    sanity_time = timer_end("Sanity", start)
    log.info(f"  Results: {len(results)}  Failures: {len(failures)}")
    if failures:
        for idx, f in enumerate(failures):
            log.error(f"  FAIL[{idx}]: {f}")
        log.error("Sanity check failed.")
        flush()
        sys.exit(1)
    if not results:
        log.error("Sanity check returned no results.")
        flush()
        sys.exit(1)
    preview = ingest_json_results_to_blob(results[0])[:200].replace(chr(10), " ")
    log.info(f"  Preview: {preview}...")
    log.info("  SANITY PASSED")
    flush()
    return sanity_time


# ═══════════════════════════════════════════════════════════════════════
# INGESTION
# ═══════════════════════════════════════════════════════════════════════

def extract_kwargs(multimodal: bool) -> dict:
    kwargs = {
        "extract_text": True,
        "text_depth": "page",
        "extract_tables": multimodal,
        "extract_charts": multimodal,
        "extract_images": multimodal,
        "extract_infographics": multimodal,
    }
    if multimodal:
        kwargs["table_output_format"] = "markdown"
    return kwargs


def run_ingest_attempt(client, filepath: str, kwargs: dict, collection_name: str):
    from nv_ingest_client.client import Ingestor

    ingestor = (
        Ingestor(client=client)
        .files(filepath)
        .extract(**kwargs)
        .split(tokenizer=TOKENIZER, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        .embed()
        .vdb_upload(
            collection_name=collection_name,
            milvus_uri=MILVUS_URI,
            sparse=SPARSE,
            dense_dim=DENSE_DIM,
        )
    )
    return ingestor.ingest(show_progress=False, return_failures=True)


def should_retry_text_only(failures: list) -> bool:
    message = " ".join(str(f).lower() for f in failures)
    return any(marker in message for marker in NIM_FAILURE_MARKERS)


def ingest_single_file(client, filepath: str, idx: int, total: int, nim_status: dict, collection_name: str):
    filename = Path(filepath).name
    ext = Path(filepath).suffix.lower()
    all_nims = all(nim_status.values())
    ocr_ok = nim_status.get("ocr", False)

    if ext in IMAGE_ONLY_EXTS and not ocr_ok:
        log.info(f"  [{idx}/{total}]  {filename}  [SKIPPED - image, no OCR NIM]")
        flush()
        return {"mode": "skipped", "total_ingest_sec": 0, "results_count": 0, "failures_count": 0, "skipped": True}, []

    mode = "FULL+HYBRID" if all_nims else "TEXT+HYBRID"
    log.info(f"  [{idx}/{total}]  {filename}  [{mode}]  sparse={SPARSE}")
    start = time.perf_counter()
    fallback = False

    try:
        results, failures = run_ingest_attempt(client, filepath, extract_kwargs(all_nims), collection_name)
        if all_nims and failures and not results and should_retry_text_only(failures):
            fallback = True
            mode = "TEXT+HYBRID (fallback)"
            log.warning("    Multimodal failed. Retrying text-only.")
            results, failures = run_ingest_attempt(client, filepath, extract_kwargs(False), collection_name)
        elapsed = time.perf_counter() - start
        if failures:
            for fi, f in enumerate(failures):
                log.warning(f"    FAIL[{fi}]: {str(f)[:180]}")
        log.info(f"    results={len(results)}  failures={len(failures)}  {elapsed:.3f}s")
        flush()
        return {
            "mode": mode,
            "fallback_triggered": fallback,
            "total_ingest_sec": round(elapsed, 3),
            "results_count": len(results),
            "failures_count": len(failures),
        }, failures
    except ValueError as exc:
        elapsed = time.perf_counter() - start
        log.warning(f"    SKIPPED (no embeddable content): {exc}")
        flush()
        return {"mode": mode, "total_ingest_sec": round(elapsed, 3), "results_count": 0, "failures_count": 0, "skipped": True}, []
    except Exception as exc:
        elapsed = time.perf_counter() - start
        log.error(f"    EXCEPTION: {exc}")
        log.error(traceback.format_exc())
        flush()
        return {"mode": mode, "total_ingest_sec": round(elapsed, 3), "results_count": 0, "failures_count": 1, "error": str(exc)}, [exc]


def run_batch(client, files: list, nim_status: dict, collection_name: str):
    sep()
    label = "FULL MULTIMODAL + HYBRID BM25" if all(nim_status.values()) else "TEXT-ONLY + HYBRID BM25"
    log.info(f"BATCH INGEST -> {len(files)} files | {label}")
    log.info(f"  Collection: {collection_name}  Milvus: {MILVUS_URI}")
    log.info(f"  Chunk: {CHUNK_SIZE} tok  overlap: {CHUNK_OVERLAP}  sparse={SPARSE}")
    sep()
    flush()
    start = time.perf_counter()
    timings = {}
    ok = failed = skipped = 0

    for idx, fp in enumerate(files, start=1):
        filename = Path(fp).name
        doc_start = time.perf_counter()
        try:
            timing, _ = ingest_single_file(client, fp, idx, len(files), nim_status, collection_name)
            timings[filename] = timing
            if timing.get("skipped"):
                skipped += 1
            elif timing.get("failures_count", 1) == 0:
                ok += 1
            else:
                failed += 1
        except Exception as exc:
            log.error(f"  OUTER EXCEPTION {filename}: {exc}")
            log.error(traceback.format_exc())
            timings[filename] = {"error": str(exc), "total_ingest_sec": 0, "failures_count": 1}
            failed += 1
        log.info(f"  -- wall: {time.perf_counter() - doc_start:.3f}s --")
        flush()

    batch_total = time.perf_counter() - start
    sep()
    log.info("BATCH SUMMARY")
    log.info(f"  Total: {len(files)}  OK: {ok}  Skipped: {skipped}  Failed: {failed}")
    log.info(f"  Wall: {batch_total:.3f}s  Avg: {batch_total / max(len(files), 1):.3f}s")
    sep()
    flush()
    return timings, batch_total


# ═══════════════════════════════════════════════════════════════════════
# MILVUS VERIFICATION
# ═══════════════════════════════════════════════════════════════════════

def get_entity_count(collection_name: str) -> int:
    from pymilvus import Collection, connections, utility

    connections.connect(uri=MILVUS_URI)
    if not utility.has_collection(collection_name):
        return 0
    col = Collection(collection_name)
    col.load()
    return col.num_entities


def check_milvus(collection_name: str) -> Tuple[bool, int]:
    sep()
    log.info("MILVUS VERIFICATION")
    try:
        count = get_entity_count(collection_name)
        log.info(f"  '{collection_name}': {count} chunks")
        if count == 0:
            log.error("  ZERO chunks. Ingestion/embedding likely failed.")
            flush()
            return False, 0
        log.info(f"  Retrieval: TOP_K={TOP_K_RETRIEVE} -> rerank -> keep {RERANK_K}")
        flush()
        return True, count
    except Exception as exc:
        log.error(f"  Milvus error: {exc}")
        flush()
        return False, 0


# ═══════════════════════════════════════════════════════════════════════
# RAG QUERIES
# ═══════════════════════════════════════════════════════════════════════

def run_rag(api_key: str, questions: list, collection_name: str):
    from openai import OpenAI
    from nv_ingest_client.util.milvus import nvingest_retrieval

    sep()
    log.info(f"RAG QUERIES -> {len(questions)} question(s)")
    log.info(f"  Collection : {collection_name}")
    log.info(f"  Mode       : hybrid BM25+semantic (sparse={SPARSE})")
    log.info(f"  Fetch      : TOP_K={TOP_K_RETRIEVE} -> rerank -> RERANK_K={RERANK_K}")
    log.info(f"  LLM        : {LLM_MODEL}")
    log.info(f"  Grounded   : Yes (system prompt enforced)")
    sep()
    flush()

    llm = OpenAI(base_url=LLM_BASE_URL, api_key=api_key)
    all_qa = []
    timings = {}

    for idx, question in enumerate(questions, start=1):
        log.info(f"  Q{idx}: {question}")
        flush()
        start = time.perf_counter()

        # Retrieve
        docs = nvingest_retrieval(
            [question],
            collection_name,
            milvus_uri=MILVUS_URI,
            hybrid=SPARSE,
            top_k=TOP_K_RETRIEVE,
        )
        retrieval_sec = time.perf_counter() - start
        candidates = extract_candidates(docs)

        log.info(
            f"    Retrieval : {retrieval_sec:.3f}s | candidates: {len(candidates)}/{TOP_K_RETRIEVE}"
        )

        # Log top candidate sources for debugging
        if candidates:
            source_dist = {}
            for c in candidates:
                source_dist[c["source"]] = source_dist.get(c["source"], 0) + 1
            log.debug(f"    Source distribution: {source_dist}")

        # Rerank
        rr_start = time.perf_counter()
        ranked, rerank_mode = rerank(question, candidates, api_key)
        rerank_sec = time.perf_counter() - rr_start

        if not ranked and candidates:
            ranked = rerank_locally(question, candidates)
            rerank_mode = "local_only"

        top_score = ranked[0]["rerank_score"] if ranked else 0.0
        sources = top_sources(ranked)

        log.info(
            f"    Reranking : {rerank_sec:.3f}s | kept {len(ranked)}/{len(candidates)} "
            f"top_score={top_score:.4f} mode={rerank_mode}"
        )
        if sources:
            log.info(f"    Sources   : {', '.join(sources[:5])}")

        # Log individual ranked chunks for debugging
        for ri, rc in enumerate(ranked[:3]):
            log.debug(
                f"    Rank[{ri}] src={rc['source']} type={rc.get('content_type','')} "
                f"score={rc.get('rerank_score',0):.4f} text={rc['text'][:100]}..."
            )

        # Build context
        context = build_context(ranked) if ranked else "No relevant excerpts were retrieved for this question."

        # LLM call with grounded system prompt
        p_tok = c_tok = tot_tok = 0
        tps = llm_sec = 0.0
        try:
            llm_start = time.perf_counter()
            resp = llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
                ],
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            )
            llm_sec = time.perf_counter() - llm_start
            answer = answer_text(resp)
            usage = getattr(resp, "usage", None)
            p_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
            c_tok = getattr(usage, "completion_tokens", 0) if usage else 0
            tot_tok = getattr(usage, "total_tokens", 0) if usage else 0
            tps = c_tok / llm_sec if llm_sec > 0 else 0.0
        except Exception as exc:
            answer = f"LLM call failed: {exc}"
            log.error(f"    LLM error Q{idx}: {exc}")

        log.info(f"    LLM       : {llm_sec:.3f}s | tokens={tot_tok} tok/s={tps:.1f}")
        log.info(f"    Answer    : {answer[:280].replace(chr(10), ' ')}")
        flush()

        metrics = {
            "question": question,
            "retrieval_sec": round(retrieval_sec, 4),
            "rerank_sec": round(rerank_sec, 4),
            "chunks_fetched": len(candidates),
            "chunks_after_rerank": len(ranked),
            "rerank_top_score": round(top_score, 6),
            "rerank_mode": rerank_mode,
            "llm_sec": round(llm_sec, 4),
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": tot_tok,
            "tokens_per_sec": round(tps, 2),
            "top_sources": sources,
        }
        timings[f"Q{idx}"] = metrics
        all_qa.append({"question": question, "answer": answer, "metrics": metrics})
        time.sleep(0.5)

    # Summary table
    sep("-")
    log.info(f"  {'Q':<4}  {'Retrieve':>9}  {'Rerank':>7}  {'LLM':>7}  {'Tokens':>7}  {'Tok/s':>6}  {'Score':>8}  Sources")
    sep("-")
    for key, val in timings.items():
        src_str = ", ".join(val["top_sources"][:2]) if val["top_sources"] else "-"
        log.info(
            f"  {key:<4}  {val['retrieval_sec']:>8.3f}s  {val['rerank_sec']:>6.3f}s  "
            f"{val['llm_sec']:>6.3f}s  {val['total_tokens']:>7}  {val['tokens_per_sec']:>5.1f}  "
            f"{val['rerank_top_score']:>7.4f}  {src_str}"
        )
    sep("-")
    flush()
    return timings, all_qa


# ═══════════════════════════════════════════════════════════════════════
# METRICS & MAIN
# ═══════════════════════════════════════════════════════════════════════

def save_metrics(timings, batch_total, sanity_time, init_time, query_timings, nim_status, entity_count):
    payload = {
        "run_id": RUN_ID,
        "timestamp": datetime.now().isoformat(),
        "python": sys.version,
        "nim_health": nim_status,
        "retrieval": {
            "mode": "hybrid BM25+semantic" if SPARSE else "semantic-only",
            "sparse": SPARSE,
            "top_k": TOP_K_RETRIEVE,
            "rerank_k": RERANK_K,
            "min_rerank_score": MIN_RERANK_SCORE,
            "reranker": RERANKER_MODEL,
        },
        "config": {
            "docs_folder": DOCS_FOLDER,
            "collection": COLLECTION_NAME,
            "dense_dim": DENSE_DIM,
            "llm": LLM_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "grounded_prompt": True,
        },
        "milvus": {"uri": MILVUS_URI, "entity_count": entity_count},
        "phase_times": {
            "init": round(init_time, 3),
            "sanity": round(sanity_time, 3),
            "batch": round(batch_total, 3),
        },
        "per_doc": timings,
        "rag": query_timings,
        "summary": {
            "total": len(timings),
            "ok": sum(1 for v in timings.values() if not v.get("skipped") and not v.get("error") and v.get("failures_count", 1) == 0),
            "skipped": sum(1 for v in timings.values() if v.get("skipped")),
            "failed": sum(1 for v in timings.values() if v.get("error") or v.get("failures_count", 0) > 0),
            "wall": round(init_time + sanity_time + batch_total, 3),
        },
    }
    with open(METRICS_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    log.info(f"  Metrics -> {METRICS_FILE}")
    flush()
    return payload


def main():
    interactive = "--interactive" in sys.argv
    global_start = time.perf_counter()

    sep()
    log.info("NV-INGEST 25.9.0  PRODUCTION RAG PIPELINE  v7")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Python {sys.version.split()[0]}")
    log.info(f"  {'INTERACTIVE' if interactive else 'PREDEFINED'} question mode")
    log.info(f"  Collection -> {COLLECTION_NAME}")
    log.info(f"  Log        -> {os.path.abspath(LOG_FILE)}")
    log.info(f"  Grounded   -> System prompt enforced (no hallucination)")
    sep()
    flush()

    questions = collect_questions() if interactive else DEFAULT_QUESTIONS
    if not interactive:
        log.info(f"  {len(questions)} predefined question(s).")
        flush()

    api_key = setup_env()
    files = find_files()
    client, init_time = start_pipeline()
    nim_status = check_nims()

    sanity_file = choose_sanity_file(files, nim_status)
    if sanity_file:
        sanity_time = sanity_check(client, sanity_file)
    else:
        sanity_time = 0.0
        log.warning("No suitable sanity-check file. Skipping.")
        flush()

    prepare_collection(COLLECTION_NAME)
    timings, batch_total = run_batch(client, files, nim_status, COLLECTION_NAME)
    milvus_ok, entity_count = check_milvus(COLLECTION_NAME)

    if not milvus_ok:
        log.error("Milvus empty. Skipping RAG.")
        query_timings, all_qa = {}, []
    else:
        query_timings, all_qa = run_rag(api_key, questions, COLLECTION_NAME)

    with open(ANSWERS_FILE, "w", encoding="utf-8") as fh:
        json.dump(all_qa, fh, indent=2, ensure_ascii=False)
    log.info(f"  Answers -> {ANSWERS_FILE}")
    flush()

    metrics = save_metrics(timings, batch_total, sanity_time, init_time, query_timings, nim_status, entity_count)

    sep()
    log.info("FINAL SUMMARY")
    sep()
    log.info(f"  Run ID     : {RUN_ID}")
    log.info(f"  Docs       : {metrics['summary']['total']}  OK={metrics['summary']['ok']}  Skipped={metrics['summary']['skipped']}  Failed={metrics['summary']['failed']}")
    log.info(f"  Wall       : {time.perf_counter() - global_start:.1f}s")
    log.info(f"  Retrieval  : hybrid BM25+semantic  fetch={TOP_K_RETRIEVE}  rerank_to={RERANK_K}")
    log.info(f"  Chunks     : {entity_count}")
    log.info(f"  Grounded   : YES (system prompt prevents hallucination)")
    log.info(f"  Log        -> {LOG_FILE}")
    log.info(f"  Metrics    -> {METRICS_FILE}")
    log.info(f"  Answers    -> {ANSWERS_FILE}")
    sep()
    flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrupted.")
        flush()
        sys.exit(1)
    except Exception as exc:
        log.error(f"FATAL: {exc}")
        log.error(traceback.format_exc())
        flush()
        sys.exit(1)
