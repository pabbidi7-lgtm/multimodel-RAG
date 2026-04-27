import logging
import os
import sys
import time
import json
import glob
import socket
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Python version guard
# ─────────────────────────────────────────────────────────────────────────────
if sys.version_info < (3, 12):
    print(
        f"[FATAL] NV-Ingest 25.9.0 requires Python 3.12+.\n"
        f"        You are running Python {sys.version}.\n"
        f"        Fix: uv venv --python 3.12 nvingest && source nvingest/bin/activate"
    )
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# pymilvus stale connection cleanup
# ─────────────────────────────────────────────────────────────────────────────
try:
    import pymilvus
    pymilvus.connections.disconnect("default")
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DOCS_FOLDER     = "Docs"
FILE_PATTERNS   = ["*.pdf", "*.docx", "*.pptx", "*.jpeg", "*.jpg", "*.png"]
MILVUS_URI      = "milvus.db"
COLLECTION_NAME = "multimodal_docs"
SPARSE          = False
DENSE_DIM       = 2048        # llama-3.2-nv-embedqa-1b-v2 output dimension
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 50
TOKENIZER       = "intfloat/e5-large-unsupervised"
LLM_MODEL       = "meta/llama-3.3-70b-instruct"
LLM_BASE_URL    = "https://integrate.api.nvidia.com/v1"
PIPELINE_WAIT_SEC = 90
RESULTS_DIR     = "Outputs"

# TOP_K = max chunks returned FROM MILVUS per RAG query.
# This is NOT per-file chunk count.
# Each file produces its own chunk count based on length and CHUNK_SIZE.
# A 4MB PDF typically produces 200-600 chunks stored in Milvus.
# TOP_K only controls how many of those are retrieved per question.
TOP_K = 10

# NIM endpoint defaults - injected into os.environ BEFORE run_pipeline()
NIM_ENV_VARS = {
    "YOLOX_HTTP_ENDPOINT"                  : "http://localhost:8000/v1/infer",
    "YOLOX_INFER_PROTOCOL"                 : "http",
    "YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT" : "http://localhost:8003/v1/infer",
    "YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL": "http",
    "YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT"  : "http://localhost:8006/v1/infer",
    "YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL" : "http",
    "OCR_HTTP_ENDPOINT"                    : "http://localhost:8009/v1/infer",
    "OCR_INFER_PROTOCOL"                   : "http",
}

DEFAULT_QUESTIONS = [
    "Why did economics and physics become early movers in open access adoption?",
    "How did arXiv influence scholarly communication in physics?",
    "Why did life sciences move more toward open access journals and APC models?",
    "What does the report mean by successive waves of open access innovation?",
    "How does the report connect open access, open data, and reproducibility?",
]

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

RUN_ID       = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE     = os.path.join(RESULTS_DIR, f"pipeline_run_{RUN_ID}.log")
METRICS_FILE = os.path.join(RESULTS_DIR, f"metrics_{RUN_ID}.json")
ANSWERS_FILE = os.path.join(RESULTS_DIR, f"answers_{RUN_ID}.json")

_fmt = logging.Formatter(
    fmt="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

_fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8", delay=False)
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)

_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO)
_ch.setFormatter(_fmt)

log = logging.getLogger("pipeline")
log.setLevel(logging.DEBUG)
log.propagate = False   # KEY: isolates us from nv_ingest root logger hijack
log.addHandler(_fh)
log.addHandler(_ch)

def log_flush():
    """Force-flush all handlers so file is written immediately."""
    for h in log.handlers:
        h.flush()

def divider(char="=", width=72):
    log.info(char * width)
    log_flush()

def phase_start(name):
    log.info(f"PHASE START : {name}")
    log_flush()
    return time.perf_counter()

def phase_end(name, t0):
    elapsed = time.perf_counter() - t0
    log.info(f"PHASE END   : {name}  ->  {elapsed:.3f}s")
    log_flush()
    return elapsed

# ─────────────────────────────────────────────────────────────────────────────
# QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────
def collect_questions():
    print("\n" + "=" * 72)
    print("  INTERACTIVE QUESTION MODE -- Enter questions, blank line to finish")
    print("=" * 72 + "\n")
    questions = []
    idx = 1
    while True:
        try:
            q = input(f"  Q{idx}: ").strip()
        except EOFError:
            break
        if q.lower() in ("", "done"):
            if not questions:
                print("  [!] Enter at least one question.")
                continue
            break
        questions.append(q)
        idx += 1
    log.info(f"Interactive mode: {len(questions)} question(s) collected.")
    log_flush()
    return questions

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT INJECTION -- ALL vars set BEFORE run_pipeline()
# ─────────────────────────────────────────────────────────────────────────────
def inject_and_verify_environment():
    divider("=")
    log.info("STEP 0 -- ENVIRONMENT INJECTION + VERIFICATION")
    divider("=")

    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        log.error("NVIDIA_API_KEY is NOT set.")
        log.error("Run: export NVIDIA_API_KEY='nvapi-...'")
        log_flush()
        sys.exit(1)
    log.info(f"  NVIDIA_API_KEY : {'*' * 8}{api_key[-6:]}")

    for var, default_val in NIM_ENV_VARS.items():
        current = os.environ.get(var, "")
        if not current:
            os.environ[var] = default_val
            log.info(f"  {var:<47} -> injected: {default_val}")
        else:
            log.info(f"  {var:<47} -> already set: {current}")

    log.info("  All NIM env vars in os.environ -- subprocess will inherit them.")
    log_flush()
    return api_key

# ─────────────────────────────────────────────────────────────────────────────
# NIM HEALTH CHECK -- returns dict, tells you exactly what to fix
# ─────────────────────────────────────────────────────────────────────────────
NIM_HEALTH_MAP = {
    "page_elements"    : ("http://localhost:8000/v1/health/ready", 8000, "yolox"),
    "graphic_elements" : ("http://localhost:8003/v1/health/ready", 8003, "yolox-graphic-elements"),
    "table_structure"  : ("http://localhost:8006/v1/health/ready", 8006, "yolox-table-structure"),
    "ocr"              : ("http://localhost:8009/v1/health/ready", 8009, "ocr"),
}

def check_nim_health():
    divider("=")
    log.info("NIM HEALTH CHECK")
    divider("=")

    nim_status = {}
    all_ok = True

    for name, (url, port, profile) in NIM_HEALTH_MAP.items():
        try:
            r = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                 "--max-time", "5", url],
                capture_output=True, text=True
            )
            code = r.stdout.strip()
            ok = code in ("200", "201")
            nim_status[name] = ok
            if ok:
                log.info(f"  OK   {name:<22} port {port}  HTTP {code}")
            else:
                log.warning(f"  FAIL {name:<22} port {port}  HTTP {code}")
                if code == "000":
                    log.warning(f"       Container not running.")
                    log.warning(f"       Fix: cd nv-ingest && docker compose --profile {profile} up -d")
                elif code == "404":
                    log.warning(f"       Container up but /v1/health/ready returned 404.")
                    log.warning(f"       Try: curl http://localhost:{port}/v1/health")
                    log.warning(f"            curl http://localhost:{port}/health/ready")
                all_ok = False
        except Exception as e:
            nim_status[name] = False
            log.warning(f"  FAIL {name:<22} port {port}  curl error: {e}")
            all_ok = False

    healthy_count = sum(nim_status.values())

    if all_ok:
        log.info("  All 4 NIMs HEALTHY -- full multimodal extraction will run.")
    else:
        log.warning(f"  {healthy_count}/4 NIMs healthy.")
        log.warning("  IMPACT: Extraction will run in TEXT-ONLY mode.")
        log.warning("  Tables, charts, images, infographics will be SKIPPED.")
        log.warning("  Text chunks will still be indexed into Milvus.")
        log.warning("  To enable full multimodal, start all NIMs:")
        log.warning("    cd nv-ingest")
        log.warning("    docker compose --profile retrieval --profile table-structure up -d")
        log.warning("    # Wait 10-15 min for model loading, then re-run.")

    log_flush()
    return nim_status

# ─────────────────────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────
def collect_files():
    divider("=")
    log.info(f"FILE DISCOVERY  ->  folder: {DOCS_FOLDER}/")

    files = []
    for pattern in FILE_PATTERNS:
        matched = sorted(glob.glob(os.path.join(DOCS_FOLDER, pattern)))
        if matched:
            log.info(f"  {pattern:<12} -> {len(matched)} file(s)")
        files.extend(matched)
    files = sorted(set(files))

    log.info(f"  Total: {len(files)} file(s)")
    for f in files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        log.info(f"    {Path(f).name:<50}  {size_mb:.2f} MB")

    if not files:
        log.error(f"No files found in '{DOCS_FOLDER}/' matching {FILE_PATTERNS}.")
        log_flush()
        sys.exit(1)

    log_flush()
    return files

# ─────────────────────────────────────────────────────────────────────────────
# PORT READINESS POLL
# ─────────────────────────────────────────────────────────────────────────────
def wait_for_port(port=7671, host="localhost", timeout=120):
    log.info(f"  Polling port {port} for pipeline readiness (max {timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                log.info(f"  Port {port} accepting connections OK")
                log_flush()
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(2)
    log.error(f"  Port {port} not ready after {timeout}s.")
    log.error("  Check: ps aux | grep nv_ingest | grep -v grep")
    log_flush()
    return False

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE INIT
# ─────────────────────────────────────────────────────────────────────────────
def start_pipeline():
    divider("=")
    log.info("PIPELINE INITIALISATION")
    divider("=")

    from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import (
        run_pipeline,
        PipelineCreationSchema,
    )
    from nv_ingest_client.client import NvIngestClient
    from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient

    t0 = phase_start("Pipeline subprocess")
    config = PipelineCreationSchema()
    run_pipeline(config, block=False, disable_dynamic_scaling=True, run_in_subprocess=True)
    init_time = phase_end("Pipeline subprocess", t0)

    if not wait_for_port(port=7671, timeout=PIPELINE_WAIT_SEC):
        log.error("Pipeline startup failed. Exiting.")
        log_flush()
        sys.exit(1)

    time.sleep(5)  # buffer for Ray actor registration

    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost",
    )
    log.info("  NvIngestClient connected  ->  localhost:7671 OK")
    log_flush()
    return client, init_time

# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
def sanity_check(client, filepath):
    from nv_ingest_client.client import Ingestor
    from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

    divider("=")
    log.info(f"SANITY CHECK (text-only)  ->  {Path(filepath).name}")
    divider("=")
    t0 = phase_start("Sanity extraction")

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
    sanity_time = phase_end("Sanity extraction", t0)

    log.info(f"  Results: {len(results)}  Failures: {len(failures)}")

    if failures:
        for i, f in enumerate(failures):
            log.error(f"  FAILURE [{i}]: {f}")
        log.error("  Sanity check FAILED. Fix above before running batch.")
        log_flush()
        sys.exit(1)

    if results:
        blob = ingest_json_results_to_blob(results[0])
        log.info(f"  Preview: {blob[:300].replace(chr(10), ' ')}...")
        log.info("  SANITY CHECK PASSED")
    else:
        log.error("  No results and no failures -- unexpected empty response.")
        log_flush()
        sys.exit(1)

    log_flush()
    return sanity_time

# ─────────────────────────────────────────────────────────────────────────────
# INGEST ONE FILE -- adapts to NIM availability
# ─────────────────────────────────────────────────────────────────────────────
def ingest_single_file(client, filepath, doc_index, total_docs, nim_status):
    from nv_ingest_client.client import Ingestor

    fname    = Path(filepath).name
    all_nims = all(nim_status.values())
    mode     = "FULL MULTIMODAL" if all_nims else "TEXT-ONLY"

    log.info(f"  [{doc_index}/{total_docs}]  {fname}  [{mode}]")

    if all_nims:
        extract_kwargs = dict(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=True,
            extract_infographics=True,
            table_output_format="markdown",
            text_depth="page",
        )
    else:
        extract_kwargs = dict(
            extract_text=True,
            extract_tables=False,
            extract_charts=False,
            extract_images=False,
            extract_infographics=False,
            text_depth="page",
        )

    t0 = time.perf_counter()
    try:
        ingestor = (
            Ingestor(client=client)
            .files(filepath)
            .extract(**extract_kwargs)
            .split(
                tokenizer=TOKENIZER,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
            # .caption() disabled -- calls cloud API per image -> rate limits on batch.
            # Re-enable only after confirming full NIM pipeline works end-to-end.
            # .caption(
            #     endpoint_url="https://integrate.api.nvidia.com/v1/chat/completions",
            #     model_name="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
            #     api_key=api_key,
            # )
            .embed()
            .vdb_upload(
                collection_name=COLLECTION_NAME,
                milvus_uri=MILVUS_URI,
                sparse=SPARSE,
                dense_dim=DENSE_DIM,
            )
        )
        results, failures = ingestor.ingest(show_progress=False, return_failures=True)
        elapsed = time.perf_counter() - t0

        if failures:
            for i, f in enumerate(failures):
                full_err = str(f)
                log.warning(f"    FAILURE [{i}] (full message):")
                for start in range(0, len(full_err), 180):
                    log.warning(f"      {full_err[start:start+180]}")

        log.info(f"    results={len(results)}  failures={len(failures)}  time={elapsed:.3f}s")
        log_flush()

        return {
            "mode"             : mode,
            "total_ingest_sec" : round(elapsed, 3),
            "results_count"    : len(results),
            "failures_count"   : len(failures),
        }, failures

    except Exception as e:
        elapsed = time.perf_counter() - t0
        log.error(f"    EXCEPTION: {e}")
        log.error(traceback.format_exc())
        log_flush()
        return {
            "mode": mode, "error": str(e),
            "total_ingest_sec": round(elapsed, 3),
            "results_count": 0, "failures_count": 1
        }, [e]

# ─────────────────────────────────────────────────────────────────────────────
# BATCH INGEST
# ─────────────────────────────────────────────────────────────────────────────
def run_batch_ingest(client, files, nim_status):
    divider("=")
    mode_label = "FULL MULTIMODAL" if all(nim_status.values()) else "TEXT-ONLY (NIMs not all healthy)"
    log.info(f"BATCH INGEST  ->  {len(files)} files  |  mode: {mode_label}")
    log.info(f"  Collection  : {COLLECTION_NAME}")
    log.info(f"  Milvus URI  : {MILVUS_URI}")
    log.info(f"  Chunk size  : {CHUNK_SIZE} tokens  overlap: {CHUNK_OVERLAP}")
    log.info(f"  Embed dim   : {DENSE_DIM}  tokenizer: {TOKENIZER}")
    log.info(f"  TOP_K={TOP_K} is the RAG retrieval count per query, NOT per-file chunk count.")
    log.info(f"  Per-file chunk count depends on document length / {CHUNK_SIZE} tokens.")
    divider("=")
    log_flush()

    batch_t0 = time.perf_counter()
    all_timings = {}
    total_failures = 0
    successful_docs = 0

    for idx, filepath in enumerate(files, start=1):
        fname  = Path(filepath).name
        doc_t0 = time.perf_counter()
        try:
            timings, failures = ingest_single_file(client, filepath, idx, len(files), nim_status)
            all_timings[fname] = timings
            total_failures    += timings["failures_count"]
            if timings["failures_count"] == 0:
                successful_docs += 1
        except Exception as e:
            log.error(f"  OUTER EXCEPTION on {fname}: {e}")
            log.error(traceback.format_exc())
            all_timings[fname] = {
                "mode": "unknown", "error": str(e),
                "total_ingest_sec": 0, "failures_count": 1
            }
            total_failures += 1

        wall = time.perf_counter() - doc_t0
        log.info(f"  -- wall: {wall:.3f}s --")
        log_flush()

    batch_total = time.perf_counter() - batch_t0

    divider("=")
    log.info("BATCH SUMMARY")
    log.info(f"  Total    : {len(files)}")
    log.info(f"  OK       : {successful_docs}")
    log.info(f"  Failed   : {total_failures}")
    log.info(f"  Total t  : {batch_total:.3f}s")
    log.info(f"  Avg/doc  : {batch_total / max(len(files), 1):.3f}s")
    if successful_docs == 0:
        log.error("  ALL FILES FAILED -- check NIM health output above.")
    divider("=")
    log_flush()
    return all_timings, batch_total

# ─────────────────────────────────────────────────────────────────────────────
# MILVUS VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def verify_milvus_populated():
    divider("=")
    log.info("MILVUS VERIFICATION")
    try:
        from pymilvus import Collection, connections, utility
        connections.connect(uri=MILVUS_URI)
        if utility.has_collection(COLLECTION_NAME):
            col   = Collection(COLLECTION_NAME)
            col.load()
            count = col.num_entities
            log.info(f"  Collection '{COLLECTION_NAME}': {count} total entities (chunks)")
            if count == 0:
                log.error("  ZERO entities -- embed/upload failed silently.")
                log.error("  Check env vars, NIM health, milvus.db file lock.")
                log_flush()
                return False
            log.info(f"  Milvus populated OK  ({count} chunks total)")
            log.info(f"  RAG will retrieve top {TOP_K} most relevant per query.")
            log_flush()
            return True
        else:
            log.error(f"  Collection '{COLLECTION_NAME}' does not exist -- vdb_upload failed.")
            log_flush()
            return False
    except Exception as e:
        log.error(f"  Milvus verification error: {e}")
        log_flush()
        return False

# ─────────────────────────────────────────────────────────────────────────────
# RAG QUERIES
# ─────────────────────────────────────────────────────────────────────────────
def run_rag_queries(api_key, questions):
    from openai import OpenAI
    from nv_ingest_client.util.milvus import nvingest_retrieval

    divider("=")
    log.info(f"RAG QUERIES  ->  {len(questions)} question(s)")
    log.info(f"  Milvus TOP_K  : {TOP_K}  (max chunks returned per query)")
    log.info(f"  LLM model     : {LLM_MODEL}")
    divider("=")
    log_flush()

    llm_client    = OpenAI(base_url=LLM_BASE_URL, api_key=api_key)
    all_qa        = []
    query_timings = {}

    for i, q in enumerate(questions, start=1):
        log.info(f"  Q{i}: {q}")
        log_flush()

        t_ret = time.perf_counter()
        retrieved_docs = nvingest_retrieval(
            [q], COLLECTION_NAME, milvus_uri=MILVUS_URI, hybrid=SPARSE, top_k=TOP_K,
        )
        retrieval_time = time.perf_counter() - t_ret

        if retrieved_docs and retrieved_docs[0]:
            context      = "\n\n".join([d["entity"]["text"] for d in retrieved_docs[0]])
            chunks_found = len(retrieved_docs[0])
        else:
            context      = "No relevant content found."
            chunks_found = 0

        log.info(
            f"    Retrieval: {retrieval_time:.3f}s | "
            f"chunks returned: {chunks_found}  (TOP_K={TOP_K})"
        )

        prompt = (
            "Use the following context to answer the question.\n"
            "If the answer is not in the context, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"
        )

        t_llm = time.perf_counter()
        try:
            completion = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.7,
            )
            llm_time          = time.perf_counter() - t_llm
            answer            = completion.choices[0].message.content
            usage             = getattr(completion, "usage", None)
            prompt_tokens     = getattr(usage, "prompt_tokens",     0) if usage else 0
            completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            total_tokens      = getattr(usage, "total_tokens",      0) if usage else 0
            tokens_per_sec    = completion_tokens / llm_time if llm_time > 0 else 0
        except Exception as e:
            log.error(f"    LLM failed Q{i}: {e}")
            llm_time = 0
            answer   = f"LLM ERROR: {e}"
            prompt_tokens = completion_tokens = total_tokens = tokens_per_sec = 0

        log.info(f"    LLM: {llm_time:.3f}s | tokens={total_tokens} | tok/s={tokens_per_sec:.1f}")
        log.info(f"    Answer: {answer[:200].replace(chr(10), ' ')}")
        log_flush()

        query_timings[f"Q{i}"] = {
            "question"          : q,
            "retrieval_sec"     : round(retrieval_time, 4),
            "chunks_returned"   : chunks_found,
            "top_k_setting"     : TOP_K,
            "llm_inference_sec" : round(llm_time, 4),
            "prompt_tokens"     : prompt_tokens,
            "completion_tokens" : completion_tokens,
            "total_tokens"      : total_tokens,
            "tokens_per_sec"    : round(tokens_per_sec, 2),
        }
        all_qa.append({"question": q, "answer": answer, "metrics": query_timings[f"Q{i}"]})
        time.sleep(1)

    return query_timings, all_qa

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def save_metrics(all_timings, batch_total, sanity_time, init_time, query_timings, nim_status):
    metrics = {
        "run_id"                 : RUN_ID,
        "run_timestamp"          : datetime.now().isoformat(),
        "python_version"         : sys.version,
        "nim_health_at_run_time" : nim_status,
        "config": {
            "docs_folder"     : DOCS_FOLDER,
            "file_patterns"   : FILE_PATTERNS,
            "collection"      : COLLECTION_NAME,
            "dense_dim"       : DENSE_DIM,
            "llm_model"       : LLM_MODEL,
            "chunk_size"      : CHUNK_SIZE,
            "chunk_overlap"   : CHUNK_OVERLAP,
            "top_k_retrieval" : TOP_K,
        },
        "nim_endpoints": {
            "page_elements"    : os.environ.get("YOLOX_HTTP_ENDPOINT",                  "NOT SET"),
            "graphic_elements" : os.environ.get("YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT", "NOT SET"),
            "table_structure"  : os.environ.get("YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT",  "NOT SET"),
            "ocr"              : os.environ.get("OCR_HTTP_ENDPOINT",                    "NOT SET"),
        },
        "phase_times_sec": {
            "pipeline_init" : round(init_time, 3),
            "sanity_check"  : round(sanity_time, 3),
            "batch_ingest"  : round(batch_total, 3),
        },
        "per_document_timings" : all_timings,
        "rag_query_timings"    : query_timings,
        "summary": {
            "total_docs"      : len(all_timings),
            "successful_docs" : sum(
                1 for v in all_timings.values()
                if "error" not in v and v.get("failures_count", 1) == 0
            ),
            "avg_ingest_sec"  : round(
                sum(v.get("total_ingest_sec", 0) for v in all_timings.values())
                / max(len(all_timings), 1), 3
            ),
            "total_wall_time_sec": round(init_time + sanity_time + batch_total, 3),
        },
    }
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"  Metrics saved -> {METRICS_FILE}")
    log_flush()
    return metrics

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def print_final_summary(metrics, init_time, sanity_time, batch_total):
    divider("=")
    log.info("FINAL RUN SUMMARY")
    divider("=")
    log.info(f"  Run ID               : {metrics['run_id']}")
    log.info(f"  Pipeline init        : {init_time:.3f}s")
    log.info(f"  Sanity check         : {sanity_time:.3f}s")
    log.info(f"  Total batch ingest   : {batch_total:.3f}s")
    log.info(f"  Total documents      : {metrics['summary']['total_docs']}")
    log.info(f"  Successful           : {metrics['summary']['successful_docs']}")
    log.info(f"  Avg per doc          : {metrics['summary']['avg_ingest_sec']:.3f}s")
    log.info(f"  Total wall time      : {metrics['summary']['total_wall_time_sec']:.3f}s")
    log.info(f"  Log file    -> {LOG_FILE}")
    log.info(f"  Metrics     -> {METRICS_FILE}")
    log.info(f"  Answers     -> {ANSWERS_FILE}")
    divider("=")
    log_flush()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    interactive_mode = "--interactive" in sys.argv

    global_start = time.perf_counter()
    divider("=")
    log.info("NV-INGEST 25.9.0  LIBRARY MODE  v3")
    log.info(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Python   : {sys.version.split('|')[0].strip()}")
    log.info(f"  Outputs  : {os.path.abspath(RESULTS_DIR)}/")
    log.info(f"  Log file : {os.path.abspath(LOG_FILE)}")
    log.info(f"  Mode     : {'INTERACTIVE' if interactive_mode else 'PREDEFINED QUESTIONS'}")
    divider("=")
    log_flush()

    # 0 — Questions first (before GPU work)
    if interactive_mode:
        questions = collect_questions()
    else:
        questions = DEFAULT_QUESTIONS
        log.info(f"  {len(questions)} predefined question(s) loaded.")
        log_flush()

    # 1 — Inject env vars BEFORE run_pipeline()
    api_key = inject_and_verify_environment()

    # 2 — Files
    files = collect_files()

    # 3 — Start pipeline subprocess
    client, init_time = start_pipeline()

    # 4 — NIM health check (determines extraction mode)
    nim_status = check_nim_health()

    # 5 — Sanity check (text-only)
    sanity_time = sanity_check(client, files[0])

    # 6 — Batch ingest (adapts to NIM availability)
    all_timings, batch_total = run_batch_ingest(client, files, nim_status)

    # 7 — Verify Milvus
    milvus_ok = verify_milvus_populated()
    if not milvus_ok:
        log.error("Skipping RAG queries -- Milvus empty. Check ingest logs.")
        query_timings, all_qa = {}, []
    else:
        # 8 — RAG queries
        query_timings, all_qa = run_rag_queries(api_key, questions)

    # 9 — Save answers
    with open(ANSWERS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, indent=2, ensure_ascii=False)
    log.info(f"  Answers saved -> {ANSWERS_FILE}")
    log_flush()

    # 10 — Metrics
    metrics = save_metrics(
        all_timings, batch_total, sanity_time, init_time, query_timings, nim_status
    )

    # 11 — Summary
    print_final_summary(metrics, init_time, sanity_time, batch_total)

    total_wall = time.perf_counter() - global_start
    log.info(f"  Total wall clock: {total_wall:.3f}s")
    log.info("  PIPELINE DONE.")
    divider("=")
    log_flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        log_flush()
        sys.exit(1)
    except Exception as e:
        log.error(f"FATAL: {e}")
        log.error(traceback.format_exc())
        log_flush()
        sys.exit(1)
