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


if sys.version_info < (3, 12):
    print(
        f"[FATAL] NV-Ingest 25.9.0 officially requires Python 3.12+.\n"
        f"        You are running Python {sys.version}.\n"
        f"        Run: uv venv --python 3.12 nvingest && source nvingest/bin/activate"
    )
    sys.exit(1)

try:
    import pymilvus
    pymilvus.connections.disconnect("default")
except Exception:
    pass  # No existing connection — this is fine, just clearing stale state

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — Edit these values before running
# ─────────────────────────────────────────────────────────────────────────────
DOCS_FOLDER        = "Docs"
FILE_PATTERNS      = ["*.pdf", "*.docx", "*.pptx", "*.jpeg", "*.jpg", "*.png"]
MILVUS_URI         = "milvus.db"
COLLECTION_NAME    = "multimodal_docs"
SPARSE             = False
DENSE_DIM          = 2048           # matches llama-3.2-nv-embedqa-1b-v2
CHUNK_SIZE         = 512
CHUNK_OVERLAP      = 50
TOKENIZER          = "intfloat/e5-large-unsupervised"

LLM_MODEL          = "meta/llama-3.3-70b-instruct"
LLM_BASE_URL       = "https://integrate.api.nvidia.com/v1"

PIPELINE_WAIT_SEC  = 90
RESULTS_DIR        = "Outputs"
TOP_K              = 10

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

# ─────────────────────────────────────────────────────────────────────────────
# PREDEFINED QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_QUESTIONS = [
    "Why did economics and physics become early movers in open access adoption?",
    "How did arXiv influence scholarly communication in physics?",
    "Why did life sciences move more toward open access journals and APC models?",
    "What does the report mean by successive waves of open access innovation?",
    "How does the report connect open access, open data, and reproducibility?",
]

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

RUN_ID        = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE      = os.path.join(RESULTS_DIR, f"pipeline_run_{RUN_ID}.log")
METRICS_FILE  = os.path.join(RESULTS_DIR, f"metrics_{RUN_ID}.json")
ANSWERS_FILE  = os.path.join(RESULTS_DIR, f"answers_{RUN_ID}.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8", delay=False),
        logging.StreamHandler(sys.stdout),   # also print to console
    ],
)
log = logging.getLogger("nv_ingest_pipeline")

def divider(char="═", width=70):
    log.info(char * width)

def phase_start(name):
    log.info(f"PHASE START : {name}")
    return time.perf_counter()

def phase_end(name, t0):
    elapsed = time.perf_counter() - t0
    log.info(f"PHASE END   : {name}  →  {elapsed:.3f}s")
    return elapsed

# ─────────────────────────────────────────────────────────────────────────────
# COLLECT QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────
def collect_questions():
    print("\n" + "═" * 70)
    print("  INTERACTIVE QUESTION MODE")
    print("  Enter questions. Press Enter on blank line when done.")
    print("═" * 70 + "\n")
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
    return questions

# ─────────────────────────────────────────────────────────────────────────────
# This function must be called as the very first thing in main()
# ─────────────────────────────────────────────────────────────────────────────
def inject_and_verify_environment():
    """
    CRITICAL: inject NIM endpoint env vars into os.environ RIGHT NOW,
    before run_pipeline() is called.  The subprocess copies os.environ
    at call time — if vars are missing here, NIMs are never reached.
    """
    divider()
    log.info("STEP 0 — ENVIRONMENT INJECTION + VERIFICATION")
    divider()

    # 1. NVIDIA_API_KEY — must already be in env (user sets this)
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        log.error("NVIDIA_API_KEY is NOT set. Run: export NVIDIA_API_KEY='nvapi-...'")
        sys.exit(1)
    log.info(f"  NVIDIA_API_KEY        : {'*' * 8}{api_key[-6:]}")

    # 2. Inject NIM endpoints — set defaults if not already exported
    for var, default_val in NIM_ENV_VARS.items():
        current = os.environ.get(var, "")
        if not current:
            os.environ[var] = default_val
            log.info(f"  {var:<47} SET TO DEFAULT: {default_val}")
        else:
            log.info(f"  {var:<47} = {current}")

    # 3. Verify all 4 NIM HTTP endpoints are set (not blank)
    nim_http_vars = [
        "YOLOX_HTTP_ENDPOINT",
        "YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT",
        "YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT",
        "OCR_HTTP_ENDPOINT",
    ]
    all_set = all(os.environ.get(v, "") for v in nim_http_vars)
    if all_set:
        log.info("  All 4 NIM HTTP endpoints are set in os.environ ✓")
    else:
        log.warning("  Some NIM endpoints are still blank after injection — check above")

    log.info("  Environment injection complete. run_pipeline() will inherit these.")
    return api_key

def check_nim_health():
    divider()
    log.info("NIM HEALTH CHECK — verifying all 4 NIM endpoints")
    divider()
    nim_ports = {
        "page-elements (8000)"    : "http://localhost:8000/v1/health/ready",
        "graphic-elements (8003)" : "http://localhost:8003/v1/health/ready",
        "table-structure (8006)"  : "http://localhost:8006/v1/health/ready",
        "ocr (8009)"              : "http://localhost:8009/v1/health/ready",
    }
    all_ok = True
    for name, url in nim_ports.items():
        try:
            r = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "5", url],
                capture_output=True, text=True
            )
            code = r.stdout.strip()
            if code in ("200", "201"):
                log.info(f"  ✓  {name}  →  HTTP {code}")
            else:
                log.warning(f"  ✗  {name}  →  HTTP {code}  (NIM may not be ready)")
                all_ok = False
        except Exception as e:
            log.warning(f"  ✗  {name}  →  curl failed: {e}")
            all_ok = False

    if all_ok:
        log.info("  All 4 NIMs are healthy and ready ✓")
    else:
        log.warning("  One or more NIMs did not respond — extraction for those modalities will fall back")
        log.warning("  Proceeding anyway — check docker ps and docker logs if results are empty")
    return all_ok

# ─────────────────────────────────────────────────────────────────────────────
# COLLECT FILES
# ─────────────────────────────────────────────────────────────────────────────
def collect_files():
    divider()
    log.info(f"FILE DISCOVERY  →  folder: {DOCS_FOLDER}/")
    log.info(f"  Formats: {', '.join(FILE_PATTERNS)}")
    files = []
    for pattern in FILE_PATTERNS:
        matched = sorted(glob.glob(os.path.join(DOCS_FOLDER, pattern)))
        if matched:
            log.info(f"  {pattern:<12} → {len(matched)} file(s)")
        files.extend(matched)
    files = sorted(set(files))
    log.info(f"  ─── Total: {len(files)} file(s) ───")
    for f in files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        log.info(f"  › {f}  ({size_mb:.2f} MB)")
    if not files:
        log.error(f"  No files found in '{DOCS_FOLDER}/' matching {FILE_PATTERNS}.")
        sys.exit(1)
    return files

# ─────────────────────────────────────────────────────────────────────────────
# PORT READINESS POLL (replaces blind sleep)
# ─────────────────────────────────────────────────────────────────────────────
def wait_for_port(port=7671, host="localhost", timeout=120):
    """Poll port 7671 until it accepts connections, up to timeout seconds."""
    log.info(f"  Waiting for pipeline port {port} to be ready (max {timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                log.info(f"  Port {port} is accepting connections ✓")
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(2)
    log.error(f"  Port {port} did not become ready within {timeout}s — pipeline subprocess may have failed")
    return False

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE INIT
# ─────────────────────────────────────────────────────────────────────────────
def start_pipeline():
    divider()
    log.info("PIPELINE INITIALISATION")
    divider()

    
    from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import (
        run_pipeline,
        PipelineCreationSchema,
    )
    from nv_ingest_client.client import Ingestor, NvIngestClient
    from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient

    t0 = phase_start("Pipeline subprocess start")
    config = PipelineCreationSchema()
    run_pipeline(
        config,
        block=False,
        disable_dynamic_scaling=True,
        run_in_subprocess=True,
    )
    init_time = phase_end("Pipeline subprocess start", t0)

    # FIX-4 — poll for port readiness instead of blind sleep
    log.info(f"  Pipeline process launched. Waiting for port 7671 (max {PIPELINE_WAIT_SEC}s)...")
    port_ready = wait_for_port(port=7671, timeout=PIPELINE_WAIT_SEC)
    if not port_ready:
        log.error("  Pipeline port 7671 never opened. Check subprocess for errors.")
        log.error("  Common causes: Ray OOM, missing env var, import error in subprocess")
        sys.exit(1)

    # Extra 5s buffer for Ray actor registration after port opens
    time.sleep(5)

    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost",
    )
    log.info("  NvIngestClient connected  →  localhost:7671 ✓")
    return client, init_time

# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK — text-only on first file
# ─────────────────────────────────────────────────────────────────────────────
def sanity_check(client, filepath):
    from nv_ingest_client.client import Ingestor
    from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

    divider()
    log.info(f"SANITY CHECK (text-only)  →  {filepath}")
    divider()
    t0 = phase_start("Text-only extraction")
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
    sanity_time = phase_end("Text-only extraction", t0)

    log.info(f"  Results  : {len(results)}")
    log.info(f"  Failures : {len(failures)}")

    if failures:
        for i, f in enumerate(failures):
            log.error(f"  FAILURE [{i}]: {f}")
        log.error("  Sanity check FAILED. Fix the above errors before full batch.")
        sys.exit(1)

    if results:
        blob = ingest_json_results_to_blob(results[0])
        log.info(f"  Text preview: {blob[:300].replace(chr(10), ' ')}...")
        log.info("  SANITY CHECK PASSED ✓")
    else:
        log.error("  No results and no failures — unexpected empty response.")
        sys.exit(1)

    return sanity_time

# ─────────────────────────────────────────────────────────────────────────────
# INGEST ONE FILE — full multimodal, NO caption 
# ─────────────────────────────────────────────────────────────────────────────
def ingest_single_file(client, filepath, doc_index, total_docs):
    from nv_ingest_client.client import Ingestor

    fname = os.path.basename(filepath)
    log.info(f"  [{doc_index}/{total_docs}] Ingesting: {fname}")

    t_ingest_start = time.perf_counter()
    ingestor = (
        Ingestor(client=client)
        .files(filepath)
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
            tokenizer=TOKENIZER,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    
        .embed()
        .vdb_upload(
            collection_name=COLLECTION_NAME,
            milvus_uri=MILVUS_URI,
            sparse=SPARSE,
            dense_dim=DENSE_DIM,
        )
    )
    results, failures = ingestor.ingest(show_progress=False, return_failures=True)
    total_ingest_time = time.perf_counter() - t_ingest_start

    timings = {
        "total_ingest_sec" : round(total_ingest_time, 3),
        "results_count"    : len(results),
        "failures_count"   : len(failures),
    }

    if failures:
        for i, f in enumerate(failures):
            log.warning(f"    FAILURE [{i}] in {fname}: {str(f)[:200]}")

    log.info(
        f"    {fname} → results={len(results)}  failures={len(failures)}  "
        f"time={total_ingest_time:.3f}s"
    )
    return timings, failures

# ─────────────────────────────────────────────────────────────────────────────
# BATCH INGEST
# ─────────────────────────────────────────────────────────────────────────────
def run_batch_ingest(client, files):
    divider()
    log.info(f"BATCH INGEST  →  {len(files)} file(s)")
    log.info(f"  Collection : {COLLECTION_NAME}")
    log.info(f"  Milvus URI : {MILVUS_URI}")
    log.info(f"  Embedder   : dense_dim={DENSE_DIM}  tokenizer={TOKENIZER}")
    divider()

    batch_t0        = time.perf_counter()
    all_timings     = {}
    total_failures  = 0
    successful_docs = 0

    for idx, filepath in enumerate(files, start=1):
        fname  = os.path.basename(filepath)
        doc_t0 = time.perf_counter()
        try:
            timings, failures = ingest_single_file(client, filepath, idx, len(files))
            all_timings[fname]  = timings
            total_failures     += timings["failures_count"]
            if timings["failures_count"] == 0:
                successful_docs += 1
        except Exception as e:
            log.error(f"  EXCEPTION on {fname}: {e}")
            log.error(traceback.format_exc())
            all_timings[fname] = {"error": str(e), "total_ingest_sec": 0, "failures_count": 1}
        doc_elapsed = time.perf_counter() - doc_t0
        log.info(f"  ── doc wall time: {doc_elapsed:.3f}s ──")

    batch_total = time.perf_counter() - batch_t0

    divider()
    log.info("BATCH INGEST SUMMARY")
    log.info(f"  Total documents  : {len(files)}")
    log.info(f"  Successful       : {successful_docs}")
    log.info(f"  Total failures   : {total_failures}")
    log.info(f"  Total batch time : {batch_total:.3f}s")
    log.info(f"  Avg per document : {batch_total / max(len(files), 1):.3f}s")
    divider()
    return all_timings, batch_total

# ─────────────────────────────────────────────────────────────────────────────
# VERIFY MILVUS HAS ENTITIES BEFORE QUERYING
# ─────────────────────────────────────────────────────────────────────────────
def verify_milvus_populated():
    divider()
    log.info("MILVUS VERIFICATION — checking entity count before RAG queries")
    try:
        from pymilvus import Collection, connections, utility
        connections.connect(uri=MILVUS_URI)
        if utility.has_collection(COLLECTION_NAME):
            col = Collection(COLLECTION_NAME)
            col.load()
            count = col.num_entities
            log.info(f"  Collection '{COLLECTION_NAME}' has {count} entities")
            if count == 0:
                log.error("  ZERO entities in Milvus — embedding/upload step failed silently")
                log.error("  Check: env vars set? NIM health? milvus.db not locked?")
                return False
            log.info("  Milvus is populated ✓")
            return True
        else:
            log.error(f"  Collection '{COLLECTION_NAME}' does not exist — vdb_upload failed")
            return False
    except Exception as e:
        log.error(f"  Milvus verification error: {e}")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# RAG QUERIES
# ─────────────────────────────────────────────────────────────────────────────
def run_rag_queries(api_key, questions):
    from openai import OpenAI
    from nv_ingest_client.util.milvus import nvingest_retrieval

    divider()
    log.info(f"RAG RETRIEVAL + LLM INFERENCE  →  {len(questions)} question(s)")
    divider()

    llm_client    = OpenAI(base_url=LLM_BASE_URL, api_key=api_key)
    all_qa        = []
    query_timings = {}

    for i, q in enumerate(questions, start=1):
        log.info(f"  Q{i}: {q[:80]}...")

        t_ret = time.perf_counter()
        retrieved_docs = nvingest_retrieval(
            [q],
            COLLECTION_NAME,
            milvus_uri=MILVUS_URI,
            hybrid=SPARSE,
            top_k=TOP_K,
        )
        retrieval_time = time.perf_counter() - t_ret

        if retrieved_docs and retrieved_docs[0]:
            context      = "\n\n".join([d["entity"]["text"] for d in retrieved_docs[0]])
            chunks_found = len(retrieved_docs[0])
        else:
            context      = "No relevant content found."
            chunks_found = 0

        log.info(f"    Retrieval: {retrieval_time:.3f}s  |  chunks: {chunks_found}")

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
            llm_time = time.perf_counter() - t_llm
            answer   = completion.choices[0].message.content

            usage             = getattr(completion, "usage", None)
            prompt_tokens     = getattr(usage, "prompt_tokens",     0) if usage else 0
            completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            total_tokens      = getattr(usage, "total_tokens",      0) if usage else 0
            tokens_per_sec    = (completion_tokens / llm_time) if llm_time > 0 else 0

        except Exception as e:
            log.error(f"    LLM call failed for Q{i}: {e}")
            llm_time = 0; answer = f"LLM ERROR: {e}"
            prompt_tokens = completion_tokens = total_tokens = tokens_per_sec = 0

        log.info(f"    LLM: {llm_time:.3f}s  |  tokens={total_tokens}  tok/s={tokens_per_sec:.1f}")
        log.info(f"    Answer preview: {answer[:120].replace(chr(10), ' ')}...")

        query_timings[f"Q{i}"] = {
            "question"          : q,
            "retrieval_sec"     : round(retrieval_time, 4),
            "chunks_found"      : chunks_found,
            "llm_inference_sec" : round(llm_time, 4),
            "prompt_tokens"     : prompt_tokens,
            "completion_tokens" : completion_tokens,
            "total_tokens"      : total_tokens,
            "tokens_per_sec"    : round(tokens_per_sec, 2),
        }
        all_qa.append({"question": q, "answer": answer, "metrics": query_timings[f"Q{i}"]})

        # Small sleep between LLM calls to avoid rate-limit 429
        time.sleep(1)

    return query_timings, all_qa

# ─────────────────────────────────────────────────────────────────────────────
# SAVE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def save_metrics(all_timings, batch_total, sanity_time, init_time, query_timings):
    metrics = {
        "run_id"        : RUN_ID,
        "run_timestamp" : datetime.now().isoformat(),
        "python_version": sys.version,
        "config": {
            "docs_folder"   : DOCS_FOLDER,
            "file_patterns" : FILE_PATTERNS,
            "collection"    : COLLECTION_NAME,
            "dense_dim"     : DENSE_DIM,
            "llm_model"     : LLM_MODEL,
            "chunk_size"    : CHUNK_SIZE,
            "chunk_overlap" : CHUNK_OVERLAP,
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
            "total_docs"     : len(all_timings),
            "successful_docs": sum(
                1 for v in all_timings.values()
                if "error" not in v and v.get("failures_count", 1) == 0
            ),
            "avg_ingest_sec" : round(
                sum(v.get("total_ingest_sec", 0) for v in all_timings.values())
                / max(len(all_timings), 1), 3
            ),
            "total_wall_time_sec": round(init_time + sanity_time + batch_total, 3),
        },
    }
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"  Metrics saved → {METRICS_FILE}")
    return metrics

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def print_final_summary(metrics, init_time, sanity_time, batch_total):
    divider("═")
    log.info("FINAL RUN SUMMARY")
    divider("═")
    log.info(f"  Run ID                  : {metrics['run_id']}")
    log.info(f"  Pipeline init time      : {init_time:.3f}s")
    log.info(f"  Sanity check time       : {sanity_time:.3f}s")
    log.info(f"  Total batch ingest time : {batch_total:.3f}s")
    log.info(f"  Total documents         : {metrics['summary']['total_docs']}")
    log.info(f"  Successful documents    : {metrics['summary']['successful_docs']}")
    log.info(f"  Avg ingest per doc      : {metrics['summary']['avg_ingest_sec']:.3f}s")
    log.info(f"  Total wall time         : {metrics['summary']['total_wall_time_sec']:.3f}s")
    divider("-")
    log.info(f"  Log file     → {LOG_FILE}")
    log.info(f"  Metrics JSON → {METRICS_FILE}")
    log.info(f"  Answers JSON → {ANSWERS_FILE}")
    divider("═")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    interactive_mode = "--interactive" in sys.argv

    global_start = time.perf_counter()
    divider("═")
    log.info("NV-INGEST 25.9.0 — LIBRARY MODE PIPELINE (FULLY CORRECTED)")
    log.info(f"Run started at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Python version : {sys.version}")
    log.info(f"Results dir    : {RESULTS_DIR}/")
    log.info(f"Log file       : {LOG_FILE}")
    log.info(f"Question mode  : {'INTERACTIVE' if interactive_mode else 'PREDEFINED'}")
    divider("═")

    # STEP 0 — Collect questions BEFORE any GPU work
    if interactive_mode:
        questions = collect_questions()
    else:
        questions = DEFAULT_QUESTIONS
        log.info(f"  Using {len(questions)} predefined question(s).")

    # STEP 1 — FIX-2: inject + verify env vars BEFORE run_pipeline()
    api_key = inject_and_verify_environment()

    # STEP 2 — Collect files
    files = collect_files()

    # STEP 3 — Start pipeline (env vars already in os.environ, subprocess inherits them)
    client, init_time = start_pipeline()

    # STEP 4 — FIX-5: NIM health check right after pipeline starts
    check_nim_health()

    # STEP 5 — Sanity check (text-only) on first file
    sanity_time = sanity_check(client, files[0])

    # STEP 6 — Full batch ingest
    all_timings, batch_total = run_batch_ingest(client, files)

    # STEP 7 — FIX-9: Verify Milvus has data before querying
    milvus_ok = verify_milvus_populated()
    if not milvus_ok:
        log.error("  Skipping RAG queries — Milvus is empty. Check ingest logs above.")
        query_timings, all_qa = {}, []
    else:
        # STEP 8 — RAG queries
        query_timings, all_qa = run_rag_queries(api_key, questions)

    # STEP 9 — Save answers
    with open(ANSWERS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, indent=2, ensure_ascii=False)
    log.info(f"  Answers saved → {ANSWERS_FILE}")

    # STEP 10 — Save metrics
    metrics = save_metrics(all_timings, batch_total, sanity_time, init_time, query_timings)

    # STEP 11 — Final summary
    print_final_summary(metrics, init_time, sanity_time, batch_total)

    total_wall = time.perf_counter() - global_start
    log.info(f"  Total wall clock time : {total_wall:.3f}s")
    log.info("  PIPELINE COMPLETED SUCCESSFULLY")
    divider("═")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        log.error(f"FATAL ERROR: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)
