"""
NV-Ingest 25.9.0 — Library Mode Pipeline
Full Logging + Timing + Latency Tracking + 100-PDF Batch Support
Target: A100 GPU with all Nemotron NIMs running via docker compose

Author : Your Team
Version: 25.9.0
"""

import logging
import os
import sys
import time
import json
import glob
import traceback
from datetime import datetime
from pathlib import Path

import pymilvus
pymilvus.connections.disconnect("default")

from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import (
    run_pipeline,
    PipelineCreationSchema,
)
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — Edit these values before running
# ─────────────────────────────────────────────────────────────────────────────
DOCS_FOLDER        = "Docs"          # folder containing your PDFs / images
FILE_PATTERN       = "*.pdf"         # change to *.jpeg, *.png, *.docx etc.
MILVUS_URI         = "milvus.db"
COLLECTION_NAME    = "multimodal_docs"
SPARSE             = False
DENSE_DIM          = 2048
CHUNK_SIZE         = 512
CHUNK_OVERLAP      = 50
TOKENIZER          = "intfloat/e5-large-unsupervised"
CAPTION_ENDPOINT   = "https://integrate.api.nvidia.com/v1/chat/completions"
CAPTION_MODEL      = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
LLM_MODEL          = "meta/llama-3.3-70b-instruct"
LLM_BASE_URL       = "https://integrate.api.nvidia.com/v1"
PIPELINE_WAIT_SEC  = 20              # seconds to wait for pipeline to init
RESULTS_DIR        = "results"       # all output logs saved here
TOP_K              = 10

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
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("nv_ingest_pipeline")

def divider(char="═", width=70):
    log.info(char * width)

def phase_start(name):
    log.info(f"▶  PHASE START : {name}")
    return time.perf_counter()

def phase_end(name, t0):
    elapsed = time.perf_counter() - t0
    log.info(f"✔  PHASE END   : {name}  →  {elapsed:.3f}s")
    return elapsed

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT CHECK
# ─────────────────────────────────────────────────────────────────────────────
def check_environment():
    divider()
    log.info("ENVIRONMENT CHECK")
    divider()

    assert "NVIDIA_API_KEY" in os.environ, (
        "NVIDIA_API_KEY not set.  Run: export NVIDIA_API_KEY='nvapi-...' "
    )
    NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
    log.info(f"  NVIDIA_API_KEY        : {'*' * 8}{NVIDIA_API_KEY[-6:]}")

    nim_vars = {
        "YOLOX_HTTP_ENDPOINT"                    : "page-elements     (port 8000)",
        "YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT"   : "graphic-elements  (port 8003)",
        "YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT"    : "table-structure   (port 8006)",
        "OCR_HTTP_ENDPOINT"                      : "ocr               (port 8009)",
    }

    all_set = True
    for var, desc in nim_vars.items():
        val = os.environ.get(var, "")
        if val:
            log.info(f"  {var:<45} = {val}")
        else:
            log.warning(f"  {var:<45} NOT SET — {desc} will NOT be invoked")
            all_set = False

    if not all_set:
        log.warning("  Some NIM endpoints are missing. Nemotron models for those")
        log.warning("  modalities will NOT run. Basic extraction only for those paths.")
    else:
        log.info("  All 4 Nemotron NIM endpoints are set. Full multimodal extraction enabled.")

    return NVIDIA_API_KEY

# ─────────────────────────────────────────────────────────────────────────────
# COLLECT FILES
# ─────────────────────────────────────────────────────────────────────────────
def collect_files():
    pattern = os.path.join(DOCS_FOLDER, FILE_PATTERN)
    files   = sorted(glob.glob(pattern))
    divider()
    log.info(f"FILE DISCOVERY  →  pattern: {pattern}")
    log.info(f"  Found {len(files)} file(s)")
    for f in files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        log.info(f"  › {f}  ({size_mb:.2f} MB)")
    if not files:
        log.error("  No files found. Check DOCS_FOLDER and FILE_PATTERN.")
        sys.exit(1)
    return files

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE INIT
# ─────────────────────────────────────────────────────────────────────────────
def start_pipeline():
    divider()
    log.info("PIPELINE INITIALISATION")
    divider()
    t0 = phase_start("Pipeline subprocess start")
    config = PipelineCreationSchema()
    run_pipeline(
        config,
        block=False,
        disable_dynamic_scaling=True,
        run_in_subprocess=True,
    )
    init_time = phase_end("Pipeline subprocess start", t0)
    log.info(f"  Waiting {PIPELINE_WAIT_SEC}s for pipeline to become ready...")
    time.sleep(PIPELINE_WAIT_SEC)
    log.info("  Pipeline ready.")

    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost",
    )
    log.info("  NvIngestClient connected  →  localhost:7671")
    return client, init_time

# ─────────────────────────────────────────────────────────────────────────────
# INGEST ONE FILE  (text-only sanity check)
# ─────────────────────────────────────────────────────────────────────────────
def sanity_check(client, filepath):
    divider()
    log.info(f"STEP 1 — SANITY CHECK (text-only)  →  {filepath}")
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
    log.info(f"  Time     : {sanity_time:.3f}s")

    if failures:
        for i, f in enumerate(failures):
            log.error(f"  FAILURE [{i}]: {f}")
        log.error("  Sanity check failed. Fix errors before running full batch.")
        sys.exit(1)

    if results:
        blob = ingest_json_results_to_blob(results[0])
        preview = blob[:400].replace("\n", " ")
        log.info(f"  Text preview: {preview}...")
        log.info("  SANITY CHECK PASSED")
    else:
        log.error("  No results and no failures — unexpected state.")
        sys.exit(1)

    return sanity_time

# ─────────────────────────────────────────────────────────────────────────────
# FULL MULTIMODAL INGEST — one file
# ─────────────────────────────────────────────────────────────────────────────
def ingest_single_file(client, filepath, api_key, doc_index, total_docs):
    fname = os.path.basename(filepath)
    log.info(f"  [{doc_index}/{total_docs}] Ingesting: {fname}")

    timings = {}

    # ── Extract ──────────────────────────────────────────────────────────────
    t_extract = phase_start(f"Extract  [{fname}]")
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
        .caption(
            endpoint_url=CAPTION_ENDPOINT,
            model_name=CAPTION_MODEL,
            api_key=api_key,
        )
        .embed()
        .vdb_upload(
            collection_name=COLLECTION_NAME,
            milvus_uri=MILVUS_URI,
            sparse=SPARSE,
            dense_dim=DENSE_DIM,
        )
    )

    t_ingest_start = time.perf_counter()
    results, failures = ingestor.ingest(show_progress=False, return_failures=True)
    total_ingest_time = time.perf_counter() - t_ingest_start

    timings["total_ingest_sec"] = round(total_ingest_time, 3)
    timings["results_count"]    = len(results)
    timings["failures_count"]   = len(failures)

    if failures:
        for i, f in enumerate(failures):
            log.warning(f"    FAILURE [{i}] in {fname}: {str(f)[:200]}")

    log.info(
        f"    {fname} → results={len(results)}  failures={len(failures)}  "
        f"time={total_ingest_time:.3f}s"
    )

    return timings, failures

# ─────────────────────────────────────────────────────────────────────────────
# BATCH INGEST — all files
# ─────────────────────────────────────────────────────────────────────────────
def run_batch_ingest(client, files, api_key):
    divider()
    log.info(f"STEP 2 — FULL MULTIMODAL BATCH INGEST  →  {len(files)} file(s)")
    log.info(f"  Collection : {COLLECTION_NAME}")
    log.info(f"  Milvus URI : {MILVUS_URI}")
    log.info(f"  Caption    : {CAPTION_MODEL}")
    log.info(f"  Embedder   : dense_dim={DENSE_DIM}  tokenizer={TOKENIZER}")
    log.info(f"  NIM Models : page-elements | graphic-elements | table-structure | OCR")
    divider()

    batch_t0          = time.perf_counter()
    all_timings       = {}
    total_failures    = 0
    successful_docs   = 0

    for idx, filepath in enumerate(files, start=1):
        fname = os.path.basename(filepath)
        doc_t0 = time.perf_counter()

        try:
            timings, failures = ingest_single_file(
                client, filepath, api_key, idx, len(files)
            )
            all_timings[fname] = timings
            total_failures    += timings["failures_count"]
            if timings["failures_count"] == 0:
                successful_docs += 1
        except Exception as e:
            log.error(f"  EXCEPTION on {fname}: {e}")
            log.error(traceback.format_exc())
            all_timings[fname] = {"error": str(e), "total_ingest_sec": 0}

        doc_elapsed = time.perf_counter() - doc_t0
        log.info(f"  ── doc wall time: {doc_elapsed:.3f}s  ──")

    batch_total = time.perf_counter() - batch_t0

    divider()
    log.info("BATCH INGEST SUMMARY")
    log.info(f"  Total documents   : {len(files)}")
    log.info(f"  Successful        : {successful_docs}")
    log.info(f"  Total failures    : {total_failures}")
    log.info(f"  Total batch time  : {batch_total:.3f}s")
    log.info(f"  Avg per document  : {batch_total / max(len(files), 1):.3f}s")
    divider()

    return all_timings, batch_total

# ─────────────────────────────────────────────────────────────────────────────
# RAG QUERY + TIMING
# ─────────────────────────────────────────────────────────────────────────────
def run_rag_queries(api_key):
    from openai import OpenAI
    from nv_ingest_client.util.milvus import nvingest_retrieval

    divider()
    log.info("STEP 3 — RAG RETRIEVAL + LLM INFERENCE")
    divider()

    queries = [
        "Why did economics and physics become early movers in open access adoption?",
        "How did arXiv influence scholarly communication in physics?",
        "Why did life sciences move more toward open access journals and APC models instead of preprints?",
        "What does the report mean by saying open access has grown through successive waves of innovation?",
        "How does the report connect open access, open data, and reproducibility?",
    ]

    llm_client = OpenAI(base_url=LLM_BASE_URL, api_key=api_key)
    all_qa     = []
    query_timings = {}

    for i, q in enumerate(queries, start=1):
        log.info(f"  Q{i}: {q[:80]}...")

        # ── Retrieval timing ──────────────────────────────────────────────────
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

        log.info(f"    Retrieval  : {retrieval_time:.3f}s  |  chunks found: {chunks_found}")

        # ── LLM inference timing ──────────────────────────────────────────────
        prompt = (
            "Use the following context to answer the question.\n"
            "If the answer is not in the context, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"
        )

        t_llm_start = time.perf_counter()
        completion  = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7,
        )
        llm_time   = time.perf_counter() - t_llm_start
        answer     = completion.choices[0].message.content

        # ── Token / latency metrics ───────────────────────────────────────────
        usage = getattr(completion, "usage", None)
        prompt_tokens     = getattr(usage, "prompt_tokens",     0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tokens      = getattr(usage, "total_tokens",      0) if usage else 0
        tokens_per_sec    = (completion_tokens / llm_time) if llm_time > 0 else 0

        log.info(
            f"    LLM Inference: {llm_time:.3f}s  |  "
            f"prompt_tokens={prompt_tokens}  completion_tokens={completion_tokens}  "
            f"tokens/sec={tokens_per_sec:.1f}"
        )
        log.info(f"    Answer preview: {answer[:150].replace(chr(10), ' ')}...")

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

    # ── Per-query summary table ───────────────────────────────────────────────
    divider("-")
    log.info(f"  {'Q':<4}  {'Retrieval':>10}  {'LLM':>10}  {'Tokens':>8}  {'Tok/s':>7}")
    divider("-")
    for k, v in query_timings.items():
        log.info(
            f"  {k:<4}  {v['retrieval_sec']:>9.3f}s  "
            f"{v['llm_inference_sec']:>9.3f}s  "
            f"{v['total_tokens']:>8}  "
            f"{v['tokens_per_sec']:>6.1f}"
        )
    divider("-")

    return query_timings, all_qa

# ─────────────────────────────────────────────────────────────────────────────
# SAVE ALL METRICS TO JSON
# ─────────────────────────────────────────────────────────────────────────────
def save_metrics(all_timings, batch_total, sanity_time, init_time, query_timings):
    metrics = {
        "run_id"               : RUN_ID,
        "run_timestamp"        : datetime.now().isoformat(),
        "config": {
            "docs_folder"      : DOCS_FOLDER,
            "file_pattern"     : FILE_PATTERN,
            "collection"       : COLLECTION_NAME,
            "dense_dim"        : DENSE_DIM,
            "caption_model"    : CAPTION_MODEL,
            "llm_model"        : LLM_MODEL,
            "chunk_size"       : CHUNK_SIZE,
            "chunk_overlap"    : CHUNK_OVERLAP,
        },
        "nim_endpoints": {
            "page_elements"    : os.environ.get("YOLOX_HTTP_ENDPOINT",                  "NOT SET"),
            "graphic_elements" : os.environ.get("YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT", "NOT SET"),
            "table_structure"  : os.environ.get("YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT",  "NOT SET"),
            "ocr"              : os.environ.get("OCR_HTTP_ENDPOINT",                    "NOT SET"),
        },
        "phase_times_sec": {
            "pipeline_init"    : round(init_time, 3),
            "sanity_check"     : round(sanity_time, 3),
            "batch_ingest"     : round(batch_total, 3),
        },
        "per_document_timings" : all_timings,
        "rag_query_timings"    : query_timings,
        "summary": {
            "total_docs"       : len(all_timings),
            "successful_docs"  : sum(1 for v in all_timings.values() if "error" not in v and v.get("failures_count", 1) == 0),
            "avg_ingest_sec"   : round(
                sum(v.get("total_ingest_sec", 0) for v in all_timings.values()) / max(len(all_timings), 1), 3
            ),
            "total_wall_time_sec" : round(init_time + sanity_time + batch_total, 3),
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
    global_start = time.perf_counter()
    divider("═")
    log.info("NV-INGEST 25.9.0 — LIBRARY MODE PIPELINE WITH FULL LOGGING")
    log.info(f"Run started at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Results dir    : {RESULTS_DIR}/")
    log.info(f"Log file       : {LOG_FILE}")
    divider("═")

    # 1. Check environment
    api_key = check_environment()

    # 2. Collect files
    files = collect_files()

    # 3. Start pipeline
    client, init_time = start_pipeline()

    # 4. Sanity check on first file
    sanity_time = sanity_check(client, files[0])

    # 5. Full batch ingest
    all_timings, batch_total = run_batch_ingest(client, files, api_key)

    # 6. RAG queries + timing
    query_timings, all_qa = run_rag_queries(api_key)

    # 7. Save answers
    with open(ANSWERS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, indent=2, ensure_ascii=False)
    log.info(f"  Answers saved → {ANSWERS_FILE}")

    # 8. Save metrics
    metrics = save_metrics(all_timings, batch_total, sanity_time, init_time, query_timings)

    # 9. Final summary
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
