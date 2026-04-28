"""
NV-Ingest 25.9.0 - Library Mode Pipeline  (v4)
Target : A100 | Python 3.12
"""

import logging, os, sys, time, json, glob, socket, subprocess, traceback
from datetime import datetime
from pathlib import Path

# Python guard
if sys.version_info < (3, 12):
    sys.exit(f"[FATAL] Python 3.12+ required. You have {sys.version.split()[0]}")

# Clear any stale pymilvus connection
try:
    import pymilvus; pymilvus.connections.disconnect("default")
except Exception:
    pass

# ── CONFIG ────────────────────────────────────────────────────────────────────
DOCS_FOLDER     = "Docs"
FILE_PATTERNS   = ["*.pdf", "*.docx", "*.pptx", "*.jpeg", "*.jpg", "*.png"]
MILVUS_URI      = "milvus.db"
COLLECTION_NAME = "multimodal_docs"
SPARSE          = False
DENSE_DIM       = 2048
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 50
TOKENIZER       = "intfloat/e5-large-unsupervised"
LLM_MODEL       = "meta/llama-3.3-70b-instruct"
LLM_BASE_URL    = "https://integrate.api.nvidia.com/v1"
PIPELINE_WAIT   = 90          # seconds to wait for port 7671
RESULTS_DIR     = "Outputs"
TOP_K           = 10          # chunks returned per RAG query, NOT per-file count

# Image-only formats that produce 0 embeddable text without OCR NIM
IMAGE_ONLY_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

NIM_ENV_VARS = {
    "YOLOX_HTTP_ENDPOINT"                   : "http://localhost:8000/v1/infer",
    "YOLOX_INFER_PROTOCOL"                  : "http",
    "YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT"  : "http://localhost:8003/v1/infer",
    "YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL" : "http",
    "YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT"   : "http://localhost:8006/v1/infer",
    "YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL"  : "http",
    "OCR_HTTP_ENDPOINT"                     : "http://localhost:8009/v1/infer",
    "OCR_INFER_PROTOCOL"                    : "http",
}

DEFAULT_QUESTIONS = [
    "Why did economics and physics become early movers in open access adoption?",
    "How did arXiv influence scholarly communication in physics?",
    "Why did life sciences move more toward open access journals and APC models?",
    "What does the report mean by successive waves of open access innovation?",
    "How does the report connect open access, open data, and reproducibility?",
]

# ── LOGGING — isolated from nv_ingest root logger hijack ─────────────────────
# nv_ingest calls configure_logging() at import time which overwrites the root
# logger handlers. Using propagate=False on our named logger prevents that.
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
RUN_ID       = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE     = os.path.join(RESULTS_DIR, f"pipeline_run_{RUN_ID}.log")
METRICS_FILE = os.path.join(RESULTS_DIR, f"metrics_{RUN_ID}.json")
ANSWERS_FILE = os.path.join(RESULTS_DIR, f"answers_{RUN_ID}.json")

_fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%Y-%m-%d %H:%M:%S")
_fh  = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8", delay=False)
_fh.setLevel(logging.DEBUG); _fh.setFormatter(_fmt)
_ch  = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO);  _ch.setFormatter(_fmt)

log = logging.getLogger("pipeline")
log.setLevel(logging.DEBUG)
log.propagate = False   # key: blocks nv_ingest from wiping our FileHandler
log.addHandler(_fh); log.addHandler(_ch)

def flush():
    for h in log.handlers: h.flush()

def sep(c="=", w=72): log.info(c * w); flush()

def tick(name): log.info(f">> {name}"); flush(); return time.perf_counter()

def tock(name, t0):
    e = time.perf_counter() - t0
    log.info(f"<< {name}  {e:.3f}s"); flush(); return e

# ── QUESTIONS ─────────────────────────────────────────────────────────────────
def collect_questions():
    print("\n" + "=" * 72)
    print("  Enter questions. Blank line or 'done' to finish.")
    print("=" * 72 + "\n")
    qs, idx = [], 1
    while True:
        try: q = input(f"  Q{idx}: ").strip()
        except EOFError: break
        if q.lower() in ("", "done"):
            if not qs: print("  [!] Need at least one question."); continue
            break
        qs.append(q); idx += 1
    log.info(f"Collected {len(qs)} question(s)."); flush()
    return qs

# ── ENVIRONMENT ───────────────────────────────────────────────────────────────
def setup_env():
    sep()
    log.info("ENVIRONMENT SETUP")
    sep()
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        log.error("NVIDIA_API_KEY not set.  export NVIDIA_API_KEY='nvapi-...'")
        flush(); sys.exit(1)
    log.info(f"  NVIDIA_API_KEY : ********{api_key[-6:]}")
    for var, default in NIM_ENV_VARS.items():
        if not os.environ.get(var):
            os.environ[var] = default
            log.info(f"  {var:<47} -> {default}")
        else:
            log.info(f"  {var:<47} (already set)")
    log.info("  All NIM vars injected into os.environ before run_pipeline().")
    flush(); return api_key

# ── NIM HEALTH ────────────────────────────────────────────────────────────────
NIM_CHECKS = {
    "page_elements"    : ("http://localhost:8000/v1/health/ready", 8000),
    "graphic_elements" : ("http://localhost:8003/v1/health/ready", 8003),
    "table_structure"  : ("http://localhost:8006/v1/health/ready", 8006),
    "ocr"              : ("http://localhost:8009/v1/health/ready", 8009),
}

def check_nims():
    sep()
    log.info("NIM HEALTH CHECK")
    sep()
    status = {}
    for name, (url, port) in NIM_CHECKS.items():
        try:
            r = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "5", url],
                capture_output=True, text=True)
            code = r.stdout.strip()
            ok = code in ("200", "201")
            status[name] = ok
            tag = "OK  " if ok else "FAIL"
            log.info(f"  {tag}  {name:<22} port {port}  HTTP {code}")
            if not ok:
                if code == "000": log.warning(f"       Container not running on port {port}")
                elif code == "404": log.warning(f"       Container up but health path returned 404")
        except Exception as e:
            status[name] = False
            log.warning(f"  FAIL {name:<22} port {port}  error: {e}")

    healthy = sum(status.values())
    if healthy == 4:
        log.info("  All 4 NIMs healthy — full multimodal extraction enabled.")
    else:
        log.warning(f"  {healthy}/4 NIMs healthy — running in TEXT-ONLY mode.")
        log.warning("  Images/tables/charts will be skipped (no OCR/YOLOX NIMs).")
        log.warning("  To enable full multimodal:")
        log.warning("    cd nv-ingest && docker compose --profile retrieval --profile table-structure up -d")
        log.warning("    Wait 10-15 min for model loading, then re-run.")
    flush(); return status

# ── FILE DISCOVERY ────────────────────────────────────────────────────────────
def find_files():
    sep()
    log.info(f"FILE DISCOVERY  ->  {DOCS_FOLDER}/")
    files = []
    for pat in FILE_PATTERNS:
        found = sorted(glob.glob(os.path.join(DOCS_FOLDER, pat)))
        if found: log.info(f"  {pat:<12} -> {len(found)}")
        files.extend(found)
    files = sorted(set(files))
    log.info(f"  Total: {len(files)} file(s)")
    for f in files:
        log.info(f"    {Path(f).name:<52} {os.path.getsize(f)/1048576:.2f} MB")
    if not files:
        log.error(f"No files found in {DOCS_FOLDER}/ matching {FILE_PATTERNS}")
        flush(); sys.exit(1)
    flush(); return files

# ── PIPELINE START ────────────────────────────────────────────────────────────
def start_pipeline():
    sep()
    log.info("PIPELINE INIT")
    sep()
    from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import (
        run_pipeline, PipelineCreationSchema)
    from nv_ingest_client.client import NvIngestClient
    from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient

    t0 = tick("Pipeline subprocess")
    run_pipeline(PipelineCreationSchema(), block=False,
                 disable_dynamic_scaling=True, run_in_subprocess=True)
    init_time = tock("Pipeline subprocess", t0)

    log.info(f"  Polling port 7671 (max {PIPELINE_WAIT}s)...")
    deadline = time.time() + PIPELINE_WAIT
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", 7671), timeout=2): break
        except (ConnectionRefusedError, OSError): time.sleep(2)
    else:
        log.error("Port 7671 never opened — pipeline subprocess crashed.")
        flush(); sys.exit(1)
    log.info("  Port 7671 ready.")
    time.sleep(5)

    client = NvIngestClient(message_client_allocator=SimpleClient,
                            message_client_port=7671, message_client_hostname="localhost")
    log.info("  NvIngestClient connected -> localhost:7671")
    flush(); return client, init_time

# ── SANITY CHECK ──────────────────────────────────────────────────────────────
def sanity_check(client, filepath):
    from nv_ingest_client.client import Ingestor
    from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob
    sep()
    log.info(f"SANITY CHECK (text-only) -> {Path(filepath).name}")
    sep()
    t0 = tick("Sanity")
    ingestor = (Ingestor(client=client).files(filepath)
                .extract(extract_text=True, extract_tables=False,
                         extract_charts=False, extract_images=False,
                         extract_infographics=False, text_depth="page"))
    results, failures = ingestor.ingest(show_progress=True, return_failures=True)
    st = tock("Sanity", t0)
    log.info(f"  Results: {len(results)}  Failures: {len(failures)}")
    if failures:
        for i, f in enumerate(failures): log.error(f"  FAIL[{i}]: {f}")
        log.error("Sanity check failed."); flush(); sys.exit(1)
    if not results:
        log.error("Empty response from pipeline."); flush(); sys.exit(1)
    log.info(f"  Preview: {ingest_json_results_to_blob(results[0])[:200].replace(chr(10),' ')}...")
    log.info("  SANITY PASSED"); flush()
    return st

# ── INGEST ONE FILE ───────────────────────────────────────────────────────────
# FIX: image-only files (jpg/jpeg/png) produce zero embeddable text in TEXT-ONLY
# mode because there is no OCR NIM. The vdb_upload() call then raises:
#   ValueError: No records with Embeddings to insert detected.
# We detect this upfront and skip image-only files when NIMs are not healthy,
# counting them as "skipped" instead of "failed" so they don't break the batch.
def ingest_single_file(client, filepath, idx, total, nim_status):
    from nv_ingest_client.client import Ingestor

    fname   = Path(filepath).name
    ext     = Path(filepath).suffix.lower()
    all_ok  = all(nim_status.values())
    ocr_ok  = nim_status.get("ocr", False)
    mode    = "FULL" if all_ok else "TEXT-ONLY"

    # Skip pure image files when OCR NIM is not running — they will produce
    # zero text, then vdb_upload raises ValueError with no embeddings.
    if ext in IMAGE_ONLY_EXTS and not ocr_ok:
        log.info(f"  [{idx}/{total}]  {fname}  [SKIPPED — image file, OCR NIM not running]")
        log.warning(f"    Start OCR NIM to index this image: docker compose --profile ocr up -d")
        flush()
        return {"mode": "skipped", "total_ingest_sec": 0,
                "results_count": 0, "failures_count": 0, "skipped": True}, []

    log.info(f"  [{idx}/{total}]  {fname}  [{mode}]")
    extract_kw = dict(extract_text=True, text_depth="page",
                      extract_tables=all_ok, extract_charts=all_ok,
                      extract_images=all_ok, extract_infographics=all_ok,
                      **({"table_output_format": "markdown"} if all_ok else {}))
    t0 = time.perf_counter()
    try:
        ingestor = (Ingestor(client=client).files(filepath)
                    .extract(**extract_kw)
                    .split(tokenizer=TOKENIZER, chunk_size=CHUNK_SIZE,
                           chunk_overlap=CHUNK_OVERLAP)
                    # .caption() disabled — calls cloud API per image → rate limits
                    # Re-enable after confirming full NIM pipeline works:
                    # .caption(endpoint_url="...", model_name="...", api_key=api_key)
                    .embed()
                    .vdb_upload(collection_name=COLLECTION_NAME, milvus_uri=MILVUS_URI,
                                sparse=SPARSE, dense_dim=DENSE_DIM))
        results, failures = ingestor.ingest(show_progress=False, return_failures=True)
        elapsed = time.perf_counter() - t0
        if failures:
            for i, f in enumerate(failures):
                msg = str(f)
                for s in range(0, len(msg), 180): log.warning(f"    FAIL[{i}]: {msg[s:s+180]}")
        log.info(f"    results={len(results)}  failures={len(failures)}  {elapsed:.3f}s")
        flush()
        return {"mode": mode, "total_ingest_sec": round(elapsed, 3),
                "results_count": len(results), "failures_count": len(failures)}, failures
    except ValueError as ve:
        # Catches "No records with Embeddings to insert detected."
        # This happens for files that extract 0 text (e.g. image-only PDFs
        # with no embedded text layer). Treat as skipped, not crashed.
        elapsed = time.perf_counter() - t0
        log.warning(f"    SKIPPED (no embeddable content extracted): {ve}")
        flush()
        return {"mode": mode, "total_ingest_sec": round(elapsed, 3),
                "results_count": 0, "failures_count": 0, "skipped": True}, []
    except Exception as e:
        elapsed = time.perf_counter() - t0
        log.error(f"    EXCEPTION: {e}")
        log.error(traceback.format_exc()); flush()
        return {"mode": mode, "total_ingest_sec": round(elapsed, 3),
                "results_count": 0, "failures_count": 1, "error": str(e)}, [e]

# ── BATCH INGEST ──────────────────────────────────────────────────────────────
def run_batch(client, files, nim_status):
    sep()
    mode_label = "FULL MULTIMODAL" if all(nim_status.values()) else "TEXT-ONLY"
    log.info(f"BATCH INGEST  ->  {len(files)} files  |  {mode_label}")
    log.info(f"  Collection: {COLLECTION_NAME}  |  Milvus: {MILVUS_URI}")
    log.info(f"  Chunk: {CHUNK_SIZE} tok  overlap: {CHUNK_OVERLAP}  embed-dim: {DENSE_DIM}")
    log.info(f"  NOTE: TOP_K={TOP_K} is the RAG retrieval limit per query, "
             f"NOT the per-file chunk count.")
    sep()
    flush()

    t0 = time.perf_counter(); timings = {}
    ok = failed = skipped = 0

    for i, fp in enumerate(files, 1):
        fname = Path(fp).name
        doc_t = time.perf_counter()
        try:
            t, _ = ingest_single_file(client, fp, i, len(files), nim_status)
            timings[fname] = t
            if t.get("skipped"):
                skipped += 1
            elif t["failures_count"] == 0:
                ok += 1
            else:
                failed += 1
        except Exception as e:
            log.error(f"  OUTER EXCEPTION {fname}: {e}")
            log.error(traceback.format_exc())
            timings[fname] = {"error": str(e), "total_ingest_sec": 0, "failures_count": 1}
            failed += 1
        log.info(f"  -- wall: {time.perf_counter()-doc_t:.3f}s --"); flush()

    batch_total = time.perf_counter() - t0
    sep()
    log.info("BATCH SUMMARY")
    log.info(f"  Total: {len(files)}  OK: {ok}  Skipped: {skipped}  Failed: {failed}")
    log.info(f"  Total time: {batch_total:.3f}s  Avg: {batch_total/max(len(files),1):.3f}s")
    if skipped:
        log.info(f"  Skipped files: image-only files without OCR NIM, or files with no text layer.")
        log.info(f"  To index them: start the OCR NIM container and re-run.")
    if ok == 0 and skipped < len(files):
        log.error("  All non-skipped files FAILED. Check NIM health above.")
    sep(); flush()
    return timings, batch_total

# ── MILVUS CHECK ──────────────────────────────────────────────────────────────
def check_milvus():
    sep(); log.info("MILVUS VERIFICATION")
    try:
        from pymilvus import Collection, connections, utility
        connections.connect(uri=MILVUS_URI)
        if not utility.has_collection(COLLECTION_NAME):
            log.error(f"  Collection '{COLLECTION_NAME}' missing — vdb_upload failed.")
            flush(); return False
        col = Collection(COLLECTION_NAME); col.load()
        n = col.num_entities
        log.info(f"  '{COLLECTION_NAME}': {n} total chunks stored")
        if n == 0:
            log.error("  ZERO chunks — embed/upload failed. Check NIM health and env vars.")
            flush(); return False
        log.info(f"  RAG will retrieve top {TOP_K} most relevant chunks per query.")
        flush(); return True
    except Exception as e:
        log.error(f"  Milvus error: {e}"); flush(); return False

# ── RAG QUERIES ───────────────────────────────────────────────────────────────
# LLM guardrail: the system prompt instructs the LLM to answer ONLY from
# retrieved context and respond with a clear "not found" message if the
# answer is absent — preventing hallucination and cross-document confusion.
LLM_SYSTEM_PROMPT = """You are a document question-answering assistant.
You have been given context chunks retrieved from a set of ingested documents.

Rules you must follow:
1. Answer ONLY using information that is explicitly present in the provided context.
2. Do NOT use any prior knowledge, assumptions, or information outside the context.
3. If the context does not contain the answer, respond with exactly:
   "This information is not found in the ingested documents."
4. Do not guess, infer, or extrapolate beyond what the context states.
5. If the question refers to a document or topic not represented in the context,
   state: "This topic is not covered in the documents that were indexed."
6. Keep your answer concise and cite specific details from the context where possible."""

def run_rag(api_key, questions):
    from openai import OpenAI
    from nv_ingest_client.util.milvus import nvingest_retrieval

    sep()
    log.info(f"RAG QUERIES  ->  {len(questions)} question(s)")
    log.info(f"  TOP_K: {TOP_K}  |  LLM: {LLM_MODEL}")
    sep(); flush()

    llm = OpenAI(base_url=LLM_BASE_URL, api_key=api_key)
    all_qa = []; timings = {}

    for i, q in enumerate(questions, 1):
        log.info(f"  Q{i}: {q}"); flush()

        t_ret = time.perf_counter()
        docs = nvingest_retrieval([q], COLLECTION_NAME, milvus_uri=MILVUS_URI,
                                  hybrid=SPARSE, top_k=TOP_K)
        ret_t = time.perf_counter() - t_ret

        if docs and docs[0]:
            ctx = "\n\n".join(d["entity"]["text"] for d in docs[0])
            n_chunks = len(docs[0])
        else:
            ctx = "No relevant content found in the indexed documents."
            n_chunks = 0

        log.info(f"    Retrieval: {ret_t:.3f}s  |  chunks: {n_chunks}/{TOP_K}")

        t_llm = time.perf_counter()
        try:
            resp = llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Context:\n{ctx}\n\nQuestion: {q}"}
                ],
                max_tokens=1024, temperature=0.1,   # low temp for factual Q&A
            )
            llm_t   = time.perf_counter() - t_llm
            answer  = resp.choices[0].message.content
            usage   = resp.usage
            p_tok   = getattr(usage, "prompt_tokens",     0)
            c_tok   = getattr(usage, "completion_tokens", 0)
            tot_tok = getattr(usage, "total_tokens",      0)
            tps     = c_tok / llm_t if llm_t > 0 else 0
        except Exception as e:
            log.error(f"    LLM error Q{i}: {e}")
            llm_t = tps = p_tok = c_tok = tot_tok = 0
            answer = "LLM call failed."

        log.info(f"    LLM: {llm_t:.3f}s  |  tokens={tot_tok}  tok/s={tps:.1f}")
        log.info(f"    Answer: {answer[:200].replace(chr(10),' ')}")
        flush()

        timings[f"Q{i}"] = {"question": q, "retrieval_sec": round(ret_t,4),
                             "chunks_returned": n_chunks, "top_k": TOP_K,
                             "llm_sec": round(llm_t,4), "prompt_tokens": p_tok,
                             "completion_tokens": c_tok, "total_tokens": tot_tok,
                             "tokens_per_sec": round(tps,2)}
        all_qa.append({"question": q, "answer": answer, "metrics": timings[f"Q{i}"]})
        time.sleep(1)

    return timings, all_qa

# ── METRICS ───────────────────────────────────────────────────────────────────
def save_metrics(timings, batch_total, sanity_t, init_t, q_timings, nim_status):
    m = {
        "run_id": RUN_ID, "timestamp": datetime.now().isoformat(),
        "python": sys.version, "nim_health": nim_status,
        "config": {"docs_folder": DOCS_FOLDER, "collection": COLLECTION_NAME,
                   "dense_dim": DENSE_DIM, "llm": LLM_MODEL,
                   "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
                   "top_k": TOP_K},
        "phase_times": {"init": round(init_t,3), "sanity": round(sanity_t,3),
                        "batch": round(batch_total,3)},
        "per_doc": timings, "rag": q_timings,
        "summary": {
            "total": len(timings),
            "ok":    sum(1 for v in timings.values()
                         if not v.get("skipped") and not v.get("error")
                         and v.get("failures_count", 1) == 0),
            "skipped": sum(1 for v in timings.values() if v.get("skipped")),
            "failed":  sum(1 for v in timings.values()
                          if v.get("error") or v.get("failures_count", 0) > 0),
            "wall": round(init_t + sanity_t + batch_total, 3),
        }
    }
    with open(METRICS_FILE, "w", encoding="utf-8") as f: json.dump(m, f, indent=2)
    log.info(f"  Metrics -> {METRICS_FILE}"); flush()
    return m

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    interactive = "--interactive" in sys.argv
    g0 = time.perf_counter()
    sep()
    log.info("NV-INGEST 25.9.0  LIBRARY MODE  v4")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
             f"Python {sys.version.split()[0]}  |  "
             f"{'INTERACTIVE' if interactive else 'PREDEFINED'} mode")
    log.info(f"  Log  -> {os.path.abspath(LOG_FILE)}")
    sep(); flush()

    questions = collect_questions() if interactive else DEFAULT_QUESTIONS
    if not interactive:
        log.info(f"  {len(questions)} predefined question(s)."); flush()

    api_key    = setup_env()
    files      = find_files()
    client, it = start_pipeline()
    nim_status = check_nims()
    sanity_t   = sanity_check(client, files[0])
    timings, bt = run_batch(client, files, nim_status)

    if not check_milvus():
        log.error("Milvus empty — skipping RAG."); q_timings, all_qa = {}, []
    else:
        q_timings, all_qa = run_rag(api_key, questions)

    with open(ANSWERS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, indent=2, ensure_ascii=False)
    log.info(f"  Answers -> {ANSWERS_FILE}"); flush()

    m = save_metrics(timings, bt, sanity_t, it, q_timings, nim_status)
    sep()
    log.info("FINAL SUMMARY")
    sep()
    log.info(f"  Run ID   : {RUN_ID}")
    log.info(f"  Docs     : {m['summary']['total']}  "
             f"OK={m['summary']['ok']}  "
             f"Skipped={m['summary']['skipped']}  "
             f"Failed={m['summary']['failed']}")
    log.info(f"  Wall     : {time.perf_counter()-g0:.1f}s total  "
             f"(init {it:.1f}s + sanity {sanity_t:.1f}s + batch {bt:.1f}s)")
    log.info(f"  Log      -> {LOG_FILE}")
    log.info(f"  Metrics  -> {METRICS_FILE}")
    log.info(f"  Answers  -> {ANSWERS_FILE}")
    sep(); flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrupted."); flush(); sys.exit(1)
    except Exception as e:
        log.error(f"FATAL: {e}"); log.error(traceback.format_exc()); flush(); sys.exit(1)
