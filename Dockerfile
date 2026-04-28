"""
NV-Ingest 25.9.0 - Library Mode Pipeline  (v5)
Hybrid retrieval : sparse BM25 + dense semantic (Milvus)
Reranking        : nvidia/llama-3.2-nv-rerankqa-1b-v2  via NVIDIA API
                   (API chosen over container — avoids VRAM contention on A100
                    where docker-compose pins reranker to CUDA_VISIBLE_DEVICES=0,
                    same device already used by page-elements/graphic-elements/
                    table-structure/OCR NIMs)
Target           : A100 | Python 3.12
"""

import logging, os, sys, time, json, glob, socket, subprocess, traceback, requests
from datetime import datetime
from pathlib import Path

if sys.version_info < (3, 12):
    sys.exit(f"[FATAL] Python 3.12+ required. You have {sys.version.split()[0]}")

try:
    import pymilvus; pymilvus.connections.disconnect("default")
except Exception:
    pass

# ── CONFIG ────────────────────────────────────────────────────────────────────
DOCS_FOLDER     = "Docs"
FILE_PATTERNS   = ["*.pdf", "*.docx", "*.pptx", "*.jpeg", "*.jpg", "*.png"]
MILVUS_URI      = "milvus.db"
COLLECTION_NAME = "multimodal_docs"
SPARSE          = True          # enables BM25 sparse vectors alongside dense
DENSE_DIM       = 2048
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 50
TOKENIZER       = "intfloat/e5-large-unsupervised"
LLM_MODEL       = "meta/llama-3.3-70b-instruct"
LLM_BASE_URL    = "https://integrate.api.nvidia.com/v1"
PIPELINE_WAIT   = 90
RESULTS_DIR     = "Outputs"

# Hybrid retrieval: fetch more candidates then rerank to RERANK_K final
TOP_K_RETRIEVE  = 20            # chunks pulled from Milvus (BM25 + semantic)
RERANK_K        = 5             # chunks kept after cross-encoder reranking

# Reranker via NVIDIA hosted API (same key, no extra GPU, no container conflict)
RERANKER_MODEL  = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
RERANKER_URL    = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3.2-nv-rerankqa-1b-v2/reranking"
RERANKER_TIMEOUT = 15           # seconds per rerank call

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
    "Why did life sciences move toward open access journals and APC models?",
    "What does the report mean by successive waves of open access innovation?",
    "How does the report connect open access, open data, and reproducibility?",
]

# ── LOGGING ───────────────────────────────────────────────────────────────────
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
log  = logging.getLogger("pipeline")
log.setLevel(logging.DEBUG); log.propagate = False
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
    sep(); log.info("ENVIRONMENT SETUP"); sep()
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
    log.info(f"  SPARSE={SPARSE} — BM25 sparse vectors enabled for hybrid retrieval")
    log.info(f"  Retrieve TOP_K={TOP_K_RETRIEVE} → rerank → keep RERANK_K={RERANK_K}")
    log.info(f"  Reranker: {RERANKER_MODEL} via NVIDIA API (no extra GPU needed)")
    flush(); return api_key

# ── NIM HEALTH ────────────────────────────────────────────────────────────────
NIM_CHECKS = {
    "page_elements"    : ("http://localhost:8000/v1/health/ready", 8000),
    "graphic_elements" : ("http://localhost:8003/v1/health/ready", 8003),
    "table_structure"  : ("http://localhost:8006/v1/health/ready", 8006),
    "ocr"              : ("http://localhost:8009/v1/health/ready", 8009),
}

def check_nims():
    sep(); log.info("NIM HEALTH CHECK"); sep()
    status = {}
    for name, (url, port) in NIM_CHECKS.items():
        try:
            r = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "5", url],
                capture_output=True, text=True)
            code = r.stdout.strip(); ok = code in ("200", "201")
            status[name] = ok
            log.info(f"  {'OK  ' if ok else 'FAIL'}  {name:<22} port {port}  HTTP {code}")
            if not ok and code == "000":
                log.warning(f"       Not running on port {port} — start with docker compose")
        except Exception as e:
            status[name] = False
            log.warning(f"  FAIL  {name:<22} port {port}  error: {e}")
    healthy = sum(status.values())
    if healthy == 4:
        log.info("  All 4 NIMs healthy — full multimodal + BM25 hybrid extraction enabled.")
    else:
        log.warning(f"  {healthy}/4 NIMs healthy — running in TEXT-ONLY + hybrid mode.")
        log.warning("  To start all NIMs:")
        log.warning("    cd nv-ingest && docker compose --profile retrieval --profile table-structure up -d")
    flush(); return status

# ── FILE DISCOVERY ────────────────────────────────────────────────────────────
def find_files():
    sep(); log.info(f"FILE DISCOVERY  ->  {DOCS_FOLDER}/")
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
    sep(); log.info("PIPELINE INIT"); sep()
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
    log.info("  Port 7671 ready."); time.sleep(5)
    client = NvIngestClient(message_client_allocator=SimpleClient,
                            message_client_port=7671, message_client_hostname="localhost")
    log.info("  NvIngestClient connected -> localhost:7671")
    flush(); return client, init_time

# ── SANITY CHECK ──────────────────────────────────────────────────────────────
def sanity_check(client, filepath):
    from nv_ingest_client.client import Ingestor
    from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob
    sep(); log.info(f"SANITY CHECK (text-only) -> {Path(filepath).name}"); sep()
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
        log.error("Empty response."); flush(); sys.exit(1)
    log.info(f"  Preview: {ingest_json_results_to_blob(results[0])[:200].replace(chr(10),' ')}...")
    log.info("  SANITY PASSED"); flush(); return st

# ── INGEST ONE FILE ───────────────────────────────────────────────────────────
def ingest_single_file(client, filepath, idx, total, nim_status):
    from nv_ingest_client.client import Ingestor
    fname  = Path(filepath).name
    ext    = Path(filepath).suffix.lower()
    all_ok = all(nim_status.values())
    ocr_ok = nim_status.get("ocr", False)
    mode   = "FULL+HYBRID" if all_ok else "TEXT+HYBRID"

    if ext in IMAGE_ONLY_EXTS and not ocr_ok:
        log.info(f"  [{idx}/{total}]  {fname}  [SKIPPED — image file, OCR NIM not running]")
        flush()
        return {"mode": "skipped", "total_ingest_sec": 0,
                "results_count": 0, "failures_count": 0, "skipped": True}, []

    log.info(f"  [{idx}/{total}]  {fname}  [{mode}]  sparse={SPARSE}")
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
                    .embed()
                    .vdb_upload(collection_name=COLLECTION_NAME, milvus_uri=MILVUS_URI,
                                sparse=SPARSE,          # BM25 sparse index stored here
                                dense_dim=DENSE_DIM))
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
        elapsed = time.perf_counter() - t0
        log.warning(f"    SKIPPED (no embeddable content): {ve}")
        flush()
        return {"mode": mode, "total_ingest_sec": round(elapsed, 3),
                "results_count": 0, "failures_count": 0, "skipped": True}, []
    except Exception as e:
        elapsed = time.perf_counter() - t0
        log.error(f"    EXCEPTION: {e}"); log.error(traceback.format_exc()); flush()
        return {"mode": mode, "total_ingest_sec": round(elapsed, 3),
                "results_count": 0, "failures_count": 1, "error": str(e)}, [e]

# ── BATCH INGEST ──────────────────────────────────────────────────────────────
def run_batch(client, files, nim_status):
    sep()
    mode_label = "FULL MULTIMODAL + HYBRID BM25" if all(nim_status.values()) else "TEXT-ONLY + HYBRID BM25"
    log.info(f"BATCH INGEST  ->  {len(files)} files  |  {mode_label}")
    log.info(f"  Collection: {COLLECTION_NAME}  Milvus: {MILVUS_URI}")
    log.info(f"  Chunk: {CHUNK_SIZE} tok  overlap: {CHUNK_OVERLAP}  sparse={SPARSE}")
    sep(); flush()

    t0 = time.perf_counter(); timings = {}
    ok = failed = skipped = 0

    for i, fp in enumerate(files, 1):
        fname = Path(fp).name; doc_t = time.perf_counter()
        try:
            t, _ = ingest_single_file(client, fp, i, len(files), nim_status)
            timings[fname] = t
            if t.get("skipped"):        skipped += 1
            elif t["failures_count"]==0: ok += 1
            else:                       failed += 1
        except Exception as e:
            log.error(f"  OUTER EXCEPTION {fname}: {e}")
            log.error(traceback.format_exc())
            timings[fname] = {"error": str(e), "total_ingest_sec": 0, "failures_count": 1}
            failed += 1
        log.info(f"  -- wall: {time.perf_counter()-doc_t:.3f}s --"); flush()

    batch_total = time.perf_counter() - t0
    sep(); log.info("BATCH SUMMARY")
    log.info(f"  Total: {len(files)}  OK: {ok}  Skipped: {skipped}  Failed: {failed}")
    log.info(f"  Total: {batch_total:.3f}s  Avg: {batch_total/max(len(files),1):.3f}s")
    sep(); flush()
    return timings, batch_total

# ── MILVUS CHECK ──────────────────────────────────────────────────────────────
def check_milvus():
    sep(); log.info("MILVUS VERIFICATION")
    try:
        from pymilvus import Collection, connections, utility
        connections.connect(uri=MILVUS_URI)
        if not utility.has_collection(COLLECTION_NAME):
            log.error(f"  Collection '{COLLECTION_NAME}' missing."); flush(); return False
        col = Collection(COLLECTION_NAME); col.load()
        n = col.num_entities
        log.info(f"  '{COLLECTION_NAME}': {n} total chunks (dense + sparse BM25 vectors)")
        if n == 0:
            log.error("  ZERO chunks — embed/upload failed."); flush(); return False
        log.info(f"  Hybrid retrieval: TOP_K={TOP_K_RETRIEVE} → rerank → RERANK_K={RERANK_K}")
        flush(); return True
    except Exception as e:
        log.error(f"  Milvus error: {e}"); flush(); return False

# ── RERANKER ─────────────────────────────────────────────────────────────────
# Using NVIDIA hosted API for reranker (not the container from docker-compose).
# Reason: the 25.9.0 docker-compose.yaml pins the reranker container to
# CUDA_VISIBLE_DEVICES=0 — the same device already running OCR, page-elements,
# graphic-elements, and table-structure NIMs. Running the container would cause
# direct VRAM contention on A100.
# The hosted API uses the same NVIDIA_API_KEY, adds ~100-200ms per query
# (negligible versus LLM inference), and avoids any GPU conflict.
# Model: nvidia/llama-3.2-nv-rerankqa-1b-v2 (same model as the container).
def rerank(query, chunks, api_key):
    """
    Cross-encoder reranking via NVIDIA API.
    Returns list of (chunk_text, score) sorted descending, trimmed to RERANK_K.
    Falls back to original order on any API error so RAG always continues.
    """
    if not chunks:
        return []
    passages = [{"text": c} for c in chunks]
    payload  = {"model": RERANKER_MODEL,
                "query": {"text": query},
                "passages": passages}
    headers  = {"Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"}
    try:
        resp = requests.post(RERANKER_URL, headers=headers,
                             json=payload, timeout=RERANKER_TIMEOUT)
        resp.raise_for_status()
        rankings = resp.json().get("rankings", [])
        # rankings = [{"index": int, "logit": float}, ...]
        ranked = sorted(rankings, key=lambda x: x.get("logit", 0), reverse=True)
        result = [(chunks[r["index"]], r.get("logit", 0.0))
                  for r in ranked[:RERANK_K] if r["index"] < len(chunks)]
        return result
    except requests.exceptions.Timeout:
        log.warning(f"    Reranker timeout after {RERANKER_TIMEOUT}s — using original order")
    except requests.exceptions.HTTPError as e:
        log.warning(f"    Reranker HTTP error {e.response.status_code}: {e.response.text[:120]}")
    except Exception as e:
        log.warning(f"    Reranker error: {e}")
    # fallback: original order, top RERANK_K
    return [(c, 0.0) for c in chunks[:RERANK_K]]

# ── RAG QUERIES ───────────────────────────────────────────────────────────────
LLM_SYSTEM_PROMPT = """You are a helpful document assistant.
Answer questions using the provided context from ingested documents.
If the context contains partial information, synthesize what is available and clearly indicate what was found.
If the context genuinely has no relevant information, say so briefly.
Be specific, cite details from context where useful, and keep answers concise."""

def run_rag(api_key, questions):
    from openai import OpenAI
    from nv_ingest_client.util.milvus import nvingest_retrieval

    sep()
    log.info(f"RAG QUERIES  ->  {len(questions)} question(s)")
    log.info(f"  Mode  : hybrid BM25+semantic (sparse={SPARSE})")
    log.info(f"  Fetch : TOP_K_RETRIEVE={TOP_K_RETRIEVE}  ->  rerank  ->  RERANK_K={RERANK_K}")
    log.info(f"  LLM   : {LLM_MODEL}")
    sep(); flush()

    llm     = OpenAI(base_url=LLM_BASE_URL, api_key=api_key)
    all_qa  = []; timings = {}

    for i, q in enumerate(questions, 1):
        log.info(f"  Q{i}: {q}"); flush()

        # ── 1. Hybrid retrieval (BM25 sparse + dense semantic) ────────────────
        t_ret = time.perf_counter()
        docs  = nvingest_retrieval(
            [q], COLLECTION_NAME, milvus_uri=MILVUS_URI,
            hybrid=SPARSE,          # True  = BM25+semantic  |  False = dense-only
            top_k=TOP_K_RETRIEVE)
        ret_t = time.perf_counter() - t_ret
        raw_chunks = [d["entity"]["text"] for d in (docs[0] if docs and docs[0] else [])]
        log.info(f"    Retrieval : {ret_t:.3f}s  |  chunks fetched: {len(raw_chunks)}/{TOP_K_RETRIEVE}"
                 f"  mode: {'hybrid BM25+semantic' if SPARSE else 'semantic-only'}")

        # ── 2. Cross-encoder reranking ────────────────────────────────────────
        t_rr = time.perf_counter()
        ranked = rerank(q, raw_chunks, api_key)
        rr_t   = time.perf_counter() - t_rr
        log.info(f"    Reranking : {rr_t:.3f}s  |  kept {len(ranked)}/{len(raw_chunks)} chunks"
                 f"  top_score={ranked[0][1]:.4f}" if ranked else "    Reranking : no chunks")

        if ranked:
            ctx = "\n\n".join(text for text, _ in ranked)
        elif raw_chunks:
            ctx = "\n\n".join(raw_chunks[:RERANK_K])   # fallback
        else:
            ctx = "No relevant content found in the indexed documents."

        # ── 3. LLM inference ──────────────────────────────────────────────────
        t_llm = time.perf_counter()
        try:
            resp = llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Context:\n{ctx}\n\nQuestion: {q}"}
                ],
                max_tokens=1024, temperature=0.3,
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

        log.info(f"    LLM       : {llm_t:.3f}s  |  tokens={tot_tok}  tok/s={tps:.1f}")
        log.info(f"    Answer    : {answer[:240].replace(chr(10),' ')}")
        flush()

        timings[f"Q{i}"] = {
            "question"          : q,
            "retrieval_sec"     : round(ret_t, 4),
            "rerank_sec"        : round(rr_t,  4),
            "chunks_fetched"    : len(raw_chunks),
            "chunks_after_rerank": len(ranked),
            "rerank_top_score"  : ranked[0][1] if ranked else 0.0,
            "llm_sec"           : round(llm_t,  4),
            "prompt_tokens"     : p_tok,
            "completion_tokens" : c_tok,
            "total_tokens"      : tot_tok,
            "tokens_per_sec"    : round(tps, 2),
        }
        all_qa.append({"question": q, "answer": answer, "metrics": timings[f"Q{i}"]})
        time.sleep(0.5)

    # summary table
    sep("-")
    log.info(f"  {'Q':<4}  {'Retrieve':>9}  {'Rerank':>7}  {'LLM':>7}  {'Tokens':>7}  {'Tok/s':>6}  Score")
    sep("-")
    for k, v in timings.items():
        log.info(f"  {k:<4}  {v['retrieval_sec']:>8.3f}s  {v['rerank_sec']:>6.3f}s"
                 f"  {v['llm_sec']:>6.3f}s  {v['total_tokens']:>7}"
                 f"  {v['tokens_per_sec']:>5.1f}  {v['rerank_top_score']:.4f}")
    sep("-")
    return timings, all_qa

# ── METRICS ───────────────────────────────────────────────────────────────────
def save_metrics(timings, batch_total, sanity_t, init_t, q_timings, nim_status):
    m = {
        "run_id": RUN_ID, "timestamp": datetime.now().isoformat(),
        "python": sys.version, "nim_health": nim_status,
        "retrieval": {
            "mode"    : "hybrid BM25+semantic" if SPARSE else "semantic-only",
            "sparse"  : SPARSE,
            "top_k"   : TOP_K_RETRIEVE,
            "rerank_k": RERANK_K,
            "reranker": RERANKER_MODEL,
        },
        "config": {
            "docs_folder" : DOCS_FOLDER,
            "collection"  : COLLECTION_NAME,
            "dense_dim"   : DENSE_DIM,
            "llm"         : LLM_MODEL,
            "chunk_size"  : CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
        "phase_times": {"init": round(init_t,3), "sanity": round(sanity_t,3),
                        "batch": round(batch_total,3)},
        "per_doc": timings, "rag": q_timings,
        "summary": {
            "total"  : len(timings),
            "ok"     : sum(1 for v in timings.values()
                           if not v.get("skipped") and not v.get("error")
                           and v.get("failures_count", 1) == 0),
            "skipped": sum(1 for v in timings.values() if v.get("skipped")),
            "failed" : sum(1 for v in timings.values()
                          if v.get("error") or v.get("failures_count", 0) > 0),
            "wall"   : round(init_t + sanity_t + batch_total, 3),
        }
    }
    with open(METRICS_FILE, "w", encoding="utf-8") as f: json.dump(m, f, indent=2)
    log.info(f"  Metrics -> {METRICS_FILE}"); flush(); return m

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    interactive = "--interactive" in sys.argv
    g0 = time.perf_counter()
    sep()
    log.info("NV-INGEST 25.9.0  LIBRARY MODE  v5  |  hybrid BM25+semantic + reranker")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Python {sys.version.split()[0]}")
    log.info(f"  {'INTERACTIVE' if interactive else 'PREDEFINED'} question mode")
    log.info(f"  Log -> {os.path.abspath(LOG_FILE)}")
    sep(); flush()

    questions         = collect_questions() if interactive else DEFAULT_QUESTIONS
    if not interactive:
        log.info(f"  {len(questions)} predefined question(s)."); flush()

    api_key           = setup_env()
    files             = find_files()
    client, init_time = start_pipeline()
    nim_status        = check_nims()
    sanity_t          = sanity_check(client, files[0])
    timings, bt       = run_batch(client, files, nim_status)

    if not check_milvus():
        log.error("Milvus empty — skipping RAG.")
        q_timings, all_qa = {}, []
    else:
        q_timings, all_qa = run_rag(api_key, questions)

    with open(ANSWERS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, indent=2, ensure_ascii=False)
    log.info(f"  Answers -> {ANSWERS_FILE}"); flush()

    m = save_metrics(timings, bt, sanity_t, init_time, q_timings, nim_status)
    sep(); log.info("FINAL SUMMARY"); sep()
    log.info(f"  Run ID   : {RUN_ID}")
    log.info(f"  Docs     : {m['summary']['total']}  "
             f"OK={m['summary']['ok']}  Skipped={m['summary']['skipped']}  "
             f"Failed={m['summary']['failed']}")
    log.info(f"  Wall     : {time.perf_counter()-g0:.1f}s  "
             f"(init {init_time:.1f}s + sanity {sanity_t:.1f}s + batch {bt:.1f}s)")
    log.info(f"  Retrieval: hybrid BM25+semantic  fetch={TOP_K_RETRIEVE}  "
             f"rerank_to={RERANK_K}  model={RERANKER_MODEL}")
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
