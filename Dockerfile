Updated [nv_ingest_pipeline_v3.py](</Users/santhoshkumar/Documents/New project/nv_ingest_pipeline_v3.py:1>). It is `481` lines now, compiles cleanly, uses a fresh Milvus collection per run, fixes reranking, and removes the separate LLM prompt layer. The LLM call now sends only one user message containing raw `Context` and `Question`.

```python
import glob, json, logging, os, re, socket, subprocess, sys, time, traceback
from datetime import datetime
from pathlib import Path
import requests

if sys.version_info < (3, 12): sys.exit(f"[FATAL] Python 3.12+ required. You have {sys.version.split()[0]}")
try:
    import pymilvus; pymilvus.connections.disconnect("default")
except Exception:
    pass

DOCS_FOLDER = "Docs"
FILE_PATTERNS = ["*.pdf", "*.PDF", "*.docx", "*.DOCX", "*.pptx", "*.PPTX", "*.jpeg", "*.JPEG", "*.jpg", "*.JPG", "*.png", "*.PNG"]
MILVUS_URI = "milvus.db"
COLLECTION_PREFIX = "multimodal_docs"
SPARSE = True
DENSE_DIM = 2048
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOKENIZER = "intfloat/e5-large-unsupervised"
LLM_MODEL = "meta/llama-3.3-70b-instruct"
LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
PIPELINE_WAIT = 90
RESULTS_DIR = "Outputs"
TOP_K_RETRIEVE = 30
RERANK_K = 8
MAX_CONTEXT_CHARS = 18000
MAX_CHUNK_CHARS = 2200
RERANKER_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
RERANKER_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-1b-v2/reranking"
RERANKER_TIMEOUT = 20
IMAGE_ONLY_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
DROP_COLLECTION_IF_EXISTS = os.environ.get("DROP_COLLECTION_IF_EXISTS", "0") == "1"
NIM_ENV_VARS = {
    "YOLOX_HTTP_ENDPOINT": "http://localhost:8000/v1/infer", "YOLOX_INFER_PROTOCOL": "http",
    "YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT": "http://localhost:8003/v1/infer", "YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL": "http",
    "YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT": "http://localhost:8006/v1/infer", "YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL": "http",
    "OCR_HTTP_ENDPOINT": "http://localhost:8009/v1/infer", "OCR_INFER_PROTOCOL": "http",
}
NIM_CHECKS = {
    "page_elements": ("http://localhost:8000/v1/health/ready", 8000, "yolox"),
    "graphic_elements": ("http://localhost:8003/v1/health/ready", 8003, "yolox-graphic-elements"),
    "table_structure": ("http://localhost:8006/v1/health/ready", 8006, "yolox-table-structure"),
    "ocr": ("http://localhost:8009/v1/health/ready", 8009, "ocr"),
}
DEFAULT_QUESTIONS = [
    "Why did economics and physics become early movers in open access adoption?",
    "How did arXiv influence scholarly communication in physics?",
    "Why did life sciences move toward open access journals and APC models?",
    "What does the report mean by successive waves of open access innovation?",
    "How does the report connect open access, open data, and reproducibility?",
]
NIM_FAILURE_MARKERS = ("extract_primitives_from_pdf", "yolox", "ocr", "table_structure", "graphic_elements", "page_elements", "connection refused", "http 404", "http 000")
STOPWORDS = {"a","an","and","are","as","at","be","by","for","from","how","in","is","it","of","on","or","that","the","this","to","was","what","when","where","which","who","why","with"}

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
COLLECTION_NAME = os.environ.get("NV_INGEST_COLLECTION_NAME") or f"{COLLECTION_PREFIX}_{RUN_ID}"
LOG_FILE = os.path.join(RESULTS_DIR, f"pipeline_run_{RUN_ID}.log")
METRICS_FILE = os.path.join(RESULTS_DIR, f"metrics_{RUN_ID}.json")
ANSWERS_FILE = os.path.join(RESULTS_DIR, f"answers_{RUN_ID}.json")
_fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%Y-%m-%d %H:%M:%S")
_fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8", delay=False); _fh.setLevel(logging.DEBUG); _fh.setFormatter(_fmt)
_ch = logging.StreamHandler(sys.stdout); _ch.setLevel(logging.INFO); _ch.setFormatter(_fmt)
log = logging.getLogger("pipeline"); log.setLevel(logging.DEBUG); log.propagate = False; log.handlers.clear(); log.addHandler(_fh); log.addHandler(_ch)

def flush():
    for h in log.handlers:
        try: h.flush()
        except Exception: pass

def sep(c="=", w=72): log.info(c * w); flush()
def timer_start(name): log.info(f">> {name}"); flush(); return time.perf_counter()
def timer_end(name, start): elapsed = time.perf_counter() - start; log.info(f"<< {name}  {elapsed:.3f}s"); flush(); return elapsed
def collapse_ws(text): return re.sub(r"\s+", " ", str(text or "")).strip()
def tok(text): return [t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if len(t) > 1 and t not in STOPWORDS]
def top_sources(items):
    seen, ordered = set(), []
    for item in items:
        src = item["source"]
        if src not in seen: seen.add(src); ordered.append(src)
    return ordered

def lexical_score(query, text):
    q, tl = tok(query), str(text).lower()
    if not q: return 0.0
    overlap = len(set(q) & set(tok(text))) / max(len(set(q)), 1)
    phrase = 1.0 if str(query).lower().strip() in tl else 0.0
    rare = sum(0.03 for term in set(q) if len(term) >= 6 and term in tl)
    long_parts = sum(0.2 for part in re.split(r"[,;:?.!]", str(query).lower()) if len(part.strip()) > 8 and part.strip() in tl)
    return overlap + phrase + rare + long_parts

def source_from_hit(hit):
    items = []
    if isinstance(hit, dict):
        items.append(hit)
        entity = hit.get("entity"); payload = hit.get("payload")
        if isinstance(entity, dict):
            items.append(entity)
            meta = entity.get("metadata")
            if isinstance(meta, dict): items.append(meta)
        if isinstance(payload, dict): items.append(payload)
    for obj in items:
        for key in ("source_name", "source", "filename", "file_name", "document_name", "doc_name", "filepath", "path", "source_id", "uri"):
            if obj.get(key): return Path(str(obj[key])).name
    return "unknown_source"

def text_from_hit(hit):
    if not isinstance(hit, dict): return ""
    entity = hit.get("entity", {})
    if isinstance(entity, dict):
        for key in ("text", "content", "chunk", "body"):
            if entity.get(key): return str(entity[key])
    for key in ("text", "content", "chunk", "body"):
        if hit.get(key): return str(hit[key])
    return ""

def extract_candidates(retrieved_docs):
    hits = retrieved_docs[0] if retrieved_docs and retrieved_docs[0] else []
    seen, out = set(), []
    for rank, hit in enumerate(hits, start=1):
        text = collapse_ws(text_from_hit(hit))
        if not text: continue
        key = text.lower()[:800]
        if key in seen: continue
        seen.add(key)
        score = None
        if isinstance(hit, dict):
            for field in ("score", "distance", "similarity"):
                val = hit.get(field)
                if isinstance(val, (int, float)): score = float(val); break
        out.append({"text": text, "source": source_from_hit(hit), "retrieval_rank": rank, "retrieval_score": score})
    return out

def collect_questions():
    print("\n" + "=" * 72); print("  Enter questions. Blank line or 'done' to finish."); print("=" * 72 + "\n")
    qs, idx = [], 1
    while True:
        try: q = input(f"  Q{idx}: ").strip()
        except EOFError: break
        if q.lower() in ("", "done"):
            if not qs: print("  [!] Need at least one question."); continue
            break
        qs.append(q); idx += 1
    if not qs:
        log.warning("No interactive questions supplied. Using default questions."); flush(); return DEFAULT_QUESTIONS
    log.info(f"Collected {len(qs)} question(s)."); flush(); return qs

def setup_env():
    sep(); log.info("ENVIRONMENT SETUP"); sep()
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        log.error("NVIDIA_API_KEY not set. export NVIDIA_API_KEY='nvapi-...'"); flush(); sys.exit(1)
    log.info(f"  NVIDIA_API_KEY : ********{api_key[-6:]}")
    for var, default in NIM_ENV_VARS.items():
        if not os.environ.get(var): os.environ[var] = default; log.info(f"  {var:<47} -> {default}")
        else: log.info(f"  {var:<47} (already set)")
    log.info(f"  Collection     : {COLLECTION_NAME}")
    log.info(f"  Retrieval      : sparse={SPARSE}  top_k={TOP_K_RETRIEVE}  rerank_k={RERANK_K}")
    log.info(f"  Reranker       : {RERANKER_MODEL}")
    flush(); return api_key

def check_nims():
    sep(); log.info("NIM HEALTH CHECK"); sep(); status = {}
    for name, (url, port, profile) in NIM_CHECKS.items():
        try:
            r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "5", url], capture_output=True, text=True)
            code = r.stdout.strip() or "000"; ok = code in ("200", "201"); status[name] = ok
            log.info(f"  {'OK  ' if ok else 'FAIL'}  {name:<22} port {port}  HTTP {code}")
            if not ok and code == "000": log.warning(f"       Fix: docker compose --profile {profile} up -d")
            elif not ok and code == "404": log.warning(f"       Ready path returned 404 on port {port}. Try /v1/health or /health/ready")
        except Exception as exc:
            status[name] = False; log.warning(f"  FAIL  {name:<22} port {port}  error: {exc}")
    healthy = sum(1 for x in status.values() if x)
    log.info("  All 4 NIMs healthy. Full multimodal extraction enabled." if healthy == 4 else f"  {healthy}/4 NIMs healthy. Text-only fallback will be used where needed.")
    flush(); return status

def find_files():
    sep(); log.info(f"FILE DISCOVERY  ->  {DOCS_FOLDER}/"); files = []
    for pattern in FILE_PATTERNS:
        found = sorted(glob.glob(os.path.join(DOCS_FOLDER, pattern)))
        if found: log.info(f"  {pattern:<12} -> {len(found)}")
        files.extend(found)
    files = sorted(set(files)); log.info(f"  Total: {len(files)} file(s)")
    for filepath in files: log.info(f"    {Path(filepath).name:<52} {os.path.getsize(filepath) / 1048576:.2f} MB")
    if not files:
        log.error(f"No files found in {DOCS_FOLDER}/ matching {FILE_PATTERNS}"); flush(); sys.exit(1)
    flush(); return files

def choose_sanity_file(files, nim_status):
    if nim_status.get("ocr", False): return files[0] if files else None
    for filepath in files:
        if Path(filepath).suffix.lower() not in IMAGE_ONLY_EXTS: return filepath
    return None

def start_pipeline():
    sep(); log.info("PIPELINE INIT"); sep()
    from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import PipelineCreationSchema, run_pipeline
    from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
    from nv_ingest_client.client import NvIngestClient
    start = timer_start("Pipeline subprocess")
    run_pipeline(PipelineCreationSchema(), block=False, disable_dynamic_scaling=True, run_in_subprocess=True)
    log.info(f"  Polling port 7671 (max {PIPELINE_WAIT}s)..."); deadline = time.time() + PIPELINE_WAIT
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", 7671), timeout=2): break
        except (ConnectionRefusedError, OSError): time.sleep(2)
    else:
        log.error("Port 7671 never opened. Pipeline subprocess likely crashed."); flush(); sys.exit(1)
    log.info("  Port 7671 ready."); time.sleep(5); init_time = timer_end("Pipeline subprocess", start)
    client = NvIngestClient(message_client_allocator=SimpleClient, message_client_port=7671, message_client_hostname="localhost")
    log.info("  NvIngestClient connected -> localhost:7671"); flush(); return client, init_time

def sanity_check(client, filepath):
    from nv_ingest_client.client import Ingestor
    from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob
    sep(); log.info(f"SANITY CHECK (text-only) -> {Path(filepath).name}"); sep(); start = timer_start("Sanity")
    ingestor = Ingestor(client=client).files(filepath).extract(extract_text=True, extract_tables=False, extract_charts=False, extract_images=False, extract_infographics=False, text_depth="page")
    results, failures = ingestor.ingest(show_progress=True, return_failures=True); sanity_time = timer_end("Sanity", start)
    log.info(f"  Results: {len(results)}  Failures: {len(failures)}")
    if failures:
        for idx, failure in enumerate(failures): log.error(f"  FAIL[{idx}]: {failure}")
        log.error("Sanity check failed."); flush(); sys.exit(1)
    if not results:
        log.error("Sanity check returned no results."); flush(); sys.exit(1)
    log.info(f"  Preview: {ingest_json_results_to_blob(results[0])[:200].replace(chr(10), ' ')}..."); log.info("  SANITY PASSED"); flush(); return sanity_time

def prepare_collection(collection_name):
    from pymilvus import connections, utility
    connections.connect(uri=MILVUS_URI)
    if utility.has_collection(collection_name):
        if DROP_COLLECTION_IF_EXISTS:
            log.warning(f"Collection '{collection_name}' exists. Dropping because DROP_COLLECTION_IF_EXISTS=1"); utility.drop_collection(collection_name)
        else:
            log.warning(f"Collection '{collection_name}' exists and will be reused.")
    else:
        log.info(f"Collection '{collection_name}' does not exist yet.")
    flush()

def extract_kwargs(multimodal):
    kwargs = {"extract_text": True, "text_depth": "page", "extract_tables": multimodal, "extract_charts": multimodal, "extract_images": multimodal, "extract_infographics": multimodal}
    if multimodal: kwargs["table_output_format"] = "markdown"
    return kwargs

def run_ingest_attempt(client, filepath, kwargs, collection_name):
    from nv_ingest_client.client import Ingestor
    ingestor = Ingestor(client=client).files(filepath).extract(**kwargs).split(tokenizer=TOKENIZER, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).embed().vdb_upload(collection_name=collection_name, milvus_uri=MILVUS_URI, sparse=SPARSE, dense_dim=DENSE_DIM)
    return ingestor.ingest(show_progress=False, return_failures=True)

def should_retry_text_only(failures):
    message = " ".join(str(f).lower() for f in failures)
    return any(marker in message for marker in NIM_FAILURE_MARKERS)

def log_failures(failures):
    for idx, failure in enumerate(failures):
        msg = str(failure)
        for start in range(0, len(msg), 180): log.warning(f"    FAIL[{idx}]: {msg[start:start + 180]}")

def ingest_single_file(client, filepath, idx, total, nim_status, collection_name):
    filename, ext = Path(filepath).name, Path(filepath).suffix.lower()
    all_nims, ocr_ok = all(nim_status.values()), nim_status.get("ocr", False)
    mode, fallback = ("FULL+HYBRID" if all_nims else "TEXT+HYBRID"), False
    if ext in IMAGE_ONLY_EXTS and not ocr_ok:
        log.info(f"  [{idx}/{total}]  {filename}  [SKIPPED - image file, OCR NIM not running]"); flush()
        return {"mode": "skipped", "total_ingest_sec": 0, "results_count": 0, "failures_count": 0, "skipped": True}, []
    log.info(f"  [{idx}/{total}]  {filename}  [{mode}]  sparse={SPARSE}"); start = time.perf_counter()
    try:
        results, failures = run_ingest_attempt(client, filepath, extract_kwargs(all_nims), collection_name)
        if all_nims and failures and not results and should_retry_text_only(failures):
            fallback, mode = True, "TEXT+HYBRID (fallback)"
            log.warning("    Multimodal stage failed. Retrying once in text-only mode."); log_failures(failures)
            results, failures = run_ingest_attempt(client, filepath, extract_kwargs(False), collection_name)
        elapsed = time.perf_counter() - start
        if failures: log_failures(failures)
        log.info(f"    results={len(results)}  failures={len(failures)}  {elapsed:.3f}s"); flush()
        return {"mode": mode, "fallback_triggered": fallback, "total_ingest_sec": round(elapsed, 3), "results_count": len(results), "failures_count": len(failures)}, failures
    except ValueError as exc:
        elapsed = time.perf_counter() - start
        log.warning(f"    SKIPPED (no embeddable content): {exc}"); flush()
        return {"mode": mode, "total_ingest_sec": round(elapsed, 3), "results_count": 0, "failures_count": 0, "skipped": True}, []
    except Exception as exc:
        elapsed = time.perf_counter() - start
        log.error(f"    EXCEPTION: {exc}"); log.error(traceback.format_exc()); flush()
        return {"mode": mode, "total_ingest_sec": round(elapsed, 3), "results_count": 0, "failures_count": 1, "error": str(exc)}, [exc]

def run_batch(client, files, nim_status, collection_name):
    sep(); label = "FULL MULTIMODAL + HYBRID BM25" if all(nim_status.values()) else "TEXT-ONLY + HYBRID BM25"
    log.info(f"BATCH INGEST  ->  {len(files)} files  |  {label}")
    log.info(f"  Collection: {collection_name}  Milvus: {MILVUS_URI}")
    log.info(f"  Chunk: {CHUNK_SIZE} tok  overlap: {CHUNK_OVERLAP}  sparse={SPARSE}"); sep(); flush()
    start, timings = time.perf_counter(), {}; ok = failed = skipped = 0
    for idx, filepath in enumerate(files, start=1):
        filename, doc_start = Path(filepath).name, time.perf_counter()
        try:
            timing, _ = ingest_single_file(client, filepath, idx, len(files), nim_status, collection_name)
            timings[filename] = timing
            if timing.get("skipped"): skipped += 1
            elif timing.get("failures_count", 1) == 0: ok += 1
            else: failed += 1
        except Exception as exc:
            log.error(f"  OUTER EXCEPTION {filename}: {exc}"); log.error(traceback.format_exc())
            timings[filename] = {"error": str(exc), "total_ingest_sec": 0, "failures_count": 1}; failed += 1
        log.info(f"  -- wall: {time.perf_counter() - doc_start:.3f}s --"); flush()
    batch_total = time.perf_counter() - start; sep(); log.info("BATCH SUMMARY")
    log.info(f"  Total: {len(files)}  OK: {ok}  Skipped: {skipped}  Failed: {failed}")
    log.info(f"  Total: {batch_total:.3f}s  Avg: {batch_total / max(len(files), 1):.3f}s"); sep(); flush()
    return timings, batch_total

def get_collection_entity_count(collection_name):
    from pymilvus import Collection, connections, utility
    connections.connect(uri=MILVUS_URI)
    if not utility.has_collection(collection_name): return 0
    col = Collection(collection_name); col.load(); return col.num_entities

def check_milvus(collection_name):
    sep(); log.info("MILVUS VERIFICATION")
    try:
        count = get_collection_entity_count(collection_name); log.info(f"  '{collection_name}': {count} total chunks")
        if count == 0:
            log.error("  ZERO chunks. Embed/upload likely failed."); flush(); return False, 0
        log.info(f"  Hybrid retrieval: TOP_K={TOP_K_RETRIEVE} -> rerank -> keep {RERANK_K}"); flush(); return True, count
    except Exception as exc:
        log.error(f"  Milvus error: {exc}"); flush(); return False, 0

def parse_rerank_payload(payload, n):
    ranked = []
    if isinstance(payload, list):
        if all(isinstance(x, (int, float)) for x in payload): ranked = [(i, float(s)) for i, s in enumerate(payload[:n])]
        elif all(isinstance(x, dict) for x in payload):
            for i, item in enumerate(payload):
                idx = item.get("index", item.get("id", i)); score = item.get("score", item.get("logit", item.get("relevance", 0.0)))
                if isinstance(idx, int) and isinstance(score, (int, float)): ranked.append((idx, float(score)))
    elif isinstance(payload, dict):
        for key in ("rankings", "results", "data", "scores"):
            if payload.get(key) is not None:
                ranked = parse_rerank_payload(payload[key], n)
                if ranked: break
    ranked = [(i, s) for i, s in ranked if 0 <= i < n]; ranked.sort(key=lambda x: x[1], reverse=True); return ranked

def rerank_with_api(query, candidates, api_key):
    if not candidates: return []
    payload = {"model": RERANKER_MODEL, "query": {"text": query}, "passages": [{"text": c["text"][:MAX_CHUNK_CHARS]} for c in candidates]}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}
    resp = requests.post(RERANKER_URL, headers=headers, json=payload, timeout=RERANKER_TIMEOUT); resp.raise_for_status()
    ranked_pairs = parse_rerank_payload(resp.json(), len(candidates))
    if not ranked_pairs: raise ValueError("Unrecognized reranker response")
    out = []
    for idx, score in ranked_pairs[:RERANK_K]:
        item = dict(candidates[idx]); item["rerank_score"] = float(score); out.append(item)
    return out

def rerank_locally(query, candidates):
    out = []
    for c in candidates:
        item = dict(c)
        item["rerank_score"] = round(lexical_score(query, c["text"]) + 0.02 * max(0, TOP_K_RETRIEVE + 1 - c["retrieval_rank"]), 6)
        out.append(item)
    out.sort(key=lambda x: x["rerank_score"], reverse=True); return out[:RERANK_K]

def rerank(query, candidates, api_key):
    try:
        return rerank_with_api(query, candidates, api_key), "api"
    except requests.exceptions.Timeout:
        log.warning(f"    Reranker timeout after {RERANKER_TIMEOUT}s. Falling back to local rerank.")
    except requests.exceptions.HTTPError as exc:
        snippet = exc.response.text[:160] if exc.response is not None else str(exc)
        log.warning(f"    Reranker HTTP error. Falling back locally. Details: {snippet}")
    except Exception as exc:
        log.warning(f"    Reranker parse/error fallback: {exc}")
    return rerank_locally(query, candidates), "local_fallback"

def build_context(items):
    blocks, chars = [], 0
    for idx, item in enumerate(items, start=1):
        block = f"[Excerpt {idx} | source={item['source']} | score={item.get('rerank_score', 0.0):.4f}]\n{item['text'][:MAX_CHUNK_CHARS]}"
        if chars + len(block) > MAX_CONTEXT_CHARS: break
        blocks.append(block); chars += len(block) + 2
    return "\n\n".join(blocks)

def answer_text(resp):
    content = getattr(resp.choices[0].message, "content", "")
    if isinstance(content, list): return "\n".join(item["text"] for item in content if isinstance(item, dict) and item.get("text")).strip()
    return str(content).strip()

def run_rag(api_key, questions, collection_name):
    from openai import OpenAI
    from nv_ingest_client.util.milvus import nvingest_retrieval
    sep(); log.info(f"RAG QUERIES  ->  {len(questions)} question(s)")
    log.info(f"  Collection : {collection_name}")
    log.info(f"  Mode       : hybrid BM25+semantic (sparse={SPARSE})")
    log.info(f"  Fetch      : TOP_K_RETRIEVE={TOP_K_RETRIEVE} -> rerank -> RERANK_K={RERANK_K}")
    log.info(f"  LLM        : {LLM_MODEL}"); sep(); flush()
    llm, all_qa, timings = OpenAI(base_url=LLM_BASE_URL, api_key=api_key), [], {}
    for idx, question in enumerate(questions, start=1):
        log.info(f"  Q{idx}: {question}"); flush(); start = time.perf_counter()
        docs = nvingest_retrieval([question], collection_name, milvus_uri=MILVUS_URI, hybrid=SPARSE, top_k=TOP_K_RETRIEVE)
        retrieval_sec = time.perf_counter() - start; candidates = extract_candidates(docs)
        log.info(f"    Retrieval : {retrieval_sec:.3f}s  |  candidates: {len(candidates)}/{TOP_K_RETRIEVE}  mode: {'hybrid BM25+semantic' if SPARSE else 'semantic-only'}")
        rr_start = time.perf_counter(); ranked, rerank_mode = rerank(question, candidates, api_key); rerank_sec = time.perf_counter() - rr_start
        if not ranked and candidates: ranked, rerank_mode = rerank_locally(question, candidates), "local_only"
        top_score = ranked[0]["rerank_score"] if ranked else 0.0
        log.info(f"    Reranking : {rerank_sec:.3f}s  |  kept {len(ranked)}/{len(candidates)}  top_score={top_score:.4f}  mode={rerank_mode}")
        if ranked: log.info(f"    Sources   : {', '.join(top_sources(ranked)[:5])}")
        context = build_context(ranked) if ranked else "No excerpts were retrieved for this question."
        p_tok = c_tok = tot_tok = 0; tps = llm_sec = 0.0
        try:
            llm_start = time.perf_counter()
            resp = llm.chat.completions.create(model=LLM_MODEL, messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}], max_tokens=900, temperature=0.2)
            llm_sec = time.perf_counter() - llm_start; answer = answer_text(resp); usage = getattr(resp, "usage", None)
            p_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
            c_tok = getattr(usage, "completion_tokens", 0) if usage else 0
            tot_tok = getattr(usage, "total_tokens", 0) if usage else 0
            tps = c_tok / llm_sec if llm_sec > 0 else 0.0
        except Exception as exc:
            answer = f"LLM call failed: {exc}"; log.error(f"    LLM error Q{idx}: {exc}")
        log.info(f"    LLM       : {llm_sec:.3f}s  |  tokens={tot_tok}  tok/s={tps:.1f}")
        log.info(f"    Answer    : {answer[:240].replace(chr(10), ' ')}"); flush()
        metrics = {
            "question": question, "retrieval_sec": round(retrieval_sec, 4), "rerank_sec": round(rerank_sec, 4),
            "chunks_fetched": len(candidates), "chunks_after_rerank": len(ranked), "rerank_top_score": round(top_score, 6),
            "rerank_mode": rerank_mode, "llm_sec": round(llm_sec, 4), "prompt_tokens": p_tok,
            "completion_tokens": c_tok, "total_tokens": tot_tok, "tokens_per_sec": round(tps, 2), "top_sources": top_sources(ranked),
        }
        timings[f"Q{idx}"] = metrics; all_qa.append({"question": question, "answer": answer, "metrics": metrics}); time.sleep(0.5)
    sep("-"); log.info(f"  {'Q':<4}  {'Retrieve':>9}  {'Rerank':>7}  {'LLM':>7}  {'Tokens':>7}  {'Tok/s':>6}  Score"); sep("-")
    for key, value in timings.items():
        log.info(f"  {key:<4}  {value['retrieval_sec']:>8.3f}s  {value['rerank_sec']:>6.3f}s  {value['llm_sec']:>6.3f}s  {value['total_tokens']:>7}  {value['tokens_per_sec']:>5.1f}  {value['rerank_top_score']:.4f}")
    sep("-"); flush(); return timings, all_qa

def save_metrics(timings, batch_total, sanity_time, init_time, query_timings, nim_status, entity_count):
    payload = {
        "run_id": RUN_ID, "timestamp": datetime.now().isoformat(), "python": sys.version, "nim_health": nim_status,
        "retrieval": {"mode": "hybrid BM25+semantic" if SPARSE else "semantic-only", "sparse": SPARSE, "top_k": TOP_K_RETRIEVE, "rerank_k": RERANK_K, "reranker": RERANKER_MODEL},
        "config": {"docs_folder": DOCS_FOLDER, "collection": COLLECTION_NAME, "dense_dim": DENSE_DIM, "llm": LLM_MODEL, "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP},
        "milvus": {"uri": MILVUS_URI, "entity_count": entity_count},
        "phase_times": {"init": round(init_time, 3), "sanity": round(sanity_time, 3), "batch": round(batch_total, 3)},
        "per_doc": timings, "rag": query_timings,
        "summary": {
            "total": len(timings),
            "ok": sum(1 for v in timings.values() if not v.get("skipped") and not v.get("error") and v.get("failures_count", 1) == 0),
            "skipped": sum(1 for v in timings.values() if v.get("skipped")),
            "failed": sum(1 for v in timings.values() if v.get("error") or v.get("failures_count", 0) > 0),
            "wall": round(init_time + sanity_time + batch_total, 3),
        },
    }
    with open(METRICS_FILE, "w", encoding="utf-8") as handle: json.dump(payload, handle, indent=2)
    log.info(f"  Metrics -> {METRICS_FILE}"); flush(); return payload

def main():
    interactive, global_start = "--interactive" in sys.argv, time.perf_counter()
    sep(); log.info("NV-INGEST 25.9.0  LIBRARY MODE  v6  |  hybrid BM25+semantic + rerank")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Python {sys.version.split()[0]}")
    log.info(f"  {'INTERACTIVE' if interactive else 'PREDEFINED'} question mode")
    log.info(f"  Collection -> {COLLECTION_NAME}"); log.info(f"  Log        -> {os.path.abspath(LOG_FILE)}"); sep(); flush()
    questions = collect_questions() if interactive else DEFAULT_QUESTIONS
    if not interactive: log.info(f"  {len(questions)} predefined question(s)."); flush()
    api_key = setup_env(); files = find_files(); client, init_time = start_pipeline(); nim_status = check_nims()
    sanity_file = choose_sanity_file(files, nim_status)
    if sanity_file: sanity_time = sanity_check(client, sanity_file)
    else: sanity_time = 0.0; log.warning("No suitable sanity-check file found. Skipping sanity check."); flush()
    prepare_collection(COLLECTION_NAME); timings, batch_total = run_batch(client, files, nim_status, COLLECTION_NAME)
    milvus_ok, entity_count = check_milvus(COLLECTION_NAME)
    if not milvus_ok: log.error("Milvus empty. Skipping RAG."); query_timings, all_qa = {}, []
    else: query_timings, all_qa = run_rag(api_key, questions, COLLECTION_NAME)
    with open(ANSWERS_FILE, "w", encoding="utf-8") as handle: json.dump(all_qa, handle, indent=2, ensure_ascii=False)
    log.info(f"  Answers -> {ANSWERS_FILE}"); flush()
    metrics = save_metrics(timings, batch_total, sanity_time, init_time, query_timings, nim_status, entity_count)
    sep(); log.info("FINAL SUMMARY"); sep()
    log.info(f"  Run ID   : {RUN_ID}")
    log.info(f"  Docs     : {metrics['summary']['total']}  OK={metrics['summary']['ok']}  Skipped={metrics['summary']['skipped']}  Failed={metrics['summary']['failed']}")
    log.info(f"  Wall     : {time.perf_counter() - global_start:.1f}s  (init {init_time:.1f}s + sanity {sanity_time:.1f}s + batch {batch_total:.1f}s)")
    log.info(f"  Retrieval: hybrid BM25+semantic  fetch={TOP_K_RETRIEVE}  rerank_to={RERANK_K}  model={RERANKER_MODEL}")
    log.info(f"  Chunks   : {entity_count}")
    log.info(f"  Log      -> {LOG_FILE}"); log.info(f"  Metrics  -> {METRICS_FILE}"); log.info(f"  Answers  -> {ANSWERS_FILE}"); sep(); flush()

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt:
        log.warning("Interrupted."); flush(); sys.exit(1)
    except Exception as exc:
        log.error(f"FATAL: {exc}"); log.error(traceback.format_exc()); flush(); sys.exit(1)
```
