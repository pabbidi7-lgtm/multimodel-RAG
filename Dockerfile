import ast
import glob
import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests


if sys.version_info[:2] not in {(3, 11), (3, 12)}:
    sys.exit(f"[FATAL] Python 3.11 or 3.12 required. You have {sys.version.split()[0]}")

try:
    import pymilvus

    pymilvus.connections.disconnect("default")
except Exception:
    pass


DOCS_FOLDER = "Docs"
FILE_PATTERNS = [
    "*.pdf",
    "*.PDF",
    "*.docx",
    "*.DOCX",
    "*.pptx",
    "*.PPTX",
    "*.jpeg",
    "*.JPEG",
    "*.jpg",
    "*.JPG",
    "*.png",
    "*.PNG",
]
RESULTS_DIR = "Outputs"
COLLECTION_PREFIX = "multimodal_docs"
MILVUS_URI = os.environ.get("MILVUS_URI", "milvus.db")
PIPELINE_WAIT = int(os.environ.get("PIPELINE_WAIT", "90"))
SPARSE = os.environ.get("SPARSE", "1") == "1"
DENSE_DIM = int(os.environ.get("DENSE_DIM", "2048"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))
TOKENIZER = os.environ.get("TOKENIZER", "intfloat/e5-large-unsupervised")
PDF_EXTRACT_METHOD = os.environ.get("PDF_EXTRACT_METHOD", "pdfium")
LLM_MODEL = os.environ.get("LLM_MODEL", "meta/llama-3.3-70b-instruct")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "nvidia/llama-nemotron-rerank-1b-v2")
RERANKER_URL = os.environ.get(
    "RERANKER_URL",
    "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-1b-v2/reranking",
)
RERANKER_TIMEOUT = int(os.environ.get("RERANKER_TIMEOUT", "20"))
ENABLE_CAPTION = os.environ.get("ENABLE_CAPTION", "0") == "1"
CAPTION_MODEL = os.environ.get("CAPTION_MODEL", "nvidia/llama-3.1-nemotron-nano-vl-8b-v1")
CAPTION_ENDPOINT_URL = os.environ.get("CAPTION_ENDPOINT_URL", f"{LLM_BASE_URL}/chat/completions")
RECREATE_COLLECTION = os.environ.get("RECREATE_COLLECTION", "1") == "1"
FAIL_ON_LOW_CHUNK_VOLUME = os.environ.get("FAIL_ON_LOW_CHUNK_VOLUME", "1") == "1"
LOW_CHUNK_FACTOR = float(os.environ.get("LOW_CHUNK_FACTOR", "3.0"))
TOP_K_RETRIEVE = int(os.environ.get("TOP_K_RETRIEVE", "50"))
DOC_ROUTE_K = int(os.environ.get("DOC_ROUTE_K", "3"))
DOC_LOCAL_CHUNKS = int(os.environ.get("DOC_LOCAL_CHUNKS", "12"))
RERANK_K = int(os.environ.get("RERANK_K", "8"))
MIN_EVIDENCE_SCORE = float(os.environ.get("MIN_EVIDENCE_SCORE", "0.12"))
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "18000"))
MAX_CHUNK_CHARS = int(os.environ.get("MAX_CHUNK_CHARS", "2200"))
IMAGE_ONLY_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

DEFAULT_QUESTIONS = [
    "Why did economics and physics become early movers in open access adoption?",
    "How did arXiv influence scholarly communication in physics?",
    "Why did life sciences move toward open access journals and APC models?",
    "What does the report mean by successive waves of open access innovation?",
    "How does the report connect open access, open data, and reproducibility?",
]

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

NIM_HEALTH_ENDPOINTS = {
    "page_elements": {
        "profile": "yolox",
        "checks": [
            ("http://localhost:8000/v1/health/ready", 8000),
            ("http://localhost:8000/v1/health", 8000),
            ("http://localhost:8000/health/ready", 8000),
        ],
    },
    "graphic_elements": {
        "profile": "yolox-graphic-elements",
        "checks": [
            ("http://localhost:8003/v1/health/ready", 8003),
            ("http://localhost:8003/v1/health", 8003),
            ("http://localhost:8003/health/ready", 8003),
        ],
    },
    "table_structure": {
        "profile": "yolox-table-structure",
        "checks": [
            ("http://localhost:8006/v1/health/ready", 8006),
            ("http://localhost:8006/v1/health", 8006),
            ("http://localhost:8006/health/ready", 8006),
        ],
    },
    "ocr": {
        "profile": "ocr",
        "checks": [
            ("http://localhost:8009/v1/health/ready", 8009),
            ("http://localhost:8009/v1/health", 8009),
            ("http://localhost:8009/health/ready", 8009),
        ],
    },
}

GROUP_RULES = {
    "finance_legal": [
        "merger",
        "agreement",
        "public common",
        "sponsor owned",
        "evercore",
        "barclays",
        "series b",
        "series c",
        "merger sub",
        "units",
        "effective time",
    ],
    "medical_policy": [
        "discharge summary",
        "history and physical",
        "telephone orders",
        "verbal orders",
        "documentation standards",
        "anesthesia",
        "hospitalization",
        "patient",
    ],
    "accessibility": [
        "wcag",
        "contrast",
        "colour",
        "color",
        "accessible",
        "large text",
        "non-text graphical",
        "figure",
    ],
    "open_access": [
        "open access",
        "arxiv",
        "economics",
        "physics",
        "apc",
        "reproducibility",
    ],
    "identity": ["nid", "drivers license", "singapore", "license"],
    "tech": ["minion", "multimodal", "brochure", "nvidia"],
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
COLLECTION_NAME = os.environ.get("NV_INGEST_COLLECTION_NAME", f"{COLLECTION_PREFIX}_{RUN_ID}")
LOG_FILE = os.path.join(RESULTS_DIR, f"pipeline_run_{RUN_ID}.log")
METRICS_FILE = os.path.join(RESULTS_DIR, f"metrics_{RUN_ID}.json")
ANSWERS_FILE = os.path.join(RESULTS_DIR, f"answers_{RUN_ID}.json")
CATALOG_FILE = os.path.join(RESULTS_DIR, f"catalog_{RUN_ID}.json")

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
    for handler in log.handlers:
        try:
            handler.flush()
        except Exception:
            pass


def sep(char="=", width=72):
    log.info(char * width)
    flush()


def timer_start(name):
    log.info(f">> {name}")
    flush()
    return time.perf_counter()


def timer_end(name, start):
    elapsed = time.perf_counter() - start
    log.info(f"<< {name}  {elapsed:.3f}s")
    flush()
    return elapsed


def collapse_ws(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def safe_dump(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if len(t) > 1 and t not in STOPWORDS]


def lexical_score(query, text):
    q_tokens = tokenize(query)
    text_lower = str(text).lower()
    if not q_tokens:
        return 0.0
    overlap = len(set(q_tokens) & set(tokenize(text))) / max(len(set(q_tokens)), 1)
    phrase = 1.0 if str(query).lower().strip() in text_lower else 0.0
    rare = sum(0.03 for term in set(q_tokens) if len(term) >= 6 and term in text_lower)
    return overlap + phrase + rare


def infer_group(text):
    hay = str(text).lower()
    scores = {group: sum(1 for term in terms if term in hay) for group, terms in GROUP_RULES.items()}
    scores = {group: score for group, score in scores.items() if score > 0}
    return max(scores, key=scores.get) if scores else "general"


def parse_jsonish(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except Exception:
                try:
                    return ast.literal_eval(text)
                except Exception:
                    return {}
    return {}


def normalize_source_name(value):
    if isinstance(value, dict):
        for key in ("source_name", "filename", "file_name", "document_name", "path", "filepath", "uri"):
            if value.get(key):
                return Path(str(value[key])).name
    if isinstance(value, str) and value:
        return Path(value).name
    return "unknown_source"


def get_hit_entity(hit):
    if isinstance(hit, dict) and isinstance(hit.get("entity"), dict):
        return hit["entity"]
    return hit if isinstance(hit, dict) else {}


def source_from_hit(hit):
    entity = get_hit_entity(hit)
    return normalize_source_name(parse_jsonish(entity.get("source", {})))


def metadata_from_hit(hit):
    entity = get_hit_entity(hit)
    return parse_jsonish(entity.get("content_metadata", {}))


def text_from_hit(hit):
    entity = get_hit_entity(hit)
    for key in ("text", "content", "chunk", "body"):
        if entity.get(key):
            return str(entity[key])
    return ""


def top_sources(items, limit=5):
    seen, output = set(), []
    for item in items:
        label = f"{item['source_name']}:p{item['page_number']}" if item.get("page_number") is not None else item["source_name"]
        if label not in seen:
            seen.add(label)
            output.append(label)
        if len(output) >= limit:
            break
    return output


def collect_questions():
    print("\n" + "=" * 72)
    print("  Enter questions. Blank line or 'done' to finish.")
    print("=" * 72 + "\n")
    questions, idx = [], 1
    while True:
        try:
            q = input(f"  Q{idx}: ").strip()
        except EOFError:
            break
        if q.lower() in ("", "done"):
            if not questions:
                print("  [!] Need at least one question.")
                continue
            break
        questions.append(q)
        idx += 1
    if not questions:
        log.warning("No interactive questions supplied. Using default questions.")
        flush()
        return DEFAULT_QUESTIONS
    log.info(f"Collected {len(questions)} question(s).")
    flush()
    return questions


def setup_env():
    sep()
    log.info("ENVIRONMENT SETUP")
    sep()
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        log.error("NVIDIA_API_KEY not set. export NVIDIA_API_KEY='nvapi-...'")
        flush()
        sys.exit(1)
    log.info(f"  NVIDIA_API_KEY : ********{api_key[-6:]}")
    for key, value in NIM_ENV_VARS.items():
        if not os.environ.get(key):
            os.environ[key] = value
            log.info(f"  {key:<47} -> {value}")
        else:
            log.info(f"  {key:<47} (already set)")
    log.info(f"  Collection     : {COLLECTION_NAME}")
    log.info(f"  Milvus URI     : {MILVUS_URI}")
    log.info(f"  Sparse/BM25    : {SPARSE}")
    log.info(f"  PDF extract    : {PDF_EXTRACT_METHOD or 'default'}")
    log.info(f"  Caption        : {'enabled' if ENABLE_CAPTION else 'disabled'}")
    flush()
    return api_key


def check_single_endpoint(url):
    result = subprocess.run(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "5", url],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "000"


def check_nims():
    sep()
    log.info("NIM HEALTH CHECK")
    sep()
    status = {}
    for name, spec in NIM_HEALTH_ENDPOINTS.items():
        observed = []
        ok = False
        for url, port in spec["checks"]:
            try:
                code = check_single_endpoint(url)
            except Exception as exc:
                code = f"ERR:{exc}"
            observed.append((url, port, code))
            if code in ("200", "201"):
                ok = True
                break
        status[name] = ok
        if ok:
            url, port, code = next(item for item in observed if item[2] in ("200", "201"))
            log.info(f"  OK    {name:<18} port {port}  {url}  HTTP {code}")
        else:
            url, port, code = observed[0]
            log.warning(f"  FAIL  {name:<18} port {port}  {url}  HTTP {code}")
            for alt_url, _, alt_code in observed[1:]:
                log.warning(f"        alt check -> {alt_url}  HTTP {alt_code}")
            log.warning(f"        Fix: docker compose --profile {spec['profile']} up -d")
    healthy = sum(1 for value in status.values() if value)
    if healthy == 4:
        log.info("  All 4 extraction NIMs healthy. Full multimodal extraction is available.")
    else:
        log.warning(f"  {healthy}/4 extraction NIMs healthy. Text-only fallback will be used where needed.")
        log.warning("  For full demo quality, start all retrieval and table-structure profiles before ingest.")
    flush()
    return status


def find_files():
    sep()
    log.info(f"FILE DISCOVERY  ->  {DOCS_FOLDER}/")
    files, sizes = [], {}
    for pattern in FILE_PATTERNS:
        found = sorted(glob.glob(os.path.join(DOCS_FOLDER, pattern)))
        if found:
            log.info(f"  {pattern:<12} -> {len(found)}")
        files.extend(found)
    files = sorted(set(files))
    log.info(f"  Total: {len(files)} file(s)")
    for filepath in files:
        size_mb = os.path.getsize(filepath) / 1048576
        sizes[Path(filepath).name] = round(size_mb, 2)
        log.info(f"    {Path(filepath).name:<52} {size_mb:.2f} MB")
    if not files:
        log.error(f"No files found in {DOCS_FOLDER}/ matching {FILE_PATTERNS}")
        flush()
        sys.exit(1)
    flush()
    return files, sizes


def choose_sanity_file(files, nim_status):
    if nim_status.get("ocr", False):
        return files[0] if files else None
    for filepath in files:
        if Path(filepath).suffix.lower() not in IMAGE_ONLY_EXTS:
            return filepath
    return None


def start_pipeline():
    sep()
    log.info("PIPELINE INIT")
    sep()
    from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import PipelineCreationSchema, run_pipeline
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


def sanity_check(client, filepath):
    from nv_ingest_client.client import Ingestor
    from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

    kwargs = {
        "extract_text": True,
        "extract_tables": False,
        "extract_charts": False,
        "extract_images": False,
        "extract_infographics": False,
        "text_depth": "page",
    }
    if PDF_EXTRACT_METHOD and filepath.lower().endswith(".pdf"):
        kwargs["extract_method"] = PDF_EXTRACT_METHOD
    sep()
    log.info(f"SANITY CHECK (text-only) -> {Path(filepath).name}")
    sep()
    start = timer_start("Sanity")
    ingestor = Ingestor(client=client).files(filepath).extract(**kwargs)
    results, failures = ingestor.ingest(show_progress=True, return_failures=True)
    sanity_time = timer_end("Sanity", start)
    log.info(f"  Results: {len(results)}  Failures: {len(failures)}")
    if failures or not results:
        for idx, failure in enumerate(failures[:10]):
            log.error(f"  FAIL[{idx}]: {failure}")
        log.error("Sanity check failed.")
        flush()
        sys.exit(1)
    preview = ingest_json_results_to_blob(results[0])[:240].replace(chr(10), " ")
    log.info(f"  Preview: {preview}...")
    log.info("  SANITY PASSED")
    flush()
    return sanity_time


def prepare_collection(collection_name):
    from pymilvus import connections, utility

    connections.connect(uri=MILVUS_URI)
    if utility.has_collection(collection_name):
        if RECREATE_COLLECTION:
            log.warning(f"Collection '{collection_name}' exists. Dropping once before batch ingest.")
            utility.drop_collection(collection_name)
        else:
            log.warning(f"Collection '{collection_name}' exists and will be appended to.")
    else:
        log.info(f"Collection '{collection_name}' does not exist yet.")
    flush()


def build_batch_ingestor(client, files, api_key, multimodal):
    from nv_ingest_client.client import Ingestor

    extract_kwargs = {
        "extract_text": True,
        "extract_tables": multimodal,
        "extract_charts": multimodal,
        "extract_images": multimodal,
        "extract_infographics": multimodal,
        "text_depth": "page",
    }
    if multimodal:
        extract_kwargs["table_output_format"] = "markdown"
    if PDF_EXTRACT_METHOD:
        extract_kwargs["extract_method"] = PDF_EXTRACT_METHOD
    ingestor = (
        Ingestor(client=client)
        .files(files)
        .extract(**extract_kwargs)
        .split(tokenizer=TOKENIZER, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    )
    if multimodal and ENABLE_CAPTION:
        ingestor = ingestor.caption(
            endpoint_url=CAPTION_ENDPOINT_URL,
            model_name=CAPTION_MODEL,
            api_key=api_key,
        )
    return ingestor.embed().vdb_upload(
        collection_name=COLLECTION_NAME,
        milvus_uri=MILVUS_URI,
        sparse=SPARSE,
        dense_dim=DENSE_DIM,
        recreate=False,
        stream=(len(files) <= 10 or MILVUS_URI.endswith(".db")),
    )


def run_batch_ingest(client, files, api_key, nim_status):
    multimodal = all(nim_status.values())
    ocr_ok = nim_status.get("ocr", False)
    active_files, skipped_files = [], []
    for filepath in files:
        if Path(filepath).suffix.lower() in IMAGE_ONLY_EXTS and not ocr_ok:
            skipped_files.append(filepath)
        else:
            active_files.append(filepath)
    sep()
    log.info(
        f"BATCH INGEST  ->  {len(active_files)} active files  |  "
        f"{'FULL MULTIMODAL + HYBRID BM25' if multimodal else 'TEXT-ONLY + HYBRID BM25'}"
    )
    log.info(f"  Collection: {COLLECTION_NAME}  Milvus: {MILVUS_URI}")
    log.info(f"  Chunk: {CHUNK_SIZE} tok  overlap: {CHUNK_OVERLAP}  sparse={SPARSE}")
    if skipped_files:
        log.warning(f"  Skipping {len(skipped_files)} image-only files because OCR NIM is not healthy.")
        for filepath in skipped_files:
            log.warning(f"    skipped -> {Path(filepath).name}")
    sep()
    flush()
    if not active_files:
        log.error("No active files remain after OCR-based filtering.")
        flush()
        sys.exit(1)
    start = timer_start("Batch ingest")
    ingestor = build_batch_ingestor(client, active_files, api_key, multimodal)
    results, failures = ingestor.ingest(show_progress=True, return_failures=True)
    batch_total = timer_end("Batch ingest", start)
    log.info(f"  Successful docs: {len(results)}  Failures: {len(failures)}")
    for idx, failure in enumerate(failures[:10]):
        log.warning(f"  FAIL[{idx}]: {failure}")
    flush()
    return {
        "active_files": active_files,
        "skipped_files": skipped_files,
        "results": results,
        "failures": failures,
        "batch_total": batch_total,
        "nim_status": nim_status,
        "multimodal_enabled": multimodal,
    }


def get_collection_entity_count(collection_name):
    from pymilvus import Collection, connections, utility

    connections.connect(uri=MILVUS_URI)
    if not utility.has_collection(collection_name):
        return 0
    collection = Collection(collection_name)
    collection.load()
    return collection.num_entities


def check_milvus(collection_name):
    sep()
    log.info("MILVUS VERIFICATION")
    try:
        count = get_collection_entity_count(collection_name)
        log.info(f"  '{collection_name}': {count} total chunks")
        if count == 0:
            log.error("  ZERO chunks. Embed/upload likely failed.")
            flush()
            return False, 0
        log.info(f"  Hybrid retrieval: TOP_K={TOP_K_RETRIEVE} -> routed docs -> rerank -> keep {RERANK_K}")
        flush()
        return True, count
    except Exception as exc:
        log.error(f"  Milvus error: {exc}")
        flush()
        return False, 0


def load_collection_records(collection_name, batch_size=1000):
    from pymilvus import MilvusClient

    client = MilvusClient(MILVUS_URI)
    iterator = client.query_iterator(
        collection_name=collection_name,
        filter="pk >= 0",
        output_fields=["source", "content_metadata", "text"],
        batch_size=batch_size,
    )
    records = []
    while True:
        batch = iterator.next()
        if not batch:
            iterator.close()
            break
        records.extend(batch)
    return records


def build_catalog(records, file_sizes):
    catalog = {}
    for record in records:
        source_blob = parse_jsonish(record.get("source", {}))
        meta = parse_jsonish(record.get("content_metadata", {}))
        source_name = normalize_source_name(source_blob)
        text = collapse_ws(record.get("text", ""))
        page_number = meta.get("page_number")
        modality = str(meta.get("type") or meta.get("content_type") or meta.get("subtype") or "text")
        entry = catalog.setdefault(
            source_name,
            {
                "source_name": source_name,
                "source_path": source_blob.get("source_name") or source_blob.get("path") or "",
                "size_mb": file_sizes.get(source_name, 0.0),
                "doc_group": "general",
                "chunk_count": 0,
                "pages": set(),
                "modalities": defaultdict(int),
                "samples": [],
                "records": [],
            },
        )
        entry["chunk_count"] += 1
        if page_number is not None:
            entry["pages"].add(page_number)
        entry["modalities"][modality] += 1
        if text and len(entry["samples"]) < 3:
            entry["samples"].append(text[:700])
        entry["records"].append(
            {
                "text": text,
                "page_number": page_number,
                "modality": modality,
                "source_name": source_name,
            }
        )
    for name, entry in catalog.items():
        entry["pages"] = sorted(entry["pages"])
        entry["page_count"] = len(entry["pages"])
        entry["modalities"] = dict(entry["modalities"])
        entry["doc_group"] = infer_group(" ".join([name] + entry["samples"]))
    return catalog


def save_catalog(catalog):
    payload = {}
    for name, entry in catalog.items():
        payload[name] = {
            "source_name": entry["source_name"],
            "source_path": entry["source_path"],
            "size_mb": entry["size_mb"],
            "doc_group": entry["doc_group"],
            "chunk_count": entry["chunk_count"],
            "page_count": entry["page_count"],
            "pages": entry["pages"],
            "modalities": entry["modalities"],
            "samples": entry["samples"],
        }
    safe_dump(CATALOG_FILE, payload)
    log.info(f"  Catalog -> {CATALOG_FILE}")
    flush()


def check_catalog_health(catalog, active_files):
    total_chunks = sum(entry["chunk_count"] for entry in catalog.values())
    expected_floor = max(10, int(max(len(active_files), 1) * LOW_CHUNK_FACTOR))
    if total_chunks < expected_floor:
        message = (
            f"Collection chunk volume is suspiciously low: {total_chunks} chunks for "
            f"{len(active_files)} active files."
        )
        if FAIL_ON_LOW_CHUNK_VOLUME:
            log.error(message)
            flush()
            return False
        log.warning(message)
    for entry in catalog.values():
        if entry["size_mb"] >= 1.0 and entry["chunk_count"] <= 2:
            log.warning(
                f"Large document has very low chunk count: {entry['source_name']} "
                f"size={entry['size_mb']}MB chunks={entry['chunk_count']}"
            )
    flush()
    return True


def extract_candidates(retrieved_docs):
    hits = retrieved_docs[0] if retrieved_docs and retrieved_docs[0] else []
    seen, output = set(), []
    for rank, hit in enumerate(hits, start=1):
        text = collapse_ws(text_from_hit(hit))
        if not text:
            continue
        meta = metadata_from_hit(hit)
        page_number = meta.get("page_number")
        modality = str(meta.get("type") or meta.get("content_type") or meta.get("subtype") or "text")
        candidate = {
            "text": text,
            "source_name": source_from_hit(hit),
            "page_number": page_number,
            "modality": modality,
            "retrieval_rank": rank,
        }
        key = (candidate["source_name"], candidate["page_number"], candidate["text"][:800])
        if key not in seen:
            seen.add(key)
            output.append(candidate)
    return output


def route_documents(question, catalog, first_pass):
    q_group = infer_group(question)
    scores = defaultdict(float)
    for name, entry in catalog.items():
        base = lexical_score(question, " ".join([name, entry["doc_group"]] + entry["samples"]))
        if q_group != "general" and entry["doc_group"] == q_group:
            base += 1.5
        if base > 0:
            scores[name] += base
    for rank, candidate in enumerate(first_pass, start=1):
        bonus = 1.0 / rank + 0.15 * lexical_score(question, candidate["text"][:800])
        if q_group != "general" and catalog.get(candidate["source_name"], {}).get("doc_group") == q_group:
            bonus += 1.0
        scores[candidate["source_name"]] += bonus
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_docs = [name for name, _ in ranked[:DOC_ROUTE_K]]
    if not top_docs and q_group != "general":
        top_docs = [name for name, entry in catalog.items() if entry["doc_group"] == q_group][:DOC_ROUTE_K]
    if not top_docs:
        top_docs = list(sorted(catalog.keys()))[:DOC_ROUTE_K]
    return q_group, top_docs, ranked


def candidate_pool(question, top_docs, catalog, first_pass):
    seen, pool = set(), []
    for candidate in first_pass:
        if candidate["source_name"] in top_docs:
            key = (candidate["source_name"], candidate.get("page_number"), candidate["text"][:500])
            if key not in seen:
                seen.add(key)
                pool.append(candidate)
    for source_name in top_docs:
        local = []
        for record in catalog.get(source_name, {}).get("records", []):
            if not record["text"]:
                continue
            item = dict(record)
            item["retrieval_rank"] = TOP_K_RETRIEVE
            item["local_score"] = lexical_score(question, record["text"])
            local.append(item)
        local.sort(key=lambda item: item["local_score"], reverse=True)
        for item in local[:DOC_LOCAL_CHUNKS]:
            key = (item["source_name"], item.get("page_number"), item["text"][:500])
            if key not in seen:
                seen.add(key)
                pool.append(item)
    if not pool:
        pool = first_pass[:RERANK_K]
    return pool


def parse_rerank_payload(payload, count):
    ranked = []
    if isinstance(payload, list):
        if all(isinstance(item, (int, float)) for item in payload):
            ranked = [(idx, float(score)) for idx, score in enumerate(payload[:count])]
        elif all(isinstance(item, dict) for item in payload):
            for idx, item in enumerate(payload):
                pos = item.get("index", item.get("id", idx))
                score = item.get("score", item.get("logit", item.get("relevance", 0.0)))
                if isinstance(pos, int) and isinstance(score, (int, float)):
                    ranked.append((pos, float(score)))
    elif isinstance(payload, dict):
        for key in ("rankings", "results", "data", "scores"):
            if payload.get(key) is not None:
                ranked = parse_rerank_payload(payload[key], count)
                if ranked:
                    break
    ranked = [(idx, score) for idx, score in ranked if 0 <= idx < count]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def rerank_with_api(question, candidates, api_key):
    if not candidates:
        return []
    payload = {
        "model": RERANKER_MODEL,
        "query": {"text": question},
        "passages": [{"text": item["text"][:MAX_CHUNK_CHARS]} for item in candidates],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}
    response = requests.post(RERANKER_URL, headers=headers, json=payload, timeout=RERANKER_TIMEOUT)
    response.raise_for_status()
    ranked_pairs = parse_rerank_payload(response.json(), len(candidates))
    if not ranked_pairs:
        raise ValueError("Unrecognized reranker response")
    output = []
    for idx, score in ranked_pairs[:RERANK_K]:
        item = dict(candidates[idx])
        item["rerank_score"] = float(score)
        output.append(item)
    return output


def rerank_locally(question, candidates):
    output = []
    for item in candidates:
        rescored = dict(item)
        rescored["rerank_score"] = round(
            lexical_score(question, item["text"])
            + 0.05 * (1 if item.get("page_number") in (0, 1, 2) else 0)
            + 0.02 * max(0, TOP_K_RETRIEVE + 1 - item.get("retrieval_rank", TOP_K_RETRIEVE)),
            6,
        )
        output.append(rescored)
    output.sort(key=lambda item: item["rerank_score"], reverse=True)
    return output[:RERANK_K]


def rerank(question, candidates, api_key):
    try:
        return rerank_with_api(question, candidates, api_key), "api"
    except requests.exceptions.Timeout:
        log.warning(f"    Reranker timeout after {RERANKER_TIMEOUT}s. Falling back to local rerank.")
    except requests.exceptions.HTTPError as exc:
        body = exc.response.text[:160] if exc.response is not None else str(exc)
        log.warning(f"    Reranker HTTP error. Falling back locally. Details: {body}")
    except Exception as exc:
        log.warning(f"    Reranker parse/error fallback: {exc}")
    return rerank_locally(question, candidates), "local_fallback"


def evidence_score(question, ranked):
    if not ranked:
        return 0.0
    return max(lexical_score(question, item["text"]) for item in ranked[: min(3, len(ranked))])


def build_llm_payload(question, ranked):
    evidence, total = [], 0
    for item in ranked:
        block = {
            "source": item["source_name"],
            "page": item.get("page_number"),
            "modality": item.get("modality", "text"),
            "score": round(item.get("rerank_score", 0.0), 4),
            "text": item["text"][:MAX_CHUNK_CHARS],
        }
        raw = json.dumps(block, ensure_ascii=False)
        if total + len(raw) > MAX_CONTEXT_CHARS:
            break
        evidence.append(block)
        total += len(raw)
    return json.dumps({"question": question, "evidence": evidence}, ensure_ascii=False)


def answer_text(response):
    content = getattr(response.choices[0].message, "content", "")
    if isinstance(content, list):
        return "\n".join(item["text"] for item in content if isinstance(item, dict) and item.get("text")).strip()
    return str(content).strip()


def mature_fallback(question, q_group, top_docs):
    hint = f" Routed docs: {', '.join(top_docs)}." if top_docs else ""
    if q_group != "general":
        return f"I could not verify a reliable answer for this {q_group.replace('_', ' ')} question from the indexed evidence in this run.{hint}"
    return f"I could not verify a reliable answer for this question from the indexed evidence in this run.{hint}"


def run_rag(api_key, questions, catalog):
    from openai import OpenAI
    from nv_ingest_client.util.milvus import nvingest_retrieval

    sep()
    log.info(f"RAG QUERIES  ->  {len(questions)} question(s)")
    log.info(f"  Collection : {COLLECTION_NAME}")
    log.info(f"  Mode       : hybrid BM25+semantic (sparse={SPARSE})")
    log.info(f"  Fetch      : TOP_K_RETRIEVE={TOP_K_RETRIEVE} -> route -> rerank -> keep {RERANK_K}")
    log.info(f"  LLM        : {LLM_MODEL}")
    sep()
    flush()
    llm = OpenAI(base_url=LLM_BASE_URL, api_key=api_key)
    timings, answers = {}, []
    for idx, question in enumerate(questions, start=1):
        log.info(f"  Q{idx}: {question}")
        flush()
        start = time.perf_counter()
        docs = nvingest_retrieval(
            [question],
            COLLECTION_NAME,
            milvus_uri=MILVUS_URI,
            hybrid=SPARSE,
            top_k=TOP_K_RETRIEVE,
            output_fields=["text", "source", "content_metadata"],
        )
        retrieval_sec = time.perf_counter() - start
        first_pass = extract_candidates(docs)
        q_group, top_docs, doc_scores = route_documents(question, catalog, first_pass)
        pool = candidate_pool(question, top_docs, catalog, first_pass)
        rr_start = time.perf_counter()
        ranked, rerank_mode = rerank(question, pool, api_key)
        rerank_sec = time.perf_counter() - rr_start
        if not ranked and pool:
            ranked, rerank_mode = rerank_locally(question, pool), "local_only"
        ev_score = evidence_score(question, ranked)
        top_score = ranked[0]["rerank_score"] if ranked else 0.0
        log.info(f"    Retrieval : {retrieval_sec:.3f}s  |  candidates={len(first_pass)}  group={q_group}")
        log.info(f"    Routing   : {', '.join(top_docs) if top_docs else 'none'}")
        log.info(
            f"    Reranking : {rerank_sec:.3f}s  |  kept {len(ranked)}/{len(pool)}  "
            f"top_score={top_score:.4f}  evidence={ev_score:.4f}  mode={rerank_mode}"
        )
        llm_sec = prompt_tokens = completion_tokens = total_tokens = 0
        tokens_per_sec = 0.0
        if ranked and ev_score >= MIN_EVIDENCE_SCORE:
            log.info(f"    Sources   : {', '.join(top_sources(ranked))}")
            payload = build_llm_payload(question, ranked)
            try:
                llm_start = time.perf_counter()
                response = llm.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": payload}],
                    max_tokens=700,
                    temperature=0.1,
                )
                llm_sec = time.perf_counter() - llm_start
                answer = answer_text(response)
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
                tokens_per_sec = completion_tokens / llm_sec if llm_sec > 0 else 0.0
                sources_line = "Sources: " + ", ".join(top_sources(ranked))
                if sources_line not in answer:
                    answer = f"{answer.rstrip()}\n\n{sources_line}"
            except Exception as exc:
                answer = f"LLM call failed: {exc}"
                log.error(f"    LLM error Q{idx}: {exc}")
        else:
            answer = mature_fallback(question, q_group, top_docs)
        log.info(f"    LLM       : {llm_sec:.3f}s  |  tokens={total_tokens}  tok/s={tokens_per_sec:.1f}")
        log.info(f"    Answer    : {answer[:260].replace(chr(10), ' ')}")
        flush()
        metrics = {
            "question": question,
            "question_group": q_group,
            "retrieval_sec": round(retrieval_sec, 4),
            "rerank_sec": round(rerank_sec, 4),
            "chunks_fetched": len(first_pass),
            "candidate_pool": len(pool),
            "chunks_after_rerank": len(ranked),
            "rerank_top_score": round(top_score, 6),
            "evidence_score": round(ev_score, 6),
            "rerank_mode": rerank_mode,
            "routed_docs": top_docs,
            "top_sources": top_sources(ranked),
            "doc_route_scores": doc_scores[:8],
            "llm_sec": round(llm_sec, 4),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_per_sec": round(tokens_per_sec, 2),
        }
        timings[f"Q{idx}"] = metrics
        answers.append({"question": question, "answer": answer, "metrics": metrics})
        time.sleep(0.5)
    return timings, answers


def save_metrics(batch_state, init_time, sanity_time, entity_count, catalog, query_timings):
    payload = {
        "run_id": RUN_ID,
        "timestamp": datetime.now().isoformat(),
        "python": sys.version,
        "collection_name": COLLECTION_NAME,
        "milvus_uri": MILVUS_URI,
        "nim_health": batch_state["nim_status"],
        "config": {
            "sparse": SPARSE,
            "dense_dim": DENSE_DIM,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "pdf_extract_method": PDF_EXTRACT_METHOD,
            "enable_caption": ENABLE_CAPTION,
        },
        "phase_times": {
            "init": round(init_time, 3),
            "sanity": round(sanity_time, 3),
            "batch": round(batch_state["batch_total"], 3),
        },
        "summary": {
            "active_files": len(batch_state["active_files"]),
            "skipped_files": len(batch_state["skipped_files"]),
            "successful_docs": len(batch_state["results"]),
            "failed_docs": len(batch_state["failures"]),
            "entity_count": entity_count,
            "catalog_docs": len(catalog),
        },
        "catalog": {
            name: {
                "doc_group": entry["doc_group"],
                "size_mb": entry["size_mb"],
                "chunk_count": entry["chunk_count"],
                "page_count": entry["page_count"],
                "modalities": entry["modalities"],
            }
            for name, entry in catalog.items()
        },
        "rag": query_timings,
    }
    safe_dump(METRICS_FILE, payload)
    log.info(f"  Metrics -> {METRICS_FILE}")
    flush()
    return payload


def main():
    interactive = "--interactive" in sys.argv
    sep()
    log.info("NV-INGEST 25.9.0  LIBRARY MODE  v7  |  batch ingest + routed hybrid retrieval")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  Python {sys.version.split()[0]}")
    log.info(f"  {'INTERACTIVE' if interactive else 'PREDEFINED'} question mode")
    log.info(f"  Collection -> {COLLECTION_NAME}")
    log.info(f"  Milvus URI -> {MILVUS_URI}")
    log.info(f"  Log        -> {os.path.abspath(LOG_FILE)}")
    sep()
    flush()
    questions = collect_questions() if interactive else DEFAULT_QUESTIONS
    if not interactive:
        log.info(f"  {len(questions)} predefined question(s).")
        flush()
    api_key = setup_env()
    files, file_sizes = find_files()
    client, init_time = start_pipeline()
    nim_status = check_nims()
    sanity_file = choose_sanity_file(files, nim_status)
    sanity_time = sanity_check(client, sanity_file) if sanity_file else 0.0
    if not sanity_file:
        log.warning("No suitable sanity-check file found. Skipping sanity check.")
        flush()
    prepare_collection(COLLECTION_NAME)
    batch_state = run_batch_ingest(client, files, api_key, nim_status)
    milvus_ok, entity_count = check_milvus(COLLECTION_NAME)
    catalog, query_timings, answers = {}, {}, []
    if milvus_ok:
        catalog = build_catalog(load_collection_records(COLLECTION_NAME), file_sizes)
        save_catalog(catalog)
        if check_catalog_health(catalog, batch_state["active_files"]):
            query_timings, answers = run_rag(api_key, questions, catalog)
        else:
            log.error("Collection health check failed. Refusing to run RAG on suspicious ingest output.")
            flush()
    else:
        log.error("Milvus empty. Skipping RAG.")
        flush()
    safe_dump(ANSWERS_FILE, answers)
    log.info(f"  Answers -> {ANSWERS_FILE}")
    metrics = save_metrics(batch_state, init_time, sanity_time, entity_count, catalog, query_timings)
    sep()
    log.info("FINAL SUMMARY")
    sep()
    log.info(f"  Run ID   : {RUN_ID}")
    log.info(f"  Active   : {len(batch_state['active_files'])}  Skipped={len(batch_state['skipped_files'])}  Failures={len(batch_state['failures'])}")
    log.info(f"  Chunks   : {entity_count}")
    log.info(f"  Catalog  : {len(catalog)} documents")
    log.info(f"  Metrics  -> {METRICS_FILE}")
    log.info(f"  Answers  -> {ANSWERS_FILE}")
    log.info(f"  Log      -> {LOG_FILE}")
    sep()
    flush()
    return metrics


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
