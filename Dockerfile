import json, os, time
import pymilvus
pymilvus.connections.disconnect("default")

from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import (
    run_pipeline, PipelineCreationSchema
)
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from openai import OpenAI
from pymilvus import MilvusClient, DataType

# ─── CONFIG ───────────────────────────────────────────────────────────────────
assert "NVIDIA_API_KEY" in os.environ, "export NVIDIA_API_KEY=..."
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]

PDF_PATH        = "Docs/Ascent_of_Open.pdf"
MILVUS_URI      = "milvus.db"
COLLECTION      = "multimodal_docs"
VLM_MODEL       = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
EMBED_MODEL     = "nvidia/nv-embedqa-e5-v5"
DENSE_DIM       = 1024

VLM_PROMPT = (
    "You are processing a document page visual for a retrieval system. "
    "Describe everything visible: all text, numbers, labels, legends, axes, "
    "data values, titles, and relationships. Be thorough — this description "
    "is the only way this content will be searchable."
)

# ─── CLIENTS ──────────────────────────────────────────────────────────────────
nvidia = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)

# ─── VLM: base64 → text ───────────────────────────────────────────────────────
def vlm_caption(b64: str) -> str:
    """Send any visual (image / chart / table / infographic) as base64 → VLM text."""
    resp = nvidia.chat.completions.create(
        model=VLM_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": VLM_PROMPT},
        ]}],
        max_tokens=768,
    )
    return resp.choices[0].message.content.strip()

# ─── EMBED ────────────────────────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    resp = nvidia.embeddings.create(model=EMBED_MODEL, input=texts, encoding_format="float")
    return [r.embedding for r in resp.data]

# ─── MILVUS SETUP ─────────────────────────────────────────────────────────────
def setup_collection(uri: str, name: str, dim: int) -> MilvusClient:
    mc = MilvusClient(uri)
    if mc.has_collection(name):
        mc.drop_collection(name)
    schema = mc.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("id",        DataType.INT64,        is_primary=True)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("text",      DataType.VARCHAR,      max_length=65535)
    schema.add_field("source",    DataType.VARCHAR,      max_length=256)
    idx = mc.prepare_index_params()
    idx.add_index("embedding", index_type="FLAT", metric_type="IP")
    mc.create_collection(name, schema=schema, index_params=idx)
    return mc

# ─── PIPELINE START ───────────────────────────────────────────────────────────
print("Starting nv-ingest pipeline...")
run_pipeline(
    PipelineCreationSchema(),
    block=False,
    disable_dynamic_scaling=True,
    run_in_subprocess=True,
)
time.sleep(15)
print("Pipeline ready.")

nv_client = NvIngestClient(
    message_client_allocator=SimpleClient,
    message_client_port=7671,
    message_client_hostname="localhost",
)

# ─── STEP 1: EXTRACT (nv-ingest handles all document parsing) ─────────────────
print("\n=== STEP 1: Extracting document ===")
results, failures = (
    Ingestor(client=nv_client)
    .files(PDF_PATH)
    .extract(
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_images=True,
        extract_infographics=True,
        table_output_format="markdown",
        text_depth="page",
    )
    .ingest(show_progress=True, return_failures=True)
)

if failures:
    print(f"[!] Extraction failures: {len(failures)}")
    for f in failures:
        print(f"  {f}")
    raise SystemExit(1)

print(f"Extracted {len(results)} result(s).")

# ─── STEP 2: VLM PASS — all visuals go through base64 → VLM ──────────────────
print("\n=== STEP 2: VLM captioning all visuals ===")

chunks: list[dict] = []   # {"text": ..., "source": ...}

VISUAL_TYPES = {"image", "structured", "infographic"}  # nv-ingest content types

for raw in results:
    docs = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(docs, list):
        docs = [docs]

    for doc in docs:
        meta   = doc.get("metadata", {})
        cmeta  = meta.get("content_metadata", {})
        ctype  = cmeta.get("type", "").lower()        # "text" | "image" | "structured" | ...
        page   = cmeta.get("page_number", 0)
        source = f"{os.path.basename(PDF_PATH)}::p{page}"

        content = meta.get("content", "")
        if not content:
            continue

        if ctype == "text":
            # Plain extracted text — keep as-is
            if content.strip():
                chunks.append({"text": content.strip(), "source": source})

        elif ctype in VISUAL_TYPES:
            # image / chart / table / infographic — send raw pixels to VLM
            print(f"  VLM [{ctype}] page {page} ...", end=" ", flush=True)
            try:
                caption = vlm_caption(content)
                chunks.append({"text": f"[{ctype.upper()} p{page}]: {caption}", "source": source})
                print("✓")
            except Exception as e:
                print(f"✗ {e}")

print(f"Total chunks after VLM pass: {len(chunks)}")

# ─── STEP 3: EMBED + VDB ─────────────────────────────────────────────────────
print("\n=== STEP 3: Embedding & uploading to Milvus ===")

mc = setup_collection(MILVUS_URI, COLLECTION, DENSE_DIM)

BATCH = 32
inserted = 0
for i in range(0, len(chunks), BATCH):
    batch   = chunks[i : i + BATCH]
    texts   = [c["text"]   for c in batch]
    sources = [c["source"] for c in batch]
    vecs    = embed_texts(texts)
    mc.insert(COLLECTION, [
        {"embedding": v, "text": t, "source": s}
        for v, t, s in zip(vecs, texts, sources)
    ])
    inserted += len(batch)
    print(f"  inserted {inserted}/{len(chunks)}")

mc.flush(COLLECTION)
print(f"\nDone. {inserted} chunks in '{COLLECTION}' → {MILVUS_URI}")

# ─── STEP 4: SMOKE-TEST QUERY ─────────────────────────────────────────────────
print("\n=== STEP 4: Test retrieval ===")

from nv_ingest_client.util.milvus import nvingest_retrieval

queries = [
    "Why did economics and physics become early movers in open access adoption?",
    "How did arXiv influence scholarly communication in physics?",
    "What does the report mean by successive waves of innovation in open access?",
]

for q in queries:
    hits = nvingest_retrieval([q], COLLECTION, milvus_uri=MILVUS_URI, hybrid=False, top_k=8)
    ctx  = "\n\n".join(h["entity"]["text"] for h in hits[0]) if hits and hits[0] else "—"

    ans = nvidia.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=[{"role": "user", "content": (
            f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer concisely."
        )}],
        max_tokens=512,
        temperature=0.5,
    )
    print(f"\nQ: {q}")
    print(f"A: {ans.choices[0].message.content.strip()}")
    print("-" * 60)
