import json, os, time, textwrap
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

PDF_PATH       = "Docs/california-drivers-license-small 1 1.jpg"
MILVUS_URI     = "milvus.db"
COLLECTION     = "multimodal_docs"
VLM_MODEL      = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
EMBED_MODEL    = "nvidia/nv-embedqa-e5-v5"
DENSE_DIM      = 1024
CHUNK_SIZE     = 512          # characters per text chunk
CHUNK_OVERLAP  = 80           # overlap between consecutive chunks
EMBED_BATCH    = 16           # safe batch size for NVIDIA embed API

VISUAL_TYPES   = {"image", "structured", "infographic"}

VLM_PROMPT = (
    "You are processing a document visual for a retrieval system. "
    "Describe EVERYTHING visible: all text, numbers, labels, legends, axes, "
    "data values, titles, and relationships. Be thorough — this description "
    "is the only way this content will be searchable."
)

# ─── CLIENTS ──────────────────────────────────────────────────────────────────
nvidia = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-window chunks."""
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def vlm_caption(b64: str) -> str:
    """Send any visual as base64 PNG → VLM descriptive text."""
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


def embed_texts(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    """Embed a batch of strings.  input_type = 'passage' | 'query'"""
    resp = nvidia.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        encoding_format="float",
        extra_body={"input_type": input_type, "truncate": "END"},
    )
    return [r.embedding for r in resp.data]


def setup_collection(uri: str, name: str, dim: int) -> MilvusClient:
    mc = MilvusClient(uri)
    if mc.has_collection(name):
        mc.drop_collection(name)
    schema = mc.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("id",        DataType.INT64,        is_primary=True)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("text",      DataType.VARCHAR,      max_length=65535)
    schema.add_field("source",    DataType.VARCHAR,      max_length=512)
    idx = mc.prepare_index_params()
    idx.add_index("embedding", index_type="FLAT", metric_type="IP")
    mc.create_collection(name, schema=schema, index_params=idx)
    return mc

# ─── PIPELINE ─────────────────────────────────────────────────────────────────
print("Starting nv-ingest pipeline...")
run_pipeline(
    PipelineCreationSchema(),
    block=False,
    disable_dynamic_scaling=True,
    run_in_subprocess=True,
)
time.sleep(15)
print("Pipeline ready.\n")

nv_client = NvIngestClient(
    message_client_allocator=SimpleClient,
    message_client_port=7671,
    message_client_hostname="localhost",
)

# ─── STEP 1: EXTRACT ──────────────────────────────────────────────────────────
print("=== STEP 1: Extracting document ===")
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
    print(f"[!] {len(failures)} extraction failure(s):")
    for f in failures:
        print(f"  {f}")
    raise SystemExit(1)

print(f"Extracted {len(results)} result(s).")

# ─── STEP 2: CHUNK TEXT  +  VLM-CAPTION VISUALS ───────────────────────────────
print("\n=== STEP 2: Chunking text  +  VLM captioning visuals ===")

chunks: list[dict] = []   # {"text": ..., "source": ...}

for raw in results:
    docs = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(docs, list):
        docs = [docs]

    for doc in docs:
        meta   = doc.get("metadata", {})
        cmeta  = meta.get("content_metadata", {})
        ctype  = cmeta.get("type", "").lower()
        page   = cmeta.get("page_number", 0)
        source = f"{os.path.basename(PDF_PATH)}::p{page}::{ctype}"
        content = meta.get("content", "")
        if not content:
            continue

        if ctype == "text":
            # ── split long pages into overlapping chunks ──────────────────
            for i, piece in enumerate(chunk_text(content.strip())):
                chunks.append({"text": piece, "source": f"{source}::c{i}"})

        elif ctype in VISUAL_TYPES:
            # ── all visuals (image / chart / table / infographic) → VLM ──
            print(f"  VLM [{ctype}] page {page} ...", end=" ", flush=True)
            try:
                caption = vlm_caption(content)
                # VLM output can be long too — chunk it as well
                for i, piece in enumerate(chunk_text(caption)):
                    chunks.append({
                        "text": f"[{ctype.upper()} p{page}]: {piece}",
                        "source": f"{source}::c{i}",
                    })
                print("✓")
            except Exception as e:
                print(f"✗  {e}")

print(f"\nTotal chunks ready for embedding: {len(chunks)}")

# ─── STEP 3: EMBED + VDB ──────────────────────────────────────────────────────
print("\n=== STEP 3: Embedding & uploading to Milvus ===")

mc = setup_collection(MILVUS_URI, COLLECTION, DENSE_DIM)
inserted = 0

for i in range(0, len(chunks), EMBED_BATCH):
    batch   = chunks[i : i + EMBED_BATCH]
    texts   = [c["text"]   for c in batch]
    sources = [c["source"] for c in batch]

    try:
        vecs = embed_texts(texts, input_type="passage")
    except Exception as e:
        print(f"  [!] embed error at batch {i}: {e} — skipping")
        continue

    mc.insert(COLLECTION, [
        {"embedding": v, "text": t, "source": s}
        for v, t, s in zip(vecs, texts, sources)
    ])
    inserted += len(batch)
    print(f"  inserted {inserted}/{len(chunks)}")

mc.flush(COLLECTION)
print(f"\n✓ {inserted} chunks stored in '{COLLECTION}' → {MILVUS_URI}")

# ─── STEP 4: RETRIEVAL TEST ───────────────────────────────────────────────────
print("\n=== STEP 4: Test retrieval + RAG ===")

queries = [
    "Is there a date of birth field shown on the card?",
    "Does the card include an address section?",
    "What is the expiry date mentioned?",
    "which country ID is this?",
]

def retrieve(q: str, top_k: int = 8) -> str:
    [qvec] = embed_texts([q], input_type="query")
    hits   = mc.search(
        COLLECTION,
        data=[qvec],
        anns_field="embedding",
        limit=top_k,
        output_fields=["text", "source"],
    )
    return "\n\n".join(h["entity"]["text"] for h in hits[0]) if hits else "—"

print("=" * 60)
for q in queries:
    ctx = retrieve(q)
    ans = nvidia.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=[{"role": "user", "content": (
            f"Use the context below to answer the question.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {q}\nAnswer concisely."
        )}],
        max_tokens=512,
        temperature=0.5,
    )
    print(f"\nQ: {q}")
    print(f"A: {ans.choices[0].message.content.strip()}")
    print("-" * 60)
    





Hi Sneha,

I’m really glad I got the chance to work with you and Sharanya. It’s been such a valuable experience, and I’ve learned a lot during this time.

I truly appreciate the way both of you review things and discuss different approaches. The subtle way you point out mistakes really made me think deeper and look at problems from multiple angles—it pushed me to improve every day.

I hope I was able to meet your expectations under your guidance. Honestly, time went by so quickly, and I genuinely wish I could continue working with your team because of how much there is to learn.

I’m very grateful for the opportunity LTM provided, and especially thankful to both of you for consistently investing your time and effort in helping me grow. Without your guidance, I wouldn’t have developed this way of thinking while building solutions.

I do wish I had a bit more time to explore things further—like running the NV Ingest full version on GPU and understanding how the Nemotron models work in depth, as well as learning more about the iScan application. Unfortunately, time wasn’t sufficient, but I’ll definitely try to explore these on my own.

I’ll continue to upskill and carry forward everything I’ve learned here.

Thank you once again, Sneha. It truly means a lot.





Hi Sharanya, good morning,

As this is the last day of my internship, I just wanted to thank you for all your support. I’m really glad I got the opportunity to work under you and Sneha.

From your end, you helped me a lot. Even with your busy schedule, you always made time to connect with me and clarify my doubts, which I truly appreciate.

I know I still have a lot to learn, but I’ve gained so much from you—especially in terms of coding, referring to documentation, and implementing solutions. All of that has been really valuable for me.

It has genuinely been a great experience working with both of you. I’ll make sure to take your feedback seriously and continue improving going forward.

Thank you very much, Sharanya, for your guidance and support—it really means a lot.
