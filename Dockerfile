import logging, os, time, glob

import pymilvus
pymilvus.connections.disconnect("default")


from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import (
    run_pipeline,
    PipelineCreationSchema
)
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

# ------------ CONFIG ------------
assert "NVIDIA_API_KEY" in os.environ, "Set env: export NVIDIA_API_KEY=..."
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]

DOCS_FOLDER    = "Docs"
FILE_PATTERNS  = ["*.pdf", "*.docx", "*.pptx", "*.jpeg", "*.jpg", "*.png"]

# Collect all files from Docs/ folder
all_files = []
for pat in FILE_PATTERNS:
    all_files.extend(glob.glob(os.path.join(DOCS_FOLDER, pat)))
all_files = sorted(set(all_files))

print(f"Found {len(all_files)} file(s) in {DOCS_FOLDER}/:")
for f in all_files:
    print(f"  {f}")

assert all_files, f"No files found in {DOCS_FOLDER}/ matching {FILE_PATTERNS}"

# Use first file for sanity check (Step 1)
first_file = all_files[0]

# ------------ START PIPELINE ------------
config = PipelineCreationSchema()

run_pipeline(
    config,
    block=False,
    disable_dynamic_scaling=True,
    run_in_subprocess=True
)

print("Waiting for pipeline to initialize...")
time.sleep(15)
print("Pipeline ready. Connecting client...")

client = NvIngestClient(
    message_client_allocator=SimpleClient,
    message_client_port=7671,
    message_client_hostname="localhost"
)

milvus_uri = "milvus.db"
collection_name = "multimodal_docs"
sparse = False

# =========================================================================
#  STEP 1: Basic text extraction (sanity check on first file only)
# =========================================================================
print(f"\n=== STEP 1: Basic text extraction (sanity check: {first_file}) ===")

ingestor = (
    Ingestor(client=client)
    .files(first_file)
    .extract(
        extract_text=True,
        extract_tables=False,
        extract_charts=False,
        extract_images=False,
        extract_infographics=False,
        text_depth="page",
    )
)

print("Starting ingestion...")
t0 = time.time()
results, failures = ingestor.ingest(show_progress=True, return_failures=True)
t1 = time.time()
print(f"Total time: {t1 - t0:.2f} seconds")
print(f"\nResults:  {len(results)}")
print(f"Failures: {len(failures)}")

if failures:
    print("\n=== STEP 1 FAILURES ===")
    for i, f in enumerate(failures):
        print(f"--- [{i}] ---\n{f}")
    print("\nFix Step 1 before proceeding.")

elif results:
    print("\n=== STEP 1 SUCCEEDED ===")
    blob = ingest_json_results_to_blob(results[0])
    print(blob[:500] + "..." if len(blob) > 500 else blob)

    # =========================================================================
    #  STEP 2: Full extraction + split + caption + embed + vdb upload
    #          Runs on ALL files in Docs/ folder
    # =========================================================================
    print(f"\n=== STEP 2: Full pipeline on all {len(all_files)} file(s) ===")

    total_ok      = 0
    total_failed  = 0
    total_skipped = 0

    # Image-only extensions produce no embeddable text without OCR NIM
    IMAGE_ONLY_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    for idx, filepath in enumerate(all_files, 1):
        ext = os.path.splitext(filepath)[1].lower()

        # Skip pure image files — they raise ValueError (no embeddings) without OCR NIM
        if ext in IMAGE_ONLY_EXTS:
            print(f"\n[{idx}/{len(all_files)}] SKIPPED (image file, needs OCR NIM): {filepath}")
            total_skipped += 1
            continue

        print(f"\n[{idx}/{len(all_files)}] Ingesting: {filepath}")

        ingestor_full = (
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
                tokenizer="intfloat/e5-large-unsupervised",
                chunk_size=512,
                chunk_overlap=50,
            )
            .caption(
                endpoint_url="https://integrate.api.nvidia.com/v1/chat/completions",
                model_name="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
                api_key=NVIDIA_API_KEY,
            )
            .embed()
            .vdb_upload(
                collection_name=collection_name,
                milvus_uri=milvus_uri,
                sparse=sparse,
                dense_dim=2048
            )
        )

        t0 = time.time()
        try:
            results_full, failures_full = ingestor_full.ingest(
                show_progress=True, return_failures=True
            )
            t1 = time.time()
            print(f"  Time: {t1 - t0:.2f}s  Results: {len(results_full)}  Failures: {len(failures_full)}")

            if failures_full:
                print(f"  FAILURES for {filepath}:")
                for i, f in enumerate(failures_full):
                    print(f"    [{i}] {str(f)[:300]}")
                total_failed += 1
            else:
                print(f"  OK — embeddings stored in Milvus.")
                total_ok += 1

        except ValueError as ve:
            # No embeddable content (e.g. image-only PDF with no text layer)
            t1 = time.time()
            print(f"  SKIPPED ({t1-t0:.2f}s) — no embeddable content: {ve}")
            total_skipped += 1
        except Exception as e:
            t1 = time.time()
            print(f"  ERROR ({t1-t0:.2f}s): {e}")
            total_failed += 1

    print(f"\n=== STEP 2 COMPLETE ===")
    print(f"  Total files : {len(all_files)}")
    print(f"  OK          : {total_ok}")
    print(f"  Skipped     : {total_skipped}  (image files without OCR NIM)")
    print(f"  Failed      : {total_failed}")
    print(f"  Milvus URI  : {milvus_uri}")
    print(f"  Collection  : {collection_name}")

    if total_ok > 0:
        # =========================================================================
        #  STEP 3: Retrieval + RAG queries
        # =========================================================================
        print("\n=== STEP 3: Querying ingested documents ===")

        from openai import OpenAI
        from nv_ingest_client.util.milvus import nvingest_retrieval

        queries = [
            "Why did economics and physics become early movers in open access adoption?",
            "How did arXiv influence scholarly communication in physics?",
            "Why did life sciences move more toward open access journals and APC models instead of preprints?",
            "What does the report mean by saying open access has grown through \"successive waves of innovation\"?",
            "How does the report connect open access, open data, and reproducibility?",
        ]

        llm_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY
        )

        print("=" * 60)
        for q in queries:
            retrieved_docs = nvingest_retrieval(
                [q],
                collection_name,
                milvus_uri=milvus_uri,
                hybrid=sparse,
                top_k=10,
            )

            if retrieved_docs and retrieved_docs[0]:
                context = "\n\n".join([doc["entity"]["text"] for doc in retrieved_docs[0]])
            else:
                context = "No relevant content found."

            prompt = f"""Use the following context to answer the question.
If the answer is not in the context, say so.

Context:
{context}

Question: {q}
Answer:"""

            completion = llm_client.chat.completions.create(
                model="meta/llama-3.3-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.7,
            )

            print(f"\nQ: {q}")
            print(f"A: {completion.choices[0].message.content}")
            print("-" * 60)
    else:
        print("\nNo files succeeded — skipping RAG queries.")

else:
    print("\nNo results and no failures — unexpected state.")
