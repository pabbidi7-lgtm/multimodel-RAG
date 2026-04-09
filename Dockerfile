
You said:
see i will explain it very clearly i was running the pipeline.py it was working in my folder no issurethe problem is the normal version that means 25.9.0 version of nv ingest library mode the nemotron ocr nemotron infographhics elmenets nemotron table extraction was not working that measn it only taking the pdf and docs but making the ocr extraction but when i was running the image like licence it was not able to get the information means these nemotron graphic elements and remaing all based on VLM not applying on the image so that there is a normal extraction as output not exact outputimport logging, os, time

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
#  STEP 1: Basic text extraction (sanity check)
# =========================================================================
print("\n=== STEP 1: Basic text extraction ===")

ingestor = (
    Ingestor(client=client)
    .files("Docs/Singapore_NID_F 1.jpeg")
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
    # =========================================================================
    print("\n=== STEP 2: Full pipeline (extract + split + caption + embed + vdb) ===")

    ingestor_full = (
        Ingestor(client=client)
        .files("Docs/Singapore_NID_F 1.jpeg")
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
            params={"split_source_types": ["PDF", "text", "html", "mp3", "docx", "pptx"]},
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

    print("Starting full ingestion...")
    t0 = time.time()
    results_full, failures_full = ingestor_full.ingest(show_progress=True, return_failures=True)
    t1 = time.time()
    print(f"Total time: {t1 - t0:.2f} seconds")
    print(f"\nResults:  {len(results_full)}")
    print(f"Failures: {len(failures_full)}")

    if failures_full:
        print("\n=== STEP 2 FAILURES ===")
        for i, f in enumerate(failures_full):
            print(f"--- [{i}] ---\n{f}")
    else:
        print("\n=== STEP 2 SUCCEEDED ===")
        print(f"Embeddings stored in Milvus Lite: {milvus_uri}")
        print(f"Collection: {collection_name}")

        # =========================================================================
        #  STEP 3: Retrieval + RAG queries
        # =========================================================================
        print("\n=== STEP 3: Querying ingested documents ===")

        from openai import OpenAI
        from nv_ingest_client.util.milvus import nvingest_retrieval

        queries = [
            "What is the name mentioned in the ID",
            "what is the service number mentioned?",
            "What is the place birth?",
            "what is her data of birth?",
            "which country is she belongs?",
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
    print("\nNo results and no failures — unexpected state.")i was figured it out we have the nemotron graphic elements and nemotron table structure are running seperatly and they are working greatso i want to host that grpc and http endpoints to my pipeline.py i dont know how to do itif that makes i thinking of then nv ingest 25.9.0 version + these grpc and http nim endpoints will work perfectly and now when we are inserting the images/ tables they will work on them tooi was explaining this to my superiors and they mentioned that check the nemo retriver github repos try to take the custom functions and add them and see how they are running try to take those functions and run themso iwas very new to this cloning and mkaing custom function need to learn but dont know at allso what is the best option now to choose or to implement but my final aim is 70% nv ingest 25.9.0 librabry mode is working but images/tables/charts. graphics parts are not working at all when this ingestion is working so need to make sure that they have to run and provide the perfect answer
Copilot said:
Copilot

Reasoning completed in 4 steps
You explained it clearly now — and yes, you’ve already found the real problem:

Your pipeline.py is fine, and nv-ingest 25.9.0 library mode is fine.
The part that is missing is proper image/document-analysis NIM wiring + image-oriented extraction settings, so image files like licenses are not going through the OCR / page-elements / table / graphic-element path the way you expect. The official library-mode flow is still run_pipeline(...) + Ingestor(...), and run_pipeline(...) loads the default libmode pipeline automatically if you don’t pass a custom pipeline config.
nvidia
+1

So the best option now is not to rewrite the whole internal pipeline or start by copying custom functions from the repo. The best option is:

Keep your current pipeline.py and the official installed nv-ingest==25.9.0.
github
+1
Wire your working OCR / page-elements / graphic-elements / table-structure NIM endpoints into the libmode environment before run_pipeline(...) starts. The health/readiness logic in NeMo Retriever checks these endpoint env vars explicitly: OCR_HTTP_ENDPOINT, YOLOX_HTTP_ENDPOINT (page-elements), YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT, and YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT.
nvidia
+1
Adjust your ingestion settings for image files, because your current code is partially turning off the image-specific path and then filtering image-derived chunks out during splitting. The Python API is designed for this kind of high-level control through Ingestor(...).extract(...).split(...).caption(...).embed()... rather than by editing internal stage YAML first.
github
+1
What is happening in your current script
Why PDFs/DOCX feel “okay” but JPEG licenses do not
NeMo Retriever supports many file types and routes them through a shared internal pipeline, but the extraction behavior differs by content type. The docs explicitly note that the default native-text method (pdfium) is fast but does not capture text from scanned/image-based pages, while ocr and pdfium_hybrid are the methods intended for scanned pages.
github
+1

For image-heavy understanding, the internal extractors rely on OCR plus object-detection NIMs:

chart extraction combines graphic-elements + OCR, not OCR alone.
github
infographic extraction also relies on OCR endpoints.
github
the overall extraction architecture includes page-elements, graphic-elements, table-structure, and OCR as distinct NIM services.
github
+1
That means if image files are going through the pipeline without the full endpoint wiring and without image-friendly extraction settings, you will get a thin generic extraction instead of the rich OCR/layout/table/chart output you want.
github
+1

Your real root cause
You already described it correctly:

the separate NIMs (graphic-elements, table-structure, etc.) work when tested separately,
but your libmode pipeline is not fully using them for image inputs,
so the ingestion result is “normal extraction,” not the richer multimodal extraction.
That strongly points to pipeline configuration / endpoint wiring / extraction settings, not to a fundamental issue with your pipeline.py. The official libmode entry point remains run_pipeline(...), and the Python API remains the supported way to request extract/split/caption/embed/vdb tasks.
nvidia
+1

Best option to choose now
✅ Recommended architecture
Use this stack:

official installed nv-ingest 25.9.0
your existing working pipeline.py
your working hosted/local NIM endpoints injected through environment variables
small changes in pipeline.py to make image ingestion actually exercise the image path
This is the lowest-risk and fastest path to a working result. The 25.9.0 release explicitly expanded library-mode support and also added support for custom/user-defined stages, but you do not need custom stages first just to make image OCR and image/table/chart extraction work.
github
+1

Do not do this first
❌ Do not start by copying random custom functions from the repo
That is useful later, but not as the first fix. The Python API already supports high-level ingestion tasks, and it also exposes .udf() for user-defined functions if you truly need custom logic in the task chain.
github

So if your superiors said “check the repo and custom functions,” the right interpretation is:

first get the official 25.9.0 image path working with the right NIM endpoints,
then later explore .udf() or custom Lambda/user-defined stages if you need special processing beyond the stock pipeline. The 25.9.0 release notes explicitly mention support for custom Lambda stages / user-defined stages.
github
+1
What to implement now
1) Export the endpoint env vars before run_pipeline(...)
The libmode pipeline discovers NIM services through environment variables. The readiness code explicitly checks these names:

OCR_HTTP_ENDPOINT
YOLOX_HTTP_ENDPOINT (page-elements)
YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT
YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT
nvidia
So, in the same terminal where you run pipeline.py, do this:




Shell
micromamba activate myenv

export NVIDIA_API_KEY="YOUR_NEW_KEY"

# OCR
export OCR_HTTP_ENDPOINT="YOUR_WORKING_OCR_HTTP_ENDPOINT"
export OCR_INFER_PROTOCOL="http"

# page elements
export YOLOX_HTTP_ENDPOINT="YOUR_WORKING_PAGE_ELEMENTS_HTTP_ENDPOINT"
export YOLOX_INFER_PROTOCOL="http"

# graphic elements
export YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT="YOUR_WORKING_GRAPHIC_ELEMENTS_HTTP_ENDPOINT"
export YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL="http"

# table structure
export YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT="YOUR_WORKING_TABLE_STRUCTURE_HTTP_ENDPOINT"
export YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL="http"

Show more lines
If you are using gRPC for some services, set the matching *_GRPC_ENDPOINT values as well; the internal pipeline config supports both gRPC and HTTP endpoint tuples for these NIMs. The library-mode pipeline and helper code are built around (grpc, http) endpoint pairs plus an infer_protocol selector.
nvidia
+1

2) Keep run_pipeline(...) exactly as you already have it
This part is correct:




Python
config = PipelineCreationSchema()

run_pipeline(
    config,
    block=False,
    disable_dynamic_scaling=True,
    run_in_subprocess=True
)
That is the documented library-mode way to start the pipeline, and if no explicit pipeline config is passed, the default library-mode pipeline is loaded automatically.
nvidia
+1

3) Fix the image ingestion settings in your current script
These are the two biggest issues in your current code:

A. Your Step 1 sanity check is misleading for an image file
In Step 1 you do:




Python
.extract(
    extract_text=True,
    extract_tables=False,
    extract_charts=False,
    extract_images=False,
    extract_infographics=False,
    text_depth="page",
)
For a JPEG license, that is not really testing the full image-analysis path. You are turning off the image/infographic/table/chart branches, even though the richer image understanding depends on those extractors and their linked OCR/detector NIMs. Chart and infographic extraction specifically depend on OCR + detector integration.
github
+1

B. Your split filter is excluding image-style outputs
In Step 2 you do:




Python
params={"split_source_types": ["PDF", "text", "html", "mp3", "docx", "pptx"]}
That can cause image-derived content to be skipped from chunking/retrieval because you are not allowing image-origin content through the splitter.

So the first practical fix is:

remove that params=... restriction,
and keep image/infographic/chart/table extraction enabled for image inputs.
4) Save raw extraction output before embedding/upload
The Python API provides save_to_disk(...) and ingest_json_results_to_blob(...), which are the right tools to inspect what the extraction stage actually produced before you blame retrieval/LLM generation.
github
+1

That gives you proof of whether OCR text, table structure, chart regions, or image captions were actually created.

What I recommend you change in pipeline.py
Here is the practical best version of your approach:

Keep the official pipeline startup



Python
config = PipelineCreationSchema()
run_pipeline(config, block=False, disable_dynamic_scaling=True, run_in_subprocess=True)
That remains unchanged.
nvidia
+1

For image files, use a real multimodal extract step
Use this style for the image file path:




Python
ingestor_full = (
    Ingestor(client=client)
    .files("Docs/Singapore_NID_F 1.jpeg")
    .extract(
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_images=True,
        extract_infographics=True,
        table_output_format="markdown",
        text_depth="page",
    )
    .save_to_disk("./debug_extract")
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
        dense_dim=2048,
    )
)

Show more lines
Why this is better
It actually enables the image/table/chart/infographic path for the JPEG.
github
+2
It removes the restrictive split filter that can hide image-derived content.
It saves raw extraction output so you can verify that OCR/layout/table/chart metadata exists before you even get to Milvus/LLM.
Where “custom functions from the repo” fit in
After the above works, then you can consider two customization levels:

Level 1 — use your own Python helper functions
Example:

a function to summarize extraction JSON,
a function to print which modalities were extracted,
a function to validate OCR text is non-empty.
This is the easiest and safest place to start.

Level 2 — use Ingestor.udf(...) or custom stages
The Python API includes a .udf() method, and 25.9.0 added support for custom Lambda/user-defined stages. Use this only after the stock image pipeline works with your endpoints.
github
+1

My final recommendation to you
✅ Choose this path
Do not rewrite the internal libmode pipeline first.
Do not start by copying repo custom functions first.

Instead:

Keep official nv-ingest 25.9.0 + your working pipeline.py.
github
+1
Inject your working OCR / page-elements / graphic-elements / table-structure NIM endpoints through env vars before startup.
nvidia
+1
Change your image ingestion settings so image/table/chart/infographic extraction is actually exercised.
github
+2
Remove the restrictive split_source_types filter.
Save extraction output to disk and inspect it before embedding/upload.
github
+1
Only after that, explore repo custom functions / .udf() / custom stages.
github
+1
That is the most realistic path to get your “70% working” system to a state where images/tables/charts/graphics also work and produce good answers.

If you want, I can do the next step very concretely:

I can rewrite your exact pipeline.py now into a final working version with:
endpoint checks,
raw extraction debug,
the corrected image settings,
no restrictive split filter,
and an interactive question loop after ingestion.
Sources
Favicon typeFavicon type







You said:
I can rewrite your exact pipeline.py now into a final working version with:

endpoint checks,
raw extraction debug,
the corrected image settings,
no restrictive split filter,
and an interactive question loop after ingestion.do it
Copilot said:
Copilot

Reasoning completed in 2 steps
Absolutely — below is a rewritten pipeline.py that keeps the official nv-ingest 25.9.0 library-mode flow (run_pipeline(...) + Ingestor(...)) intact, adds endpoint checks, saves raw extraction output for debugging, uses image-friendly extraction settings, removes the restrictive split filter, and gives you an interactive Q&A loop after ingestion. The official library-mode entry point is run_pipeline(...), and the Python API is designed around Ingestor(...).extract(...).split(...).caption(...).embed()....
nvidia
+2

This version also checks the specific NIM endpoint environment variables that NeMo Retriever’s readiness code expects for OCR, page-elements, graphic-elements, and table-structure.
nvidia
+1

✅ Final pipeline.py
Replace your current file with this.




Python
import json
import os
import sys
import time
from pathlib import Path

import pymilvus
from openai import OpenAI

from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import (
    run_pipeline,
    PipelineCreationSchema
)
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob
from nv_ingest_client.util.milvus import nvingest_retrieval


# -----------------------------
# CONFIG
# -----------------------------
pymilvus.connections.disconnect("default")

# Required auth
assert "NVIDIA_API_KEY" in os.environ, "Set env: export NVIDIA_API_KEY=..."
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]

# Required NIM endpoints for image/document multimodal extraction
REQUIRED_ENV_VARS = [
    "OCR_HTTP_ENDPOINT",
    "YOLOX_HTTP_ENDPOINT",  # page-elements
    "YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT",
    "YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT",
]

OPTIONAL_ENV_VARS = [
    "OCR_INFER_PROTOCOL",
    "YOLOX_INFER_PROTOCOL",
    "YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL",
    "YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL",
    "VLM_CAPTION_ENDPOINT",
    "VLM_CAPTION_MODEL_NAME",
]

INPUT_FILE = "Docs/Singapore_NID_F 1.jpeg"
MILVUS_URI = "milvus.db"
COLLECTION_NAME = "multimodal_docs"
SPARSE = False

DEBUG_DIR_STEP1 = "./debug_step1_extract"
DEBUG_DIR_STEP2 = "./debug_step2_full"


# -----------------------------
# HELPERS
# -----------------------------
def print_env_status():
    print("\n=== ENVIRONMENT CHECK ===")
    missing = []
    for key in REQUIRED_ENV_VARS:
        value = os.environ.get(key)
        if value:
            print(f"[OK]   {key} = {value}")
        else:
            print(f"[MISS] {key} is not set")
            missing.append(key)

    for key in OPTIONAL_ENV_VARS:
        value = os.environ.get(key)
        if value:
            print(f"[INFO] {key} = {value}")

    if missing:
        raise RuntimeError(
            "Missing required NIM endpoint env vars: "
            + ", ".join(missing)
            + "\nSet them before running pipeline.py"
        )
    print("=" * 60)


def print_failures(title, failures):
    print(f"\n=== {title} FAILURES ===")
    if not failures:
        print("No failures.")
        return

    for i, f in enumerate(failures):
        print(f"\n--- FAILURE [{i}] ---")
        print(f)


def print_result_preview(title, results):
    print(f"\n=== {title} RESULT PREVIEW ===")
    if not results:
        print("No results returned.")
        return

    try:
        blob = ingest_json_results_to_blob(results[0])
        print(blob[:1500] + "..." if len(blob) > 1500 else blob)
    except Exception as exc:
        print(f"Could not convert result to blob preview: {exc}")
        print("Top-level result object type:", type(results[0]))
        if isinstance(results[0], dict):
            print("Top-level keys:", list(results[0].keys()))


def inspect_debug_dir(debug_dir):
    print(f"\n=== DEBUG FILE CHECK: {debug_dir} ===")
    p = Path(debug_dir)
    if not p.exists():
        print("Debug directory not found.")
        return

    files = sorted(p.rglob("*"))
    json_files = [f for f in files if f.is_file() and f.suffix.lower() == ".json"]
    other_files = [f for f in files if f.is_file() and f.suffix.lower() != ".json"]

    print(f"Total files: {len([f for f in files if f.is_file()])}")
    print(f"JSON files : {len(json_files)}")
    print(f"Other files: {len(other_files)}")

    if json_files:
        print("Sample JSON files:")
        for f in json_files[:5]:
            print(" -", f)


def build_context(retrieved_docs):
    if not retrieved_docs or not retrieved_docsreturn "No relevant content found."

    chunks = []
    for doc in retrieved_docstry:
            txt = doc["entity"].get("text", "")
            if txt and txt.strip():
                chunks.append(txt.strip())
        except Exception:
            continue

    return "\n\n".join(chunks) if chunks else "No relevant content found."


def interactive_qa_loop(llm_client):
    print("\n=== STEP 3: INTERACTIVE QUESTION LOOP ===")
    print("Ask questions about the ingested document.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        q = input("Q: ").strip()

        if q.lower() in {"exit", "quit", "q"}:
            print("Exiting interactive loop.")
            break

        if not q:
            print("Please enter a question.\n")
            continue

        retrieved_docs = nvingest_retrieval(
            [q],
            COLLECTION_NAME,
            milvus_uri=MILVUS_URI,
            hybrid=SPARSE,
            top_k=10,
        )

        context = build_context(retrieved_docs)

        prompt = f"""Use the following context to answer the question.
If the answer is not in the context, say so.

Context:
{context}

Question: {q}
Answer:"""

        completion = llm_client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.2,
        )

        print("\nA:", completion.choices[0].message.content)
        print("-" * 60)


# -----------------------------
# MAIN
# -----------------------------
def main():
    print_env_status()

    # Start official libmode pipeline
    config = PipelineCreationSchema()

    run_pipeline(
        config,
        block=False,
        disable_dynamic_scaling=True,
        run_in_subprocess=True
    )

    print("\nWaiting for pipeline to initialize...")
    time.sleep(15)
    print("Pipeline ready. Connecting client...")

    client = NvIngestClient(
        message_client_allocator=SimpleClient,
        message_client_port=7671,
        message_client_hostname="localhost"
    )

    # ---------------------------------------------------------
    # STEP 1: RAW EXTRACTION DEBUG FOR IMAGE INPUT
    # ---------------------------------------------------------
    print("\n=== STEP 1: RAW EXTRACTION DEBUG ===")
    print(f"Input file: {INPUT_FILE}")

    ingestor_debug = (
        Ingestor(client=client)
        .files(INPUT_FILE)
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=True,
            extract_infographics=True,
            table_output_format="markdown",
            text_depth="page",
        )
        .save_to_disk(DEBUG_DIR_STEP1)
    )

    print("Starting Step 1 debug extraction...")
    t0 = time.time()
    results_step1, failures_step1 = ingestor_debug.ingest(
        show_progress=True,
        return_failures=True
    )
    t1 = time.time()

    print(f"Step 1 total time: {t1 - t0:.2f} seconds")
    print(f"Step 1 results : {len(results_step1)}")
    print(f"Step 1 failures: {len(failures_step1)}")

    if failures_step1:
        print_failures("STEP 1", failures_step1)

    print_result_preview("STEP 1", results_step1)
    inspect_debug_dir(DEBUG_DIR_STEP1)

    if not results_step1:
        raise SystemExit(
            "\nNo Step 1 results returned. Fix extraction / endpoint setup before proceeding."
        )

    # ---------------------------------------------------------
    # STEP 2: FULL MULTIMODAL INGESTION
    # ---------------------------------------------------------
    print("\n=== STEP 2: FULL MULTIMODAL INGESTION ===")

    ingestor_full = (
        Ingestor(client=client)
        .files(INPUT_FILE)
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=True,
            extract_infographics=True,
            table_output_format="markdown",
            text_depth="page",
        )
        .save_to_disk(DEBUG_DIR_STEP2)
        .split(
            tokenizer="intfloat/e5-large-unsupervised",
            chunk_size=512,
            chunk_overlap=50,
            # IMPORTANT:
            # No restrictive split_source_types filter here.
            # Let image-derived content also be chunked.
        )
        .caption(
            endpoint_url=os.environ.get(
                "VLM_CAPTION_ENDPOINT",
                "https://integrate.api.nvidia.com/v1/chat/completions"
            ),
            model_name=os.environ.get(
                "VLM_CAPTION_MODEL_NAME",
                "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
            ),
            api_key=NVIDIA_API_KEY,
        )
        .embed()
        .vdb_upload(
            collection_name=COLLECTION_NAME,
            milvus_uri=MILVUS_URI,
            sparse=SPARSE,
            dense_dim=2048
        )
    )

    print("Starting Step 2 full ingestion...")
    t0 = time.time()
    results_step2, failures_step2 = ingestor_full.ingest(
        show_progress=True,
        return_failures=True
    )
    t1 = time.time()

    print(f"Step 2 total time: {t1 - t0:.2f} seconds")
    print(f"Step 2 results : {len(results_step2)}")
    print(f"Step 2 failures: {len(failures_step2)}")

    if failures_step2:
        print_failures("STEP 2", failures_step2)

    print_result_preview("STEP 2", results_step2)
    inspect_debug_dir(DEBUG_DIR_STEP2)

    if not results_step2:
        raise SystemExit(
            "\nNo Step 2 results returned. Check OCR / page-elements / graphic-elements / table-structure endpoints."
        )

    print("\n=== STEP 2 SUCCEEDED ===")
    print(f"Embeddings stored in Milvus Lite: {MILVUS_URI}")
    print(f"Collection: {COLLECTION_NAME}")

    # ---------------------------------------------------------
    # STEP 3: INTERACTIVE Q&A
    # ---------------------------------------------------------
    llm_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY
    )

    interactive_qa_loop(llm_client)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}")
        sys.exit(1)

Show more lines
✅ Why this version is better
This version keeps the official library-mode startup unchanged (run_pipeline(...)), which is the documented way to launch the Ray-based libmode pipeline, and it continues to use the Ingestor Python API as intended.
nvidia
+2

It also checks for the exact multimodal NIM environment variables the service expects for readiness:

OCR_HTTP_ENDPOINT
YOLOX_HTTP_ENDPOINT
YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT
YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT
nvidia
+1
And, importantly for your image/license use case:

it enables text + tables + charts + images + infographics together for the image input,
it removes the restrictive split filter that was preventing image-derived content from being chunked,
and it saves raw extraction output so you can verify whether OCR / table / chart / infographic data was actually produced before embedding and retrieval. The Python API explicitly supports .save_to_disk(...), .extract(...), .split(...), .caption(...), and .vdb_upload(...).
github
+1
Also, chart extraction in NeMo Retriever relies on graphic-elements + OCR, and infographic extraction relies on OCR, so enabling all related extractors and ensuring those endpoint vars are present is important for image-heavy documents.
github
+1

✅ Run it like this
From the folder where your pipeline.py is located:




Shell
micromamba activate myenv

export NVIDIA_API_KEY="YOUR_KEY"

export OCR_HTTP_ENDPOINT="YOUR_WORKING_OCR_HTTP_ENDPOINT"
export OCR_INFER_PROTOCOL="http"

export YOLOX_HTTP_ENDPOINT="YOUR_WORKING_PAGE_ELEMENTS_HTTP_ENDPOINT"
export YOLOX_INFER_PROTOCOL="http"

export YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT="YOUR_WORKING_GRAPHIC_ELEMENTS_HTTP_ENDPOINT"
export YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL="http"

export YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT="YOUR_WORKING_TABLE_STRUCTURE_HTTP_ENDPOINT"
export YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL="http"

taskset -c 0-7 python pipeline.py

Show more lines
Those endpoint variables are exactly the kind of service wiring the libmode pipeline expects to discover and use for multimodal extraction.
nvidia
+1

✅ What to inspect after running
After Step 1 and Step 2, check:

./debug_step1_extract
./debug_step2_full
If those folders contain JSON results with OCR text / image metadata / chart or table content, then the pipeline is actually using the image-oriented extraction path. The Python API is built to let you inspect or save those ingestion results directly.
github
+1

If you want, next I can help you with one more very useful step:

I can give you a small endpoint smoke-test script
that checks each of your OCR / page-elements / graphic-elements / table-structure endpoints before you run the pipeline, so you know exactly which one is failing or unauthorized.

Sources
Favicon typeFavicon type








pdf
Oxford
You said:
so that means if i going to insert any type of file format inside this pipeline it will work inside those files let say too many diagrams/charts/images also this works by actiavting these nemotron models?fro this type of documents alaso it works that you are saying?also show me the demo that you prepared the pipeline with the example dont hallucinate be genuine no sugart coating cause it has to work perfectly and exactly dont say yes for my satisfaction check clearly and think too
Copilot said:
Copilot

Reasoning completed in 5 steps
Yes — but I want to be very precise and not overpromise:

Short, honest answer
No, this does not mean “every file format” will work perfectly.
What it does mean is:

For the file types that NeMo Retriever / nv-ingest supports (for example PDF, DOCX, PPTX, HTML, JPEG, PNG, TIFF, BMP, MP3, and some early-access video formats), your pipeline.py + the correct NIM endpoint wiring is the right architecture for multimodal extraction.
github
+1
For image-heavy or scan-heavy documents (licenses, scanned PDFs, charts, diagrams, tables inside images), you only get the good multimodal behavior if the right services are actually reachable: OCR, page-elements, graphic-elements, and table-structure. Those are explicitly separate services in the NeMo Retriever stack and readiness checks.
conda
+1
It will not be “perfect on every document” automatically. Results still depend on the document quality, whether the file type is supported, whether the endpoints are correctly configured and authorized, and whether your extraction settings actually allow image-derived content to flow into splitting/embedding/retrieval.
github
+1
So the genuine answer is:

Yes, this is the correct way to make image/table/chart/graphic extraction available in library mode — but no, I cannot honestly say it will be perfect on every document type or every low-quality image.

What will work, and what will not
What should work if configured correctly
If you keep the official nv-ingest 25.9.0 library-mode pipeline and correctly wire these endpoint variables:

OCR_HTTP_ENDPOINT
YOLOX_HTTP_ENDPOINT (page-elements)
YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT
YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT
github
+1
then you are enabling the intended multimodal pipeline path for:

scanned/image-based text via OCR,
charts and graphics via graphic-elements + OCR,
tables via table-structure + OCR,
layout/page regions via page-elements,
semantic image captioning via the caption stage in the Python API.
github
+2
That is exactly the missing part in the “70% works, but images/tables/charts do not” situation you described.

What will still not be guaranteed
Even with all of that enabled:

a blurry ID card photo can still OCR poorly,
tiny labels inside diagrams can still be missed,
a low-quality scan may still produce weak extraction,
and some documents will require OCR-focused extraction methods rather than native-text extraction. The docs explicitly distinguish pdfium (fast, native text only) from ocr and pdfium_hybrid for scanned/image-based pages.
github
+1
So the correct promise is:

This setup is the right setup to make those modalities work.
It is not a magic guarantee of perfect extraction on every file.

Your uploaded Oxford.pdf is actually a very good demo case
From the uploaded PDF, I can confirm it is a figure-accessibility guide titled “Making Figures Accessible! Journals Edition”, and it explicitly contains sections for:

Use of Colour
Colour Contrast
Line Graphs
Grouped Bar Graphs
Scatter Plots
Pie Charts
Diagrams
Alt Text
PDF type
Oxford
The file also clearly includes many figure/diagram/chart examples across the document, including sections on line graphs (pages 12–17), grouped bar graphs (pages 18–22), scatter plots (pages 23–27), pie charts (pages 28–32), diagrams (pages 33–38), and alt text examples (pages 39–47).
PDF type
Oxford

So if your pipeline is properly wired for multimodal extraction, this is exactly the kind of PDF where you should see better results than plain text-only extraction.

Demo: what I genuinely expect from the rewritten pipeline on Oxford.pdf
Below is the exact kind of demo I mean — not hand-wavy, but tied to the file you uploaded and the pipeline I prepared.

1) Example run setup
Terminal setup



Shell
micromamba activate myenv

export NVIDIA_API_KEY="YOUR_KEY"

export OCR_HTTP_ENDPOINT="YOUR_WORKING_OCR_HTTP_ENDPOINT"
export OCR_INFER_PROTOCOL="http"

export YOLOX_HTTP_ENDPOINT="YOUR_WORKING_PAGE_ELEMENTS_HTTP_ENDPOINT"
export YOLOX_INFER_PROTOCOL="http"

export YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT="YOUR_WORKING_GRAPHIC_ELEMENTS_HTTP_ENDPOINT"
export YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL="http"

export YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT="YOUR_WORKING_TABLE_STRUCTURE_HTTP_ENDPOINT"
export YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL="http"

taskset -c 0-7 python pipeline.py

Show more lines
Those are the exact endpoint categories the service checks for multimodal document-analysis readiness.
github
+1

2) Example input change for your demo
In your rewritten script, for this demo, use:




Python
INPUT_FILE = "Oxford.pdf"
or, if the file is in a different directory, use the correct full/relative path.

3) What Step 1 should show
With raw extraction debug turned on, the pipeline should at minimum recover the major text structure of the PDF, such as:

“Making Figures Accessible!” on the cover page,
the contents page with sections like “Use of Colour”, “Colour Contrast”, “Line Graphs”, “Grouped Bar Graphs”, “Scatter Plots”, “Pie Charts”, “Diagrams”, and “Alt Text”.
PDF type
Oxford
If Step 1 cannot even recover those headings cleanly, then the OCR / page-analysis path is not healthy enough yet.

4) What Step 2 should improve
With full multimodal extraction enabled (extract_text=True, extract_tables=True, extract_charts=True, extract_images=True, extract_infographics=True) and no restrictive split filter, I would expect retrieval to work for content like:

Example query 1
Question:
“What accessibility guidance does the document give for pie charts?”

Ground-truth content in the file:
The PDF says pie charts should either:

add divider lines and label the slices,
use patterns instead of just colour,
or use a table instead of a pie chart if differences are marginal.
PDF type
Oxford
If your pipeline is working correctly, retrieval should bring back those sections from pages 28–32.

Example query 2
Question:
“What WCAG contrast ratios does the guide recommend?”

Ground-truth content in the file:
The guide states:

3:1 minimum contrast for graphical/non-text objects,
4.5:1 for normal text,
3:1 for large text.
PDF type
Oxford
That is a very concrete retrieval check.

Example query 3
Question:
“What makes alt text good according to this document?”

Ground-truth content in the file:
The PDF says good alt text should be concise, consistent, unique, clear, relevant, simple, inclusive, standalone, and complete.
PDF type
Oxford

That is another clear validation query.

What I would accept as a “successful demo”
I would say the pipeline is behaving correctly on Oxford.pdf if you see all three of these:

✅ 1. Raw extraction output contains the expected section headings
For example:

“Making Figures Accessible!”
“Line Graphs”
“Pie Charts”
“Diagrams”
“Alt Text”
PDF type
Oxford
✅ 2. Full ingestion inserts more than a tiny number of usable chunks
If you again get something like just 1 element inserted into Milvus, that is a red flag for image-rich content. It suggests the multimodal path is still not contributing much.

✅ 3. Retrieval answers are grounded in actual file content
For example:

Pie chart guidance returns divider lines / patterns / table alternative,
PDF type
Oxford
contrast ratio answer returns 3:1 and 4.5:1,
PDF type
Oxford
alt text answer returns concise/clear/relevant/etc.
PDF type
Oxford
If that happens, then the pipeline is doing real retrieval over your ingested PDF, not just hallucinating.

What I would not claim yet
I would not honestly tell you:

“Now any file type will work perfectly,”
“All diagrams/charts/images will always be extracted exactly,”
or “Once you set the endpoints, everything is solved.”
That would be misleading.

The truthful version is:

Supported file types can go through the library-mode pipeline.
github
+1
Image-heavy/scanned docs require the right OCR + detector NIM wiring.
conda
+1
Your current issue is consistent with missing/ineffective multimodal endpoint use, not with the high-level Python API design itself.
nvidia
+1
Once you wire those NIMs and use the corrected extraction settings, you are finally giving the official 25.9.0 pipeline a fair chance to work properly on images/charts/tables/graphics.
github
+1
My clear recommendation now
Use this as your validation order
Phase 1 — Validate on Oxford.pdf
Why this file? Because it contains many chart/diagram/figure-oriented sections and lets you test retrieval against known text in the document.
PDF type
Oxford

Phase 2 — Validate on your license / image documents
Once Oxford.pdf behaves correctly, move to the image-heavy documents like licenses.
That is where OCR quality and endpoint coverage will matter the most. OCR and image/document-analysis services are the critical pieces there.
conda
+1

Bottom line
My genuine answer
Yes, your rewritten pipeline + correct NIM endpoint wiring is the right solution for making images/tables/charts/graphics work inside the official nv-ingest 25.9.0 library-mode flow.
nvidia
+2

But my equally genuine warning
No, I cannot honestly promise “perfect” results on every document or every file format.
I can say this is the correct architecture, the correct API usage pattern, and the right next implementation step for the exact failure mode you described.
github
+2

If you want, I can do one more concrete thing next:

I can give you an Oxford.pdf-specific test version of pipeline.py
with:

INPUT_FILE = "Oxford.pdf",
5 exact validation questions,
and the exact success criteria you should expect from the run.
Sources
Favicon typeFavicon typeFavicon type







You said:
so for thsoe grpc and http nim endpoinst should i need to paste the every model api key from the buil.nvidia.com and export them in bashor what exactly need to setup them in env and code and bash?
Copilot said:
Copilot

Reasoning completed in 3 steps
Yes — you do not need a separate API key for every model in the normal NVIDIA-hosted setup. In NVIDIA’s own guidance for hosted NIM/API Catalog usage, you typically use one NVIDIA API key and call different model endpoints with that same key, as long as your account/org has access to those services. NVIDIA’s blueprint docs explicitly say you need a single API key for accessing NIM services and that the same key can be used across different model API endpoints.
github
+1

The key and the endpoint URL are two different things:

API key = who you are / what you’re allowed to call.
github
+1
Endpoint URL = where that OCR / page-elements / graphic-elements / table-structure service is running.
github
+2
So the honest answer is:

No, don’t paste a different Build API key for every model.
Usually you export one NVIDIA_API_KEY, and then export the correct endpoint URLs for the NIMs you want the pipeline to use.
github
+1

Exactly what you need to set
There are 2 possible setups, and this is the important distinction.

Case A — You are using NVIDIA-hosted endpoints (cloud URLs like https://ai.api.nvidia.com/...)
In this case:

you usually need one NVIDIA_API_KEY,
github
+1
you export the HTTP endpoint URLs for OCR / page-elements / graphic-elements / table-structure,
github
+2
and you set the protocol to http for those cloud endpoints. The readiness/config logic in NeMo Retriever expects env vars like OCR_HTTP_ENDPOINT, YOLOX_HTTP_ENDPOINT, YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT, and YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT.
github
+1
In this setup, do this in Bash



Shell
export NVIDIA_API_KEY="YOUR_ONE_NVIDIA_API_KEY"

# OCR
export OCR_HTTP_ENDPOINT="YOUR_OCR_HTTP_URL"
export OCR_INFER_PROTOCOL="http"

# Page elements
export YOLOX_HTTP_ENDPOINT="YOUR_PAGE_ELEMENTS_HTTP_URL"
export YOLOX_INFER_PROTOCOL="http"

# Graphic elements
export YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT="YOUR_GRAPHIC_ELEMENTS_HTTP_URL"
export YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL="http"

# Table structure
export YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT="YOUR_TABLE_STRUCTURE_HTTP_URL"
export YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL="http"

Show more lines
Then your Python code can stay as:




Python
assert "NVIDIA_API_KEY" in os.environ
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
and run_pipeline(...) will pick up the endpoint env vars from the shell environment when it starts the library-mode pipeline. The library-mode docs say run_pipeline(...) is the primary entry point, and the default libmode pipeline is loaded automatically when you do not pass a custom pipeline config.
nvidia
+1

Case B — You are using self-hosted / locally running NIMs
If your OCR / graphic-elements / table-structure services are running on your own VM, Docker, or another internal machine, then the pipeline env vars should point to your local or internal service URLs, not NVIDIA cloud URLs. The object-detection and OCR NIM docs show that these services expose inference endpoints like /v1/infer when self-hosted, typically on your own host/port.
nvidia
+1

In this setup, do this in Bash



Shell
export NVIDIA_API_KEY="YOUR_ONE_NVIDIA_API_KEY"
# This may still be needed if your pipeline's caption/embed steps use NVIDIA-hosted endpoints.

# OCR running locally
export OCR_HTTP_ENDPOINT="http://localhost:8010"
export OCR_INFER_PROTOCOL="http"

# Page elements running locally
export YOLOX_HTTP_ENDPOINT="http://localhost:8000"
export YOLOX_INFER_PROTOCOL="http"

# Graphic elements running locally
export YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT="http://localhost:8004"
export YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL="http"

# Table structure running locally
export YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT="http://localhost:8007"
export YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL="http"

Show more lines
If the services are remote on another host, replace localhost with that host/IP.

Important
For self-hosted NIMs, the NVIDIA_API_KEY is generally not a “per-model runtime key” for your local endpoint. It is usually used to pull/run hosted NVIDIA services or for other NVIDIA-hosted calls (such as caption/embedding if you are still using NVIDIA-hosted APIs). Your own local endpoint URL is what the pipeline talks to during extraction. NVIDIA’s “run anywhere” and deploy docs distinguish between using NVIDIA-hosted endpoints for prototyping and self-hosting downloadable NIMs on your own infrastructure.
deepwiki
+1

Do you need to set gRPC too?
Short answer:
Only if you are actually using gRPC endpoints.
If you are calling cloud URLs like https://ai.api.nvidia.com/..., you should normally use the HTTP endpoint env vars and set *_INFER_PROTOCOL=http. The helper/config pattern in NeMo Retriever supports endpoint pairs (grpc, http) plus an infer_protocol, so you only need gRPC values if your deployment really exposes gRPC and you want the pipeline to use it.
nvidia
+1

If you are not using gRPC
You can leave the gRPC env vars unset.

If you are using gRPC
Then you would also export things like:




Shell
export OCR_GRPC_ENDPOINT="your-grpc-host:port"
export YOLOX_GRPC_ENDPOINT="your-grpc-host:port"
export YOLOX_GRAPHIC_ELEMENTS_GRPC_ENDPOINT="your-grpc-host:port"
export YOLOX_TABLE_STRUCTURE_GRPC_ENDPOINT="your-grpc-host:port"
and switch the corresponding *_INFER_PROTOCOL to grpc. The libmode config pattern clearly expects both gRPC and HTTP endpoint slots.
nvidia

What to put in code vs bash
Put this in Bash / shell
NVIDIA_API_KEY
OCR_HTTP_ENDPOINT
YOLOX_HTTP_ENDPOINT
YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT
YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT
*_INFER_PROTOCOL values
github
+1
That way, your pipeline is configurable without hardcoding secrets and URLs into Python.

Keep this in Python
Your Python should only read the API key from the environment and run the official libmode pipeline:




Python
assert "NVIDIA_API_KEY" in os.environ, "Set env: export NVIDIA_API_KEY=..."
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]

config = PipelineCreationSchema()
run_pipeline(
    config,
    block=False,
    disable_dynamic_scaling=True,
    run_in_subprocess=True
)
That is the official library-mode pattern in the docs.
nvidia
+1

What you should not do
❌ Do not do this
do not paste a different API key for OCR, another for graphic-elements, another for table-structure, etc., unless your org has some very unusual internal auth design. NVIDIA’s public guidance is built around a single API key for hosted service access.
github
+1
do not hardcode many keys directly into your Python if you can avoid it.
do not assume gRPC is required just because the config supports it. Use HTTP if your working endpoint tests are HTTP.
nvidia
+2
The exact setup I recommend for you
Since you said the separate NIMs are already working well, the best practical move is:

1) Reuse one API key



Shell
export NVIDIA_API_KEY="YOUR_ONE_WORKING_KEY"
2) Export the actual endpoint URLs you already tested successfully
Use the exact URLs/hosts that worked in your standalone tests:




Shell
export OCR_HTTP_ENDPOINT="PASTE_THE_OCR_URL_YOU_TESTED"
export OCR_INFER_PROTOCOL="http"

export YOLOX_HTTP_ENDPOINT="PASTE_THE_PAGE_ELEMENTS_URL_YOU_TESTED"
export YOLOX_INFER_PROTOCOL="http"

export YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT="PASTE_THE_GRAPHIC_ELEMENTS_URL_YOU_TESTED"
export YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL="http"

export YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT="PASTE_THE_TABLE_STRUCTURE_URL_YOU_TESTED"
export YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL="http"
3) Run your pipeline



Shell
taskset -c 0-7 python pipeline.py
That is the cleanest and most correct approach for your current situation. The NeMo Retriever stack is designed to discover those services through env vars, and the library-mode pipeline startup is already correct in your script.
github
+2

A small concrete example
If your setup is:

OCR on http://localhost:8010
page-elements on http://localhost:8000
graphic-elements on http://localhost:8004
table-structure on http://localhost:8007
then do exactly this:




Shell
micromamba activate myenv

export NVIDIA_API_KEY="nvapi-REPLACE_WITH_YOURS"

export OCR_HTTP_ENDPOINT="http://localhost:8010"
export OCR_INFER_PROTOCOL="http"

export YOLOX_HTTP_ENDPOINT="http://localhost:8000"
export YOLOX_INFER_PROTOCOL="http"

export YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT="http://localhost:8004"
export YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL="http"

export YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT="http://localhost:8007"
export YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL="http"

taskset -c 0-7 python pipeline.py

Show more lines
If instead they are NVIDIA-hosted URLs, replace only the URLs, not the logic.

Final bottom line
The exact truthful answer
No, you do not normally need one Build API key per model.
You usually need one NVIDIA_API_KEY, and then you must export the correct NIM endpoint URLs and protocols so the library-mode pipeline knows where OCR, page-elements, graphic-elements, and table-structure are running.
github
+2

For you specifically
The best setup is:

one API key in env,
one endpoint URL per service in env,
protocol = http unless you are truly using gRPC,
keep your Python code mostly unchanged.
nvidia
+2
