thing is as you shared the script of nv ingest on feb that was cleanly working and i also added a rag agent top of it 
so when we are connected with the abrav right he provided the public ID for input so where the Nemotron Models comes to picture - graphics elements/table structure/ocr V1 so these are not integrated to the script that you have shared thats a library model where suites for 100 pdfs OCR will run backend for images and attach the text as caption for that imgaes the script wont delas with the images/infographics/tables layout and pixels of it 
 
we really need those Nemotron models to run then only it satisfies exactly as architecture shows in documentation so to run those we need GPU and need to pull those and configure it dedicatedly need 8-12GB for each model to run 
then it satifies the NV ingest whole architetcure

this message was shared to my lead 
basically the nv ingest 25.9.0 version of library mode exactly the code i was ran is library mode script which is
import logging, os, time

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
    print("\nNo results and no failures — unexpected state.")

this code so as i given the demo of this script with the rag agent combining of both the images that means driving lisence like that we have provided as input but it was not giving the exact ouput i was worked that images seperatly with the VLM model then it is working 

so they said that you are doing something worng that means the architecture itself is allowing the nemotron models like page elements/infographics/ocr and table structure V1 right why it is not applicable for this

so that whole architetcure is not works in library mode actually casue to run those models we need to run those models in docker or helm through GPU without gpu only the above script wont able to run those architetcure i already pasted the architetcure 

these are my finding so i was explained them clearly to run on images/charts/infographics we need to access those nemotron models which not supported for the above script that script is jsut a frontend part which suits for 100pdfs thats it without gpu released by nvidia

so they said exactly - okay. please document the findings
 so i need to document the findings so task is show me exactly where these mentioned in nvidia or any documentation so that i can say this is the proof
this findings never listed in any documentation so find those and help me out
