import logging, os, time

from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline
from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import PipelineCreationSchema
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

# Start the pipeline subprocess for library mode
config = PipelineCreationSchema()
run_pipeline(config, block=False, disable_dynamic_scaling=True, run_in_subprocess=True)

client = NvIngestClient(
    message_client_allocator=SimpleClient,
    message_client_port=7671,
    message_client_hostname="localhost"
)

milvus_uri = "milvus.db"
collection_name = "medical_docs"
sparse = False

# do content extraction from files
ingestor = (
    Ingestor(client=client)
    .files("Docs/PK0016.pdf")
    .extract(
        extract_text=True,
        extract_tables=True,
        extract_charts=True,
        extract_images=True,
        table_output_format="markdown",
        extract_infographics=True,
        text_depth="page"
    ).embed()
    .vdb_upload(
        collection_name=collection_name,
        milvus_uri=milvus_uri,
        sparse=sparse,
        dense_dim=2048
    )
)

print("Starting ingestion..")
t0 = time.time()
results, failures = ingestor.ingest(show_progress=True, return_failures=True)
t1 = time.time()
print(f"Total time: {t1 - t0} seconds")

print(ingest_json_results_to_blob(results[0]))

if failures:
    print(f"There were {len(failures)} failures. Sample: {failures[0]}")

# =========================================================================
#  RETRIEVAL + RAG
# =========================================================================
from openai import OpenAI
from nv_ingest_client.util.milvus import nvingest_retrieval

queries = [
    "What are all the test results that are outside the normal biological reference interval?",
    "Based on the kidney function test and eGFR classification table, what is the patient's GFR category?",
    "What is the patient's HbA1c value and is this prediabetic or diabetic per ADA guidelines?",
    "Summarize the ultrasound whole abdomen findings and what tests were advised?",
    "What are the lipid profile results and classify each as optimal, borderline high, or high?",
]

llm_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

print("\n" + "=" * 60)
for q in queries:
    retrieved_docs = nvingest_retrieval(
        [q],
        collection_name,
        milvus_uri=milvus_uri,
        hybrid=sparse,
        top_k=10,
    )

    extract = "\n\n".join([doc["entity"]["text"] for doc in retrieved_docs[0]])

    prompt = f"Using the following content: {extract}\n\n Answer the user query: {q}"

    completion = llm_client.chat.completions.create(
        model="meta/llama-3.3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )

    print(f"\nQ: {q}")
    print(f"A: {completion.choices[0].message.content}")
    print("-" * 60)
