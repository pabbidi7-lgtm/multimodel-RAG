import logging, os, time

import pymilvus
import ray

# ------------ CONFIG ------------
assert "NVIDIA_API_KEY" in os.environ, "Set env: export NVIDIA_API_KEY=..."

# DO NOT call ray.init() manually — pipeline manages Ray internally
pymilvus.connections.disconnect("default")

from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import (
    run_pipeline,
    PipelineCreationSchema
)
from nv_ingest_api.util.logging.configuration import configure_logging as configure_local_logging
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

# Pipeline configuration
config = PipelineCreationSchema()

# Start the pipeline subprocess
run_pipeline(
    config,
    block=False,
    disable_dynamic_scaling=True,
    run_in_subprocess=True
)

# Wait for pipeline subprocess to fully initialize before sending jobs
print("Waiting for pipeline to initialize...")
time.sleep(15)
print("Pipeline ready. Connecting client...")

client = NvIngestClient(
    message_client_allocator=SimpleClient,
    message_client_port=7671,
    message_client_hostname="localhost"
)

milvus_uri = "milvus.db"
collection_name = "test"
sparse = False

# -------------------------------------------------------
# STEP 1: Minimal extraction first (text only, no embed/vdb)
# If this works, move to STEP 2
# -------------------------------------------------------
print("\n=== STEP 1: Testing basic text extraction ===")

ingestor = (
    Ingestor(client=client)
    .files("Docs/PK0016.pdf")
    .extract(
        extract_text=True,
        extract_tables=False,
        extract_charts=False,
        extract_images=False,
        extract_infographics=False,
        text_depth="page",
    )
    # No .embed() or .vdb_upload() yet
)

print("Starting ingestion...")
t0 = time.time()
results, failures = ingestor.ingest(show_progress=True, return_failures=True)
t1 = time.time()
print(f"Total time: {t1 - t0:.2f} seconds")

# Detailed failure reporting
print(f"\nResults:  {len(results)}")
print(f"Failures: {len(failures)}")

if failures:
    print("\n=== FAILURE DETAILS ===")
    for i, f in enumerate(failures):
        print(f"\n--- Failure [{i}] ---")
        print(f)
    print("\nFix the above failure before proceeding to STEP 2.")

elif results:
    print("\n=== STEP 1 SUCCEEDED ===")
    print(ingest_json_results_to_blob(results[0]))

    # -------------------------------------------------------
    # STEP 2: Full extraction + embed + vdb upload
    # Only runs if STEP 1 succeeded
    # -------------------------------------------------------
    print("\n=== STEP 2: Full extraction + embed + vdb upload ===")

    ingestor_full = (
        Ingestor(client=client)
        .files("Docs/PK0016.pdf")
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=True,
            extract_infographics=True,
            table_output_format="markdown",
            text_depth="page",
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
        print("\n=== FULL PIPELINE FAILURE DETAILS ===")
        for i, f in enumerate(failures_full):
            print(f"\n--- Failure [{i}] ---")
            print(f)
    else:
        print("\n=== STEP 2 SUCCEEDED ===")
        print(ingest_json_results_to_blob(results_full[0]))

else:
    print("\nNo results and no failures returned — unexpected state.")
    print("Check if the pipeline subprocess is still alive and port 7671 is reachable.")
