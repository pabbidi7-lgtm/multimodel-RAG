import logging, os, time

import pymilvus
import ray
ray.init(num_cpus=8)

pymilvus.connections.disconnect("default")


from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import (
    run_pipeline,
    PipelineCreationSchema
)

from nv_ingest_api.util.logging.configuration import configure_logging as configure_local_logging
from nv_ingest_client.client import Ingestor, NvIngestClient
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.util.process_json_files import ingest_json_results_to_blob

# ------------ CONFIG ------------
# Make sure your key is set before running.
assert "NVIDIA_API_KEY" in os.environ, "Set env: export NVIDIA_API_KEY=..."

# Pipeline configuration
config = PipelineCreationSchema()

# Start the pipeline subprocess for library mode
run_pipeline(
    config,
    block=False,
    disable_dynamic_scaling=True,
    run_in_subprocess=True
)

client = NvIngestClient(
    message_client_allocator=SimpleClient,
    message_client_port=7671,
    message_client_hostname="localhost"
)

milvus_uri = "milvus.db"
collection_name = "test"
sparse = False

# Ingestor pipeline
ingestor = (
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

print("Starting ingestion..")
t0 = time.time()

results, failures = ingestor.ingest(show_progress=True, return_failures=True)

t1 = time.time()
print(f"Total time: {t1 - t0} seconds")

print(ingest_json_results_to_blob(results[0]))

if failures:
    print(f"There were {len(failures)} failures. Sample: {failures[0]}")



(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ taskset -c 0-7 python pipeline.py
2026-03-26 15:25:30,222 INFO worker.py:2004 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265 
/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/ray/_private/worker.py:2052: FutureWarning: Tip: In future versions of Ray, Ray will no longer override accelerator visible devices env var if num_gpus=0 or num_gpus=None (default). To enable this behavior and turn off this error message, set RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
  warnings.warn(
2026-03-26 15:25:37.207290829 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/VMBUS:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
INFO:nv_ingest_api.util.system.hardware_info:Detected 32 logical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 16 physical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 8 cores via os.sched_getaffinity.
INFO:nv_ingest_api.util.system.hardware_info:Raw CPU limit determined: 8.00 (Method: sched_affinity)
INFO:nv_ingest_api.util.system.hardware_info:Effective CPU core limit determined: 8.00 (Method: sched_affinity)
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Launching pipeline in Python subprocess using multiprocessing.
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Pipeline subprocess started (PID=2221496)
Starting ingestion..
Processing:   0%|                                                                            | 0/1 [00:00<?, ?doc/s]INFO:nv_ingest_client.client.client:Starting batch processing for 1 jobs with batch size 32.



NVIDIA_API_KEY =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
