taskset -c 0-7 streamlit run rag_agent.py --server.port 8501

Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.


  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://10.198.133.186:8501
  External URL: http://20.193.234.109:8501

2026-04-01 12:49:12.708345110 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/MSFT1000:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
INFO:nv_ingest_api.util.system.hardware_info:Detected 32 logical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 16 physical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 8 cores via os.sched_getaffinity.
INFO:nv_ingest_api.util.system.hardware_info:Raw CPU limit determined: 8.00 (Method: sched_affinity)
INFO:nv_ingest_api.util.system.hardware_info:Effective CPU core limit determined: 8.00 (Method: sched_affinity)
INFO:rag_agent:Milvus collection exists: rag_documents
Ingesting: ./Docs/minion-tech.pdf
2026-04-01 12:49:17,146 WARNING worker.py:1659 -- SIGTERM handler is not set because current thread is not the main thread.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1775047757.224522  252716 fork_posix.cc:77] Other threads are currently calling into gRPC, skipping fork() handlers
2026-04-01 12:49:25,651 INFO worker.py:2004 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265 
/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/ray/_private/worker.py:2052: FutureWarning: Tip: In future versions of Ray, Ray will no longer override accelerator visible devices env var if num_gpus=0 or num_gpus=None (default). To enable this behavior and turn off this error message, set RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
  warnings.warn(
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Launching pipeline in Python subprocess using multiprocessing.
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Pipeline subprocess started (PID=254674)
INFO:rag_agent:NV-Ingest pipeline started
INFO:rag_agent:Waiting for broker localhost:7671 …
INFO:rag_agent:Broker ready!
Processing:   0%|                                                                       | 0/1 [00:00<?, ?doc/s]INFO:nv_ingest_client.client.client:Starting batch processing for 1 jobs with batch size 32.
Ingesting: ./Docs/minion-tech.pdf
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Launching pipeline in Python subprocess using multiprocessing.
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Pipeline subprocess started (PID=259114)
INFO:rag_agent:NV-Ingest pipeline started
INFO:rag_agent:Waiting for broker localhost:7671 …
INFO:rag_agent:Broker ready!
                                                                                                              INFO:nv_ingest_client.client.client:Starting batch processing for 1 jobs with batch size 32./1 [00:00<?, ?doc/s]
Ingesting: ./Docs/minion-tech.pdf
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Launching pipeline in Python subprocess using multiprocessing.
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Pipeline subprocess started (PID=259722)
INFO:rag_agent:NV-Ingest pipeline started
INFO:rag_agent:Waiting for broker localhost:7671 …
INFO:rag_agent:Broker ready!
                                                                                                              INFO:nv_ingest_client.client.client:Starting batch processing for 1 jobs with batch size 32.
Processi^BIngesting: ./Docs/minion-tech.pdf
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Launching pipeline in Python subprocess using multiprocessing.
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Pipeline subprocess started (PID=261591)
INFO:rag_agent:NV-Ingest pipeline started
INFO:rag_agent:Waiting for broker localhost:7671 …
INFO:rag_agent:Broker ready!
                                   INFO:nv_ingest_client.client.client:Starting batch processing for 1 jobs with batch size 32. 0/1 [00:00<?, ?d
^C  Stopping...
ERROR:nv_ingest_client.client.client:fetch_job_result_async failed for batch (1 jobs): cannot schedule new futures after shutdown
Traceback (most recent call last):
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/client/client.py", line 404, in run
    batch_futures_dict = self.client.fetch_job_result_async(current_batch_job_indices, data_only=False)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/client/client.py", line 1159, in fetch_job_result_async
    future = self._worker_pool.submit(self.fetch_job_result_cli, job_id, data_only)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/concurrent/futures/thread.py", line 171, in submit
    raise RuntimeError('cannot schedule new futures after shutdown')
RuntimeError: cannot schedule new futures after shutdown
WARNING:nv_ingest_client.client.client:Marking all 1 jobs in failed fetch initiation batch as failed.
ERROR:nv_ingest_client.client.client:Initiation failed for 0: Fetch initiation failed for batch: cannot schedule new futures after shutdown
INFO:nv_ingest_client.client.client:Batch processing finished. Success: 0, Failures: 1. Total accounted for: 1/1
Processing:   0%|                                                                       | 0/1 [06:21<?, ?doc/s]
WARNING:nv_ingest_client.client.interface:Job was not completely successful. 0 out of 1 records completed successfully. Uploading successful results to vector database.
{
  "files": [
    "minion-tech.pdf"
  ],
  "chunks_uploaded": 0,
  "failures": 1,
  "elapsed_ms": 381347
}

============================================================
Q: Using the balance sheet and profit & loss statement, calculate the debt-to-equity ratio and return on equity. Is the company financially healthy?

ERROR:nv_ingest_client.client.client:fetch_job_result_async failed for batch (1 jobs): cannot schedule new futures after shutdown
Traceback (most recent call last):
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/client/client.py", line 404, in run
    batch_futures_dict = self.client.fetch_job_result_async(current_batch_job_indices, data_only=False)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/client/client.py", line 1159, in fetch_job_result_async
    future = self._worker_pool.submit(self.fetch_job_result_cli, job_id, data_only)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/concurrent/futures/thread.py", line 171, in submit
    raise RuntimeError('cannot schedule new futures after shutdown')
RuntimeError: cannot schedule new futures after shutdown
WARNING:nv_ingest_client.client.client:Marking all 1 jobs in failed fetch initiation batch as failed.
ERROR:nv_ingest_client.client.client:Initiation failed for 0: Fetch initiation failed for batch: cannot schedule new futures after shutdown
INFO:nv_ingest_client.client.client:Batch processing finished. Success: 0, Failures: 1. Total accounted for: 1/1
Processing:   0%| | 0/1 [02:07<?, ?d
WARNING:nv_ingest_client.client.interface:Job was not completely successful. 0 out of 1 records completed successfully. Uploading successful results to vector database.
{
  "files": [
    "minion-tech.pdf"
  ],
  "chunks_uploaded": 0,
  "failures": 1,
  "elapsed_ms": 127308
}

============================================================
Q: Using the balance sheet and profit & loss statement, calculate the debt-to-equity ratio and return on equity. Is the company financially healthy?

INFO:rag_agent:Classifier → type=calculation top_k=20 flags=[] [686.5ms]
ERROR:nv_ingest_client.client.client:fetch_job_result_async failed for batch (1 jobs): cannot schedule new futures after shutdown
Traceback (most recent call last):
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/client/client.py", line 404, in run
    batch_futures_dict = self.client.fetch_job_result_async(current_batch_job_indices, data_only=False)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/client/client.py", line 1159, in fetch_job_result_async
    future = self._worker_pool.submit(self.fetch_job_result_cli, job_id, data_only)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/concurrent/futures/thread.py", line 171, in submit
    raise RuntimeError('cannot schedule new futures after shutdown')
RuntimeError: cannot schedule new futures after shutdown
WARNING:nv_ingest_client.client.client:Marking all 1 jobs in failed fetch initiation batch as failed.
ERROR:nv_ingest_client.client.client:Initiation failed for 0: Fetch initiation failed for batch: cannot schedule new futures after shutdown
INFO:nv_ingest_client.client.client:Batch processing finished. Success: 0, Failures: 1. Total accounted for: 1/1
Processing:   0%|                                                                       | 0/1 [16:56<?, ?doc/s]
WARNING:nv_ingest_client.client.interface:Job was not completely successful. 0 out of 1 records completed successfully. Uploading successful results to vector database.
{
  "files": [
    "minion-tech.pdf"
  ],
  "chunks_uploaded": 0,
  "failures": 1,
  "elapsed_ms": 1016392
}

============================================================
Q: Using the balance sheet and profit & loss statement, calculate the debt-to-equity ratio and return on equity. Is the company financially healthy?

INFO:rag_agent:Classifier → type=calculation top_k=20 flags=[] [432.7ms]
