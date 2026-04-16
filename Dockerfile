taskset -c 0-7 python vlm_nvingest.py
2026-04-16 05:47:42.102812902 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/MSFT1000:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
INFO:nv_ingest_api.util.system.hardware_info:Detected 32 logical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 16 physical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 8 cores via os.sched_getaffinity.
INFO:nv_ingest_api.util.system.hardware_info:Raw CPU limit determined: 8.00 (Method: sched_affinity)
INFO:nv_ingest_api.util.system.hardware_info:Effective CPU core limit determined: 8.00 (Method: sched_affinity)
Starting nv-ingest pipeline...
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Launching pipeline in Python subprocess using multiprocessing.
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Pipeline subprocess started (PID=52562)
Pipeline ready.

=== STEP 1: Extracting document ===
Processing:   0%|                                                                       | 0/1 [00:00<?, ?doc/s]INFO:nv_ingest_client.client.client:Starting batch processing for 1 jobs with batch size 32.
Processing: 100%|███████████████████████████████████████████████████████████████| 1/1 [00:31<00:00, 31.11s/doc]INFO:nv_ingest_client.client.client:Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
Processing: 100%|███████████████████████████████████████████████████████████████| 1/1 [00:31<00:00, 31.11s/doc]
Extracted 1 result(s).

=== STEP 2: VLM captioning all visuals ===
  VLM [image] page 0 ... INFO:httpx:HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
✓
Total chunks after VLM pass: 1

=== STEP 3: Embedding & uploading to Milvus ===
INFO:httpx:HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 400 Bad Request"
Traceback (most recent call last):
  File "/home/clouduser01/jaswanth/vlm_nvingest.py", line 161, in <module>
    vecs    = embed_texts(texts)
              ^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/jaswanth/vlm_nvingest.py", line 50, in embed_texts
    resp = nvidia.embeddings.create(model=EMBED_MODEL, input=texts, encoding_format="float")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/openai/resources/embeddings.py", line 136, in create
    return self._post(
           ^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/openai/_base_client.py", line 1297, in post
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/openai/_base_client.py", line 1070, in request
    raise self._make_status_error_from_response(err.response) from None
openai.BadRequestError: Error code: 400 - {'error': "'input_type' parameter is required for asymmetric models"}
Killed subprocess group 52562
E20260416 05:48:52.168097 59144 server.cpp:47] [SERVER][BlockLock][milvus] Process exit
^C
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ 
