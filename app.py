 taskset -c 0-7 python pipeline2.py
2026-03-27 05:35:16.879869695 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/VMBUS:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
INFO:nv_ingest_api.util.system.hardware_info:Detected 32 logical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 16 physical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 8 cores via os.sched_getaffinity.
INFO:nv_ingest_api.util.system.hardware_info:Raw CPU limit determined: 8.00 (Method: sched_affinity)
INFO:nv_ingest_api.util.system.hardware_info:Effective CPU core limit determined: 8.00 (Method: sched_affinity)
INFO:__main__:Created collection
INFO:__main__:Ingesting: ['./Docs/minion-tech.pdf']
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Launching pipeline in Python subprocess using multiprocessing.
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Pipeline subprocess started (PID=2613993)
INFO:__main__:Pipeline started...
INFO:__main__:Waiting for broker localhost:7671…
INFO:__main__:Broker ready!
Processing:   0%|                                                                                  | 0/1 [00:00<?, ?doc/s]INFO:nv_ingest_client.client.client:Starting batch processing for 1 jobs with batch size 32.
Processing: 100%|██████████████████████████████████████████████████████████████████████████| 1/1 [01:03<00:00, 63.67s/doc]INFO:nv_ingest_client.client.client:Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
Processing: 100%|██████████████████████████████████████████████████████████████████████████| 1/1 [01:03<00:00, 63.68s/doc]
INFO:nv_ingest_client.client.util.processing:Saved 42 extraction items for './Docs/minion-tech.pdf' to ./temp_ingest/minion-tech.pdf.results.jsonl
INFO:nv_ingest_client.util.vdb.milvus:40 elements to insert to milvus
INFO:nv_ingest_client.util.vdb.milvus:threshold for streaming is 1000
INFO:nv_ingest_client.util.vdb.milvus:streamed 40 records
INFO:nv_ingest_client.client.interface:Purging saved results from disk after successful VDB upload.
INFO:nv_ingest_client.client.interface:Purged 1 saved result file(s).
INFO:__main__:Extracted 0 chunks → Milvus

Ingested 0 chunks

Q: Using the balance sheet and profit & loss statement, calculate the debt-to-equity ratio and return on equity. Is the company financially healthy?
Traceback (most recent call last):
  File "/home/clouduser01/jaswanth/pipeline2.py", line 207, in <module>
    print(f"A: {rag_chatbot(q)}")
                ^^^^^^^^^^^^^^
  File "/home/clouduser01/jaswanth/pipeline2.py", line 151, in rag_chatbot
    ctx = retrieve(query)
          ^^^^^^^^^^^^^^^
  File "/home/clouduser01/jaswanth/pipeline2.py", line 147, in retrieve
    return [h.entity.get("text") for h in hits]
            ^^^^^^^^
AttributeError: 'dict' object has no attribute 'entity'
Killed subprocess group 2613993
E20260327 05:36:35.269445 2613929 server.cpp:47] [SERVER][BlockLock][milvus] Process exit
