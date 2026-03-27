taskset -c 0-7 python pipeline.py
2026-03-27 02:38:57.407275009 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/VMBUS:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
INFO:nv_ingest_api.util.system.hardware_info:Detected 32 logical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 16 physical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 8 cores via os.sched_getaffinity.
INFO:nv_ingest_api.util.system.hardware_info:Raw CPU limit determined: 8.00 (Method: sched_affinity)
INFO:nv_ingest_api.util.system.hardware_info:Effective CPU core limit determined: 8.00 (Method: sched_affinity)
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Launching pipeline in Python subprocess using multiprocessing.
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Pipeline subprocess started (PID=2499246)
Waiting for pipeline to initialize...
Pipeline ready. Connecting client...

=== STEP 1: Basic text extraction ===
Starting ingestion...
Processing:   0%|                                                                                  | 0/1 [00:00<?, ?doc/s]INFO:nv_ingest_client.client.client:Starting batch processing for 1 jobs with batch size 32.
Processing: 100%|██████████████████████████████████████████████████████████████████████████| 1/1 [00:31<00:00, 31.13s/doc]INFO:nv_ingest_client.client.client:Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
Processing: 100%|██████████████████████████████████████████████████████████████████████████| 1/1 [00:31<00:00, 31.13s/doc]
Total time: 31.13 seconds

Results:  1
Failures: 0

=== STEP 1 SUCCEEDED ===
Urine R/M Urine Sample
Accession No: DEMO_BARCODE Collected On: 21-Jan-25 13:40 Received On: 21-Jan-25 14:31 Approved On: 21-Jan-25 17:23
Observation Result Unit Biological Ref. Interval Method
Physical Examination
Urine Quantity 7.5 mL 7 - 8 Physical Examination
Urine Colour Pale Yellow Pale Yellow Physical Examination
Urinary Transparency Clear Clear Physical Examination
Biochemical Examination
Urinary pH 5.5 pH 6 .0 - 8.0 pH bromothymol blue
Urinary Specific Gravity 1.025 1.005 - 1.0...

=== STEP 2: Full pipeline (extract + split + caption + embed + vdb) ===
Starting full ingestion...
Processing:   0%|                                                                                  | 0/1 [00:00<?, ?doc/s]INFO:nv_ingest_client.client.client:Starting batch processing for 1 jobs with batch size 32.
Processing: 100%|██████████████████████████████████████████████████████████████████████████| 1/1 [00:31<00:00, 31.17s/doc]INFO:nv_ingest_client.client.client:Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
Processing: 100%|██████████████████████████████████████████████████████████████████████████| 1/1 [00:31<00:00, 31.17s/doc]
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1774579216.544735 2499106 dns_resolver_ares.cc:358] no server name supplied in dns URI
E0000 00:00:1774579216.544831 2499106 legacy_channel.cc:89] channel stack builder failed: UNKNOWN: the target uri is not valid: dns:///
^CTraceback (most recent call last):
  File "/home/clouduser01/jaswanth/pipeline.py", line 118, in <module>
    results_full, failures_full = ingestor_full.ingest(show_progress=True, return_failures=True)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/client/interface.py", line 594, in ingest
    self._vdb_bulk_upload.run(results)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/util/vdb/milvus.py", line 1946, in run
    self.create_index(collection_name=collection_name, **create_params)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/util/vdb/milvus.py", line 1908, in create_index
    return create_nvingest_collection(collection_name, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/util/vdb/milvus.py", line 447, in create_nvingest_collection
    client = MilvusClient(milvus_uri)
             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/pymilvus/milvus_client/milvus_client.py", line 88, in __init__
    self._handler = self._manager.get_or_create(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/pymilvus/client/connection_manager.py", line 472, in get_or_create
    return self._create_shared(config, client, timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/pymilvus/client/connection_manager.py", line 496, in _create_shared
    handler._wait_for_channel_ready(timeout=timeout)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/pymilvus/client/grpc_handler.py", line 225, in _wait_for_channel_ready
    grpc.channel_ready_future(self._channel).result(timeout=timeout)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/grpc/_utilities.py", line 162, in result
    self._block(timeout)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/grpc/_utilities.py", line 102, in _block
    self._condition.wait()
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/threading.py", line 355, in wait
    waiter.acquire()
KeyboardInterrupt
Killed subprocess group 2499246
^C
