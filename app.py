 taskset -c 0-7 python pipeline.py
2026-03-26 15:46:07.818713136 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/VMBUS:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
INFO:nv_ingest_api.util.system.hardware_info:Detected 32 logical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 16 physical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 8 cores via os.sched_getaffinity.
INFO:nv_ingest_api.util.system.hardware_info:Raw CPU limit determined: 8.00 (Method: sched_affinity)
INFO:nv_ingest_api.util.system.hardware_info:Effective CPU core limit determined: 8.00 (Method: sched_affinity)
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Launching pipeline in Python subprocess using multiprocessing.
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Pipeline subprocess started (PID=2231322)
Starting ingestion..
Processing:   0%|                                                                            | 0/1 [00:00<?, ?doc/s]INFO:nv_ingest_client.client.client:Starting batch processing for 1 jobs with batch size 32.
Processing: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:31<00:00, 31.12s/doc]INFO:nv_ingest_client.client.client:Batch processing finished. Success: 0, Failures: 1. Total accounted for: 1/1
Processing: 100%|████████████████████████████████████████████████████████████████████| 1/1 [00:31<00:00, 31.12s/doc]
WARNING:nv_ingest_client.client.interface:Job was not completely successful. 0 out of 1 records completed successfully. Uploading successful results to vector database.
Total time: 31.12706232070923 seconds
Traceback (most recent call last):
  File "/home/clouduser01/jaswanth/pipeline.py", line 76, in <module>
    print(ingest_json_results_to_blob(results[0]))
                                      ~~~~~~~^^^
IndexError: list index out of range
Killed subprocess group 2231322
