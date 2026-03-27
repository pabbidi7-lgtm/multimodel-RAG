 uv pip show pymilvus milvus-lite
Using Python 3.12.13 environment at: /home/clouduser01/micromamba/envs/myenv
Name: milvus-lite
Version: 2.4.12
Location: /home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages
Requires: tqdm
Required-by: pymilvus
---
Name: pymilvus
Version: 2.5.4
Location: /home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages
Requires: grpcio, milvus-lite, pandas, protobuf, python-dotenv, setuptools, ujson
Required-by: nv-ingest
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ taskset -c 0-7 python pipeline.py
2026-03-27 02:47:49.424036403 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/VMBUS:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
INFO:nv_ingest_api.util.system.hardware_info:Detected 32 logical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 16 physical cores via psutil.
INFO:nv_ingest_api.util.system.hardware_info:Detected 8 cores via os.sched_getaffinity.
INFO:nv_ingest_api.util.system.hardware_info:Raw CPU limit determined: 8.00 (Method: sched_affinity)
INFO:nv_ingest_api.util.system.hardware_info:Effective CPU core limit determined: 8.00 (Method: sched_affinity)
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Launching pipeline in Python subprocess using multiprocessing.
INFO:nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners:Pipeline subprocess started (PID=2516595)
Waiting for pipeline to initialize...
Pipeline ready. Connecting client...

=== STEP 1: Basic text extraction ===
Starting ingestion...
Processing:   0%|                                                                                  | 0/1 [00:00<?, ?doc/s]INFO:nv_ingest_client.client.client:Starting batch processing for 1 jobs with batch size 32.
Processing: 100%|██████████████████████████████████████████████████████████████████████████| 1/1 [00:31<00:00, 31.12s/doc]INFO:nv_ingest_client.client.client:Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
Processing: 100%|██████████████████████████████████████████████████████████████████████████| 1/1 [00:31<00:00, 31.12s/doc]
Total time: 31.12 seconds

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
Traceback (most recent call last):
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/pymilvus/model/__init__.py", line 24, in _load_milvus_model
    import milvus_model
ModuleNotFoundError: No module named 'milvus_model'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/clouduser01/jaswanth/pipeline.py", line 108, in <module>
    .vdb_upload(
     ^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/client/interface.py", line 880, in vdb_upload
    op_cls = get_vdb_op_cls(vdb_op)
             ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/util/vdb/__init__.py", line 13, in get_vdb_op_cls
    from nv_ingest_client.util.vdb.milvus import Milvus
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/util/vdb/milvus.py", line 40, in <module>
    from pymilvus.model.sparse import BM25EmbeddingFunction
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/pymilvus/model/__init__.py", line 10, in __getattr__
    self._load_milvus_model()
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/pymilvus/model/__init__.py", line 33, in _load_milvus_model
    raise ImportError(err_str) from e
ImportError: The 'milvus_model' package is not installed. For installation, use 'pip install pymilvus[model]'. For more information, please visit https://github.com/milvus-io/milvus-model.
Killed subprocess group 2516595
