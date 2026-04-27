taskset -c 0-7 python pipelinecp.py --interactive
[I 2026-04-27T06:43:10.347] Centralized logging configured (console only) console_level=INFO log_dir=none (NEMO_EVALUATOR_LOG_DIR not set) logger=nemo_evaluator.logging.utils
2026-04-27 06:43:10  INFO      ========================================================================
2026-04-27 06:43:10  INFO      NV-INGEST 25.9.0  LIBRARY MODE  v3
2026-04-27 06:43:10  INFO        Started  : 2026-04-27 06:43:10
2026-04-27 06:43:10  INFO        Python   : 3.12.13
2026-04-27 06:43:10  INFO        Outputs  : /home/clouduser01/jaswanth/Outputs/
2026-04-27 06:43:10  INFO        Log file : /home/clouduser01/jaswanth/Outputs/pipeline_run_20260427_064310.log
2026-04-27 06:43:10  INFO        Mode     : INTERACTIVE
2026-04-27 06:43:10  INFO      ========================================================================

========================================================================
  INTERACTIVE QUESTION MODE -- Enter questions, blank line to finish
========================================================================

  Q1: What cash consideration will each Public Common Unit receive at the effective time of the merger?
  Q2: What are the signature requirements for every medical record entry under this documentation standards policy?
  Q3: What percentage of the outstanding Common Units did DCP Midstream and the General Partner own when they delivered written consent approving the merger?
  Q4: What key capabilities or features does Infinity Ensure provide for cloud governance and cost optimization?
  Q5: Within how many days must ambulatory EMR documentation be complete after the patient encounter, and within how many days must the inpatient medical record be completed after discharge?
  Q6: What business outcomes or savings are described in the brochure’s customer examples?
  Q7: What does the policy say about texting medical orders and using email to transmit orders?
  Q8: Which units or securities will remain outstanding and unaffected immediately following the merger?
  Q9: What is Infinity Ensure, and what main problem does it aim to solve for cloud environments?
  Q10: What happens to PDI’s ownership interest in Merger Sub at the effective time of the merger, and how is that amount determined?
  Q11: What are the timing requirements for inpatient history and physicals (H&Ps), including updates on the day of a procedure requiring anesthesia?
  Q12: How does the brochure describe Infinity Ensure’s governance capabilities in terms of checks, remediation, reporting, and access management?
  Q13: Who are the filing persons listed in the Schedule 13E-3 transaction statement, and what role does Merger Sub play in the transaction?
  Q14: According to the policy, what parts of the encounter may medical students document, and what must the physician still personally perform, review, or attest to?
  Q15: What are the main categories of cloud risks or operational concerns that Infinity Ensure is designed to address?
  Q16: What concrete outcomes are described in the customer examples for cloud portfolio optimization and compliance/cost optimization?
  Q17: What information must be included in the immediate post-operative progress note, and by when must the detailed dictated procedure report be placed in the record?
  Q18: What approvals or conditions were required from limited partners for the merger, and how was that requirement satisfied on January 5, 2023? [merger_agreement 1 | PDF]
  Q19: 
2026-04-27 06:51:31  INFO      Interactive mode: 18 question(s) collected.
2026-04-27 06:51:31  INFO      ========================================================================
2026-04-27 06:51:31  INFO      STEP 0 -- ENVIRONMENT INJECTION + VERIFICATION
2026-04-27 06:51:31  INFO      ========================================================================
2026-04-27 06:51:31  INFO        NVIDIA_API_KEY : ********jMJZNZ
2026-04-27 06:51:31  INFO        YOLOX_HTTP_ENDPOINT                             -> injected: http://localhost:8000/v1/infer
2026-04-27 06:51:31  INFO        YOLOX_INFER_PROTOCOL                            -> injected: http
2026-04-27 06:51:31  INFO        YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT            -> injected: http://localhost:8003/v1/infer
2026-04-27 06:51:31  INFO        YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL           -> injected: http
2026-04-27 06:51:31  INFO        YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT             -> injected: http://localhost:8006/v1/infer
2026-04-27 06:51:31  INFO        YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL            -> injected: http
2026-04-27 06:51:31  INFO        OCR_HTTP_ENDPOINT                               -> injected: http://localhost:8009/v1/infer
2026-04-27 06:51:31  INFO        OCR_INFER_PROTOCOL                              -> injected: http
2026-04-27 06:51:31  INFO        All NIM env vars in os.environ -- subprocess will inherit them.
2026-04-27 06:51:31  INFO      ========================================================================
2026-04-27 06:51:31  INFO      FILE DISCOVERY  ->  folder: Docs/
2026-04-27 06:51:31  INFO        *.pdf        -> 9 file(s)
2026-04-27 06:51:31  INFO        *.jpeg       -> 2 file(s)
2026-04-27 06:51:31  INFO        *.jpg        -> 2 file(s)
2026-04-27 06:51:31  INFO        *.png        -> 1 file(s)
2026-04-27 06:51:31  INFO        Total: 14 file(s)
2026-04-27 06:51:31  INFO          Ascent_of_Open.pdf                                  4.30 MB
2026-04-27 06:51:31  INFO          DOC-20260407-WA0009..pdf                            0.10 MB
2026-04-27 06:51:31  INFO          Driving.jpg                                         0.02 MB
2026-04-27 06:51:31  INFO          Infinity-Ensure-Brochure.pdf                        0.69 MB
2026-04-27 06:51:31  INFO          Oxford.pdf                                          3.99 MB
2026-04-27 06:51:31  INFO          PK0016.pdf                                          0.32 MB
2026-04-27 06:51:31  INFO          Screenshot (1).png                                  0.09 MB
2026-04-27 06:51:31  INFO          Singapore_NID_B 1.jpeg                              0.16 MB
2026-04-27 06:51:31  INFO          Singapore_NID_F 1.jpeg                              0.17 MB
2026-04-27 06:51:31  INFO          california-drivers-license-small 1 1.jpg            0.04 MB
2026-04-27 06:51:31  INFO          invoice-0-4.pdf                                     0.06 MB
2026-04-27 06:51:31  INFO          minion-tech.pdf                                     10.54 MB
2026-04-27 06:51:31  INFO          multimodal_test.pdf                                 0.13 MB
2026-04-27 06:51:31  INFO          policy-2.pdf                                        0.83 MB
2026-04-27 06:51:31  INFO      ========================================================================
2026-04-27 06:51:31  INFO      PIPELINE INITIALISATION
2026-04-27 06:51:31  INFO      ========================================================================
2026-04-27 06:51:38.624180731 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/MSFT1000:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
[I 2026-04-27T06:51:39.423] Detected 32 logical cores via psutil.
[I 2026-04-27T06:51:39.423] Detected 16 physical cores via psutil.
[I 2026-04-27T06:51:39.423] Detected 8 cores via os.sched_getaffinity.
[I 2026-04-27T06:51:39.423] Raw CPU limit determined: 8.00 (Method: sched_affinity)
[I 2026-04-27T06:51:39.424] Effective CPU core limit determined: 8.00 (Method: sched_affinity)
2026-04-27 06:51:39  INFO      PHASE START : Pipeline subprocess
[I 2026-04-27T06:51:39.476] Launching pipeline in Python subprocess using multiprocessing.
[I 2026-04-27T06:51:39.487] Pipeline subprocess started (PID=106229)
2026-04-27 06:51:39  INFO      PHASE END   : Pipeline subprocess  ->  0.011s
2026-04-27 06:51:39  INFO        Polling port 7671 for pipeline readiness (max 90s)...
2026-04-27 06:51:51  INFO        Port 7671 accepting connections OK
2026-04-27 06:51:56  INFO        NvIngestClient connected  ->  localhost:7671 OK
2026-04-27 06:51:56  INFO      ========================================================================
2026-04-27 06:51:56  INFO      NIM HEALTH CHECK
2026-04-27 06:51:56  INFO      ========================================================================
2026-04-27 06:51:56  WARNING     FAIL page_elements          port 8000  HTTP 404
2026-04-27 06:51:56  WARNING          Container up but /v1/health/ready returned 404.
2026-04-27 06:51:56  WARNING          Try: curl http://localhost:8000/v1/health
2026-04-27 06:51:56  WARNING               curl http://localhost:8000/health/ready
2026-04-27 06:51:56  WARNING     FAIL graphic_elements       port 8003  HTTP 000
2026-04-27 06:51:56  WARNING          Container not running.
2026-04-27 06:51:56  WARNING          Fix: cd nv-ingest && docker compose --profile yolox-graphic-elements up -d
2026-04-27 06:51:56  WARNING     FAIL table_structure        port 8006  HTTP 000
2026-04-27 06:51:56  WARNING          Container not running.
2026-04-27 06:51:56  WARNING          Fix: cd nv-ingest && docker compose --profile yolox-table-structure up -d
2026-04-27 06:51:56  WARNING     FAIL ocr                    port 8009  HTTP 000
2026-04-27 06:51:56  WARNING          Container not running.
2026-04-27 06:51:56  WARNING          Fix: cd nv-ingest && docker compose --profile ocr up -d
2026-04-27 06:51:56  WARNING     0/4 NIMs healthy.
2026-04-27 06:51:56  WARNING     IMPACT: Extraction will run in TEXT-ONLY mode.
2026-04-27 06:51:56  WARNING     Tables, charts, images, infographics will be SKIPPED.
2026-04-27 06:51:56  WARNING     Text chunks will still be indexed into Milvus.
2026-04-27 06:51:56  WARNING     To enable full multimodal, start all NIMs:
2026-04-27 06:51:56  WARNING       cd nv-ingest
2026-04-27 06:51:56  WARNING       docker compose --profile retrieval --profile table-structure up -d
2026-04-27 06:51:56  WARNING       # Wait 10-15 min for model loading, then re-run.
2026-04-27 06:51:56  INFO      ========================================================================
2026-04-27 06:51:56  INFO      SANITY CHECK (text-only)  ->  Ascent_of_Open.pdf
2026-04-27 06:51:56  INFO      ========================================================================
2026-04-27 06:51:56  INFO      PHASE START : Sanity extraction
Processing:   0%|                                                                       | 0/1 [00:00<?, ?doc/s][I 2026-04-27T06:51:56.699] Starting batch processing for 1 jobs with batch size 32.
Processing: 100%|███████████████████████████████████████████████████████████████| 1/1 [00:31<00:00, 31.40s/doc][I 2026-04-27T06:52:28.096] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
Processing: 100%|███████████████████████████████████████████████████████████████| 1/1 [00:31<00:00, 31.40s/doc]
2026-04-27 06:52:28  INFO      PHASE END   : Sanity extraction  ->  31.449s
2026-04-27 06:52:28  INFO        Results: 1  Failures: 0
 in, nurture and suppor...nce is a technology company working to make research more efficient. We invest 
2026-04-27 06:52:28  INFO        SANITY CHECK PASSED
2026-04-27 06:52:28  INFO      ========================================================================
2026-04-27 06:52:28  INFO      BATCH INGEST  ->  14 files  |  mode: TEXT-ONLY (NIMs not all healthy)
2026-04-27 06:52:28  INFO        Collection  : multimodal_docs
2026-04-27 06:52:28  INFO        Milvus URI  : milvus.db
2026-04-27 06:52:28  INFO        Chunk size  : 512 tokens  overlap: 50
2026-04-27 06:52:28  INFO        Embed dim   : 2048  tokenizer: intfloat/e5-large-unsupervised
2026-04-27 06:52:28  INFO        TOP_K=10 is the RAG retrieval count per query, NOT per-file chunk count.
2026-04-27 06:52:28  INFO        Per-file chunk count depends on document length / 512 tokens.
2026-04-27 06:52:28  INFO      ========================================================================
2026-04-27 06:52:28  INFO        [1/14]  Ascent_of_Open.pdf  [TEXT-ONLY]
[I 2026-04-27T06:52:28.124] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:52:35.312] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
[I 2026-04-27T06:52:36.529] 16 elements to insert to milvus
[I 2026-04-27T06:52:36.530] threshold for streaming is 1000
[I 2026-04-27T06:52:36.883] streamed 16 records
[I 2026-04-27T06:52:36.883] Purging saved results from disk after successful VDB upload.
[W 2026-04-27T06:52:36.883] Purge requested, but save_to_disk was not configured. No files to purge.
2026-04-27 06:52:36  INFO          results=1  failures=0  time=8.786s
2026-04-27 06:52:36  INFO        -- wall: 8.786s --
2026-04-27 06:52:36  INFO        [2/14]  DOC-20260407-WA0009..pdf  [TEXT-ONLY]
[I 2026-04-27T06:52:36.885] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:52:39.975] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
[I 2026-04-27T06:52:40.540] 2 elements to insert to milvus
[I 2026-04-27T06:52:40.540] threshold for streaming is 1000
[I 2026-04-27T06:52:40.640] streamed 2 records
[I 2026-04-27T06:52:40.640] Purging saved results from disk after successful VDB upload.
[W 2026-04-27T06:52:40.640] Purge requested, but save_to_disk was not configured. No files to purge.
2026-04-27 06:52:40  INFO          results=1  failures=0  time=3.756s
2026-04-27 06:52:40  INFO        -- wall: 3.757s --
2026-04-27 06:52:40  INFO        [3/14]  Driving.jpg  [TEXT-ONLY]
[I 2026-04-27T06:52:40.641] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:52:41.727] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
2026-04-27 06:52:42  ERROR         EXCEPTION: No records with Embeddings to insert detected.
2026-04-27 06:52:42  ERROR     Traceback (most recent call last):
  File "/home/clouduser01/jaswanth/pipelinecp.py", line 427, in ingest_single_file
    results, failures = ingestor.ingest(show_progress=False, return_failures=True)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/client/interface.py", line 594, in ingest
    self._vdb_bulk_upload.run(results)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/util/vdb/milvus.py", line 1947, in run
    self.write_to_index(records, **write_params)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/util/vdb/milvus.py", line 1912, in write_to_index
    write_to_nvingest_collection(records, collection_name=collection_name, **kwargs)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/util/vdb/milvus.py", line 999, in write_to_nvingest_collection
    raise ValueError("No records with Embeddings to insert detected.")
ValueError: No records with Embeddings to insert detected.

2026-04-27 06:52:42  INFO        -- wall: 1.812s --
2026-04-27 06:52:42  INFO        [4/14]  Infinity-Ensure-Brochure.pdf  [TEXT-ONLY]
[I 2026-04-27T06:52:42.455] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:52:45.563] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
[I 2026-04-27T06:52:46.132] 5 elements to insert to milvus
[I 2026-04-27T06:52:46.132] threshold for streaming is 1000
[I 2026-04-27T06:52:46.271] streamed 5 records
[I 2026-04-27T06:52:46.271] Purging saved results from disk after successful VDB upload.
[W 2026-04-27T06:52:46.271] Purge requested, but save_to_disk was not configured. No files to purge.
2026-04-27 06:52:46  INFO          results=1  failures=0  time=3.818s
2026-04-27 06:52:46  INFO        -- wall: 3.819s --
2026-04-27 06:52:46  INFO        [5/14]  Oxford.pdf  [TEXT-ONLY]
[I 2026-04-27T06:52:46.285] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:52:49.480] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
[I 2026-04-27T06:52:50.926] 47 elements to insert to milvus
[I 2026-04-27T06:52:50.926] threshold for streaming is 1000
[I 2026-04-27T06:52:51.776] streamed 47 records
[I 2026-04-27T06:52:51.777] Purging saved results from disk after successful VDB upload.
[W 2026-04-27T06:52:51.777] Purge requested, but save_to_disk was not configured. No files to purge.
2026-04-27 06:52:51  INFO          results=1  failures=0  time=5.505s
2026-04-27 06:52:51  INFO        -- wall: 5.506s --
2026-04-27 06:52:51  INFO        [6/14]  PK0016.pdf  [TEXT-ONLY]
[I 2026-04-27T06:52:51.780] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:52:58.885] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
[I 2026-04-27T06:52:59.445] 13 elements to insert to milvus
[I 2026-04-27T06:52:59.446] threshold for streaming is 1000
[I 2026-04-27T06:52:59.708] streamed 13 records
[I 2026-04-27T06:52:59.708] Purging saved results from disk after successful VDB upload.
[W 2026-04-27T06:52:59.708] Purge requested, but save_to_disk was not configured. No files to purge.
2026-04-27 06:52:59  INFO          results=1  failures=0  time=7.930s
2026-04-27 06:52:59  INFO        -- wall: 7.930s --
2026-04-27 06:52:59  INFO        [7/14]  Screenshot (1).png  [TEXT-ONLY]
[I 2026-04-27T06:52:59.710] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:53:02.798] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
2026-04-27 06:53:04  ERROR         EXCEPTION: No records with Embeddings to insert detected.
2026-04-27 06:53:04  ERROR     Traceback (most recent call last):
  File "/home/clouduser01/jaswanth/pipelinecp.py", line 427, in ingest_single_file
    results, failures = ingestor.ingest(show_progress=False, return_failures=True)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/client/interface.py", line 594, in ingest
    self._vdb_bulk_upload.run(results)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/util/vdb/milvus.py", line 1947, in run
    self.write_to_index(records, **write_params)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/util/vdb/milvus.py", line 1912, in write_to_index
    write_to_nvingest_collection(records, collection_name=collection_name, **kwargs)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/util/vdb/milvus.py", line 999, in write_to_nvingest_collection
    raise ValueError("No records with Embeddings to insert detected.")
ValueError: No records with Embeddings to insert detected.

2026-04-27 06:53:04  INFO        -- wall: 5.145s --
2026-04-27 06:53:04  INFO        [8/14]  Singapore_NID_B 1.jpeg  [TEXT-ONLY]
[I 2026-04-27T06:53:04.855] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:53:11.947] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
2026-04-27 06:53:12  ERROR         EXCEPTION: No records with Embeddings to insert detected.
2026-04-27 06:53:12  ERROR     Traceback (most recent call last):
  File "/home/clouduser01/jaswanth/pipelinecp.py", line 427, in ingest_single_file
    results, failures = ingestor.ingest(show_progress=False, return_failures=True)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/client/interface.py", line 594, in ingest
    self._vdb_bulk_upload.run(results)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/util/vdb/milvus.py", line 1947, in run
    self.write_to_index(records, **write_params)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/util/vdb/milvus.py", line 1912, in write_to_index
    write_to_nvingest_collection(records, collection_name=collection_name, **kwargs)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/util/vdb/milvus.py", line 999, in write_to_nvingest_collection
    raise ValueError("No records with Embeddings to insert detected.")
ValueError: No records with Embeddings to insert detected.

2026-04-27 06:53:12  INFO        -- wall: 7.664s --
2026-04-27 06:53:12  INFO        [9/14]  Singapore_NID_F 1.jpeg  [TEXT-ONLY]
[I 2026-04-27T06:53:12.519] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:53:15.609] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
2026-04-27 06:53:16  ERROR         EXCEPTION: No records with Embeddings to insert detected.
2026-04-27 06:53:16  ERROR     Traceback (most recent call last):
  File "/home/clouduser01/jaswanth/pipelinecp.py", line 427, in ingest_single_file
    results, failures = ingestor.ingest(show_progress=False, return_failures=True)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/client/interface.py", line 594, in ingest
    self._vdb_bulk_upload.run(results)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/util/vdb/milvus.py", line 1947, in run
    self.write_to_index(records, **write_params)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/util/vdb/milvus.py", line 1912, in write_to_index
    write_to_nvingest_collection(records, collection_name=collection_name, **kwargs)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/util/vdb/milvus.py", line 999, in write_to_nvingest_collection
    raise ValueError("No records with Embeddings to insert detected.")
ValueError: No records with Embeddings to insert detected.

2026-04-27 06:53:16  INFO        -- wall: 3.647s --
2026-04-27 06:53:16  INFO        [10/14]  california-drivers-license-small 1 1.jpg  [TEXT-ONLY]
[I 2026-04-27T06:53:16.166] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:53:17.253] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
2026-04-27 06:53:17  ERROR         EXCEPTION: No records with Embeddings to insert detected.
2026-04-27 06:53:17  ERROR     Traceback (most recent call last):
  File "/home/clouduser01/jaswanth/pipelinecp.py", line 427, in ingest_single_file
    results, failures = ingestor.ingest(show_progress=False, return_failures=True)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/client/interface.py", line 594, in ingest
    self._vdb_bulk_upload.run(results)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/util/vdb/milvus.py", line 1947, in run
    self.write_to_index(records, **write_params)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/util/vdb/milvus.py", line 1912, in write_to_index
    write_to_nvingest_collection(records, collection_name=collection_name, **kwargs)
  File "/home/clouduser01/micromamba/envs/myenv/lib/python3.12/site-packages/nv_ingest_client/util/vdb/milvus.py", line 999, in write_to_nvingest_collection
    raise ValueError("No records with Embeddings to insert detected.")
ValueError: No records with Embeddings to insert detected.

2026-04-27 06:53:17  INFO        -- wall: 1.648s --
2026-04-27 06:53:17  INFO        [11/14]  invoice-0-4.pdf  [TEXT-ONLY]
[I 2026-04-27T06:53:17.814] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:53:20.902] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
[I 2026-04-27T06:53:21.474] 2 elements to insert to milvus
[I 2026-04-27T06:53:21.474] threshold for streaming is 1000
[I 2026-04-27T06:53:21.574] streamed 2 records
[I 2026-04-27T06:53:21.574] Purging saved results from disk after successful VDB upload.
[W 2026-04-27T06:53:21.574] Purge requested, but save_to_disk was not configured. No files to purge.
2026-04-27 06:53:21  INFO          results=1  failures=0  time=3.762s
2026-04-27 06:53:21  INFO        -- wall: 3.762s --
2026-04-27 06:53:21  INFO        [12/14]  minion-tech.pdf  [TEXT-ONLY]
[I 2026-04-27T06:53:21.613] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:53:24.983] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
[I 2026-04-27T06:53:25.568] 22 elements to insert to milvus
[I 2026-04-27T06:53:25.568] threshold for streaming is 1000
[I 2026-04-27T06:53:25.971] streamed 22 records
[I 2026-04-27T06:53:25.972] Purging saved results from disk after successful VDB upload.
[W 2026-04-27T06:53:25.972] Purge requested, but save_to_disk was not configured. No files to purge.
2026-04-27 06:53:25  INFO          results=1  failures=0  time=4.397s
2026-04-27 06:53:25  INFO        -- wall: 4.398s --
2026-04-27 06:53:25  INFO        [13/14]  multimodal_test.pdf  [TEXT-ONLY]
[I 2026-04-27T06:53:25.974] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:53:33.065] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
[I 2026-04-27T06:53:33.627] 3 elements to insert to milvus
[I 2026-04-27T06:53:33.627] threshold for streaming is 1000
[I 2026-04-27T06:53:33.752] streamed 3 records
[I 2026-04-27T06:53:33.752] Purging saved results from disk after successful VDB upload.
[W 2026-04-27T06:53:33.752] Purge requested, but save_to_disk was not configured. No files to purge.
2026-04-27 06:53:33  INFO          results=1  failures=0  time=7.779s
2026-04-27 06:53:33  INFO        -- wall: 7.780s --
2026-04-27 06:53:33  INFO        [14/14]  policy-2.pdf  [TEXT-ONLY]
[I 2026-04-27T06:53:33.756] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:53:40.876] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
[I 2026-04-27T06:53:41.450] 12 elements to insert to milvus
[I 2026-04-27T06:53:41.450] threshold for streaming is 1000
[I 2026-04-27T06:53:41.723] streamed 12 records
[I 2026-04-27T06:53:41.723] Purging saved results from disk after successful VDB upload.
[W 2026-04-27T06:53:41.723] Purge requested, but save_to_disk was not configured. No files to purge.
2026-04-27 06:53:41  INFO          results=1  failures=0  time=7.971s
2026-04-27 06:53:41  INFO        -- wall: 7.971s --
2026-04-27 06:53:41  INFO      ========================================================================
2026-04-27 06:53:41  INFO      BATCH SUMMARY
2026-04-27 06:53:41  INFO        Total    : 14
2026-04-27 06:53:41  INFO        OK       : 9
2026-04-27 06:53:41  INFO        Failed   : 5
2026-04-27 06:53:41  INFO        Total t  : 73.627s
2026-04-27 06:53:41  INFO        Avg/doc  : 5.259s
2026-04-27 06:53:41  INFO      ========================================================================
2026-04-27 06:53:41  INFO      ========================================================================
2026-04-27 06:53:41  INFO      MILVUS VERIFICATION
2026-04-27 06:53:41  INFO        Collection 'multimodal_docs': 12 total entities (chunks)
2026-04-27 06:53:41  INFO        Milvus populated OK  (12 chunks total)
2026-04-27 06:53:41  INFO        RAG will retrieve top 10 most relevant per query.
2026-04-27 06:53:41  INFO      ========================================================================
2026-04-27 06:53:41  INFO      RAG QUERIES  ->  18 question(s)
2026-04-27 06:53:41  INFO        Milvus TOP_K  : 10  (max chunks returned per query)
2026-04-27 06:53:41  INFO        LLM model     : meta/llama-3.3-70b-instruct
2026-04-27 06:53:41  INFO      ========================================================================
2026-04-27 06:53:41  INFO        Q1: What cash consideration will each Public Common Unit receive at the effective time of the merger?
[I 2026-04-27T06:53:43.529] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2026-04-27 06:53:43  INFO          Retrieval: 1.918s | chunks returned: 10  (TOP_K=10)
[I 2026-04-27T06:53:45.465] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-04-27 06:53:45  INFO          LLM: 1.812s | tokens=4607 | tok/s=23.2
2026-04-27 06:53:45  INFO          Answer: The answer is not in the context. The provided context is about the University of Toledo Medical Center's Documentation Standards Policy, and it does not mention anything about a merger or cash consid
2026-04-27 06:53:46  INFO        Q2: What are the signature requirements for every medical record entry under this documentation standards policy?
[I 2026-04-27T06:53:46.695] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2026-04-27 06:53:46  INFO          Retrieval: 0.347s | chunks returned: 10  (TOP_K=10)
[I 2026-04-27T06:53:48.368] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-04-27 06:53:48  INFO          LLM: 1.548s | tokens=4613 | tok/s=31.7
2026-04-27 06:53:48  INFO          Answer: Every medical record entry must be timed, dated, and its author identified with either electronic or ink signature as defined in hospital policy #3364-100-53-18. All written entries in the medical rec
2026-04-27 06:53:49  INFO        Q3: What percentage of the outstanding Common Units did DCP Midstream and the General Partner own when they delivered written consent approving the merger?
[I 2026-04-27T06:53:49.606] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2026-04-27 06:53:49  INFO          Retrieval: 0.367s | chunks returned: 10  (TOP_K=10)
[I 2026-04-27T06:53:55.137] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-04-27 06:53:55  INFO          LLM: 5.401s | tokens=4686 | tok/s=8.9
2026-04-27 06:53:55  INFO          Answer: The answer is not in the context. The provided context appears to be a medical policy document and does not mention DCP Midstream, the General Partner, or the merger. It discusses documentation standa
2026-04-27 06:53:56  INFO        Q4: What key capabilities or features does Infinity Ensure provide for cloud governance and cost optimization?
[I 2026-04-27T06:53:56.381] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2026-04-27 06:53:56  INFO          Retrieval: 0.369s | chunks returned: 10  (TOP_K=10)
[I 2026-04-27T06:53:58.362] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-04-27 06:53:58  INFO          LLM: 1.855s | tokens=4689 | tok/s=19.4
2026-04-27 06:53:58  INFO          Answer: The answer is not in the context. The provided context appears to be a policy document related to medical record documentation standards, and it does not mention Infinity Ensure or its features.
2026-04-27 06:53:59  INFO        Q5: Within how many days must ambulatory EMR documentation be complete after the patient encounter, and within how many days must the inpatient medical record be completed after discharge?
[I 2026-04-27T06:53:59.594] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2026-04-27 06:53:59  INFO          Retrieval: 0.358s | chunks returned: 10  (TOP_K=10)
[I 2026-04-27T06:54:00.876] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-04-27 06:54:00  INFO          LLM: 1.156s | tokens=4973 | tok/s=32.9
2026-04-27 06:54:00  INFO          Answer: According to the context, ambulatory EMR documentation must be complete within 7 days of the patient encounter. The inpatient medical record must be completed within 30 days of discharge.
2026-04-27 06:54:01  INFO        Q6: What business outcomes or savings are described in the brochure’s customer examples?
[I 2026-04-27T06:54:02.124] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2026-04-27 06:54:02  INFO          Retrieval: 0.368s | chunks returned: 10  (TOP_K=10)
[I 2026-04-27T06:54:03.912] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-04-27 06:54:03  INFO          LLM: 1.666s | tokens=4318 | tok/s=30.0
2026-04-27 06:54:03  INFO          Answer: The context provided does not contain any information about a brochure or customer examples, nor does it mention business outcomes or savings. The context appears to be a medical documentation standar
2026-04-27 06:54:04  INFO        Q7: What does the policy say about texting medical orders and using email to transmit orders?
[I 2026-04-27T06:54:05.131] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2026-04-27 06:54:05  INFO          Retrieval: 0.346s | chunks returned: 10  (TOP_K=10)
[I 2026-04-27T06:54:07.848] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-04-27 06:54:07  INFO          LLM: 2.589s | tokens=4645 | tok/s=31.3
2026-04-27 06:54:07  INFO          Answer: According to the policy, texting orders is not permitted per CMS and TJC regulations/standards. However, providers can communicate with other healthcare team members via secure and encrypted text thro
2026-04-27 06:54:08  INFO        Q8: Which units or securities will remain outstanding and unaffected immediately following the merger?
[I 2026-04-27T06:54:09.188] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2026-04-27 06:54:09  INFO          Retrieval: 0.355s | chunks returned: 10  (TOP_K=10)
[I 2026-04-27T06:54:23.767] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-04-27 06:54:23  INFO          LLM: 14.458s | tokens=4631 | tok/s=2.4
2026-04-27 06:54:23  INFO          Answer: The answer is not in the context. The context provided appears to be related to medical documentation standards and policies, and does not mention a merger or units/securities.
2026-04-27 06:54:24  INFO        Q9: What is Infinity Ensure, and what main problem does it aim to solve for cloud environments?
[I 2026-04-27T06:54:25.024] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2026-04-27 06:54:25  INFO          Retrieval: 0.389s | chunks returned: 10  (TOP_K=10)
[I 2026-04-27T06:54:30.450] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-04-27 06:54:30  INFO          LLM: 5.293s | tokens=4623 | tok/s=10.8
2026-04-27 06:54:30  INFO          Answer: The context provided does not mention "Infinity Ensure" or its relation to cloud environments. The context appears to be a policy document for the University of Toledo Medical Center, focusing on docu
2026-04-27 06:54:31  INFO        Q10: What happens to PDI’s ownership interest in Merger Sub at the effective time of the merger, and how is that amount determined?
[I 2026-04-27T06:54:31.701] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2026-04-27 06:54:31  INFO          Retrieval: 0.254s | chunks returned: 10  (TOP_K=10)
[I 2026-04-27T06:54:35.046] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-04-27 06:54:35  INFO          LLM: 3.341s | tokens=4661 | tok/s=25.7
2026-04-27 06:54:35  INFO          Answer: The answer is not in the context. The provided context appears to be a medical policy document, specifically the "Documentation Standards" policy for the University of Toledo Medical Center (UTMC). It
2026-04-27 06:54:36  INFO        Q11: What are the timing requirements for inpatient history and physicals (H&Ps), including updates on the day of a procedure requiring anesthesia?
[I 2026-04-27T06:54:36.282] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2026-04-27 06:54:36  INFO          Retrieval: 0.364s | chunks returned: 10  (TOP_K=10)
[I 2026-04-27T06:54:44.012] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-04-27 06:54:44  INFO          LLM: 7.556s | tokens=4878 | tok/s=27.8
2026-04-27 06:54:44  INFO          Answer: The timing requirements for inpatient history and physicals (H&Ps) are as follows:   1. An H&P must be completed no more than 30 days before or within 24 hours of admission, but prior to surgery or a 
2026-04-27 06:54:45  INFO        Q12: How does the brochure describe Infinity Ensure’s governance capabilities in terms of checks, remediation, reporting, and access management?
[I 2026-04-27T06:54:45.310] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2026-04-27 06:54:45  INFO          Retrieval: 0.379s | chunks returned: 10  (TOP_K=10)
[I 2026-04-27T06:54:47.539] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-04-27 06:54:47  INFO          LLM: 2.147s | tokens=4626 | tok/s=25.2
2026-04-27 06:54:47  INFO          Answer: The context provided does not mention Infinity Ensure or its governance capabilities. The context appears to be a policy document related to documentation standards for medical records at the Universi
2026-04-27 06:54:48  INFO        Q13: Who are the filing persons listed in the Schedule 13E-3 transaction statement, and what role does Merger Sub play in the transaction?
[I 2026-04-27T06:54:48.768] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2026-04-27 06:54:48  INFO          Retrieval: 0.354s | chunks returned: 10  (TOP_K=10)
[I 2026-04-27T06:55:09.846] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-04-27 06:55:09  INFO          LLM: 20.952s | tokens=4361 | tok/s=2.9
2026-04-27 06:55:09  INFO          Answer: The provided context does not contain information about the Schedule 13E-3 transaction statement, filing persons, or Merger Sub. The context appears to be related to medical record documentation stand
2026-04-27 06:55:10  INFO        Q14: According to the policy, what parts of the encounter may medical students document, and what must the physician still personally perform, review, or attest to?
[I 2026-04-27T06:55:11.077] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2026-04-27 06:55:11  INFO          Retrieval: 0.354s | chunks returned: 10  (TOP_K=10)
[I 2026-04-27T06:55:22.434] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-04-27 06:55:22  INFO          LLM: 11.233s | tokens=4526 | tok/s=12.8
2026-04-27 06:55:22  INFO          Answer: According to the policy, medical students may document the following parts of the encounter:   1. Review of the System 2. Past, Family and Social History 3. History of Present Illness 4. Physical Exam
2026-04-27 06:55:23  INFO        Q15: What are the main categories of cloud risks or operational concerns that Infinity Ensure is designed to address?
[I 2026-04-27T06:55:23.672] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2026-04-27 06:55:23  INFO          Retrieval: 0.363s | chunks returned: 10  (TOP_K=10)
[I 2026-04-27T06:55:48.164] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-04-27 06:55:48  INFO          LLM: 24.367s | tokens=4595 | tok/s=1.1
2026-04-27 06:55:48  INFO          Answer: The answer is not in the context. The provided text is related to medical documentation standards and does not mention Infinity Ensure or cloud risks.
2026-04-27 06:55:49  INFO        Q16: What concrete outcomes are described in the customer examples for cloud portfolio optimization and compliance/cost optimization?
[I 2026-04-27T06:55:49.399] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2026-04-27 06:55:49  INFO          Retrieval: 0.362s | chunks returned: 10  (TOP_K=10)
[I 2026-04-27T06:56:05.108] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-04-27 06:56:05  INFO          LLM: 15.580s | tokens=4626 | tok/s=3.8
2026-04-27 06:56:05  INFO          Answer: The provided context does not contain information related to cloud portfolio optimization and compliance/cost optimization, or customer examples. The context appears to be a medical policy document ou
2026-04-27 06:56:06  INFO        Q17: What information must be included in the immediate post-operative progress note, and by when must the detailed dictated procedure report be placed in the record?
[I 2026-04-27T06:56:06.340] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2026-04-27 06:56:06  INFO          Retrieval: 0.366s | chunks returned: 10  (TOP_K=10)
[I 2026-04-27T06:56:12.117] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-04-27 06:56:12  INFO          LLM: 5.642s | tokens=4794 | tok/s=35.6
2026-04-27 06:56:12  INFO          Answer: The immediate post-operative progress note must include at a minimum:  1. Pre-operative and post-operative diagnosis;  2. Name of the specific procedure(s) performed, may include relevant findings;  3
2026-04-27 06:56:13  INFO        Q18: What approvals or conditions were required from limited partners for the merger, and how was that requirement satisfied on January 5, 2023? [merger_agreement 1 | PDF]
[I 2026-04-27T06:56:13.446] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2026-04-27 06:56:13  INFO          Retrieval: 0.456s | chunks returned: 10  (TOP_K=10)
[I 2026-04-27T06:56:15.423] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
2026-04-27 06:56:15  INFO          LLM: 1.368s | tokens=4631 | tok/s=32.9
2026-04-27 06:56:15  INFO          Answer: The context provided does not mention anything about a merger, limited partners, or a specific date of January 5, 2023. Therefore, the answer to the question is: The information is not in the context.
2026-04-27 06:56:16  INFO        Answers saved -> Outputs/answers_20260427_064310.json
2026-04-27 06:56:16  INFO        Metrics saved -> Outputs/metrics_20260427_064310.json
2026-04-27 06:56:16  INFO      ========================================================================
2026-04-27 06:56:16  INFO      FINAL RUN SUMMARY
2026-04-27 06:56:16  INFO      ========================================================================
2026-04-27 06:56:16  INFO        Run ID               : 20260427_064310
2026-04-27 06:56:16  INFO        Pipeline init        : 0.011s
2026-04-27 06:56:16  INFO        Sanity check         : 31.449s
2026-04-27 06:56:16  INFO        Total batch ingest   : 73.627s
2026-04-27 06:56:16  INFO        Total documents      : 14
2026-04-27 06:56:16  INFO        Successful           : 9
2026-04-27 06:56:16  INFO        Avg per doc          : 5.257s
2026-04-27 06:56:16  INFO        Total wall time      : 105.087s
2026-04-27 06:56:16  INFO        Log file    -> Outputs/pipeline_run_20260427_064310.log
2026-04-27 06:56:16  INFO        Metrics     -> Outputs/metrics_20260427_064310.json
2026-04-27 06:56:16  INFO        Answers     -> Outputs/answers_20260427_064310.json
2026-04-27 06:56:16  INFO      ========================================================================
2026-04-27 06:56:16  INFO        Total wall clock: 785.631s
2026-04-27 06:56:16  INFO        PIPELINE DONE.
2026-04-27 06:56:16  INFO      ========================================================================
Killed subprocess group 106229
E20260427 06:56:17.495389 112755 server.cpp:47] [SERVER][BlockLock][milvus] Process exit
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ 
