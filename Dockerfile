taskset -c 0-7 python pipelinecp.py
[I 2026-04-24T10:58:59.984] Centralized logging configured (console only) console_level=INFO log_dir=none (NEMO_EVALUATOR_LOG_DIR not set) logger=nemo_evaluator.logging.utils
2026-04-24 10:59:07.477777139 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/MSFT1000:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
[I 2026-04-24T10:59:08.316] Detected 32 logical cores via psutil.
[I 2026-04-24T10:59:08.316] Detected 16 physical cores via psutil.
[I 2026-04-24T10:59:08.316] Detected 8 cores via os.sched_getaffinity.
[I 2026-04-24T10:59:08.317] Raw CPU limit determined: 8.00 (Method: sched_affinity)
[I 2026-04-24T10:59:08.317] Effective CPU core limit determined: 8.00 (Method: sched_affinity)
[I 2026-04-24T10:59:08.371] ══════════════════════════════════════════════════════════════════════
[I 2026-04-24T10:59:08.371] NV-INGEST 25.9.0 — LIBRARY MODE PIPELINE WITH FULL LOGGING
[I 2026-04-24T10:59:08.371] Run started at : 2026-04-24 10:59:08
[I 2026-04-24T10:59:08.371] Results dir    : results/
[I 2026-04-24T10:59:08.371] Log file       : results/pipeline_run_20260424_105908.log
[I 2026-04-24T10:59:08.371] ══════════════════════════════════════════════════════════════════════
[I 2026-04-24T10:59:08.371] ══════════════════════════════════════════════════════════════════════
[I 2026-04-24T10:59:08.371] ENVIRONMENT CHECK
[I 2026-04-24T10:59:08.371] ══════════════════════════════════════════════════════════════════════
[I 2026-04-24T10:59:08.371]   NVIDIA_API_KEY        : ********jMJZNZ
[W 2026-04-24T10:59:08.371]   YOLOX_HTTP_ENDPOINT                           NOT SET — page-elements     (port 8000) will NOT be invoked
[W 2026-04-24T10:59:08.371]   YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT          NOT SET — graphic-elements  (port 8003) will NOT be invoked
[W 2026-04-24T10:59:08.371]   YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT           NOT SET — table-structure   (port 8006) will NOT be invoked
[W 2026-04-24T10:59:08.371]   OCR_HTTP_ENDPOINT                             NOT SET — ocr               (port 8009) will NOT be invoked
[W 2026-04-24T10:59:08.371]   Some NIM endpoints are missing. Nemotron models for those
[W 2026-04-24T10:59:08.372]   modalities will NOT run. Basic extraction only for those paths.
[I 2026-04-24T10:59:08.372] ══════════════════════════════════════════════════════════════════════
[I 2026-04-24T10:59:08.372] FILE DISCOVERY  →  pattern: Docs/*.pdf
[I 2026-04-24T10:59:08.372]   Found 9 file(s)
[I 2026-04-24T10:59:08.372]   › Docs/Ascent_of_Open.pdf  (4.30 MB)
[I 2026-04-24T10:59:08.372]   › Docs/DOC-20260407-WA0009..pdf  (0.10 MB)
[I 2026-04-24T10:59:08.372]   › Docs/Infinity-Ensure-Brochure.pdf  (0.69 MB)
[I 2026-04-24T10:59:08.372]   › Docs/Oxford.pdf  (3.99 MB)
[I 2026-04-24T10:59:08.372]   › Docs/PK0016.pdf  (0.32 MB)
[I 2026-04-24T10:59:08.372]   › Docs/invoice-0-4.pdf  (0.06 MB)
[I 2026-04-24T10:59:08.372]   › Docs/minion-tech.pdf  (10.54 MB)
[I 2026-04-24T10:59:08.372]   › Docs/multimodal_test.pdf  (0.13 MB)
[I 2026-04-24T10:59:08.372]   › Docs/policy-2.pdf  (0.83 MB)
[I 2026-04-24T10:59:08.373] ══════════════════════════════════════════════════════════════════════
[I 2026-04-24T10:59:08.373] PIPELINE INITIALISATION
[I 2026-04-24T10:59:08.373] ══════════════════════════════════════════════════════════════════════
[I 2026-04-24T10:59:08.373] PHASE START : Pipeline subprocess start
[I 2026-04-24T10:59:08.373] Launching pipeline in Python subprocess using multiprocessing.
[I 2026-04-24T10:59:08.388] Pipeline subprocess started (PID=188563)
[I 2026-04-24T10:59:08.389] PHASE END   : Pipeline subprocess start  →  0.016s
[I 2026-04-24T10:59:08.389]   Waiting 20s for pipeline to become ready...
[I 2026-04-24T10:59:28.389]   Pipeline ready.
[I 2026-04-24T10:59:28.393]   NvIngestClient connected  →  localhost:7671
[I 2026-04-24T10:59:28.393] ══════════════════════════════════════════════════════════════════════
[I 2026-04-24T10:59:28.393] STEP 1 — SANITY CHECK (text-only)  →  Docs/Ascent_of_Open.pdf
[I 2026-04-24T10:59:28.393] ══════════════════════════════════════════════════════════════════════
[I 2026-04-24T10:59:28.393] PHASE START : Text-only extraction
Processing:   0%|                                                                       | 0/1 [00:00<?, ?doc/s][I 2026-04-24T10:59:28.911] Starting batch processing for 1 jobs with batch size 32.
Processing: 100%|███████████████████████████████████████████████████████████████| 1/1 [00:31<00:00, 31.30s/doc][I 2026-04-24T11:00:00.215] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
Processing: 100%|███████████████████████████████████████████████████████████████| 1/1 [00:31<00:00, 31.31s/doc]
[I 2026-04-24T11:00:00.216] PHASE END   : Text-only extraction  →  31.822s
[I 2026-04-24T11:00:00.216]   Results  : 1
[I 2026-04-24T11:00:00.216]   Failures : 0
[I 2026-04-24T11:00:00.216]   Time     : 31.822s
 research process more open and...e businesses and technologies that make all parts of the nt. We invest 
[I 2026-04-24T11:00:00.216]   SANITY CHECK PASSED
[I 2026-04-24T11:00:00.216] ══════════════════════════════════════════════════════════════════════
[I 2026-04-24T11:00:00.216] STEP 2 — FULL MULTIMODAL BATCH INGEST  →  9 file(s)
[I 2026-04-24T11:00:00.217]   Collection : multimodal_docs
[I 2026-04-24T11:00:00.217]   Milvus URI : milvus.db
[I 2026-04-24T11:00:00.217]   Caption    : nvidia/llama-3.1-nemotron-nano-vl-8b-v1
[I 2026-04-24T11:00:00.217]   Embedder   : dense_dim=2048  tokenizer=intfloat/e5-large-unsupervised
[I 2026-04-24T11:00:00.217]   NIM Models : page-elements | graphic-elements | table-structure | OCR
[I 2026-04-24T11:00:00.217] ══════════════════════════════════════════════════════════════════════
[I 2026-04-24T11:00:00.217]   [1/9] Ingesting: Ascent_of_Open.pdf
[I 2026-04-24T11:00:00.217] PHASE START : Extract  [Ascent_of_Open.pdf]
[I 2026-04-24T11:00:00.568] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-24T11:00:31.825] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
[I 2026-04-24T11:00:35.676] 45 elements to insert to milvus
[I 2026-04-24T11:00:35.677] threshold for streaming is 1000
[I 2026-04-24T11:00:36.740] streamed 45 records
[I 2026-04-24T11:00:36.741] Purging saved results from disk after successful VDB upload.
[W 2026-04-24T11:00:36.741] Purge requested, but save_to_disk was not configured. No files to purge.
[I 2026-04-24T11:00:36.741]     Ascent_of_Open.pdf → results=1  failures=0  time=36.173s
[I 2026-04-24T11:00:36.743]   ── doc wall time: 36.526s  ──
[I 2026-04-24T11:00:36.743]   [2/9] Ingesting: DOC-20260407-WA0009..pdf
[I 2026-04-24T11:00:36.743] PHASE START : Extract  [DOC-20260407-WA0009..pdf]
[I 2026-04-24T11:00:36.744] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-24T11:00:51.850] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
[I 2026-04-24T11:00:52.418] 6 elements to insert to milvus
[I 2026-04-24T11:00:52.418] threshold for streaming is 1000
[I 2026-04-24T11:00:52.578] streamed 6 records
[I 2026-04-24T11:00:52.578] Purging saved results from disk after successful VDB upload.
[W 2026-04-24T11:00:52.578] Purge requested, but save_to_disk was not configured. No files to purge.
[I 2026-04-24T11:00:52.578]     DOC-20260407-WA0009..pdf → results=1  failures=0  time=15.834s
[I 2026-04-24T11:00:52.578]   ── doc wall time: 15.835s  ──
[I 2026-04-24T11:00:52.578]   [3/9] Ingesting: Infinity-Ensure-Brochure.pdf
[I 2026-04-24T11:00:52.579] PHASE START : Extract  [Infinity-Ensure-Brochure.pdf]
[I 2026-04-24T11:00:52.581] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-24T11:01:23.730] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
[I 2026-04-24T11:01:24.307] 18 elements to insert to milvus
[I 2026-04-24T11:01:24.308] threshold for streaming is 1000
[I 2026-04-24T11:01:24.643] streamed 18 records
[I 2026-04-24T11:01:24.644] Purging saved results from disk after successful VDB upload.
[W 2026-04-24T11:01:24.644] Purge requested, but save_to_disk was not configured. No files to purge.
[I 2026-04-24T11:01:24.644]     Infinity-Ensure-Brochure.pdf → results=1  failures=0  time=32.063s
[I 2026-04-24T11:01:24.645]   ── doc wall time: 32.066s  ──
[I 2026-04-24T11:01:24.645]   [4/9] Ingesting: Oxford.pdf
[I 2026-04-24T11:01:24.645] PHASE START : Extract  [Oxford.pdf]
[I 2026-04-24T11:01:24.670] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-24T11:04:35.620] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
[I 2026-04-24T11:04:36.504] 435 elements to insert to milvus
[I 2026-04-24T11:04:36.504] threshold for streaming is 1000
[I 2026-04-24T11:04:45.479] streamed 435 records
[I 2026-04-24T11:04:45.485] Purging saved results from disk after successful VDB upload.
[W 2026-04-24T11:04:45.486] Purge requested, but save_to_disk was not configured. No files to purge.
[I 2026-04-24T11:04:45.486]     Oxford.pdf → results=1  failures=0  time=200.816s
[I 2026-04-24T11:04:45.499]   ── doc wall time: 200.854s  ──
[I 2026-04-24T11:04:45.499]   [5/9] Ingesting: PK0016.pdf
[I 2026-04-24T11:04:45.499] PHASE START : Extract  [PK0016.pdf]
[I 2026-04-24T11:04:45.501] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-24T11:05:16.662] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
[I 2026-04-24T11:05:17.268] 46 elements to insert to milvus
[I 2026-04-24T11:05:17.268] threshold for streaming is 1000
[I 2026-04-24T11:05:18.970] streamed 46 records
[I 2026-04-24T11:05:18.971] Purging saved results from disk after successful VDB upload.
[W 2026-04-24T11:05:18.971] Purge requested, but save_to_disk was not configured. No files to purge.
[I 2026-04-24T11:05:18.971]     PK0016.pdf → results=1  failures=0  time=33.471s
[I 2026-04-24T11:05:18.974]   ── doc wall time: 33.475s  ──
[I 2026-04-24T11:05:18.974]   [6/9] Ingesting: invoice-0-4.pdf
[I 2026-04-24T11:05:18.974] PHASE START : Extract  [invoice-0-4.pdf]
[I 2026-04-24T11:05:18.975] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-24T11:05:34.075] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
[I 2026-04-24T11:05:34.693] 7 elements to insert to milvus
[I 2026-04-24T11:05:34.693] threshold for streaming is 1000
[I 2026-04-24T11:05:34.865] streamed 7 records
[I 2026-04-24T11:05:34.865] Purging saved results from disk after successful VDB upload.
[W 2026-04-24T11:05:34.865] Purge requested, but save_to_disk was not configured. No files to purge.
[I 2026-04-24T11:05:34.865]     invoice-0-4.pdf → results=1  failures=0  time=15.890s
[I 2026-04-24T11:05:34.865]   ── doc wall time: 15.892s  ──
[I 2026-04-24T11:05:34.866]   [7/9] Ingesting: minion-tech.pdf
[I 2026-04-24T11:05:34.866] PHASE START : Extract  [minion-tech.pdf]
[I 2026-04-24T11:05:34.964] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-24T11:06:38.860] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1777028798.864390  188484 chttp2_transport.cc:1182] unix:/tmp/tmpfvap5r3z_milvus.db.sock: Got goaway [11] err=UNAVAILABLE:GOAWAY received; Error code: 11; Debug Text: too_many_pings {created_time:"2026-04-24T11:06:38.86438384+00:00", http2_error:11, grpc_status:14}
E0000 00:00:1777028798.864538  188484 chttp2_transport.cc:1210] unix:/tmp/tmpfvap5r3z_milvus.db.sock: Received a GOAWAY with error code ENHANCE_YOUR_CALM and debug data equal to "too_many_pings". Current keepalive time (before throttling): 10000ms
[I 2026-04-24T11:06:39.465] 40 elements to insert to milvus
[I 2026-04-24T11:06:39.465] threshold for streaming is 1000
[I 2026-04-24T11:06:40.223] streamed 40 records
[I 2026-04-24T11:06:40.224] Purging saved results from disk after successful VDB upload.
[W 2026-04-24T11:06:40.224] Purge requested, but save_to_disk was not configured. No files to purge.
[I 2026-04-24T11:06:40.224]     minion-tech.pdf → results=1  failures=0  time=65.261s
[I 2026-04-24T11:06:40.225]   ── doc wall time: 65.359s  ──
[I 2026-04-24T11:06:40.225]   [8/9] Ingesting: multimodal_test.pdf
[I 2026-04-24T11:06:40.225] PHASE START : Extract  [multimodal_test.pdf]
[I 2026-04-24T11:06:40.226] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-24T11:07:11.334] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
[I 2026-04-24T11:07:11.921] 13 elements to insert to milvus
[I 2026-04-24T11:07:11.921] threshold for streaming is 1000
[I 2026-04-24T11:07:12.201] streamed 13 records
[I 2026-04-24T11:07:12.202] Purging saved results from disk after successful VDB upload.
[W 2026-04-24T11:07:12.202] Purge requested, but save_to_disk was not configured. No files to purge.
[I 2026-04-24T11:07:12.202]     multimodal_test.pdf → results=1  failures=0  time=31.976s
[I 2026-04-24T11:07:12.202]   ── doc wall time: 31.978s  ──
[I 2026-04-24T11:07:12.202]   [9/9] Ingesting: policy-2.pdf
[I 2026-04-24T11:07:12.202] PHASE START : Extract  [policy-2.pdf]
[I 2026-04-24T11:07:12.207] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-24T11:07:43.342] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
[I 2026-04-24T11:07:43.934] 23 elements to insert to milvus
[I 2026-04-24T11:07:43.934] threshold for streaming is 1000
[I 2026-04-24T11:07:44.378] streamed 23 records
[I 2026-04-24T11:07:44.379] Purging saved results from disk after successful VDB upload.
[W 2026-04-24T11:07:44.379] Purge requested, but save_to_disk was not configured. No files to purge.
[I 2026-04-24T11:07:44.379]     policy-2.pdf → results=1  failures=0  time=32.172s
[I 2026-04-24T11:07:44.380]   ── doc wall time: 32.177s  ──
[I 2026-04-24T11:07:44.380] ══════════════════════════════════════════════════════════════════════
[I 2026-04-24T11:07:44.380] BATCH INGEST SUMMARY
[I 2026-04-24T11:07:44.380]   Total documents   : 9
[I 2026-04-24T11:07:44.380]   Successful        : 9
[I 2026-04-24T11:07:44.380]   Total failures    : 0
[I 2026-04-24T11:07:44.380]   Total batch time  : 464.163s
[I 2026-04-24T11:07:44.380]   Avg per document  : 51.574s
[I 2026-04-24T11:07:44.380] ══════════════════════════════════════════════════════════════════════
[I 2026-04-24T11:07:44.385] ══════════════════════════════════════════════════════════════════════
[I 2026-04-24T11:07:44.385] STEP 3 — RAG RETRIEVAL + LLM INFERENCE
[I 2026-04-24T11:07:44.385] ══════════════════════════════════════════════════════════════════════
[I 2026-04-24T11:07:44.396]   Q1: Why did economics and physics become early movers in open access adoption?...
[I 2026-04-24T11:07:47.360] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
[I 2026-04-24T11:07:47.495]     Retrieval  : 3.098s  |  chunks found: 10
[I 2026-04-24T11:07:52.885] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
[I 2026-04-24T11:07:52.894]     LLM Inference: 5.398s  |  prompt_tokens=3700  completion_tokens=38  tokens/sec=7.0
[I 2026-04-24T11:07:52.894]     Answer preview: The answer is not in the context. The provided context is about "Regulatory Guidelines for Telecommunication of Medical Orders" and does not mention e...
[I 2026-04-24T11:07:52.894]   Q2: How did arXiv influence scholarly communication in physics?...
[I 2026-04-24T11:07:53.113] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
[I 2026-04-24T11:07:53.243]     Retrieval  : 0.348s  |  chunks found: 10
[I 2026-04-24T11:07:54.586] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
[I 2026-04-24T11:07:54.587]     LLM Inference: 1.344s  |  prompt_tokens=3314  completion_tokens=41  tokens/sec=30.5
[I 2026-04-24T11:07:54.587]     Answer preview: The answer is not in the context. The provided context is about "Regulatory Guidelines for Telecommunication of Medical Orders" and does not mention a...
[I 2026-04-24T11:07:54.587]   Q3: Why did life sciences move more toward open access journals and APC models inste...
[I 2026-04-24T11:07:54.820] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
[I 2026-04-24T11:07:54.936]     Retrieval  : 0.348s  |  chunks found: 10
[I 2026-04-24T11:08:02.416] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
[I 2026-04-24T11:08:02.417]     LLM Inference: 7.481s  |  prompt_tokens=3808  completion_tokens=48  tokens/sec=6.4
[I 2026-04-24T11:08:02.417]     Answer preview: The answer is not in the context. The provided context is about regulatory guidelines for telecommunication of medical orders and documentation standa...
[I 2026-04-24T11:08:02.417]   Q4: What does the report mean by saying open access has grown through successive wav...
[I 2026-04-24T11:08:02.675] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
[I 2026-04-24T11:08:02.790]     Retrieval  : 0.372s  |  chunks found: 10
[I 2026-04-24T11:08:04.827] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
[I 2026-04-24T11:08:04.827]     LLM Inference: 2.037s  |  prompt_tokens=4227  completion_tokens=63  tokens/sec=30.9
[I 2026-04-24T11:08:04.828]     Answer preview: The answer is not in the context. The provided context appears to be a medical documentation standards policy, and it does not mention "open access" o...
[I 2026-04-24T11:08:04.828]   Q5: How does the report connect open access, open data, and reproducibility?...
[I 2026-04-24T11:08:05.069] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
[I 2026-04-24T11:08:05.214]     Retrieval  : 0.386s  |  chunks found: 10
[I 2026-04-24T11:08:08.180] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
[I 2026-04-24T11:08:08.181]     LLM Inference: 2.966s  |  prompt_tokens=4155  completion_tokens=70  tokens/sec=23.6
[I 2026-04-24T11:08:08.181]     Answer preview: The context provided does not contain information about how the report connects open access, open data, and reproducibility. The context appears to be...
[I 2026-04-24T11:08:08.181] ----------------------------------------------------------------------
[I 2026-04-24T11:08:08.181]   Q      Retrieval         LLM    Tokens    Tok/s
[I 2026-04-24T11:08:08.182] ----------------------------------------------------------------------
[I 2026-04-24T11:08:08.182]   Q1        3.098s      5.398s      3738     7.0
[I 2026-04-24T11:08:08.182]   Q2        0.348s      1.344s      3355    30.5
[I 2026-04-24T11:08:08.182]   Q3        0.348s      7.481s      3856     6.4
[I 2026-04-24T11:08:08.182]   Q4        0.372s      2.037s      4290    30.9
[I 2026-04-24T11:08:08.182]   Q5        0.386s      2.966s      4225    23.6
[I 2026-04-24T11:08:08.182] ----------------------------------------------------------------------
[I 2026-04-24T11:08:08.182]   Answers saved → results/answers_20260424_105908.json
[I 2026-04-24T11:08:08.183]   Metrics saved → results/metrics_20260424_105908.json
[I 2026-04-24T11:08:08.183] ══════════════════════════════════════════════════════════════════════
[I 2026-04-24T11:08:08.183] FINAL RUN SUMMARY
[I 2026-04-24T11:08:08.183] ══════════════════════════════════════════════════════════════════════
[I 2026-04-24T11:08:08.183]   Run ID                  : 20260424_105908
[I 2026-04-24T11:08:08.183]   Pipeline init time      : 0.016s
[I 2026-04-24T11:08:08.183]   Sanity check time       : 31.822s
[I 2026-04-24T11:08:08.183]   Total batch ingest time : 464.163s
[I 2026-04-24T11:08:08.184]   Total documents         : 9
[I 2026-04-24T11:08:08.184]   Successful documents    : 9
[I 2026-04-24T11:08:08.184]   Avg ingest per doc      : 51.517s
[I 2026-04-24T11:08:08.184]   Total wall time         : 496.001s
[I 2026-04-24T11:08:08.184] ----------------------------------------------------------------------
[I 2026-04-24T11:08:08.184]   Log file     → results/pipeline_run_20260424_105908.log
[I 2026-04-24T11:08:08.184]   Metrics JSON → results/metrics_20260424_105908.json
[I 2026-04-24T11:08:08.184]   Answers JSON → results/answers_20260424_105908.json
[I 2026-04-24T11:08:08.184] ══════════════════════════════════════════════════════════════════════
[I 2026-04-24T11:08:08.184]   Total wall clock time : 539.814s
[I 2026-04-24T11:08:08.184]   PIPELINE COMPLETED SUCCESSFULLY
[I 2026-04-24T11:08:08.184] ══════════════════════════════════════════════════════════════════════
Killed subprocess group 188563
E20260424 11:08:09.266176 195336 server.cpp:47] [SERVER][BlockLock][milvus] Process exit
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ 
