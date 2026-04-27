taskset -c 0-7 python pipelinecp.py
[I 2026-04-27T06:18:21.029] Centralized logging configured (console only) console_level=INFO log_dir=none (NEMO_EVALUATOR_LOG_DIR not set) logger=nemo_evaluator.logging.utils
[I 2026-04-27T06:18:25.906] ══════════════════════════════════════════════════════════════════════
[I 2026-04-27T06:18:25.906] NV-INGEST 25.9.0 — LIBRARY MODE PIPELINE (FULLY CORRECTED)
[I 2026-04-27T06:18:25.906] Run started at : 2026-04-27 06:18:25
[I 2026-04-27T06:18:25.906] Python version : 3.12.13 | packaged by Anaconda, Inc. | (main, Mar 19 2026, 20:20:58) [GCC 14.3.0]
[I 2026-04-27T06:18:25.906] Results dir    : Outputs/
[I 2026-04-27T06:18:25.906] Log file       : Outputs/pipeline_run_20260427_061825.log
[I 2026-04-27T06:18:25.906] Question mode  : PREDEFINED
[I 2026-04-27T06:18:25.906] ══════════════════════════════════════════════════════════════════════
[I 2026-04-27T06:18:25.906]   Using 5 predefined question(s).
[I 2026-04-27T06:18:25.907] ══════════════════════════════════════════════════════════════════════
[I 2026-04-27T06:18:25.907] STEP 0 — ENVIRONMENT INJECTION + VERIFICATION
[I 2026-04-27T06:18:25.907] ══════════════════════════════════════════════════════════════════════
[I 2026-04-27T06:18:25.907]   NVIDIA_API_KEY        : ********jMJZNZ
[I 2026-04-27T06:18:25.907]   YOLOX_HTTP_ENDPOINT                             SET TO DEFAULT: http://localhost:8000/v1/infer
[I 2026-04-27T06:18:25.907]   YOLOX_INFER_PROTOCOL                            SET TO DEFAULT: http
[I 2026-04-27T06:18:25.907]   YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT            SET TO DEFAULT: http://localhost:8003/v1/infer
[I 2026-04-27T06:18:25.907]   YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL           SET TO DEFAULT: http
[I 2026-04-27T06:18:25.907]   YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT             SET TO DEFAULT: http://localhost:8006/v1/infer
[I 2026-04-27T06:18:25.907]   YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL            SET TO DEFAULT: http
[I 2026-04-27T06:18:25.907]   OCR_HTTP_ENDPOINT                               SET TO DEFAULT: http://localhost:8009/v1/infer
[I 2026-04-27T06:18:25.907]   OCR_INFER_PROTOCOL                              SET TO DEFAULT: http
[I 2026-04-27T06:18:25.907]   All 4 NIM HTTP endpoints are set in os.environ ✓
[I 2026-04-27T06:18:25.907]   Environment injection complete. run_pipeline() will inherit these.
[I 2026-04-27T06:18:25.907] ══════════════════════════════════════════════════════════════════════
[I 2026-04-27T06:18:25.907] FILE DISCOVERY  →  folder: Docs/
[I 2026-04-27T06:18:25.907]   Formats: *.pdf, *.docx, *.pptx, *.jpeg, *.jpg, *.png
[I 2026-04-27T06:18:25.908]   *.pdf        → 9 file(s)
[I 2026-04-27T06:18:25.908]   *.jpeg       → 2 file(s)
[I 2026-04-27T06:18:25.908]   *.jpg        → 2 file(s)
[I 2026-04-27T06:18:25.909]   *.png        → 1 file(s)
[I 2026-04-27T06:18:25.909]   ─── Total: 14 file(s) ───
[I 2026-04-27T06:18:25.909]   › Docs/Ascent_of_Open.pdf  (4.30 MB)
[I 2026-04-27T06:18:25.909]   › Docs/DOC-20260407-WA0009..pdf  (0.10 MB)
[I 2026-04-27T06:18:25.909]   › Docs/Driving.jpg  (0.02 MB)
[I 2026-04-27T06:18:25.909]   › Docs/Infinity-Ensure-Brochure.pdf  (0.69 MB)
[I 2026-04-27T06:18:25.909]   › Docs/Oxford.pdf  (3.99 MB)
[I 2026-04-27T06:18:25.909]   › Docs/PK0016.pdf  (0.32 MB)
[I 2026-04-27T06:18:25.909]   › Docs/Screenshot (1).png  (0.09 MB)
[I 2026-04-27T06:18:25.909]   › Docs/Singapore_NID_B 1.jpeg  (0.16 MB)
[I 2026-04-27T06:18:25.909]   › Docs/Singapore_NID_F 1.jpeg  (0.17 MB)
[I 2026-04-27T06:18:25.909]   › Docs/california-drivers-license-small 1 1.jpg  (0.04 MB)
[I 2026-04-27T06:18:25.909]   › Docs/invoice-0-4.pdf  (0.06 MB)
[I 2026-04-27T06:18:25.909]   › Docs/minion-tech.pdf  (10.54 MB)
[I 2026-04-27T06:18:25.909]   › Docs/multimodal_test.pdf  (0.13 MB)
[I 2026-04-27T06:18:25.909]   › Docs/policy-2.pdf  (0.83 MB)
[I 2026-04-27T06:18:25.910] ══════════════════════════════════════════════════════════════════════
[I 2026-04-27T06:18:25.910] PIPELINE INITIALISATION
[I 2026-04-27T06:18:25.910] ══════════════════════════════════════════════════════════════════════
2026-04-27 06:19:02.815666165 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/MSFT1000:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
[I 2026-04-27T06:19:04.624] Detected 32 logical cores via psutil.
[I 2026-04-27T06:19:04.624] Detected 16 physical cores via psutil.
[I 2026-04-27T06:19:04.624] Detected 8 cores via os.sched_getaffinity.
[I 2026-04-27T06:19:04.624] Raw CPU limit determined: 8.00 (Method: sched_affinity)
[I 2026-04-27T06:19:04.625] Effective CPU core limit determined: 8.00 (Method: sched_affinity)
[I 2026-04-27T06:19:04.724] PHASE START : Pipeline subprocess start
[I 2026-04-27T06:19:04.724] Launching pipeline in Python subprocess using multiprocessing.
[I 2026-04-27T06:19:04.735] Pipeline subprocess started (PID=64638)
[I 2026-04-27T06:19:04.735] PHASE END   : Pipeline subprocess start  →  0.011s
[I 2026-04-27T06:19:04.735]   Pipeline process launched. Waiting for port 7671 (max 90s)...
[I 2026-04-27T06:19:04.735]   Waiting for pipeline port 7671 to be ready (max 90s)...
[I 2026-04-27T06:19:18.750]   Port 7671 is accepting connections ✓
[I 2026-04-27T06:19:23.752]   NvIngestClient connected  →  localhost:7671 ✓
[I 2026-04-27T06:19:23.752] ══════════════════════════════════════════════════════════════════════
[I 2026-04-27T06:19:23.752] NIM HEALTH CHECK — verifying all 4 NIM endpoints
[I 2026-04-27T06:19:23.752] ══════════════════════════════════════════════════════════════════════
[W 2026-04-27T06:19:23.773]   ✗  page-elements (8000)  →  HTTP 404  (NIM may not be ready)
[W 2026-04-27T06:19:23.789]   ✗  graphic-elements (8003)  →  HTTP 000  (NIM may not be ready)
[W 2026-04-27T06:19:23.803]   ✗  table-structure (8006)  →  HTTP 000  (NIM may not be ready)
[W 2026-04-27T06:19:23.818]   ✗  ocr (8009)  →  HTTP 000  (NIM may not be ready)
[W 2026-04-27T06:19:23.818]   One or more NIMs did not respond — extraction for those modalities will fall back
[W 2026-04-27T06:19:23.818]   Proceeding anyway — check docker ps and docker logs if results are empty
[I 2026-04-27T06:19:23.973] ══════════════════════════════════════════════════════════════════════
[I 2026-04-27T06:19:23.973] SANITY CHECK (text-only)  →  Docs/Ascent_of_Open.pdf
[I 2026-04-27T06:19:23.973] ══════════════════════════════════════════════════════════════════════
[I 2026-04-27T06:19:23.974] PHASE START : Text-only extraction
Processing:   0%|                                                                       | 0/1 [00:00<?, ?doc/s][I 2026-04-27T06:19:24.051] Starting batch processing for 1 jobs with batch size 32.
Processing: 100%|███████████████████████████████████████████████████████████████| 1/1 [00:31<00:00, 31.30s/doc][I 2026-04-27T06:19:55.348] Batch processing finished. Success: 1, Failures: 0. Total accounted for: 1/1
Processing: 100%|███████████████████████████████████████████████████████████████| 1/1 [00:31<00:00, 31.30s/doc]
[I 2026-04-27T06:19:55.348] PHASE END   : Text-only extraction  →  31.375s
[I 2026-04-27T06:19:55.348]   Results  : 1
[I 2026-04-27T06:19:55.348]   Failures : 0
 in, nurture and suppor...nce is a technology company working to make research more efficient. We invest 
[I 2026-04-27T06:19:55.349]   SANITY CHECK PASSED ✓
[I 2026-04-27T06:19:55.349] ══════════════════════════════════════════════════════════════════════
[I 2026-04-27T06:19:55.349] BATCH INGEST  →  14 file(s)
[I 2026-04-27T06:19:55.349]   Collection : multimodal_docs
[I 2026-04-27T06:19:55.349]   Milvus URI : milvus.db
[I 2026-04-27T06:19:55.349]   Embedder   : dense_dim=2048  tokenizer=intfloat/e5-large-unsupervised
[I 2026-04-27T06:19:55.349] ══════════════════════════════════════════════════════════════════════
[I 2026-04-27T06:19:55.349]   [1/14] Ingesting: Ascent_of_Open.pdf
[I 2026-04-27T06:19:55.400] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:20:10.575] Batch processing finished. Success: 0, Failures: 1. Total accounted for: 1/1
[W 2026-04-27T06:20:10.575] Job was not completely successful. 0 out of 1 records completed successfully. Uploading successful results to vector database.
[W 2026-04-27T06:20:10.575]     FAILURE [0] in Ascent_of_Open.pdf: ('1:Docs/Ascent_of_Open.pdf', '[]: failed\nFailed to process the message.\n↪ Event that caused this failure: annotation::5a368d8e-08d7-429f-a0e8-5b7a9ac3b060 -> Error in on_data: extract_primitives_fr
[I 2026-04-27T06:20:10.575]     Ascent_of_Open.pdf → results=0  failures=1  time=15.226s
[I 2026-04-27T06:20:10.576]   ── doc wall time: 15.226s ──
[I 2026-04-27T06:20:10.576]   [2/14] Ingesting: DOC-20260407-WA0009..pdf
[I 2026-04-27T06:20:10.582] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:20:17.672] Batch processing finished. Success: 0, Failures: 1. Total accounted for: 1/1
[W 2026-04-27T06:20:17.672] Job was not completely successful. 0 out of 1 records completed successfully. Uploading successful results to vector database.
[W 2026-04-27T06:20:17.672]     FAILURE [0] in DOC-20260407-WA0009..pdf: ('2:Docs/DOC-20260407-WA0009..pdf', '[]: failed\nFailed to process the message.\n↪ Event that caused this failure: annotation::ea7c4449-a103-4a30-928c-daf24ba2e585 -> Error in on_data: extract_primiti
[I 2026-04-27T06:20:17.672]     DOC-20260407-WA0009..pdf → results=0  failures=1  time=7.096s
[I 2026-04-27T06:20:17.672]   ── doc wall time: 7.097s ──
[I 2026-04-27T06:20:17.672]   [3/14] Ingesting: Driving.jpg
[I 2026-04-27T06:20:17.820] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:20:24.908] Batch processing finished. Success: 0, Failures: 1. Total accounted for: 1/1
[W 2026-04-27T06:20:24.908] Job was not completely successful. 0 out of 1 records completed successfully. Uploading successful results to vector database.
[W 2026-04-27T06:20:24.908]     FAILURE [0] in Driving.jpg: ('3:Docs/Driving.jpg', "[]: failed\nFailed to process the message.\n↪ Event that caused this failure: annotation::896e8eb5-9aa9-4f54-8aca-37c7fdb97fb7 -> Error in on_data: extract_primitives_from_imag
[I 2026-04-27T06:20:24.908]     Driving.jpg → results=0  failures=1  time=7.236s
[I 2026-04-27T06:20:24.908]   ── doc wall time: 7.236s ──
[I 2026-04-27T06:20:24.908]   [4/14] Ingesting: Infinity-Ensure-Brochure.pdf
[I 2026-04-27T06:20:24.917] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:20:40.023] Batch processing finished. Success: 0, Failures: 1. Total accounted for: 1/1
[W 2026-04-27T06:20:40.023] Job was not completely successful. 0 out of 1 records completed successfully. Uploading successful results to vector database.
[W 2026-04-27T06:20:40.023]     FAILURE [0] in Infinity-Ensure-Brochure.pdf: ('4:Docs/Infinity-Ensure-Brochure.pdf', '[]: failed\nFailed to process the message.\n↪ Event that caused this failure: annotation::de073afb-0577-46c1-a9b2-9e882e9d4129 -> Error in on_data: extract_pri
[I 2026-04-27T06:20:40.023]     Infinity-Ensure-Brochure.pdf → results=0  failures=1  time=15.115s
[I 2026-04-27T06:20:40.023]   ── doc wall time: 15.115s ──
[I 2026-04-27T06:20:40.023]   [5/14] Ingesting: Oxford.pdf
[I 2026-04-27T06:20:40.063] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:21:43.224] Batch processing finished. Success: 0, Failures: 1. Total accounted for: 1/1
[W 2026-04-27T06:21:43.224] Job was not completely successful. 0 out of 1 records completed successfully. Uploading successful results to vector database.
[W 2026-04-27T06:21:43.224]     FAILURE [0] in Oxford.pdf: ('5:Docs/Oxford.pdf', '[]: failed\nFailed to process the message.\n↪ Event that caused this failure: annotation::0f302379-7006-456a-82ff-6454b0b9b51d -> Error in on_data: extract_primitives_from_pdf: 
[I 2026-04-27T06:21:43.224]     Oxford.pdf → results=0  failures=1  time=63.201s
[I 2026-04-27T06:21:43.225]   ── doc wall time: 63.201s ──
[I 2026-04-27T06:21:43.225]   [6/14] Ingesting: PK0016.pdf
[I 2026-04-27T06:21:43.234] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:21:58.332] Batch processing finished. Success: 0, Failures: 1. Total accounted for: 1/1
[W 2026-04-27T06:21:58.332] Job was not completely successful. 0 out of 1 records completed successfully. Uploading successful results to vector database.
[W 2026-04-27T06:21:58.332]     FAILURE [0] in PK0016.pdf: ('6:Docs/PK0016.pdf', '[]: failed\nFailed to process the message.\n↪ Event that caused this failure: annotation::5fb6a11a-6e8f-4c87-91b4-175b62bece8e -> Error in on_data: extract_primitives_from_pdf: 
[I 2026-04-27T06:21:58.332]     PK0016.pdf → results=0  failures=1  time=15.107s
[I 2026-04-27T06:21:58.332]   ── doc wall time: 15.108s ──
[I 2026-04-27T06:21:58.333]   [7/14] Ingesting: Screenshot (1).png
[I 2026-04-27T06:21:58.373] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:22:01.461] Batch processing finished. Success: 0, Failures: 1. Total accounted for: 1/1
[W 2026-04-27T06:22:01.461] Job was not completely successful. 0 out of 1 records completed successfully. Uploading successful results to vector database.
[W 2026-04-27T06:22:01.461]     FAILURE [0] in Screenshot (1).png: ('7:Docs/Screenshot (1).png', "[]: failed\nFailed to process the message.\n↪ Event that caused this failure: annotation::77b18a5f-08b1-4ec9-afd1-21795725e852 -> Error in on_data: extract_primitives_fr
[I 2026-04-27T06:22:01.461]     Screenshot (1).png → results=0  failures=1  time=3.129s
[I 2026-04-27T06:22:01.461]   ── doc wall time: 3.129s ──
[I 2026-04-27T06:22:01.461]   [8/14] Ingesting: Singapore_NID_B 1.jpeg
[I 2026-04-27T06:22:01.467] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:22:04.557] Batch processing finished. Success: 0, Failures: 1. Total accounted for: 1/1
[W 2026-04-27T06:22:04.557] Job was not completely successful. 0 out of 1 records completed successfully. Uploading successful results to vector database.
[W 2026-04-27T06:22:04.557]     FAILURE [0] in Singapore_NID_B 1.jpeg: ('8:Docs/Singapore_NID_B 1.jpeg', "[]: failed\nFailed to process the message.\n↪ Event that caused this failure: annotation::8ac5fa40-69d1-4ba6-8d9e-d60f8acf3512 -> Error in on_data: extract_primitive
[I 2026-04-27T06:22:04.557]     Singapore_NID_B 1.jpeg → results=0  failures=1  time=3.096s
[I 2026-04-27T06:22:04.557]   ── doc wall time: 3.096s ──
[I 2026-04-27T06:22:04.557]   [9/14] Ingesting: Singapore_NID_F 1.jpeg
[I 2026-04-27T06:22:04.563] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:22:07.654] Batch processing finished. Success: 0, Failures: 1. Total accounted for: 1/1
[W 2026-04-27T06:22:07.654] Job was not completely successful. 0 out of 1 records completed successfully. Uploading successful results to vector database.
[W 2026-04-27T06:22:07.654]     FAILURE [0] in Singapore_NID_F 1.jpeg: ('9:Docs/Singapore_NID_F 1.jpeg', "[]: failed\nFailed to process the message.\n↪ Event that caused this failure: annotation::acc296c8-d089-4615-adbb-32d5bdb32412 -> Error in on_data: extract_primitive
[I 2026-04-27T06:22:07.654]     Singapore_NID_F 1.jpeg → results=0  failures=1  time=3.097s
[I 2026-04-27T06:22:07.654]   ── doc wall time: 3.097s ──
[I 2026-04-27T06:22:07.654]   [10/14] Ingesting: california-drivers-license-small 1 1.jpg
[I 2026-04-27T06:22:07.659] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:22:10.747] Batch processing finished. Success: 0, Failures: 1. Total accounted for: 1/1
[W 2026-04-27T06:22:10.747] Job was not completely successful. 0 out of 1 records completed successfully. Uploading successful results to vector database.
[W 2026-04-27T06:22:10.747]     FAILURE [0] in california-drivers-license-small 1 1.jpg: ('10:Docs/california-drivers-license-small 1 1.jpg', "[]: failed\nFailed to process the message.\n↪ Event that caused this failure: annotation::1b73111a-5c33-4159-8f55-c8ec747f275e -> Error in on_data
[I 2026-04-27T06:22:10.747]     california-drivers-license-small 1 1.jpg → results=0  failures=1  time=3.093s
[I 2026-04-27T06:22:10.747]   ── doc wall time: 3.093s ──
[I 2026-04-27T06:22:10.747]   [11/14] Ingesting: invoice-0-4.pdf
[I 2026-04-27T06:22:10.754] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:22:13.842] Batch processing finished. Success: 0, Failures: 1. Total accounted for: 1/1
[W 2026-04-27T06:22:13.842] Job was not completely successful. 0 out of 1 records completed successfully. Uploading successful results to vector database.
[W 2026-04-27T06:22:13.842]     FAILURE [0] in invoice-0-4.pdf: ('11:Docs/invoice-0-4.pdf', '[]: failed\nFailed to process the message.\n↪ Event that caused this failure: annotation::f9263a31-b42a-4391-94ad-463ae23618c2 -> Error in on_data: extract_primitives_from
[I 2026-04-27T06:22:13.842]     invoice-0-4.pdf → results=0  failures=1  time=3.095s
[I 2026-04-27T06:22:13.842]   ── doc wall time: 3.095s ──
[I 2026-04-27T06:22:13.842]   [12/14] Ingesting: minion-tech.pdf
[I 2026-04-27T06:22:13.961] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:22:29.324] Batch processing finished. Success: 0, Failures: 1. Total accounted for: 1/1
[W 2026-04-27T06:22:29.324] Job was not completely successful. 0 out of 1 records completed successfully. Uploading successful results to vector database.
[W 2026-04-27T06:22:29.324]     FAILURE [0] in minion-tech.pdf: ('12:Docs/minion-tech.pdf', '[]: failed\nFailed to process the message.\n↪ Event that caused this failure: annotation::5b7996c9-9ac2-4a9e-806c-71de450f4518 -> Error in on_data: extract_primitives_from
[I 2026-04-27T06:22:29.324]     minion-tech.pdf → results=0  failures=1  time=15.482s
[I 2026-04-27T06:22:29.324]   ── doc wall time: 15.482s ──
[I 2026-04-27T06:22:29.324]   [13/14] Ingesting: multimodal_test.pdf
[I 2026-04-27T06:22:29.331] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:22:36.421] Batch processing finished. Success: 0, Failures: 1. Total accounted for: 1/1
[W 2026-04-27T06:22:36.421] Job was not completely successful. 0 out of 1 records completed successfully. Uploading successful results to vector database.
[W 2026-04-27T06:22:36.421]     FAILURE [0] in multimodal_test.pdf: ('13:Docs/multimodal_test.pdf', '[]: failed\nFailed to process the message.\n↪ Event that caused this failure: annotation::174cb14b-4fd0-46a5-b42e-d2c0d601d905 -> Error in on_data: extract_primitives_
[I 2026-04-27T06:22:36.421]     multimodal_test.pdf → results=0  failures=1  time=7.097s
[I 2026-04-27T06:22:36.421]   ── doc wall time: 7.097s ──
[I 2026-04-27T06:22:36.421]   [14/14] Ingesting: policy-2.pdf
[I 2026-04-27T06:22:36.434] Starting batch processing for 1 jobs with batch size 32.
[I 2026-04-27T06:22:51.544] Batch processing finished. Success: 0, Failures: 1. Total accounted for: 1/1
[W 2026-04-27T06:22:51.544] Job was not completely successful. 0 out of 1 records completed successfully. Uploading successful results to vector database.
[W 2026-04-27T06:22:51.544]     FAILURE [0] in policy-2.pdf: ('14:Docs/policy-2.pdf', '[]: failed\nFailed to process the message.\n↪ Event that caused this failure: annotation::9d8531be-df82-4cf4-87b1-280135ddc66e -> Error in on_data: extract_primitives_from_pd
[I 2026-04-27T06:22:51.544]     policy-2.pdf → results=0  failures=1  time=15.123s
[I 2026-04-27T06:22:51.544]   ── doc wall time: 15.123s ──
[I 2026-04-27T06:22:51.544] ══════════════════════════════════════════════════════════════════════
[I 2026-04-27T06:22:51.545] BATCH INGEST SUMMARY
[I 2026-04-27T06:22:51.545]   Total documents  : 14
[I 2026-04-27T06:22:51.545]   Successful       : 0
[I 2026-04-27T06:22:51.545]   Total failures   : 14
[I 2026-04-27T06:22:51.545]   Total batch time : 176.195s
[I 2026-04-27T06:22:51.545]   Avg per document : 12.585s
[I 2026-04-27T06:22:51.545] ══════════════════════════════════════════════════════════════════════
[I 2026-04-27T06:22:51.545] ══════════════════════════════════════════════════════════════════════
[I 2026-04-27T06:22:51.545] MILVUS VERIFICATION — checking entity count before RAG queries
[I 2026-04-27T06:22:53.222]   Collection 'multimodal_docs' has 23 entities
[I 2026-04-27T06:22:53.223]   Milvus is populated ✓
[I 2026-04-27T06:22:53.227] ══════════════════════════════════════════════════════════════════════
[I 2026-04-27T06:22:53.227] RAG RETRIEVAL + LLM INFERENCE  →  5 question(s)
[I 2026-04-27T06:22:53.227] ══════════════════════════════════════════════════════════════════════
[I 2026-04-27T06:22:53.246]   Q1: Why did economics and physics become early movers in open access adoption?...
[I 2026-04-27T06:22:59.303] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
[I 2026-04-27T06:22:59.454]     Retrieval: 6.208s  |  chunks: 10
[I 2026-04-27T06:23:13.961] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
[I 2026-04-27T06:23:13.970]     LLM: 14.515s  |  tokens=3766  tok/s=4.5
[I 2026-04-27T06:23:13.970]     Answer preview: The answer is not in the context. The provided context is about "Regulatory Guidelines for Telecommunication of Medical ...
[I 2026-04-27T06:23:14.970]   Q2: How did arXiv influence scholarly communication in physics?...
[I 2026-04-27T06:23:15.218] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
[I 2026-04-27T06:23:15.345]     Retrieval: 0.374s  |  chunks: 10
[I 2026-04-27T06:23:17.491] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
[I 2026-04-27T06:23:17.492]     LLM: 2.146s  |  tokens=3397  tok/s=38.7
[I 2026-04-27T06:23:17.492]     Answer preview: The answer is not in the context. The context provided is related to regulatory guidelines for telecommunication of medi...
[I 2026-04-27T06:23:18.492]   Q3: Why did life sciences move more toward open access journals and APC models?...
[I 2026-04-27T06:23:18.793] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
[I 2026-04-27T06:23:19.007]     Retrieval: 0.514s  |  chunks: 10
[I 2026-04-27T06:23:20.704] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
[I 2026-04-27T06:23:20.704]     LLM: 1.697s  |  tokens=3824  tok/s=34.2
[I 2026-04-27T06:23:20.704]     Answer preview: The answer to the question "Why did life sciences move more toward open access journals and APC models?" is not in the p...
[I 2026-04-27T06:23:21.705]   Q4: What does the report mean by successive waves of open access innovation?...
[I 2026-04-27T06:23:21.940] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
[I 2026-04-27T06:23:22.065]     Retrieval: 0.360s  |  chunks: 10
[I 2026-04-27T06:23:36.006] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
[I 2026-04-27T06:23:36.006]     LLM: 13.941s  |  tokens=3868  tok/s=4.5
[I 2026-04-27T06:23:36.006]     Answer preview: The context provided does not mention "successive waves of open access innovation". The text appears to be a medical doc...
[I 2026-04-27T06:23:37.007]   Q5: How does the report connect open access, open data, and reproducibility?...
[I 2026-04-27T06:23:37.237] HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
[I 2026-04-27T06:23:37.367]     Retrieval: 0.360s  |  chunks: 10
[I 2026-04-27T06:23:38.611] HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
[I 2026-04-27T06:23:38.612]     LLM: 1.244s  |  tokens=4197  tok/s=33.8
[I 2026-04-27T06:23:38.612]     Answer preview: The answer is not in the context. The provided context appears to be a medical policy document outlining documentation s...
[I 2026-04-27T06:23:39.612]   Answers saved → Outputs/answers_20260427_061825.json
[I 2026-04-27T06:23:39.613]   Metrics saved → Outputs/metrics_20260427_061825.json
[I 2026-04-27T06:23:39.613] ══════════════════════════════════════════════════════════════════════
[I 2026-04-27T06:23:39.613] FINAL RUN SUMMARY
[I 2026-04-27T06:23:39.613] ══════════════════════════════════════════════════════════════════════
[I 2026-04-27T06:23:39.613]   Run ID                  : 20260427_061825
[I 2026-04-27T06:23:39.613]   Pipeline init time      : 0.011s
[I 2026-04-27T06:23:39.613]   Sanity check time       : 31.375s
[I 2026-04-27T06:23:39.613]   Total batch ingest time : 176.195s
[I 2026-04-27T06:23:39.614]   Total documents         : 14
[I 2026-04-27T06:23:39.614]   Successful documents    : 0
[I 2026-04-27T06:23:39.614]   Avg ingest per doc      : 12.585s
[I 2026-04-27T06:23:39.614]   Total wall time         : 207.581s
[I 2026-04-27T06:23:39.614] ----------------------------------------------------------------------
[I 2026-04-27T06:23:39.614]   Log file     → Outputs/pipeline_run_20260427_061825.log
[I 2026-04-27T06:23:39.614]   Metrics JSON → Outputs/metrics_20260427_061825.json
[I 2026-04-27T06:23:39.614]   Answers JSON → Outputs/answers_20260427_061825.json
[I 2026-04-27T06:23:39.614] ══════════════════════════════════════════════════════════════════════
[I 2026-04-27T06:23:39.614]   Total wall clock time : 313.708s
[I 2026-04-27T06:23:39.614]   PIPELINE COMPLETED SUCCESSFULLY
[I 2026-04-27T06:23:39.614] ══════════════════════════════════════════════════════════════════════
Killed subprocess group 64638
E20260427 06:23:40.702111 78818 server.cpp:47] [SERVER][BlockLock][milvus] Process exit
