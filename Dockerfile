2026-04-28 06:49:12  INFO      ========================================================================
2026-04-28 06:49:12  INFO      NV-INGEST 25.9.0  LIBRARY MODE  v6  |  hybrid BM25+semantic + rerank
2026-04-28 06:49:12  INFO        2026-04-28 06:49:12  |  Python 3.12.13
2026-04-28 06:49:12  INFO        INTERACTIVE question mode
2026-04-28 06:49:12  INFO        Collection -> multimodal_docs_20260428_064912
2026-04-28 06:49:12  INFO        Log        -> /home/clouduser01/jaswanth/Outputs/pipeline_run_20260428_064912.log
2026-04-28 06:49:12  INFO      ========================================================================
2026-04-28 06:49:57  INFO      Collected 3 question(s).
2026-04-28 06:49:57  INFO      ========================================================================
2026-04-28 06:49:57  INFO      ENVIRONMENT SETUP
2026-04-28 06:49:57  INFO      ========================================================================
2026-04-28 06:49:57  INFO        NVIDIA_API_KEY : ********jMJZNZ
2026-04-28 06:49:57  INFO        YOLOX_HTTP_ENDPOINT                             -> http://localhost:8000/v1/infer
2026-04-28 06:49:57  INFO        YOLOX_INFER_PROTOCOL                            -> http
2026-04-28 06:49:57  INFO        YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT            -> http://localhost:8003/v1/infer
2026-04-28 06:49:57  INFO        YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL           -> http
2026-04-28 06:49:57  INFO        YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT             -> http://localhost:8006/v1/infer
2026-04-28 06:49:57  INFO        YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL            -> http
2026-04-28 06:49:57  INFO        OCR_HTTP_ENDPOINT                               -> http://localhost:8009/v1/infer
2026-04-28 06:49:57  INFO        OCR_INFER_PROTOCOL                              -> http
2026-04-28 06:49:57  INFO        Collection     : multimodal_docs_20260428_064912
2026-04-28 06:49:57  INFO        Retrieval      : sparse=True  top_k=30  rerank_k=8
2026-04-28 06:49:57  INFO        Reranker       : nvidia/llama-nemotron-rerank-1b-v2
2026-04-28 06:49:57  INFO      ========================================================================
2026-04-28 06:49:57  INFO      FILE DISCOVERY  ->  Docs/
2026-04-28 06:49:57  INFO        *.pdf        -> 10
2026-04-28 06:49:57  INFO        *.jpeg       -> 2
2026-04-28 06:49:57  INFO        *.jpg        -> 2
2026-04-28 06:49:57  INFO        *.png        -> 1
2026-04-28 06:49:57  INFO        Total: 15 file(s)
2026-04-28 06:49:57  INFO          Ascent_of_Open.pdf                                   4.30 MB
2026-04-28 06:49:57  INFO          DOC-20260407-WA0009..pdf                             0.10 MB
2026-04-28 06:49:57  INFO          Driving.jpg                                          0.02 MB
2026-04-28 06:49:57  INFO          Infinity-Ensure-Brochure.pdf                         0.69 MB
2026-04-28 06:49:57  INFO          Oxford.pdf                                           3.99 MB
2026-04-28 06:49:57  INFO          PK0016.pdf                                           0.32 MB
2026-04-28 06:49:57  INFO          Screenshot (1).png                                   0.09 MB
2026-04-28 06:49:57  INFO          Singapore_NID_B 1.jpeg                               0.16 MB
2026-04-28 06:49:57  INFO          Singapore_NID_F 1.jpeg                               0.17 MB
2026-04-28 06:49:57  INFO          california-drivers-license-small 1 1.jpg             0.04 MB
2026-04-28 06:49:57  INFO          invoice-0-4.pdf                                      0.06 MB
2026-04-28 06:49:57  INFO          merger_agreement 1.pdf                               0.55 MB
2026-04-28 06:49:57  INFO          minion-tech.pdf                                      10.54 MB
2026-04-28 06:49:57  INFO          multimodal_test.pdf                                  0.13 MB
2026-04-28 06:49:57  INFO          policy-2.pdf                                         0.83 MB
2026-04-28 06:49:57  INFO      ========================================================================
2026-04-28 06:49:57  INFO      PIPELINE INIT
2026-04-28 06:49:57  INFO      ========================================================================
2026-04-28 06:50:05  INFO      >> Pipeline subprocess
2026-04-28 06:50:05  INFO        Polling port 7671 (max 90s)...
2026-04-28 06:50:19  INFO        Port 7671 ready.
2026-04-28 06:50:24  INFO      << Pipeline subprocess  19.029s
2026-04-28 06:50:24  INFO        NvIngestClient connected -> localhost:7671
2026-04-28 06:50:24  INFO      ========================================================================
2026-04-28 06:50:24  INFO      NIM HEALTH CHECK
2026-04-28 06:50:24  INFO      ========================================================================
2026-04-28 06:50:24  INFO        FAIL  page_elements          port 8000  HTTP 404
2026-04-28 06:50:24  WARNING          Ready path returned 404 on port 8000. Try /v1/health or /health/ready
2026-04-28 06:50:24  INFO        FAIL  graphic_elements       port 8003  HTTP 000
2026-04-28 06:50:24  WARNING          Fix: docker compose --profile yolox-graphic-elements up -d
2026-04-28 06:50:24  INFO        FAIL  table_structure        port 8006  HTTP 000
2026-04-28 06:50:24  WARNING          Fix: docker compose --profile yolox-table-structure up -d
2026-04-28 06:50:24  INFO        FAIL  ocr                    port 8009  HTTP 000
2026-04-28 06:50:24  WARNING          Fix: docker compose --profile ocr up -d
2026-04-28 06:50:24  INFO        0/4 NIMs healthy. Text-only fallback will be used where needed.
2026-04-28 06:50:24  INFO      ========================================================================
2026-04-28 06:50:24  INFO      SANITY CHECK (text-only) -> Ascent_of_Open.pdf
2026-04-28 06:50:24  INFO      ========================================================================
2026-04-28 06:50:24  INFO      >> Sanity
2026-04-28 06:50:56  INFO      << Sanity  31.321s
2026-04-28 06:50:56  INFO        Results: 1  Failures: 0
2026-04-28 06:50:56  INFO        Preview: Digital Research Reports
 The Ascent of Open Access
 An analysis of the Open Access landscape since the turn of the millennium
 Daniel W Hook, Ian Calvert and Mark Hahnel
 JANUARY 2019 Digital Science...
2026-04-28 06:50:56  INFO        SANITY PASSED
2026-04-28 06:50:56  INFO      Collection 'multimodal_docs_20260428_064912' does not exist yet.
2026-04-28 06:50:56  INFO      ========================================================================
2026-04-28 06:50:56  INFO      BATCH INGEST  ->  15 files  |  TEXT-ONLY + HYBRID BM25
2026-04-28 06:50:56  INFO        Collection: multimodal_docs_20260428_064912  Milvus: milvus.db
2026-04-28 06:50:56  INFO        Chunk: 512 tok  overlap: 50  sparse=True
2026-04-28 06:50:56  INFO      ========================================================================
2026-04-28 06:50:56  INFO        [1/15]  Ascent_of_Open.pdf  [TEXT+HYBRID]  sparse=True
2026-04-28 06:51:06  INFO          results=1  failures=0  9.310s
2026-04-28 06:51:06  INFO        -- wall: 9.310s --
2026-04-28 06:51:06  INFO        [2/15]  DOC-20260407-WA0009..pdf  [TEXT+HYBRID]  sparse=True
2026-04-28 06:51:14  INFO          results=1  failures=0  8.285s
2026-04-28 06:51:14  INFO        -- wall: 8.285s --
2026-04-28 06:51:14  INFO        [3/15]  Driving.jpg  [SKIPPED - image file, OCR NIM not running]
2026-04-28 06:51:14  INFO        -- wall: 0.000s --
2026-04-28 06:51:14  INFO        [4/15]  Infinity-Ensure-Brochure.pdf  [TEXT+HYBRID]  sparse=True
2026-04-28 06:51:23  INFO          results=1  failures=0  8.825s
2026-04-28 06:51:23  INFO        -- wall: 8.825s --
2026-04-28 06:51:23  INFO        [5/15]  Oxford.pdf  [TEXT+HYBRID]  sparse=True
2026-04-28 06:51:32  INFO          results=1  failures=0  9.436s
2026-04-28 06:51:32  INFO        -- wall: 9.437s --
2026-04-28 06:51:32  INFO        [6/15]  PK0016.pdf  [TEXT+HYBRID]  sparse=True
2026-04-28 06:51:41  INFO          results=1  failures=0  9.152s
2026-04-28 06:51:41  INFO        -- wall: 9.152s --
2026-04-28 06:51:41  INFO        [7/15]  Screenshot (1).png  [SKIPPED - image file, OCR NIM not running]
2026-04-28 06:51:41  INFO        -- wall: 0.000s --
2026-04-28 06:51:41  INFO        [8/15]  Singapore_NID_B 1.jpeg  [SKIPPED - image file, OCR NIM not running]
2026-04-28 06:51:41  INFO        -- wall: 0.000s --
2026-04-28 06:51:41  INFO        [9/15]  Singapore_NID_F 1.jpeg  [SKIPPED - image file, OCR NIM not running]
2026-04-28 06:51:41  INFO        -- wall: 0.000s --
2026-04-28 06:51:41  INFO        [10/15]  california-drivers-license-small 1 1.jpg  [SKIPPED - image file, OCR NIM not running]
2026-04-28 06:51:41  INFO        -- wall: 0.000s --
2026-04-28 06:51:41  INFO        [11/15]  invoice-0-4.pdf  [TEXT+HYBRID]  sparse=True
2026-04-28 06:51:51  INFO          results=1  failures=0  9.118s
2026-04-28 06:51:51  INFO        -- wall: 9.118s --
2026-04-28 06:51:51  INFO        [12/15]  merger_agreement 1.pdf  [TEXT+HYBRID]  sparse=True
2026-04-28 06:51:59  INFO          results=1  failures=0  8.841s
2026-04-28 06:51:59  INFO        -- wall: 8.842s --
2026-04-28 06:51:59  INFO        [13/15]  minion-tech.pdf  [TEXT+HYBRID]  sparse=True
2026-04-28 06:52:08  INFO          results=1  failures=0  9.031s
2026-04-28 06:52:08  INFO        -- wall: 9.031s --
2026-04-28 06:52:08  INFO        [14/15]  multimodal_test.pdf  [TEXT+HYBRID]  sparse=True
2026-04-28 06:52:13  INFO          results=1  failures=0  4.420s
2026-04-28 06:52:13  INFO        -- wall: 4.421s --
2026-04-28 06:52:13  INFO        [15/15]  policy-2.pdf  [TEXT+HYBRID]  sparse=True
2026-04-28 06:52:22  INFO          results=1  failures=0  8.751s
2026-04-28 06:52:22  INFO        -- wall: 8.752s --
2026-04-28 06:52:22  INFO      ========================================================================
2026-04-28 06:52:22  INFO      BATCH SUMMARY
2026-04-28 06:52:22  INFO        Total: 15  OK: 10  Skipped: 5  Failed: 0
2026-04-28 06:52:22  INFO        Total: 85.175s  Avg: 5.678s
2026-04-28 06:52:22  INFO      ========================================================================
2026-04-28 06:52:22  INFO      ========================================================================
2026-04-28 06:52:22  INFO      MILVUS VERIFICATION
2026-04-28 06:52:22  INFO        'multimodal_docs_20260428_064912': 12 total chunks
2026-04-28 06:52:22  INFO        Hybrid retrieval: TOP_K=30 -> rerank -> keep 8
2026-04-28 06:52:22  INFO      ========================================================================
2026-04-28 06:52:22  INFO      RAG QUERIES  ->  3 question(s)
2026-04-28 06:52:22  INFO        Collection : multimodal_docs_20260428_064912
2026-04-28 06:52:22  INFO        Mode       : hybrid BM25+semantic (sparse=True)
2026-04-28 06:52:22  INFO        Fetch      : TOP_K_RETRIEVE=30 -> rerank -> RERANK_K=8
2026-04-28 06:52:22  INFO        LLM        : meta/llama-3.3-70b-instruct
2026-04-28 06:52:22  INFO      ========================================================================
2026-04-28 06:52:22  INFO        Q1: At the Effective Time of the merger, how are Public Common Units, Sponsor Owned Units, and the Series B / Series C Preferred Units treated differently, and what exactly does PDI receive in exchange for its ownership interest in Merger Sub
2026-04-28 06:52:23  INFO          Retrieval : 1.610s  |  candidates: 12/30  mode: hybrid BM25+semantic
2026-04-28 06:52:24  INFO          Reranking : 0.794s  |  kept 8/12  top_score=-22.7500  mode=api
2026-04-28 06:52:24  INFO          Sources   : policy-2.pdf', 'source_location': '', 'source_type': 'PDF', 'collection_id': '', 'date_created': '2019-03-06T16:38:27', 'last_modified': '2019-03-08T14:27:37', 'summary': '', 'partition_id': -1, 'access_level': -1, 'custom_content': None}
2026-04-28 06:52:30  INFO          LLM       : 6.066s  |  tokens=4012  tok/s=25.4
2026-04-28 06:52:30  INFO          Answer    : The provided excerpts appear to be from a medical policy document, specifically "3364-87-42 – Documentation Standards." The document outlines the standards for medical record documentation, including who is authorized to make entries, the c
2026-04-28 06:52:31  INFO        Q2: hat are the WCAG minimum contrast requirements recommended in the guide for non-text graphical objects, normal text, and large text, and how does the guide define large text?
2026-04-28 06:52:31  INFO          Retrieval : 0.402s  |  candidates: 12/30  mode: hybrid BM25+semantic
2026-04-28 06:52:31  INFO          Reranking : 0.301s  |  kept 8/12  top_score=-18.2031  mode=api
2026-04-28 06:52:31  INFO          Sources   : policy-2.pdf', 'source_location': '', 'source_type': 'PDF', 'collection_id': '', 'date_created': '2019-03-06T16:38:27', 'last_modified': '2019-03-08T14:27:37', 'summary': '', 'partition_id': -1, 'access_level': -1, 'custom_content': None}
2026-04-28 06:52:39  INFO          LLM       : 7.665s  |  tokens=3955  tok/s=25.4
2026-04-28 06:52:39  INFO          Answer    : The provided text does not mention the WCAG minimum contrast requirements. It appears to be a policy document related to medical record-keeping and documentation standards at the University of Toledo Medical Center (UTMC).   However, accord
2026-04-28 06:52:39  INFO        Q3: What must be included in a discharge summary, who is ultimately responsible for completing it, what is the recommended completion timeline, and in what situation may a final progress note be used instead of a full discharge summary?
2026-04-28 06:52:40  INFO          Retrieval : 0.422s  |  candidates: 12/30  mode: hybrid BM25+semantic
2026-04-28 06:52:40  INFO          Reranking : 0.401s  |  kept 8/12  top_score=10.2422  mode=api
2026-04-28 06:52:40  INFO          Sources   : policy-2.pdf', 'source_location': '', 'source_type': 'PDF', 'collection_id': '', 'date_created': '2019-03-06T16:38:27', 'last_modified': '2019-03-08T14:27:37', 'summary': '', 'partition_id': -1, 'access_level': -1, 'custom_content': None}
2026-04-28 06:52:44  INFO          LLM       : 4.223s  |  tokens=4204  tok/s=42.6
2026-04-28 06:52:44  INFO          Answer    : A discharge summary must include the following information: 1. Reason for hospitalization 2. Provisional, primary, secondary, and final diagnoses 3. Significant findings 4. Procedures and treatment provided 5. Patient's discharge condition 
2026-04-28 06:52:45  INFO      ------------------------------------------------------------------------
2026-04-28 06:52:45  INFO        Q      Retrieve   Rerank      LLM   Tokens   Tok/s  Score
2026-04-28 06:52:45  INFO      ------------------------------------------------------------------------
2026-04-28 06:52:45  INFO        Q1       1.610s   0.794s   6.066s     4012   25.4  -22.7500
2026-04-28 06:52:45  INFO        Q2       0.402s   0.301s   7.665s     3955   25.4  -18.2031
2026-04-28 06:52:45  INFO        Q3       0.422s   0.401s   4.223s     4204   42.6  10.2422
2026-04-28 06:52:45  INFO      ------------------------------------------------------------------------
2026-04-28 06:52:45  INFO        Answers -> Outputs/answers_20260428_064912.json
2026-04-28 06:52:45  INFO        Metrics -> Outputs/metrics_20260428_064912.json
2026-04-28 06:52:45  INFO      ========================================================================
2026-04-28 06:52:45  INFO      FINAL SUMMARY
2026-04-28 06:52:45  INFO      ========================================================================
2026-04-28 06:52:45  INFO        Run ID   : 20260428_064912
2026-04-28 06:52:45  INFO        Docs     : 15  OK=10  Skipped=5  Failed=0
2026-04-28 06:52:45  INFO        Wall     : 213.4s  (init 19.0s + sanity 31.3s + batch 85.2s)
2026-04-28 06:52:45  INFO        Retrieval: hybrid BM25+semantic  fetch=30  rerank_to=8  model=nvidia/llama-nemotron-rerank-1b-v2
2026-04-28 06:52:45  INFO        Chunks   : 12
2026-04-28 06:52:45  INFO        Log      -> Outputs/pipeline_run_20260428_064912.log
2026-04-28 06:52:45  INFO        Metrics  -> Outputs/metrics_20260428_064912.json
2026-04-28 06:52:45  INFO        Answers  -> Outputs/answers_20260428_064912.json
2026-04-28 06:52:45  INFO      ========================================================================
