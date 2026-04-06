taskset -c 0-7 python rag_agent.py

╔═══════════════════════════════════════════════════════════╗
║  Enterprise Multimodal RAG Agent v3                       ║
║  NV-Ingest 25.9.0 + LangGraph + NVIDIA Mistral Reranker   ║
║                                                           ║
║  Modalities: text · table · caption · chart · diagram     ║
║              infographic · OCR · handwritten · invoice    ║
║                                                           ║
║  Graph: guardrail → intent_router → multi_expander →      ║
║         hybrid_retriever → nvidia_reranker →              ║
║         [adaptive_rescue?] → generator                    ║
║                                                           ║
║  NEVER silences — always generates best-effort answer     ║
║  Reranker: nvidia/nv-rerankqa-mistral-4b-v3 (API)         ║
╚═══════════════════════════════════════════════════════════╝

  ✓ NVIDIA_API_KEY  : nvapi-xvdRyUyAG…
  ✓ Milvus DB       : ./milvus_rag_v3.db
  ✓ Collection      : rag_documents_v3
  ✓ Embed model     : nvidia/nv-embedqa-e5-v5
  ✓ Reranker        : nvidia/nv-rerankqa-mistral-4b-v3  (NVIDIA API)
  ✓ LLM primary     : meta/llama-3.3-70b-instruct
  ✓ LLM fallback    : nvidia/llama-3.1-nemotron-70b-instruct
  ✓ Confidence HIGH : ≥0.45  MEDIUM: ≥0.15  RESCUE: <0.1
  ✓ Retrieval top-k : 60   Rerank top-k: 20   Context: 8
  ✓ Min gen score   : 0.0 (NEVER silences)
  ▶ Auto-ingesting 1 file(s)…
  ▶ Importing NV-Ingest (loads Ray internally)…
2026-04-06 13:17:19.148271709 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/MSFT1000:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
  ✓ NV-Ingest imported (8.6s)
  ▶ Launching pipeline subprocess…
  ⚠ First run takes 2–5 min. Please wait.
  ▶ Waiting for broker localhost:7671...
  ✓ Broker ready
  ✓ Pipeline ready (8.6s)
  ▶ Ingesting 1 file(s)…
  ▶   → Ascent_of_Open.pdf (4399 KB)
  ▶ Running: load → extract → split → caption → embed → vdb_upload
Processing: 100%|█████████████████████████████████████████████████████████████████████████| 1/1 [01:03<00:00, 63.24s/doc]
  ✓ 1 chunks ingested in 65,198ms (0 failures)

Commands:
  ingest <path> [path2 …]   Ingest file(s)
  ingest --reset <path>     Reset collection then ingest
  stats                     Show chunk count
  reset                     Clear collection
  history                   Show conversation memory
  clear                     Clear conversation memory
  quit                      Exit


Q: Based on the way these chord diagrams are described, why does the report caution that the diagrams are “unnormalised” and involve “n‑fold counting,” and how could this affect an interpretation comparing collaboration intensity between countries over time?
  ▶ Query: Based on the way these chord diagrams are described, why does the report caution that the diagrams are “unnormalised” and involve “n‑fold counting,” and how could this affect an interpretation comparing collaboration intensity between countries over time?

  ▶ [N1] guardrail clean='Based on the way these chord diagrams are described, why doe' (2ms)
  ▶ [N2] intent_router modality=diagram (0ms)
  ▶ [N3] expander 5 variants (27169ms)
  ▶ Connecting to Milvus: ./milvus_rag_v3.db
  ✓ Collection 'rag_documents_v3' exists
  ▶ [N4] retriever 45 unique chunks from 5 variants | modalities={'text': 28, 'diagram': 13, 'table': 3, 'chart': 1} (1817ms)
API attempt 1 HTTP error: 404 Client Error: Not Found for url: https://ai.api.nvidia.com/v1/ranking | URL: https://ai.api.nvidia.com/v1/ranking
Body: 404 page not found

API attempt 2 HTTP error: 404 Client Error: Not Found for url: https://ai.api.nvidia.com/v1/ranking | URL: https://ai.api.nvidia.com/v1/ranking
Body: 404 page not found

API attempt 3 HTTP error: 404 Client Error: Not Found for url: https://ai.api.nvidia.com/v1/ranking | URL: https://ai.api.nvidia.com/v1/ranking
Body: 404 page not found

NVIDIA reranker batch failed: 404 Client Error: Not Found for url: https://ai.api.nvidia.com/v1/ranking — using fallback uniform scores
  ▶ [N5] reranker 20 chunks | top_score=0.1600 | MEDIUM (3179ms)
  ▶ [ROUTE] top_score=0.1600 → generator
  ▶   Generating with llama-3.3-70b-instruct…
  ▶ [N6] generator llama-3.3-70b-instruct | mode=normal | rescue=False (2840ms)

╔── ANSWER ────────────────────────────────────────────────╗
║ MEDIUM CONFIDENCE  │  llama-3.3-70b-instruct  │  35,019ms
╠──────────────────────────────────────────────────────────╣
║ The provided documents do not contain this information. 
║ 
║ The closest relevant information found is in Chunk 8,
║ which mentions a diagram (Figure 4) related to the
║ trend in the number of publications by Open Access type
║ for all publications. It also mentions that Bronze and
║ Gold routes are the fastest-growing channels, but it
║ does not provide information on why the report would
║ caution that the diagrams are "unnormalised" and
║ involve "n‑fold counting," or how this could affect an
║ interpretation comparing collaboration intensity
║ between countries over time. 
║ 
║ Additionally, Chunk 4 provides a list of years and
║ numbers, but it does not provide any context or
║ explanation related to the caution about "unnormalised"
║ diagrams or "n‑fold counting." 
║ 
║ Therefore, without more specific information, it is not
║ possible to provide a detailed answer to the question.
╠──────────────────────────────────────────────────────────╣

  Latency breakdown:
         guardrail  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  2ms
     intent_router  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0ms
          expander  ████████████████████████████  27169ms
         retriever  █░░░░░░░░░░░░░░░░░░░░░░░░░░░  1817ms
          reranker  ███░░░░░░░░░░░░░░░░░░░░░░░░░  3179ms
         generator  ██░░░░░░░░░░░░░░░░░░░░░░░░░░  2840ms

  Sources (8 chunks passed to LLM):
    #1 medium  [TEXT]  score=0.1600
    The content of the image is a logo, which belongs to BIICRAFT. Logos are identifiers for organizatio…
    #2 medium  [TEXT]  score=0.1600
    Digital Science is a technology company working to make research more efficient. We invest 
in, nur…
    #3 medium  [TEXT]  score=0.1600
    This image features the logo of a company named "Transcriptic." The logo is minimalist and clearly d…
    #4 medium  [TEXT]  score=0.1600
    the fastest growing channels.   2000 - 2001 - 2002 - 2003 - 2004 - 2005 - 2006 - 2007 - 2008 - 2009 …
    #5 medium  [TEXT]  score=0.1600
    This image showcases a collection of logos from various companies within the pharmaceutical and life…
╚══════════════════════════════════════════════════════════╝


Q: Why does the report argue that “Open Access, Funded, Internationally Collaborative” papers outperform other categories despite forming a very small percentage of total output, and what mistaken conclusion could a reader draw if they focused only on publication volume without considering this chart structure?
  ▶ Query: Why does the report argue that “Open Access, Funded, Internationally Collaborative” papers outperform other categories despite forming a very small percentage of total output, and what mistaken conclusion could a reader draw if they focused only on publication volume without considering this chart structure?
  ▶ Memory: 1 turn(s) in context

  ▶ [N1] guardrail clean='Why does the report argue that “Open Access, Funded, Interna' (0ms)
  ▶ [N2] intent_router modality=chart (0ms)
API attempt 1 connection error: HTTPSConnectionPool(host='integrate.api.nvidia.com', port=443): Read timed out. (read timeout=120) | URL: https://integrate.api.nvidia.com/v1/chat/completions
  ▶ [N3] expander 5 variants (198926ms)
  ▶ [N4] retriever 45 unique chunks from 5 variants | modalities={'text': 28, 'diagram': 13, 'chart': 1, 'table': 3} (2161ms)
API attempt 1 HTTP error: 404 Client Error: Not Found for url: https://ai.api.nvidia.com/v1/ranking | URL: https://ai.api.nvidia.com/v1/ranking
Body: 404 page not found

API attempt 2 HTTP error: 404 Client Error: Not Found for url: https://ai.api.nvidia.com/v1/ranking | URL: https://ai.api.nvidia.com/v1/ranking
Body: 404 page not found

API attempt 3 HTTP error: 404 Client Error: Not Found for url: https://ai.api.nvidia.com/v1/ranking | URL: https://ai.api.nvidia.com/v1/ranking
Body: 404 page not found

NVIDIA reranker batch failed: 404 Client Error: Not Found for url: https://ai.api.nvidia.com/v1/ranking — using fallback uniform scores
  ▶ [N5] reranker 20 chunks | top_score=0.1600 | MEDIUM (3272ms)
  ▶ [ROUTE] top_score=0.1600 → generator
  ▶   Generating with llama-3.3-70b-instruct…
  ▶ [N6] generator llama-3.3-70b-instruct | mode=normal | rescue=False (12702ms)

╔── ANSWER ────────────────────────────────────────────────╗
║ MEDIUM CONFIDENCE  │  llama-3.3-70b-instruct  │  217,067ms
╠──────────────────────────────────────────────────────────╣
║ The provided documents do not contain this information.
║ The closest relevant information found is in Chunk 8,
║ which mentions a diagram (implied, but not directly
║ related to the question) and discusses trends in
║ publications, but it does not provide information on
║ why the report argues that “Open Access, Funded,
║ Internationally Collaborative” papers outperform other
║ categories. Additionally, the chunks provided contain
║ descriptions of images, logos, and comparisons of
║ cross-exports between countries, but they do not
║ provide context or explanation related to the
║ performance of "Open Access, Funded, Internationally
║ Collaborative" papers or the potential mistaken
║ conclusion a reader could draw from focusing only on
║ publication volume. Therefore, without more specific
║ information, it is not possible to provide a detailed
║ answer to the question.
╠──────────────────────────────────────────────────────────╣

  Latency breakdown:
         guardrail  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0ms
     intent_router  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0ms
          expander  ████████████████████████████  198926ms
         retriever  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  2161ms
          reranker  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  3272ms
         generator  █░░░░░░░░░░░░░░░░░░░░░░░░░░░  12702ms

  Sources (8 chunks passed to LLM):
    #1 medium  [TEXT]  score=0.1600
    Caption: "A pixelated sunrise over a serene, green rural landscape, where the beauty of the countrys…
    #2 medium  [TEXT]  score=0.1600
    GRID…
    #3 medium  [TEXT]  score=0.1600
    The content of the image is a logo, which belongs to BIICRAFT. Logos are identifiers for organizatio…
    #4 medium  [TEXT]  score=0.1600
    This image displays a comparison of the number of cross-exports at the product level between differe…
    #5 medium  [TEXT]  score=0.1600
    The content of the image appears to be a logo, possibly representing a company or a product due to i…
╚══════════════════════════════════════════════════════════╝


Q: explain why increased openness can simultaneously improve reproducibility and increase the risk of misuse or misinterpretation of research data.
  ▶ Query: explain why increased openness can simultaneously improve reproducibility and increase the risk of misuse or misinterpretation of research data.
  ▶ Memory: 2 turn(s) in context

  ▶ [N1] guardrail clean='why increased openness can simultaneously improve reproducib' (0ms)
  ▶ [N2] intent_router modality=text (0ms)
  ▶ [N3] expander 4 variants (77789ms)
  ▶ [N4] retriever 45 unique chunks from 4 variants | modalities={'text': 28, 'diagram': 13, 'chart': 1, 'table': 3} (1601ms)
API attempt 1 HTTP error: 404 Client Error: Not Found for url: https://ai.api.nvidia.com/v1/ranking | URL: https://ai.api.nvidia.com/v1/ranking
Body: 404 page not found

API attempt 2 HTTP error: 404 Client Error: Not Found for url: https://ai.api.nvidia.com/v1/ranking | URL: https://ai.api.nvidia.com/v1/ranking
Body: 404 page not found

API attempt 3 HTTP error: 404 Client Error: Not Found for url: https://ai.api.nvidia.com/v1/ranking | URL: https://ai.api.nvidia.com/v1/ranking
Body: 404 page not found

NVIDIA reranker batch failed: 404 Client Error: Not Found for url: https://ai.api.nvidia.com/v1/ranking — using fallback uniform scores
  ▶ [N5] reranker 20 chunks | top_score=0.1600 | MEDIUM (3779ms)
  ▶ [ROUTE] top_score=0.1600 → generator
  ▶   Generating with llama-3.3-70b-instruct…
  ▶ [N6] generator llama-3.3-70b-instruct | mode=normal | rescue=False (15048ms)

╔── ANSWER ────────────────────────────────────────────────╗
║ MEDIUM CONFIDENCE  │  llama-3.3-70b-instruct  │  98,222ms
╠──────────────────────────────────────────────────────────╣
║ The provided documents do not contain this information.
║ The closest relevant information found is in Chunk 3,
║ which mentions the fastest-growing channels, including
║ Bronze, Pure Gold, Hybrid, Green (Submitted), Green
║ (Published), and Green (Accepted), but it does not
║ provide information on why increased openness can
║ simultaneously improve reproducibility and increase the
║ risk of misuse or misinterpretation of research data.
║ Additionally, Chunk 6 discusses the global distribution
║ of people proficient in English, and Chunk 8 mentions a
║ company logo, but neither provides context or
║ explanation related to the potential benefits and risks
║ of increased openness in research data. Therefore,
║ without more specific information, it is not possible
║ to provide a detailed answer to the question.
╠──────────────────────────────────────────────────────────╣

  Latency breakdown:
         guardrail  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0ms
     intent_router  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0ms
          expander  ████████████████████████████  77789ms
         retriever  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1601ms
          reranker  █░░░░░░░░░░░░░░░░░░░░░░░░░░░  3779ms
         generator  █████░░░░░░░░░░░░░░░░░░░░░░░  15048ms

  Sources (8 chunks passed to LLM):
    #1 medium  [TEXT]  score=0.1600
    The content of the image appears to be a logo, possibly representing a company or a product due to i…
    #2 medium  [TEXT]  score=0.1600
    This image displays a comparison of the number of cross-exports at the product level between differe…
    #3 medium  [TEXT]  score=0.1600
    the fastest growing channels.   2000 - 2001 - 2002 - 2003 - 2004 - 2005 - 2006 - 2007 - 2008 - 2009 …
    #4 medium  [TEXT]  score=0.1600
    Caption: "A pixelated sunrise over a serene, green rural landscape, where the beauty of the countrys…
    #5 medium  [TEXT]  score=0.1600
    This image can be captioned as "Abstract Geometric Logo: A Vivid Display of Interlinked Squares in O…
╚══════════════════════════════════════════════════════════╝


Q: ^C
Session ended.
Killed subprocess group 563654
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ taskset -c 0-7 python rag_agent.py

╔═══════════════════════════════════════════════════════════╗
║  Enterprise Multimodal RAG Agent v3                       ║
║  NV-Ingest 25.9.0 + LangGraph + NVIDIA Mistral Reranker   ║
║                                                           ║
║  Modalities: text · table · caption · chart · diagram     ║
║              infographic · OCR · handwritten · invoice    ║
║                                                           ║
║  Graph: guardrail → intent_router → multi_expander →      ║
║         hybrid_retriever → nvidia_reranker →              ║
║         [adaptive_rescue?] → generator                    ║
║                                                           ║
║  NEVER silences — always generates best-effort answer     ║
║  Reranker: nvidia/nv-rerankqa-mistral-4b-v3 (API)         ║
╚═══════════════════════════════════════════════════════════╝

  ✓ NVIDIA_API_KEY  : nvapi-xvdRyUyAG…
  ✓ Milvus DB       : ./milvus_rag_v3.db
  ✓ Collection      : rag_documents_v3
  ✓ Embed model     : nvidia/nv-embedqa-e5-v5
  ✓ Reranker        : nvidia/nv-rerankqa-mistral-4b-v3  (NVIDIA API)
  ✓ LLM primary     : meta/llama-3.3-70b-instruct
  ✓ LLM fallback    : nvidia/llama-3.1-nemotron-70b-instruct
  ✓ Confidence HIGH : ≥0.45  MEDIUM: ≥0.15  RESCUE: <0.1
  ✓ Retrieval top-k : 60   Rerank top-k: 20   Context: 8
  ✓ Min gen score   : 0.0 (NEVER silences)
  ▶ Auto-ingesting 1 file(s)…
  ▶ Importing NV-Ingest (loads Ray internally)…
2026-04-06 13:41:29.639842109 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/MSFT1000:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
  ✓ NV-Ingest imported (9.8s)
  ▶ Launching pipeline subprocess…
  ⚠ First run takes 2–5 min. Please wait.
  ▶ Waiting for broker localhost:7671...
  ✓ Broker ready
  ✓ Pipeline ready (9.8s)
  ▶ Ingesting 1 file(s)…
  ▶   → Ascent_of_Open.pdf (4399 KB)
  ▶ Running: load → extract → split → caption → embed → vdb_upload
Processing: 100%|█████████████████████████████████████████████████████████████████████████| 1/1 [01:03<00:00, 63.24s/doc]
  ✓ 1 chunks ingested in 65,253ms (0 failures)

Commands:
  ingest <path> [path2 …]   Ingest file(s)
  ingest --reset <path>     Reset collection then ingest
  stats                     Show chunk count
  reset                     Clear collection
  history                   Show conversation memory
  clear                     Clear conversation memory
  quit                      Exit


Q: Why does the report argue that “Open Access, Funded, Internationally Collaborative” papers outperform other categories despite forming a very small percentage of total output, and what mistaken conclusion could a reader draw if they focused only on publication volume without considering this chart structure?
  ▶ Query: Why does the report argue that “Open Access, Funded, Internationally Collaborative” papers outperform other categories despite forming a very small percentage of total output, and what mistaken conclusion could a reader draw if they focused only on publication volume without considering this chart structure?

  ▶ [N1] guardrail clean='Why does the report argue that “Open Access, Funded, Interna' (1ms)
  ▶ [N2] intent_router modality=chart (0ms)
  ▶ [N3] expander 5 variants (63770ms)
  ▶ Connecting to Milvus: ./milvus_rag_v3.db
  ✓ Collection 'rag_documents_v3' exists
  ▶ [N4] retriever 45 unique chunks from 5 variants | modalities={'text': 28, 'diagram': 13, 'chart': 1, 'table': 3} (2448ms)
API attempt 1 HTTP error: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking | URL: https://integrate.api.nvidia.com/v1/ranking
Body: 404 page not found

API attempt 2 HTTP error: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking | URL: https://integrate.api.nvidia.com/v1/ranking
Body: 404 page not found

API attempt 3 HTTP error: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking | URL: https://integrate.api.nvidia.com/v1/ranking
Body: 404 page not found

NVIDIA reranker batch failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking — using fallback uniform scores
  ▶ [N5] reranker 20 chunks | top_score=0.1600 | MEDIUM (3182ms)
  ▶ [ROUTE] top_score=0.1600 → generator
  ▶   Generating with llama-3.3-70b-instruct…
  ▶ [N6] generator llama-3.3-70b-instruct | mode=normal | rescue=False (34183ms)

╔── ANSWER ────────────────────────────────────────────────╗
║ MEDIUM CONFIDENCE  │  llama-3.3-70b-instruct  │  103,595ms
╠──────────────────────────────────────────────────────────╣
║ The provided documents do not contain this information.
║ The closest relevant information found is related to
║ the descriptions of various logos and images, including
║ those for BIICRAFT, Tetrascience, and Symplectic, as
║ well as a comparison of cross-exports between
║ countries. There is no mention of a report arguing
║ about the performance of "Open Access, Funded,
║ Internationally Collaborative" papers or the potential
║ mistaken conclusions a reader could draw from focusing
║ solely on publication volume.
╠──────────────────────────────────────────────────────────╣

  Latency breakdown:
         guardrail  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  1ms
     intent_router  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0ms
          expander  ████████████████████████████  63770ms
         retriever  █░░░░░░░░░░░░░░░░░░░░░░░░░░░  2448ms
          reranker  █░░░░░░░░░░░░░░░░░░░░░░░░░░░  3182ms
         generator  ███████████████░░░░░░░░░░░░░  34183ms

  Sources (8 chunks passed to LLM):
    #1 medium  [TEXT]  score=0.1600
    Caption: "A pixelated sunrise over a serene, green rural landscape, where the beauty of the countrys…
    #2 medium  [TEXT]  score=0.1600
    GRID…
    #3 medium  [TEXT]  score=0.1600
    The content of the image is a logo, which belongs to BIICRAFT. Logos are identifiers for organizatio…
    #4 medium  [TEXT]  score=0.1600
    The content of the image appears to be a logo, possibly representing a company or a product due to i…
    #5 medium  [TEXT]  score=0.1600
    This image displays a comparison of the number of cross-exports at the product level between differe…
╚══════════════════════════════════════════════════════════╝


Q: 
