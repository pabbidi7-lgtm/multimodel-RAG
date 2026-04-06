taskset -c 0-7 python rag_agent.py

+===========================================================+
|  Enterprise Multimodal RAG Agent v2                       |
|  NV-Ingest 25.9.0 + LangGraph                             |
|                                                           |
|  Supports: PDF, DOCX, PPTX, XLSX, images, handwritten,   |
|  invoices, identity docs, tables, charts, captions        |
|                                                           |
|  Nodes: guardrail → expander → retriever → reranker →     |
|         quality_gate → generator                          |
|  Memory: last 3 conversation turns                        |
|  Retry: hallucination x1                                  |
+===========================================================+

  OK NVIDIA_API_KEY: nvapi-BEJBdSJ-K...
  OK Milvus DB: ./milvus_rag.db
  OK Collection: rag_documents
  OK Embed URL: https://integrate.api.nvidia.com/v1/embeddings
  OK Reranker: cross-encoder/ms-marco-MiniLM-L-12-v2
  OK Confidence thresholds: high>=-3.0, medium>=-8.0
  OK Quality gate: skip generation if top score < -10.0
  OK Retrieval top-k: 40, Rerank top-k: 15, Context: 6
  OK LLM: meta/llama-3.3-70b-instruct → fallback: nvidia/llama-3.1-nemotron-70b-instruct
  > Auto-ingesting 1 file(s)...
  > Importing NV-Ingest (loads Ray internally)...
2026-04-06 12:04:28.455673811 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/MSFT1000:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
  OK NV-Ingest imported (8.2s)
  > Launching pipeline subprocess...
  > First run takes 2-5 min. Please wait.
  > Waiting for broker localhost:7671...
  OK Broker ready
  OK Pipeline ready (8.2s)
  > Ingesting 1 file(s)...
  >   -> Oxford.pdf (4087 KB)
  > Running: load → extract → split → caption → embed → vdb_upload
Processing: 100%|████████████████████████████████████████████████████████████████████████| 1/1 [03:42<00:00, 222.21s/doc]
  OK 0 chunks ingested in 222,207ms (1 failures)

Commands:
  ingest <path>        Ingest a file (any format)
  ingest <p1> <p2> ... Ingest multiple files
  stats                Show chunk count in Milvus
  reset                Clear the collection
  history              Show current conversation memory
  clear                Clear conversation memory
  quit                 Exit


Q: Describe a scenario where a pie chart might still be acceptable despite marginal differences, and contrast it with a case from the document where a table is objectively superior. Explain which visual cues fail in each case and why.
  > Query: Describe a scenario where a pie chart might still be acceptable despite marginal differences, and contrast it with a case from the document where a table is objectively superior. Explain which visual cues fail in each case and why.

  > Node 1 guardrail: intent=table (2ms)
  > Node 2 expander: 3 variants (26918ms)
  > Connecting to Milvus: ./milvus_rag.db
  OK Collection 'rag_documents' exists
  > Node 3 retriever: 18 unique chunks from 3 variants (1786ms)
Loading weights: 100%|███████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 3226.68it/s]
BertForSequenceClassification LOAD REPORT from: cross-encoder/ms-marco-MiniLM-L-12-v2
Key                          | Status     |  | 
-----------------------------+------------+--+-
bert.embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Batches: 100%|█████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.75s/it]
  > Node 4 reranker: quality gate failed (top=-11.01) — skipping generation
  > Generating with llama-3.3-70b-instruct...
  > Node 5 generator: llama-3.3-70b-instruct (24608ms)

+-- ANSWER --------------------------------------------------+
| LOW CONFIDENCE  |  llama-3.3-70b-instruct  |  60,788ms
+------------------------------------------------------------+
| The provided documents do not contain this information.
+------------------------------------------------------------+

  Latency:
       guardrail  ..............................  2ms
        expander  ##############################  26918ms
       retriever  #.............................  1786ms
        reranker  ########......................  7465ms
       generator  ###########################...  24608ms

  Sources (6 chunks):
    Chunk 1  low  [text]  score=-11.011
    This image could be captioned as "Financial Calculations and Analyses." The calc...
    Chunk 2  low  [text]  score=-11.179
    How LTIMindtree can help...
    Chunk 3  low  [text]  score=-11.203
    Infinity Ensure
One stop mul-cloud plaorm for opmized 
Governance and FinOp...
    Chunk 4  low  [text]  score=-11.249
    Caption: "A Serene Skyscape: Earth's layer of cotton candy meets the vast cosmic...

Q: Given a six‑panel biomedical figure containing charts, images, and annotations, how should alt text be structured to balance conciseness, completeness, and non‑redundancy—while ensuring that a screen‑reader user can reconstruct the scientific conclusion without seeing the figure?
  > Query: Given a six‑panel biomedical figure containing charts, images, and annotations, how should alt text be structured to balance conciseness, completeness, and non‑redundancy—while ensuring that a screen‑reader user can reconstruct the scientific conclusion without seeing the figure?
  > Memory: 1 turn(s) in context

  > Node 1 guardrail: intent=image_caption (0ms)
  > Node 2 expander: 3 variants (55886ms)
  > Node 3 retriever: 18 unique chunks from 3 variants (1918ms)
Batches: 100%|█████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.08s/it]
  > Node 4 reranker: 15 chunks, low (2082ms)
  > Generating with llama-3.3-70b-instruct...
  > Node 5 generator: llama-3.3-70b-instruct (745ms)

+-- ANSWER --------------------------------------------------+
| LOW CONFIDENCE  |  llama-3.3-70b-instruct  |  60,635ms
+------------------------------------------------------------+
| The provided documents do not contain this information.
+------------------------------------------------------------+

  Latency:
       guardrail  ..............................  0ms
        expander  ##############################  55886ms
       retriever  #.............................  1918ms
        reranker  #.............................  2082ms
       generator  ..............................  745ms

  Sources (6 chunks):
    Chunk 1  low  [text]  score=-9.184
    This image could be captioned as "Financial Calculations and Analyses." The calc...
    Chunk 2  low  [text]  score=-10.202
    The image appears to be a logo or a title frame for an event, product, or compan...
    Chunk 3  low  [text]  score=-11.081
    How LTIMindtree can help...
    Chunk 4  low  [text]  score=-11.133
    This image could be captioned as "Cloud Computing Engagement," conveying the act...

Q: In a scenario where figure labels must be placed within a coloured graphical region due to space constraints, how does the document implicitly resolve the conflict between contrast‑ratio compliance and text‑placement guidance, and what hierarchy of accessibility principles emerges from this resolution?
  > Query: In a scenario where figure labels must be placed within a coloured graphical region due to space constraints, how does the document implicitly resolve the conflict between contrast‑ratio compliance and text‑placement guidance, and what hierarchy of accessibility principles emerges from this resolution?
  > Memory: 2 turn(s) in context

  > Node 1 guardrail: intent=image_caption (0ms)
  > Node 2 expander: 3 variants (15283ms)
  > Node 3 retriever: 18 unique chunks from 3 variants (1120ms)
Batches: 100%|█████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.13s/it]
  > Node 4 reranker: quality gate failed (top=-11.13) — skipping generation
  > Generating with llama-3.3-70b-instruct...
  > Node 5 generator: llama-3.3-70b-instruct (31393ms)

+-- ANSWER --------------------------------------------------+
| LOW CONFIDENCE  |  llama-3.3-70b-instruct  |  49,936ms
+------------------------------------------------------------+
| The provided documents do not contain this information.
+------------------------------------------------------------+

  Latency:
       guardrail  ..............................  0ms
        expander  ##############................  15284ms
       retriever  #.............................  1120ms
        reranker  ##............................  2135ms
       generator  ##############################  31393ms

  Sources (6 chunks):
    Chunk 1  low  [text]  score=-11.133
    The image appears to be a logo or a title frame for an event, product, or compan...
    Chunk 2  low  [text]  score=-11.248
    LTIMindtree One stop multi-cloud plattorm Challenges in cloud for optimized Gove...
    Chunk 3  low  [text]  score=-11.254
    The client intended to implement and adhere to NIST (National 
Institute of Sta...
    Chunk 4  low  [text]  score=-11.270
    In the present day, every business strategy is 
subjected to a cloud strategy. ...

Q: Based solely on the textual explanations (not examples), explain why WCAG contrast compliance is treated as a necessary but insufficient condition for accessibility in the document, and identify the additional non‑visual factors the guide prioritizes to ensure comprehension.
  > Query: Based solely on the textual explanations (not examples), explain why WCAG contrast compliance is treated as a necessary but insufficient condition for accessibility in the document, and identify the additional non‑visual factors the guide prioritizes to ensure comprehension.
  > Memory: 3 turn(s) in context

  > Node 1 guardrail: intent=text (0ms)
  > Node 2 expander: 3 variants (10554ms)
  > Node 3 retriever: 18 unique chunks from 3 variants (1155ms)
Batches: 100%|█████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.30s/it]
  > Node 4 reranker: 15 chunks, low (2304ms)
  > Generating with llama-3.3-70b-instruct...
  > Node 5 generator: llama-3.3-70b-instruct (36779ms)

+-- ANSWER --------------------------------------------------+
| LOW CONFIDENCE  |  llama-3.3-70b-instruct  |  50,796ms
+------------------------------------------------------------+
| The provided documents do not contain this information.
+------------------------------------------------------------+

  Latency:
       guardrail  ..............................  0ms
        expander  ########......................  10554ms
       retriever  ..............................  1155ms
        reranker  #.............................  2304ms
       generator  ##############################  36779ms

  Sources (6 chunks):
    Chunk 1  low  [text]  score=-9.586
    This image could be captioned as "Cloud Computing Engagement," conveying the act...
    Chunk 2  low  [text]  score=-10.310
    LTIMindtree Our cloud monitoring services span across governance policies, cost ...
    Chunk 3  low  [table]  score=-10.498
    |  | LTIMindtree |
|  | The client intended to implement and adhere to NIST (Nat...
    Chunk 4  low  [text]  score=-10.508
    The client intended to implement and adhere to NIST (National 
Institute of Sta...

Q: ^C
Session ended.
Killed subprocess group 475790
