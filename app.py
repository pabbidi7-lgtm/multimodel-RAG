 taskset -c 0-7 python rag_agent.py --demo --demo-pdf ./Docs/minion-tech.pdf

+===========================================================+
|  NV-Ingest 25.9.0 + LangGraph RAG Agent                  |
|                                                           |
|  Nodes: classifier -> retriever -> reranker -> generator  |
|  Retry: LOW confidence x1  |  hallucination x1           |
+===========================================================+

  OK NVIDIA_API_KEY: nvapi-BEJBdSJ-K...
  OK Milvus DB: ./milvus_rag.db
  OK Collection: rag_documents
  OK Reranker URL: https://integrate.api.nvidia.com/v1/ranking

============================================================
  DEMO MODE
============================================================
  > Document: minion-tech.pdf


============================================================
  STARTING NV-INGEST PIPELINE
============================================================
  > Importing NV-Ingest (loads Ray internally)...
2026-04-01 15:24:30.282134195 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/MSFT1000:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
  OK NV-Ingest imported (7.8s)
  > Launching pipeline subprocess...
  > First run takes 2-5 min. Please wait.
  > Waiting for broker localhost:7671...
  OK Broker ready
  OK Pipeline ready (15.8s)
  > Ingesting 1 file(s)...
  >   -> minion-tech.pdf (10791 KB)
  > Running: load -> extract -> split -> caption -> embed -> vdb_upload
Processing: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1/1 [01:03<00:00, 63.85s/doc]
  OK 0 chunks ingested in 63,856ms (1 failures)


============================================================
  QUESTION 1 of 5
============================================================
  > Query: Using the balance sheet and P&L statement, calculate the debt-to-equity ratio and return on equity (ROE). Based on these metrics, is Gru's Enterprises financially healthy?

  > Node 1 classifier: type=calculation (6211ms)
  > Connecting to Milvus: ./milvus_rag.db
  OK Collection 'rag_documents' exists
  > Node 2 retriever: 20 chunks (916ms)
API attempt 1 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking
API attempt 2 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking
  ERR Reranker failed (404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking) -- using vector order
  > Node 3 reranker: 8 chunks, medium (3158ms)

+-- ANSWER --------------------------------------------------+
| MEDIUM CONFIDENCE  |    |  10,293ms
+------------------------------------------------------------+
| 
+------------------------------------------------------------+

  Latency:
      classifier  ##############################  6211ms
       retriever  ####..........................  916ms
        reranker  ###############...............  3158ms


============================================================
  QUESTION 2 of 5
============================================================
  > Query: Compare the gross margin percentages of Freeze Ray, Rocket Boots, and Bubble Gun. Which product has the highest net margin and why might that be the case given the cost structure?

  > Node 1 classifier: type=comparison (522ms)
  > Node 2 retriever: 20 chunks (408ms)
API attempt 1 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking
API attempt 2 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking
  ERR Reranker failed (404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking) -- using vector order
  > Node 3 reranker: 8 chunks, medium (3222ms)

+-- ANSWER --------------------------------------------------+
| MEDIUM CONFIDENCE  |    |  4,155ms
+------------------------------------------------------------+
| 
+------------------------------------------------------------+

  Latency:
      classifier  ####..........................  522ms
       retriever  ###...........................  408ms
        reranker  ##############################  3222ms


============================================================
  QUESTION 3 of 5
============================================================
  > Query: Based on the cash flow statement, the company shows cash at beginning of period as negative $1,034,000 but ends at $500,000. Explain how this was achieved and whether the financing strategy is sustainable.

  > Node 1 classifier: type=factual (469ms)
  > Node 2 retriever: 12 chunks (357ms)
API attempt 1 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking
API attempt 2 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking
  ERR Reranker failed (404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking) -- using vector order
  > Node 3 reranker: 8 chunks, medium (3127ms)

+-- ANSWER --------------------------------------------------+
| MEDIUM CONFIDENCE  |    |  3,957ms
+------------------------------------------------------------+
| 
+------------------------------------------------------------+

  Latency:
      classifier  ####..........................  469ms
       retriever  ###...........................  357ms
        reranker  ##############################  3127ms


============================================================
  QUESTION 4 of 5
============================================================
  > Query: The company has total revenue of $4.2M but net income of only $80K. Walk through the full cost waterfall from revenue to net income, identifying which expense category consumes the largest share.
