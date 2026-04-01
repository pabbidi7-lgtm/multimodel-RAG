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
  > Document: minion-tech.pdf

  > Importing NV-Ingest (loads Ray internally)...
2026-04-01 15:49:02.347062931 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/MSFT1000:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
  OK NV-Ingest imported (7.7s)
  > Launching pipeline subprocess...
  > First run takes 2-5 min. Please wait.
  > Waiting for broker localhost:7671...
  OK Broker ready
  OK Pipeline ready (15.7s)
  > Ingesting 1 file(s)...
  >   -> minion-tech.pdf (10791 KB)
  > Running: load -> extract -> split -> caption -> embed -> vdb_upload
Processing: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1/1 [01:03<00:00, 63.66s/doc]
  OK 1 chunks ingested in 65,640ms (0 failures)

  > [1/5] Using the balance sheet and P&L statement, calculate the debt-to-equity ratio and return on equity (ROE). Based on these metrics, is Gru's Enterprises financially healthy?

  > Node 1 classifier: type=calculation (363ms)
  > Connecting to Milvus: ./milvus_rag.db
  OK Collection 'rag_documents' exists
  > Node 2 retriever: 20 chunks (347ms)
API attempt 1 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking | URL: https://integrate.api.nvidia.com/v1/ranking
Response body: 404 page not found

API attempt 2 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking | URL: https://integrate.api.nvidia.com/v1/ranking
Response body: 404 page not found

API attempt 3 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking | URL: https://integrate.api.nvidia.com/v1/ranking
Response body: 404 page not found

  ERR Reranker failed (404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking) -- using vector order
  > Node 3 reranker: 8 chunks, medium (3145ms)

+-- ANSWER --------------------------------------------------+
| MEDIUM CONFIDENCE  |    |  3,863ms
+------------------------------------------------------------+
| 
+------------------------------------------------------------+

  Latency:
      classifier  ###...........................  363ms
       retriever  ###...........................  347ms
        reranker  ##############################  3145ms

  > [2/5] Compare the gross margin percentages of Freeze Ray, Rocket Boots, and Bubble Gun. Which product has the highest net margin and why might that be the case given the cost structure?

  > Node 1 classifier: type=comparison (16580ms)
  > Node 2 retriever: 20 chunks (332ms)
API attempt 1 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking | URL: https://integrate.api.nvidia.com/v1/ranking
Response body: 404 page not found

API attempt 2 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking | URL: https://integrate.api.nvidia.com/v1/ranking
Response body: 404 page not found

API attempt 3 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking | URL: https://integrate.api.nvidia.com/v1/ranking
Response body: 404 page not found

  ERR Reranker failed (404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking) -- using vector order
  > Node 3 reranker: 8 chunks, medium (3149ms)

+-- ANSWER --------------------------------------------------+
| MEDIUM CONFIDENCE  |    |  20,064ms
+------------------------------------------------------------+
| 
+------------------------------------------------------------+

  Latency:
      classifier  ##############################  16580ms
       retriever  ..............................  332ms
        reranker  #####.........................  3149ms

  > [3/5] Based on the cash flow statement, the company shows cash at beginning of period as negative $1,034,000 but ends at $500,000. Explain how this was achieved and whether the financing strategy is sustainable.

  > Node 1 classifier: type=factual (501ms)
  > Node 2 retriever: 12 chunks (329ms)
API attempt 1 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking | URL: https://integrate.api.nvidia.com/v1/ranking
Response body: 404 page not found

API attempt 2 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking | URL: https://integrate.api.nvidia.com/v1/ranking
Response body: 404 page not found

API attempt 3 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking | URL: https://integrate.api.nvidia.com/v1/ranking
Response body: 404 page not found

  ERR Reranker failed (404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking) -- using vector order
  > Node 3 reranker: 8 chunks, medium (3188ms)

+-- ANSWER --------------------------------------------------+
| MEDIUM CONFIDENCE  |    |  4,022ms
+------------------------------------------------------------+
| 
+------------------------------------------------------------+

  Latency:
      classifier  ####..........................  501ms
       retriever  ###...........................  329ms
        reranker  ##############################  3188ms

  > [4/5] The company has total revenue of $4.2M but net income of only $80K. Walk through the full cost waterfall from revenue to net income, identifying which expense category consumes the largest share.

  > Node 1 classifier: type=calculation (489ms)
  > Node 2 retriever: 20 chunks (348ms)
API attempt 1 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking | URL: https://integrate.api.nvidia.com/v1/ranking
Response body: 404 page not found

API attempt 2 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking | URL: https://integrate.api.nvidia.com/v1/ranking
Response body: 404 page not found
