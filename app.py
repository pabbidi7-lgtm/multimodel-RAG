taskset -c 0-7 python rag_agent.py --demo --demo-pdf ./Docs/minion-tech.pdf

+===========================================================+
|  NV-Ingest 25.9.0 + LangGraph RAG Agent                  |
|                                                           |
|  Nodes: classifier -> retriever -> reranker -> generator  |
|  Retry: LOW confidence x1  |  hallucination x1           |
+===========================================================+

  OK NVIDIA_API_KEY: nvapi-gCbDQq9wb...
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
2026-04-01 14:56:18.048069736 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/MSFT1000:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
  OK NV-Ingest imported (7.9s)
  > Launching pipeline subprocess...
  > First run takes 2-5 min. Please wait.
  > Waiting for broker localhost:7671...
  OK Broker ready
  OK Pipeline ready (15.9s)
  > Ingesting 1 file(s)...
  >   -> minion-tech.pdf (10791 KB)
  > Running: load -> extract -> split -> caption -> embed -> vdb_upload
Processing: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1/1 [01:35<00:00, 95.69s/doc]
  OK 1 chunks ingested in 97,579ms (0 failures)


============================================================
  QUESTION 1 of 5
============================================================
  > Query: Using the balance sheet and P&L statement, calculate the debt-to-equity ratio and return on equity (ROE). Based on these metrics, is Gru's Enterprises financially healthy?

  > Node 1 classifier: type=calculation (718ms)
  > Connecting to Milvus: ./milvus_rag.db
  OK Collection 'rag_documents' exists
  > Node 2 retriever: 20 chunks (353ms)
API attempt 1 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking
API attempt 2 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking
  ERR Reranker failed (404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking) -- using vector order
  > Node 3 reranker: 8 chunks, medium (3128ms)

+-- ANSWER --------------------------------------------------+
| MEDIUM CONFIDENCE  |    |  4,206ms
+------------------------------------------------------------+
| 
+------------------------------------------------------------+

  Latency:
      classifier  ######........................  718ms
       retriever  ###...........................  353ms
        reranker  ##############################  3128ms


============================================================
  QUESTION 2 of 5
============================================================
  > Query: Compare the gross margin percentages of Freeze Ray, Rocket Boots, and Bubble Gun. Which product has the highest net margin and why might that be the case given the cost structure?

  > Node 1 classifier: type=comparison (417ms)
  > Node 2 retriever: 20 chunks (341ms)
API attempt 1 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking
API attempt 2 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking
  ERR Reranker failed (404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking) -- using vector order
  > Node 3 reranker: 8 chunks, medium (3139ms)

+-- ANSWER --------------------------------------------------+
| MEDIUM CONFIDENCE  |    |  3,901ms
+------------------------------------------------------------+
| 
+------------------------------------------------------------+

  Latency:
      classifier  ###...........................  417ms
       retriever  ###...........................  341ms
        reranker  ##############################  3139ms


============================================================
  QUESTION 3 of 5
============================================================
  > Query: Based on the cash flow statement, the company shows cash at beginning of period as negative $1,034,000 but ends at $500,000. Explain how this was achieved and whether the financing strategy is sustainable.

  > Node 1 classifier: type=factual (468ms)
  > Node 2 retriever: 12 chunks (480ms)
API attempt 1 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking
API attempt 2 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking
  ERR Reranker failed (404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking) -- using vector order
  > Node 3 reranker: 8 chunks, medium (3134ms)

+-- ANSWER --------------------------------------------------+
| MEDIUM CONFIDENCE  |    |  4,085ms
+------------------------------------------------------------+
| 
+------------------------------------------------------------+

  Latency:
      classifier  ####..........................  468ms
       retriever  ####..........................  480ms
        reranker  ##############################  3134ms


============================================================
  QUESTION 4 of 5
============================================================
  > Query: The company has total revenue of $4.2M but net income of only $80K. Walk through the full cost waterfall from revenue to net income, identifying which expense category consumes the largest share.

  > Node 1 classifier: type=calculation (14788ms)
  > Node 2 retriever: 20 chunks (345ms)
API attempt 1 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking
API attempt 2 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking
  ERR Reranker failed (404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking) -- using vector order
  > Node 3 reranker: 8 chunks, medium (3161ms)

+-- ANSWER --------------------------------------------------+
| MEDIUM CONFIDENCE  |    |  18,298ms
+------------------------------------------------------------+
| 
+------------------------------------------------------------+

  Latency:
      classifier  ##############################  14788ms
       retriever  ..............................  345ms
        reranker  ######........................  3161ms


============================================================
  QUESTION 5 of 5
============================================================
  > Query: If the proposed $2M investment is secured with the projected 25% revenue increase over 3 years, what would the projected revenue be in year 3? Would the 15% annual profitability growth bring net income above $150K by then?

  > Node 1 classifier: type=calculation (374ms)
  > Node 2 retriever: 20 chunks (351ms)
API attempt 1 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking
API attempt 2 failed: 404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking
  ERR Reranker failed (404 Client Error: Not Found for url: https://integrate.api.nvidia.com/v1/ranking) -- using vector order
  > Node 3 reranker: 8 chunks, medium (3131ms)

+-- ANSWER --------------------------------------------------+
| MEDIUM CONFIDENCE  |    |  3,859ms
+------------------------------------------------------------+
| 
+------------------------------------------------------------+

  Latency:
      classifier  ###...........................  374ms
       retriever  ###...........................  351ms
        reranker  ##############################  3131ms

Killed subprocess group 317573
E20260401 14:58:40.511188 323830 server.cpp:47] [SERVER][BlockLock][milvus] Process exit
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ 
