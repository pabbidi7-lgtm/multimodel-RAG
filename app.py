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
  OK Embed URL: https://integrate.api.nvidia.com/v1/embeddings
  OK Reranker URL: cross-encoder/ms-marco-MiniLM-L-12-v2
  OK LLM URL: https://integrate.api.nvidia.com/v1/chat/completions
  > Document: minion-tech.pdf

  > Importing NV-Ingest (loads Ray internally)...
2026-04-01 17:27:33.443191595 [W:onnxruntime:Default, device_discovery.cc:132 GetPciBusId] Skipping pci_bus_id for PCI path at "/sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0004:00/MSFT1000:00/5620e0c7-8062-4dce-aeb7-520c7ef76171" because filename ""5620e0c7-8062-4dce-aeb7-520c7ef76171"" dit not match expected pattern of [0-9a-f]+:[0-9a-f]+:[0-9a-f]+[.][0-9a-f]+
  OK NV-Ingest imported (7.6s)
  > Launching pipeline subprocess...
  > First run takes 2-5 min. Please wait.
  > Waiting for broker localhost:7671...
  OK Broker ready
  OK Pipeline ready (15.6s)
  > Ingesting 1 file(s)...
  >   -> minion-tech.pdf (10791 KB)
  > Running: load -> extract -> split -> caption -> embed -> vdb_upload
Processing: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1/1 [01:03<00:00, 63.71s/doc]
  OK 1 chunks ingested in 65,634ms (0 failures)

  > [1/5] Using the balance sheet and P&L statement, calculate the debt-to-equity ratio and return on equity (ROE). Based on these metrics, is Gru's Enterprises financially healthy?

  > Node 1 classifier: type=calculation (4132ms)
  > Connecting to Milvus: ./milvus_rag.db
  OK Collection 'rag_documents' exists
  > Node 2 retriever: 20 chunks (351ms)
config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████| 791/791 [00:00<00:00, 3.75MB/s]
model.safetensors: 100%|███████████████████████████████████████████████████████████████████████████████| 133M/133M [00:05<00:00, 23.9MB/s]
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████| 201/201 [00:00<00:00, 3850.83it/s]
BertForSequenceClassification LOAD REPORT from: cross-encoder/ms-marco-MiniLM-L-12-v2
Key                          | Status     |  | 
-----------------------------+------------+--+-
bert.embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
tokenizer_config.json: 1.33kB [00:00, 3.69MB/s]
vocab.txt: 232kB [00:00, 3.52MB/s]
tokenizer.json: 711kB [00:00, 21.0MB/s]
special_tokens_map.json: 100%|████████████████████████████████████████████████████████████████████████████| 132/132 [00:00<00:00, 473kB/s]
README.md: 3.68kB [00:00, 12.0MB/s]
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.46s/it]
  > Node 3 reranker: 8 chunks, low (26203ms)
  > Generating with llama-3.3-70b-instruct...
  > Node 4 generator: llama-3.3-70b-instruct (25120ms)
  > LOW confidence -> reformulating (retry 1)
  > Node 1 classifier: type=calculation (14510ms)
  > Node 2 retriever: 20 chunks (412ms)
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.02s/it]
  > Node 3 reranker: 8 chunks, low (2024ms)
  > Generating with llama-3.3-70b-instruct...
  > Node 4 generator: llama-3.3-70b-instruct (5897ms)

+-- ANSWER --------------------------------------------------+
| LOW CONFIDENCE  |  llama-3.3-70b-instruct  |  78,658ms
| retried 1x
+------------------------------------------------------------+
| To calculate the debt-to-equity ratio and return on
| equity (ROE), we will use the values from the Balance
| Sheet and Profit and Loss Statement.
| 
| 1. Debt-to-Equity Ratio:
|    - Total Liabilities: $2,200,000
|    - Equity: $300,000
|    - Debt-to-Equity Ratio = Total Liabilities / Equity
|    - Debt-to-Equity Ratio = $2,200,000 / $300,000
|    - Debt-to-Equity Ratio = 7.33
| 
| 2. Return on Equity (ROE):
|    - Net Income: $80,000
|    - Equity: $300,000
|    - ROE = Net Income / Equity
|    - ROE = $80,000 / $300,000
|    - ROE = 0.2667 or 26.67%
| 
| Based on these metrics:
| - A debt-to-equity ratio of 7.33 indicates that Gru's
| Enterprises has a high level of debt compared to its
| equity, which could be a sign of financial risk.
| - A return on equity (ROE) of 26.67% suggests that the
| company is generating a significant return on its
| equity, which is a positive sign.
| 
| However, considering both metrics together, while the
| ROE is positive, the high debt-to-equity ratio may
| indicate that Gru's Enterprises is not entirely
| financially healthy due to its high reliance on debt.
| This could potentially lead to financial instability if
| not managed properly.
+------------------------------------------------------------+

  Latency:
      classifier  ##############################  14510ms
       retriever  ..............................  412ms
        reranker  ####..........................  2024ms
       generator  ############..................  5897ms

  Sources (6 chunks):
    Chunk 1  low  score=-9.455
    minion-tech.md 2024-01-06
1 / 22
Business Analysis Document
Gru's Enterprises...
    Chunk 2  low  score=-9.791
    minion-tech.md 2024-01-06
4 / 22
1. Executive Summary
Overview of Gru's Busin...
    Chunk 3  low  score=-6.745
    minion-tech.md 2024-01-06
2 / 22
Table of Contents
1. Executive Summary
Over...
    Chunk 4  low  score=-4.761
    | Financial Statements |  |
| 1.Balance Sheet |  |
|  | Details assets, liabilit...

  > [2/5] Compare the gross margin percentages of Freeze Ray, Rocket Boots, and Bubble Gun. Which product has the highest net margin and why might that be the case given the cost structure?

  > Node 1 classifier: type=comparison (1872ms)
  > Node 2 retriever: 20 chunks (407ms)
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.00s/it]
  > Node 3 reranker: 8 chunks, high (1010ms)
  > Generating with llama-3.3-70b-instruct...
  > Node 4 generator: llama-3.3-70b-instruct (15655ms)

+-- ANSWER --------------------------------------------------+
| HIGH CONFIDENCE  |  llama-3.3-70b-instruct  |  18,947ms
+------------------------------------------------------------+
| To compare the gross margin percentages of Freeze Ray,
| Rocket Boots, and Bubble Gun, we will examine the
| provided data:
| 
| 1. Freeze Ray: 70% Gross Margin
| 2. Rocket Boots: 60% Gross Margin
| 3. Bubble Gun: 66.7% Gross Margin
| 
| Comparison of Gross Margin Percentages:
| - Highest: Freeze Ray at 70%
| - Middle: Bubble Gun at 66.7%
| - Lowest: Rocket Boots at 60%
| 
| To determine which product has the highest net margin,
| we will look at the provided net margin percentages:
| 
| 1. Freeze Ray: 30% Net Margin
| 2. Rocket Boots: 26.7% Net Margin
| 3. Bubble Gun: 33.3% Net Margin
| 
| Comparison of Net Margin Percentages:
| - Highest: Bubble Gun at 33.3%
| - Middle: Freeze Ray at 30%
| - Lowest: Rocket Boots at 26.7%
| 
| The Bubble Gun has the highest net margin at 33.3%.
| This might be the case because, given its cost
| structure, the Bubble Gun has a relatively lower cost
| ($80,000) compared to its sales ($300,000), which could
| contribute to its higher net margin percentage.
| Additionally, its profit ($100,000) as a percentage of
| sales is higher than that of Rocket Boots, indicating
| more efficient cost management or pricing strategy for
| the Bubble Gun.
+------------------------------------------------------------+

  Latency:
      classifier  ###...........................  1872ms
       retriever  ..............................  407ms
        reranker  #.............................  1010ms
       generator  ##############################  15655ms

  Sources (6 chunks):
    Chunk 1  high  score=3.923
    | Best-selling Products Analysis | Best-selling Products Analysis | Best-selling...
    Chunk 2  high  score=4.506
    minion-tech.md 2024-01-06
19 / 22
Best-selling Products Analysis
Product Cost...
    Chunk 3  low  score=-0.838
    minion-tech.md 2024-01-06
6 / 22
3. Product Portfolio
Description of Key Prod...
    Chunk 4  medium  score=0.149
    minion-tech.md 2024-01-06
4 / 22
1. Executive Summary
Overview of Gru's Busin...

  > [3/5] Based on the cash flow statement, the company shows cash at beginning of period as negative $1,034,000 but ends at $500,000. Explain how this was achieved and whether the financing strategy is sustainable.

  > Node 1 classifier: type=factual (451ms)
  > Node 2 retriever: 12 chunks (374ms)
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.51s/it]
  > Node 3 reranker: 8 chunks, medium (1514ms)
  > Generating with llama-3.3-70b-instruct...
  > Node 4 generator: llama-3.3-70b-instruct (27708ms)

+-- ANSWER --------------------------------------------------+
| MEDIUM CONFIDENCE  |  llama-3.3-70b-instruct  |  30,051ms
+------------------------------------------------------------+
| To understand how the company achieved a positive cash
| balance at the end of the period, we need to analyze
| the cash flow statement provided in the context.
| 
| 1. **Cash at Beginning of Period**: -$1,034,000
| 2. **Net Increase in Cash**: $1,534,000
| 3. **Cash at End of Period**: $500,000
| 
| Let's calculate the change in cash:
| 
| **Initial Cash**: -$1,034,000
| **Net Increase in Cash**: +$1,534,000
| **Final Cash**: -$1,034,000 + $1,534,000 = $500,000
| 
| This calculation confirms that the company indeed ended
| the period with $500,000 in cash, starting from a
| negative balance.
| 
| Now, let's examine the components of the cash flow
| statement to understand the sources of this increase:
| 
| - **Operating Activities**: Net Cash from Operating
| Activities is $34,000. This is the result of:
|   - Net Income: $84,000
|   - Adjustments to Reconcile Net Income:
|     - Depreciation: $50,000
|     - Changes in Working Capital: -$100,000
|   - Total: $84,000 + $50,000 - $100,000 = $34,000
| 
| - **Investing Activities**: Net Cash from Investing
| Activities is -$200,000, due to Capital Expenditures of
| -$200,000.
| 
| - **Financing Activities**: Net Cash from Financing
| Activities is $1,700,000, resulting from Long-term Debt
| Financing of $1,700,000.
| 
| The significant increase in cash is primarily due to
| the financing activities, specifically the $1,700,000
| from long-term debt financing. This, combined with the
| $34,000 from operating activities, and despite the
| -$200,000 from investing activities, led to a net
| increase in cash of $1,534,000.
| 
| **Sustainability of the Financing Strategy**:
| 
| The reliance on long-term debt financing to achieve a
| positive cash balance raises questions about the
| sustainability of this strategy. While debt financing
| can provide necessary capital for operations and
| investments, excessive reliance on debt can lead to
| increased financial risk, including higher interest
| expenses and the potential for default if cash flows
| are insufficient to service the debt.
| 
| In the provided context, the company's balance sheet
| shows a significant long-term debt of $1,700,000, which
| matches the amount of long-term debt financing in the
| cash flow statement. This indicates that the company
| has indeed increased its debt burden.
| 
| Without information on the company's ability to
| generate sufficient cash flows to service this debt,
| it's challenging to definitively state whether this
| financing strategy is sustainable. However, it's clear
| that the company needs to carefully manage its debt and
| ensure that future cash flows from operations can cover
| interest payments and potentially debt repayment to
| avoid financial distress.
| 
| The provided documents do not contain detailed
| information on the company's future cash flow
| projections, interest rates on the debt, or other
| financial metrics that would be necessary for a
| comprehensive assessment of the sustainability of this
| financing strategy.
+------------------------------------------------------------+

  Latency:
      classifier  ..............................  451ms
       retriever  ..............................  374ms
        reranker  #.............................  1514ms
       generator  ##############################  27708ms

  Sources (6 chunks):
    Chunk 1  high  score=4.555
    minion-tech.md 2024-01-06
16 / 22
2. Cash Flow Statement
Outlines the cash in...
    Chunk 2  medium  score=1.074
    | 2.Cash Flow Statement |  |
| o Outlines the cash inflows and outflows over the...
    Chunk 3  low  score=-4.204
    | Financial Statements |  |
| 1.Balance Sheet |  |
|  | Details assets, liabilit...
    Chunk 4  low  score=-6.583
    minion-tech.md 2024-01-06
17 / 22
3. Profit and Loss Statement
Outlines the p...

  > [4/5] The company has total revenue of $4.2M but net income of only $80K. Walk through the full cost waterfall from revenue to net income, identifying which expense category consumes the largest share.

  > Node 1 classifier: type=calculation (419ms)
  > Node 2 retriever: 20 chunks (332ms)
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.81s/it]
  > Node 3 reranker: 8 chunks, low (1818ms)
  > Generating with llama-3.3-70b-instruct...
  > Node 4 generator: llama-3.3-70b-instruct (10577ms)
  > LOW confidence -> reformulating (retry 1)
  > Node 1 classifier: type=calculation (420ms)
  > Node 2 retriever: 20 chunks (342ms)
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.04s/it]
  > Node 3 reranker: 8 chunks, low (2046ms)
  > Generating with llama-3.3-70b-instruct...
  > Node 4 generator: llama-3.3-70b-instruct (13802ms)

+-- ANSWER --------------------------------------------------+
| LOW CONFIDENCE  |  llama-3.3-70b-instruct  |  29,762ms
| retried 1x
+------------------------------------------------------------+
| To walk through the full cost waterfall from revenue to
| net income, we will analyze the given Profit and Loss
| Statement for the Year 2023.
| 
| 1. **Revenue**: The company starts with a total revenue
| of $4,200,000.
| 
| 2. **Cost of Goods Sold (COGS)**: The COGS is
| $2,200,000, which includes $1,200,000 in Material Costs
| and $1,000,000 in Labor (Manufacturing) costs.
| 
| 3. **Gross Profit**: Subtracting COGS from Revenue
| gives us the Gross Profit.
|    - Gross Profit = Revenue - COGS
|    - Gross Profit = $4,200,000 - $2,200,000
|    - Gross Profit = $2,000,000
| 
| 4. **Operating Expenses**: The total Operating Expenses
| are $1,800,000, which includes $600,000 for R&D,
| $300,000 for Marketing, and $900,000 for General and
| Administrative expenses.
| 
| 5. **Operating Income**: Subtracting Operating Expenses
| from Gross Profit gives us the Operating Income.
|    - Operating Income = Gross Profit - Operating
| Expenses
|    - Operating Income = $2,000,000 - $1,800,000
|    - Operating Income = $200,000
| 
| 6. **Other Expenses**: The company has $100,000 in
| Other Expenses, which is solely Interest Expense.
| 
| 7. **Net Income Before Taxes**: Subtracting Other
| Expenses from Operating Income gives us the Net Income
| Before Taxes.
|    - Net Income Before Taxes = Operating Income - Other
| Expenses
|    - Net Income Before Taxes = $200,000 - $100,000
|    - Net Income Before Taxes = $100,000
| 
| 8. **Taxes**: The company pays 20% in Taxes.
|    - Taxes = 20% of Net Income Before Taxes
|    - Taxes = 0.20 * $100,000
|    - Taxes = $20,000
| 
| 9. **Net Income**: Finally, subtracting Taxes from Net
| Income Before Taxes gives us the Net Income.
|    - Net Income = Net Income Before Taxes - Taxes
|    - Net Income = $100,000 - $20,000
|    - Net Income = $80,000
| 
| From the given data, the largest expense category
| consuming the largest share of the revenue is the
| **Cost of Goods Sold (COGS)**, which is $2,200,000.
| This is followed by **Operating Expenses** at
| $1,800,000. The COGS category alone consumes more than
| half of the company's revenue, significantly impacting
| the net income.
+------------------------------------------------------------+

  Latency:
      classifier  ..............................  420ms
       retriever  ..............................  342ms
        reranker  ####..........................  2046ms
       generator  ##############################  13802ms

  Sources (6 chunks):
    Chunk 1  low  score=-3.575
    | minion-tech.md |  |
| 3.Profit and Loss Statement |  |
| oOutlines the profit ...
    Chunk 2  low  score=-2.735
    minion-tech.md 2024-01-06
17 / 22
3. Profit and Loss Statement
Outlines the p...
    Chunk 3  low  score=-6.842
    | 2.Cash Flow Statement |  |
| o Outlines the cash inflows and outflows over the...
    Chunk 4  low  score=-6.528
    | Financial Statements |  |
| 1.Balance Sheet |  |
|  | Details assets, liabilit...

  > [5/5] If the proposed $2M investment is secured with the projected 25% revenue increase over 3 years, what would the projected revenue be in year 3? Would the 15% annual profitability growth bring net income above $150K by then?

  > Node 1 classifier: type=calculation (461ms)
  > Node 2 retriever: 20 chunks (373ms)
Batches: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.05s/it]
  > Node 3 reranker: 8 chunks, medium (2059ms)
  > Generating with llama-3.3-70b-instruct...
  > Node 4 generator: llama-3.3-70b-instruct (7493ms)

+-- ANSWER --------------------------------------------------+
| MEDIUM CONFIDENCE  |  llama-3.3-70b-instruct  |  10,388ms
+------------------------------------------------------------+
| To calculate the projected revenue in year 3, we first
| need to establish the current revenue. According to the
| Profit and Loss Statement for the Year 2023, the Total
| Revenue is $4,200,000.
| 
| Given that the investment is expected to result in a
| 25% increase in annual revenue over the next three
| years, we calculate the revenue for each year as
| follows:
| 
| Year 1: $4,200,000 * 1.25 = $5,250,000
| Year 2: $5,250,000 * 1.25 = $6,562,500
| Year 3: $6,562,500 * 1.25 = $8,203,125
| 
| Therefore, the projected revenue in year 3 would be
| $8,203,125.
| 
| To determine if the 15% annual profitability growth
| would bring the net income above $150,000 by year 3, we
| first need to calculate the net income growth rate. The
| current net income is $80,000.
| 
| Year 1: $80,000 * 1.15 = $92,000
| Year 2: $92,000 * 1.15 = $105,800
| Year 3: $105,800 * 1.15 = $121,570
| 
| By year 3, the net income would be $121,570, which is
| below $150,000. Therefore, the 15% annual profitability
| growth would not bring the net income above $150,000 by
| year 3.
+------------------------------------------------------------+

  Latency:
      classifier  #.............................  461ms
       retriever  #.............................  373ms
        reranker  ########......................  2059ms
       generator  ##############################  7493ms

  Sources (6 chunks):
    Chunk 1  medium  score=0.327
    minion-tech.md 2024-01-06
13 / 22
9. Investment Appeal
Investment Needs
To a...
    Chunk 2  low  score=-7.456
    | 2.Cash Flow Statement |  |
| o Outlines the cash inflows and outflows over the...
    Chunk 3  low  score=-7.133
    minion-tech.md 2024-01-06
16 / 22
2. Cash Flow Statement
Outlines the cash in...
    Chunk 4  low  score=-7.157
    | minion-tech.md |  |
| 3.Profit and Loss Statement |  |
| oOutlines the profit ...
