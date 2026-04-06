
  ## Node 1: node_guardrail

  Purpose:

  - validate query
  - strip prompt injection patterns
  - clean noisy filler words
  - detect likely intent

  Example input:
  Can you tell me what the document says about colour contrast?

  It may clean to:
  colour contrast?

  Intent detection:

  - if query contains table, row, column -> table
  - if it contains image, figure, diagram -> image_caption
  - if it contains chart, graph, plot -> chart
  - else text

  Logic:
  This improves retrieval because filler phrases dilute embeddings.

  Example:
  Bad query for embedding:
  Can you please tell me what the document says about

  Better query:
  colour contrast

  That usually retrieves better.

  Limitation:
  Intent is only annotation here. It does not change retrieval strategy much.

  ———

  ## Node 2: node_query_expander

  Purpose:
  Generate 2 paraphrases of the cleaned query.

  Example:
  Original:
  colour contrast

  Possible paraphrases:

  - guidance on color contrast
  - what does the document recommend about contrast between colors

  Final variants:

  - original
  - paraphrase 1
  - paraphrase 2

  Logic:
  Documents and users often use different wording.
  Expansion increases recall.

  Why this helps:
  The PDF might say:

  - “contrast ratio”
  - “lightness differences”
  - “WCAG AA minimum”
    instead of “colour contrast”

  So paraphrases catch vocabulary mismatch.

  Limitation:
  This is still text-centric expansion. It does not create region-aware visual search.

  ———

  ## Node 3: node_retriever

  Purpose:
  Run Milvus search for each query variant and merge results.

  Flow:

  1. For each variant, embed it.
  2. Search Milvus top RETRIEVAL_TOP_K.
  3. Extract returned chunk text.
  4. Deduplicate by SHA256 hash.
  5. Build raw_chunks

  Example:
  Variant 1 retrieves 40 chunks
  Variant 2 retrieves 40 chunks
  Variant 3 retrieves 40 chunks

  After dedup:
  maybe 62 unique chunks remain

  Logic:
  Multiple searches broaden recall.

  Why hashing:
  The same chunk may be returned for several variants.
  You only want unique chunks before reranking.

  Limitation:
  This retrieves only output_fields=["text"]

  So you lose useful metadata like:

  - page number
  - source file
  - figure id
  - chunk type from ingest
  - bounding box
  - caption linkage

  That weakens enterprise grounding.

  ———

  ## Node 4: node_reranker

  Purpose:
  Take raw_chunks and reorder them by relevance using a cross-encoder.

  Flow:

  1. Create (query, passage) pairs.
  2. Predict logits.
  3. Sort descending.
  4. Assign confidence labels:
      - high
      - medium
      - low
  5. If top score < MIN_GENERATION_SCORE, set quality_gate_failed=True

  Example:
  If top rerank scores are:

  - -1.2
  - -2.6
  - -6.9
  - -9.5

  Then:

  - top two are high
  - third is medium
  - fourth is low

  Logic:
  The reranker is much better than pure vector similarity at deciding “does this chunk answer the question?”

  ### Why confidence score exists

  The confidence label here is not model truthfulness.
  It is retrieval relevance confidence.

  That means:

  - high = top chunk strongly matches query according to reranker
  - low = top chunk is probably weak/noisy

  It is a retrieval-quality signal, not a guarantee that the answer is correct.

  ### Why thresholds are designed this way

  The model used is:
  cross-encoder/ms-marco-MiniLM-L-12-v2

  This model emits logits, not probabilities.
  So raw numbers like -2.1 or -7.3 are meaningful only relative to calibration.

  The code comments say earlier thresholds were too strict or mismatched.
  So these bands were introduced to fit observed score distributions.

  That is reasonable, but still empirical.
  You should validate them on your own documents.

  ### Why quality gate exists

  If all retrieved chunks are poor, sending them to the LLM causes:

  - hallucinations
  - generic answers
  - wasted latency

  So the quality gate says:
  “if the best evidence is too weak, don’t generate.”

  This is a good design for enterprise settings where false positives are expensive.

  ———

  ## Node 5: node_generator

  Purpose:
  Generate the final answer from the top reranked chunks.

  Flow:

  1. If prompt injection detected, block.
  2. If quality gate failed, return “not found.”
  3. Take top MAX_CONTEXT chunks.
  4. Build prompt with chunk labels and scores.
  5. Prepend last 3 conversation turns.
  6. Call primary LLM.
  7. If it fails, call fallback.
  8. Detect simple hallucination phrases.

  Example prompt structure:

  Previous conversation:
  User: ...
  Assistant: ...

  Context:
  [Chunk 1 | confidence=high | score=-1.2]
  ...

  Question: What does the document say about colour contrast?
  Answer:

  Logic:
  This is standard answer synthesis on top of retrieved evidence.

  ### Why memory is included

  If user asks:

  - What does the document say about line graphs?
    then next asks:
  - What about grouped bar graphs?

  Memory lets the second question inherit prior context.

  Limitation:
  This is conversational memory, not retrieval-grounded memory.
  If the earlier answer was wrong, that error can propagate.

  ———

  # Routing and graph behavior

  ## route_after_guardrail

  If:

  - prompt_injection_detected
  - empty_query

  Then skip normal retrieval path and go straight to generator.

  Else:

  - go to expander

  Logic:
  Short-circuit invalid queries.

  ———

  ## get_graph()

  Creates the LangGraph state machine:

  - START -> guardrail
  - guardrail -> expander or generator
  - expander -> retriever
  - retriever -> reranker
  - reranker -> generator
  - generator -> END

  Why LangGraph is used:
  It gives a clean node-based execution model instead of one long procedural function.

  Benefits:

  - easier debugging
  - easier to extend
  - easier to add branches/retries/rescue nodes later

  ———

  # Retry logic outside the graph

  ## _prepare_retry_state

  If generated answer contains one of the hallucination phrases:

  - mark possible hallucination
  - create stricter query
  - rerun graph once

  Example retried query:
  original query -- answer using ONLY exact facts stated in the document. Do not infer.

  Logic:
  This is a cheap second-pass correction.

  Limitation:
  Again, phrase-based hallucination detection is weak.

  ———

  # Interactive loop

  ## interactive_loop()

  Commands:

  - ingest
  - stats
  - reset
  - history
  - clear
  - quit

  This makes the system a CLI agent rather than a one-shot script.

  Conversation history is stored and trimmed to last 10 turns.

  ———

  # Why This Is A “Single Agent”

  You called it single-agent. That is correct.

  Why?

  Because there is one orchestrated decision flow with one shared state.
  The nodes are not independent agents with separate planning.
  They are pipeline stages.

  So this is better described as:

  - single-agent RAG workflow
    or
  - graph-orchestrated retrieval agent

  not a true multi-agent system.

  ———

  # Why It’s Good For Tables And Text

  Three main reasons:

  ## 1. Text is naturally retrievable

  PDF prose becomes chunks well.

  ## 2. Tables can be serialized to text

  You use:

  table_output_format="markdown"

  That is excellent for RAG, because LLMs handle markdown tables relatively well.

  ## 3. Reranker works best on text

  Cross-encoders score text pairs.
  They are strongest when the passage already explicitly contains the answer.

  That is exactly what text and tables provide.

  ———

  # Why It’s Not Strong For Images, Charts, Diagrams, Captions

  Three exact reasons.

  ## 1. The final answer model is not visually grounded

  At question time, the LLM only sees text chunks, not actual page images or figure crops.

  ## 2. Visual meaning is compressed into captions

  A caption model may summarize:

  - “bar chart comparing categories”
    but miss:
  - exact bar heights
  - legend-color mapping
  - overlapping markers
  - arrow directions
  - region-level structure

  ## 3. Your split config is still text/chart-centric

  "split_source_types": ["text", "table", "chart"]

  This suggests the pipeline is optimized more for chunkable text-like content than for figure-region retrieval.

  ———

  # Do You Need GPU For Better Image/Chart QA?

  If you want strong local multimodal performance, yes, GPU is very useful.
  But the key issue is still architecture.

  To get Claude/GPT-like behavior you need:

  - page/figure image extraction
  - figure-level indexing
  - metadata with page/bbox
  - retrieval of page or figure image
  - vision-capable model at answer time

  Then GPU becomes relevant if you run that model locally.

  If you use hosted multimodal APIs, you may not need local GPU, but you still need that architecture.

  ———

  # Why Confidence Was Designed Like This

  The confidence labels are trying to answer:

  “How strong is the retrieval evidence?”

  Not:
  “How certain is the answer semantically?”

  That is why they come from reranker score bands.

  Why useful:

  - user feedback
  - source ranking visibility
  - generation gating
  - safer behavior on low-quality retrieval

  Why imperfect:

  - high reranker score can still lead to wrong answer if chunk is incomplete
  - low score does not always mean impossible answer
  - score calibration is model-dependent and corpus-dependent

  So these confidence labels are operational heuristics, not truth metrics.

  ———

  # Concrete Example End-to-End

  Suppose the user asks:

  What does the document recommend for grouped bar graphs?

  Flow:

  1. guardrail
      - cleans the query
      - intent likely becomes chart
  2. expander
      - creates paraphrases like:
      - guidance for grouped bar graphs
      - how should grouped bar charts be made accessible
  3. retriever
      - embeds each variant
      - Milvus returns related chunks
  4. reranker
      - ranks the chunks
      - top chunks likely contain the “Grouped Bar Graphs” section
  5. generator
      - reads top 6 chunks
      - answers using those chunks only

  This will likely work because the PDF contains explanatory text for grouped bar graphs.

  Now ask:

  In the grouped bar graph figure, which color corresponds to the left-most series under deuteranopia?

  This is much weaker, because:

  - answer depends on precise visual mapping
  - caption may not encode that
  - extracted text may omit it
  - final LLM never sees the figure image directly

  ———

  # Bottom Line

  This code is a well-structured text-centric enterprise RAG pipeline with multimodal ingestion support.

  It is strong at:

  - document QA over prose
  - section lookup
  - many table questions
  - broad figure/caption summaries when extraction is good

  It is weak at:

  - exact image reasoning
  - fine chart reading
  - complex diagram topology
  - spatial/region-level visual understanding

  So the short answer is:

  - NV-Ingest is used because it is the ingestion engine that converts messy enterprise documents into chunks, captions,
    embeddings, and vector-store records.
  - The pipeline is good with text and tables because those become explicit textual evidence.
  - It is not strong with images/charts/diagrams because the architecture relies on textified visual content, not true
    image reasoning at query time.
  - GPU helps only if you also redesign the system to use a vision model over actual page/figure images during retrieval
    or generation.
