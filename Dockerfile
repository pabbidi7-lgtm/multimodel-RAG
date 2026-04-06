
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
--------------------------------------------------------------------------------------------------------------------------


Current Architecture

                        INGESTION PHASE
  ┌──────────────────────────────────────────────────────────────────────┐
  │  File(s)                                                            │
  │    PDF / DOCX / PPTX / XLSX / images                                │
  └──────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ NV-Ingest                                                            │
  │  1. load()                                                           │
  │  2. extract(text, tables, charts, images, infographics)             │
  │  3. split(text/table/chart chunks)                                  │
  │  4. caption(visuals -> text description)                            │
  │  5. embed(all chunks)                                               │
  │  6. vdb_upload()                                                    │
  └──────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ Milvus                                                               │
  │ Stores mostly text-like chunks + embeddings                          │
  │ Current code retrieves only: text                                    │
  └──────────────────────────────────────────────────────────────────────┘


                        QUERY / AGENT PHASE
  START
    │
    ▼
  [N1 guardrail]
    - clean query
    - detect injection
    - detect likely intent
    │
    ▼
  [N2 expander]
    - original query
    - 2 paraphrases
    │
    ▼
  [N3 retriever]
    - embed each variant
    - Milvus ANN search
    - merge + dedupe chunks
    │
    ▼
  [N4 reranker]
    - cross-encoder scores query/chunk pairs
    - assigns high/medium/low confidence
    - optional quality gate
    │
    ▼
  [N5 generator]
    - last 3 turns of memory
    - top 6 chunks
    - primary LLM, fallback LLM
    │
    ▼
   END

  Why This Version Works Better For Text And Tables

  - Text survives ingestion almost directly.
  - Tables can be serialized into markdown-like text, which LLMs handle well.
  - Retrieval and reranking are both text-based.
  - The final LLM answers from text chunks, so if the source is already textual, the pipeline is aligned.

  Why It Struggles With Images / Charts / Diagrams

  - Images are converted into captions, not preserved as first-class evidence for QA.
  - The agent does not retrieve page images or figure crops at query time.
  - The generator never sees pixels, only caption/extracted text.
  - Fine visual details get lost:
      - spatial layout
      - arrow direction
      - legend/color mapping
      - exact bar/line relationships
      - small labels inside figures

  So the weakness is not “NV-Ingest is bad.” It is that your system is caption-grounded instead of vision-grounded.

  Why Use NV-Ingest At All

  NV-Ingest is still the right choice here because it solves the hard ingestion layer:

  - OCR / text extraction
  - table extraction
  - chart/image extraction hooks
  - chunking
  - caption generation
  - embedding pipeline integration
  - vector DB upload

  Without it, you would need to build a full enterprise document parsing pipeline yourself.

  What NV-Ingest gives you:

  - multimodal preprocessing

  What it does not automatically give you:

  - GPT/Claude-style visual reasoning at answer time

  ———

  Recommended v3

  This is the minimum serious upgrade.

  INGEST
    PDF
     │
     ▼
  [Parser / NV-Ingest]
    - text chunks
    - table chunks
    - figure crops
    - page images
    - chart regions
    - captions
    - OCR per visual region
     │
     ▼
  [Indexing Layer]
    - child chunks:
        text chunk
        table chunk
        caption chunk
        OCR chunk
        chart-data chunk
    - parent objects:
        page
        figure
        section
     │
     ▼
  [Milvus + Metadata Store]
    Store:
    - embedding
    - text
    - file_path
    - page_num
    - figure_id
    - section_title
    - source_type
    - bbox
    - parent_page_id

  Query graph:

  START
    │
    ▼
  [N1 guardrail]
    │
    ▼
  [N2 intent router]
    - text / table / figure / chart / diagram / mixed
    │
    ▼
  [N3 query expander]
    │
    ▼
  [N4 hybrid retrieval]
    - dense retrieval
    - sparse/BM25 retrieval
    - metadata filters
    - parent-child join
    │
    ▼
  [N5 modality reranker]
    - text rerank
    - table preference if table query
    - figure preference if figure query
    │
    ▼
  [N6 evidence builder]
    Build final bundle:
    - top chunks
    - linked figure caption
    - linked nearby paragraph
    - page image / figure crop if needed
    │
    ▼
  [N7 answerer]
    - text-only LLM if enough
    - vision LLM if visual grounding needed
    │
    ▼
   END

  What v3 fixes

  - keeps figure/page metadata
  - links chunks back to pages/figures
  - adds BM25 from the start
  - lets answer step escalate to vision only when needed

  ———

  Recommended v4: GPT/Claude-like Multimodal RAG

  This is the version you actually want for complex PDFs.

                           MULTIMODAL INGEST
  PDF
   │
   ├─ page rasterization --------------------------------------┐
   ├─ text extraction                                          │
   ├─ table extraction                                         │
   ├─ chart/figure detection                                   │
   ├─ OCR on figure regions                                    │
   └─ caption / structured visual summary                      │
                                                               ▼
                       ┌───────────────────────────────────────────────┐
                       │ Unified Evidence Store                        │
                       │                                               │
                       │ text chunks                                   │
                       │ tables                                        │
                       │ captions                                      │
                       │ OCR snippets                                  │
                       │ chart data / axes / legends                   │
                       │ figure crops                                  │
                       │ page images                                   │
                       │ metadata: page, bbox, section, figure id      │
                       └───────────────────────────────────────────────┘

  Query graph:

  START
    │
    ▼
  [N1 guardrail]
    │
    ▼
  [N2 query understanding]
    - classify:
      factual text?
      table lookup?
      chart reading?
      diagram reasoning?
      mixed?
    │
    ▼
  [N3 decomposition]
    Example:
    "What does Figure 3 show and what warning is given in nearby text?"
    becomes:
    - find Figure 3
    - inspect figure
    - inspect nearby text
    - combine
    │
    ▼
  [N4 retrieval planner]
    choose evidence types:
    - text chunks
    - tables
    - page image
    - figure crop
    - OCR region
    - caption
    │
    ▼
  [N5 retrieval]
    - dense + sparse + metadata + figure/page lookup
    │
    ▼
  [N6 evidence fusion]
    assemble one grounded bundle:
    - page 12 figure crop
    - caption
    - nearby paragraph
    - OCR labels
    - section heading
    │
    ▼
  [N7 multimodal reasoner]
    vision-capable model reads:
    - image(s)
    - text evidence
    - user question
    │
    ▼
  [N8 grounded answer + citations]
    - answer
    - page number
    - figure id
    - confidence by evidence type
    │
    ▼
   END

  Why this feels like GPT/Claude

  - it can inspect the actual figure at answer time
  - it combines text and image evidence in one reasoning step
  - it cites page/figure-level evidence
  - it does not depend on caption quality alone

  ———

  Exact Improvements I’d Make

  1. Store metadata in Milvus or sidecar DB:

  - file_path
  - page_num
  - figure_id
  - bbox
  - section_title
  - source_type
  - parent_id

  2. Change retrieval output from just text to full evidence objects.
  3. Add page rasterization and figure crops during ingest.
  4. Run OCR on figure regions, not only full-page text extraction.
  5. Add chart-structure extraction if charts matter:

  - title
  - axis labels
  - legend labels
  - bar/line labels

  6. Use true hybrid retrieval:

  - dense
  - BM25
  - metadata filter
  - reciprocal rank fusion

  7. Add a visual-escalation rule:

  - if question mentions figure/chart/diagram/image/layout/color/arrow/legend
  - answer with a vision model over retrieved page/figure images

  8. Make confidence evidence-based:

  - retrieval confidence
  - visual grounding confidence
  - citation completeness
  - not just reranker score

  ———

  Confidence Design In Better Form

  Your current confidence is:

  - based on reranker score only

  That is useful but incomplete.

  A better enterprise confidence model is:

  final_confidence =
    retrieval_score
    + citation_completeness
    + modality_match
    + agreement_between_text_and_visual_evidence
    - ambiguity_penalty

  Example:

  - High retrieval score but no figure crop for a chart question -> confidence should drop
  - Medium retrieval score but direct figure grounding + nearby text agreement -> confidence can rise

  ———

  Practical Answer To “Do I Need GPU?”

  - For current v2: not necessarily. It is mostly text-RAG plus preprocessing.
  - For v3/v4 with local vision reasoning: yes, GPU becomes very useful.
  - If you use hosted multimodal APIs, local GPU is less necessary, but architecture changes are still required.

  So GPU is not the first blocker.
  The first blocker is that the current system does not retrieve and reason over visual evidence directly.

  ———
