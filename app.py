━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 0 — CONFIGURATION (PipelineCreationSchema)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NVIDIA_API_KEY → injected to ALL NIM endpoints
  yolox_auth_token, paddle_auth_token, auth_token
  broker: localhost:7671
  taskset: cores 0-7 (prevents Ray conflict)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 1 — NV-INGEST PIPELINE (extraction only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  .load()
    └─ page split + rasterize
  .extract()
    ├─ nemoretriever-page-elements-v2  → bbox per element
    ├─ nemoretriever-ocr-v1            → text from regions
    ├─ nemoretriever-table-structure-v1→ rows/cols/cells
    └─ nemoretriever-graphic-elements-v1→ chart elements
  .caption()
    └─ nemotron-nano-vl-8b-v1          → image descriptions

  OUTPUT: raw JSON with content_type, bbox, page_number,
          base64 image, captions — NO split/embed/vdb

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 2 — METADATA ENRICHMENT (your code owns this)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Per chunk, attach:

  CORE:
    doc_id          → MD5(source_file + ingested_at)
    chunk_id        → MD5(doc_id + page + type + text[:40])
    content_type    → text / table / table_row / image / chart
    page_number     → from NV-Ingest bbox metadata
    source_file     → original filename
    language        → detected or default "en"

  TIME:
    ingested_at     → datetime.utcnow().isoformat()
    pipeline_version→ "nv-ingest-25.9.0"
    embedding_model → "llama-3.2-nv-embedqa-1b-v2"

  SPATIAL:
    bbox            → [x1, y1, x2, y2] from NV-Ingest
    bbox_page_dims  → [page_w, page_h]

  CONTENT-SPECIFIC:
    [TEXT]
      section_title → nearest heading above (y-coord scan)
      word_count    → len(text.split())

    [TABLE]
      table_id      → chunk_id of parent table
      row_count     → number of data rows
      col_count     → number of columns
      headers       → list of column header strings
      table_summary → first 200 chars of table

    [IMAGE/CHART]
      image_type    → "raster" or "vector" (SVG detected)
      caption       → VLM output
      image_path    → S3/MinIO URL (not b64 in Milvus!)
      width/height  → from bbox dims


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 3 — MILVUS (3 SEPARATE COLLECTIONS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  text_collection
  ├── id (INT64, PK auto)
  ├── embedding (FLOAT_VECTOR 2048)
  ├── sparse_vector (SPARSE_FLOAT_VECTOR)
  ├── text (VARCHAR 65535)
  ├── doc_id (VARCHAR 64)         ← for dedup + filtering
  ├── chunk_id (VARCHAR 32)
  ├── page_number (INT64)
  ├── section_title (VARCHAR 512)
  ├── source_file (VARCHAR 512)
  ├── language (VARCHAR 8)
  ├── word_count (INT64)
  ├── ingested_at (VARCHAR 32)
  └── pipeline_version (VARCHAR 32)

  table_collection
  ├── id, embedding, sparse_vector
  ├── text (full table markdown)
  ├── doc_id, chunk_id, page_number, source_file
  ├── table_id (VARCHAR 32)
  ├── row_count (INT64)
  ├── col_count (INT64)
  ├── headers (VARCHAR 2048)      ← JSON encoded list
  └── table_summary (VARCHAR 512)

  image_collection
  ├── id, embedding, sparse_vector
  ├── text (caption, searchable)
  ├── doc_id, chunk_id, page_number, source_file
  ├── image_path (VARCHAR 1024)   ← S3/MinIO URL
  ├── image_type (VARCHAR 16)     ← "raster" / "vector"
  ├── caption (VARCHAR 4096)
  ├── bbox (VARCHAR 256)
  └── width, height (INT64)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 4 — LANGGRAPH RAG AGENT (6 nodes, not 4-5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  NODE 1: query_classifier
    Input : raw query string
    Does  : LLM call (llama-3.3-70b, 256 tokens, temp=0)
            classifies → {text, table, image, all}
            also detects: needs_calculation, needs_comparison
    Why   : avoids wasteful retrieval from wrong collections
            "show me the chart" → image only (not text)
            "total amount" → table first (not image)
    Output: {collections: [...], query_type, intent}

  NODE 2: parallel_retriever (runs 3 sub-retrievals in parallel)
    Input : query + collections list from node 1
    Does  : asyncio.gather() → simultaneous retrieval from
            whichever collections were flagged
            Each: dense(top30) + sparse(top30) → RRF → top10
    Why   : parallel not sequential = 3x lower latency
            if image collection not needed, skip entirely
    Output: {text_hits, table_hits, image_hits}

  NODE 3: result_merger
    Input : all hits from parallel retriever
    Does  : merge with priority weighting:
              table_hits  → weight 1.3  (structured data)
              text_hits   → weight 1.0  (baseline)
              image_hits  → weight 0.8  (supplementary)
            deduplicate by chunk_id
            limit to top 30 merged candidates
    Why   : tables should score higher for factual queries
            images are supporting context
    Output: merged_candidates list

  NODE 4: reranker
    Input : query + merged_candidates
    Does  : llama-3.2-nv-rerankqa-1b-v2
            scores each candidate with logit
            classifies HIGH/MEDIUM/LOW confidence
            if ALL candidates are LOW → triggers retry loop
    Why   : cross-encoder sees query+passage together
            catches semantic mismatches dense missed
    Output: reranked_chunks + confidence_map

  NODE 5: context_builder
    Input : reranked_chunks (top 8)
    Does  : builds structured prompt context:
              [CHUNK | TYPE | PAGE | SOURCE | CONFIDENCE]
              text chunks  → raw text
              table chunks → markdown preserved
              image chunks → caption + image_path reference
            applies guardrails (injection, length check)
    Why   : structure in context = better LLM grounding
            type labels help LLM cite correctly
    Output: formatted context string + source list

  NODE 6: generator (with fallback + self-check loop)
    Input : context + query
    Does  :
      1. Call PRIMARY: meta/llama-3.3-70b-instruct
         temp=0.3, max_tokens=1024
      2. Check answer for hallucination phrases
      3. If hallucination detected OR primary fails:
           → call FALLBACK: llama-3.1-nemotron-70b
      4. If confidence was LOW from reranker:
           → loop back to node 1 with reformulated query
             (max 1 retry to prevent infinite loop)
      5. Output guardrail check
    Why   : self-correction loop = GPT-o1 style reasoning
            single retry prevents hallucination amplification
    Output: RAGResponse with answer, sources, confidence,
            model_used, fallback_used, latency_ms

  EDGES (the LangGraph difference):
    query_classifier → parallel_retriever
    parallel_retriever → result_merger
    result_merger → reranker
    reranker → context_builder  (if HIGH/MEDIUM confidence)
    reranker → query_classifier (if ALL LOW → retry once)
    context_builder → generator
    generator → END             (if answer OK)
    generator → query_classifier (if hallucination → retry)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAYER 5 — RESPONSE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  answer
  model_used + fallback_used
  confidence: HIGH / MEDIUM / LOW
  sources: [{chunk_id, page, type, score, file}]
  latency_ms (per node breakdown)
  guardrail_flags
  retry_count
