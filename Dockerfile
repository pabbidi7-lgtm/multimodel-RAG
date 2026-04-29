# Enterprise RAG Application: Journey from Static to Multimodal

**A comprehensive documentation of building production-grade RAG systems using NVIDIA's Technology Stack**

📍 **Status**: Architecture Documented | Library Mode ✅ | Microservices Ready | Agentic Workflows ✅ | Evaluation Frameworks ✅ | A100 Deployment (Pending)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Journey Overview](#journey-overview)
3. [Architecture Fundamentals](#architecture-fundamentals)
4. [Core Approaches](#core-approaches)
5. [Ingestion Pipeline (Library Mode)](#ingestion-pipeline-library-mode)
6. [Agentic Workflows](#agentic-workflows)
7. [Multimodal RAG (VLM Integration)](#multimodal-rag-vlm-integration)
8. [Evaluation Frameworks](#evaluation-frameworks)
9. [Deployment Models](#deployment-models)
10. [Performance & Monitoring](#performance--monitoring)
11. [Future Roadmap](#future-roadmap)

---

## Executive Summary

This repository documents the complete evolution of a RAG (Retrieval-Augmented Generation) system built on NVIDIA's cutting-edge AI stack. The project demonstrates:

- **Hybrid Architectures**: Library mode (local) vs Microservices (distributed) vs Agentic workflows
- **Multimodal Processing**: Text, images, charts, tables, handwritten documents
- **Intelligent Routing**: LangGraph-based state machines for adaptive retrieval and generation
- **Enterprise-Grade Evaluation**: RAGAS metrics, NeMo evaluator, custom faithfulness scoring
- **Production Readiness**: Timestamps, latency tracking, performance monitoring

**Key Technologies Used**:
- 🔬 **NV Ingest 25.9.0** - Document extraction & chunking
- 🧠 **Nemotron Models** - LLM inference (70B, nano VL-8B)
- 🎯 **LangGraph** - Agentic workflows & state management
- 🗄️ **Milvus Lite** - Local vector database
- 🏛️ **NVIDIA APIs** - Cloud-based embeddings & generation
- 📊 **RAGAS + NeMo Evaluator** - Quality assessment frameworks

---

## Journey Overview

### Phase 1: Static RAG (Foundation)
**Goal**: Build basic retrieval-augmented generation with text documents

```
┌──────────────────────────────────────────────────────────┐
│ Input Document                                           │
│ (PDF, DOCX, TXT)                                         │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│ Text Extraction → Chunking → Embedding                   │
│ (Basic pipeline, no models)                              │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│ Milvus Vector Store                                      │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│ Query → Similarity Search → LLM Generation → Answer      │
└──────────────────────────────────────────────────────────┘
```

**Learnings**:
- Embedding quality directly impacts retrieval accuracy
- Chunking strategy (size + overlap) affects context window
- Need for reranking to improve precision

**Files**: `test.py` (API endpoint testing)

---

### Phase 2: Discovering NVIDIA's Stack (Architecture Shift)
**Goal**: Learn Nemotron models, NIM services, and microservices patterns

**Realization**: Two execution models emerged!

```
Initial Understanding:
  Broker/Client Architecture
  ├─ Broker: Your Python script (rag_agent.py)
  ├─ Server: Docker containers with models
  └─ Communication: HTTP ports (7671, 8000)
  
Discovery:
  Library Mode Also Possible
  ├─ No separate server needed
  ├─ Models as libraries in-process
  └─ Simpler deployment, no network overhead
```

**Files**: `docker-compose.yml`, `Dockerfile`, `ARCHITECTURE_INTERPRETATION.txt`

---

### Phase 3: Library Mode Implementation (Local Execution)
**Goal**: Build efficient local ingestion without external servers

**Result**: Three variations of pipelines

```
pipeline.py     → Full-featured (text + tables + charts + images + captions)
pipeline1.py    → Variant for specific use cases
pipeline2.py    → Another implementation approach
```

**Key Achievement**: Proved NV Ingest can run as a library with Ray backend locally

**Files**: `pipeline.py`, `pipeline1.py`, `pipeline2.py`

---

### Phase 4: Intelligent Routing (Agentic Workflows)
**Goal**: Build agents that KNOW when and what to retrieve

**Implementation**: Two LangGraph-based agents

```
rag_agent.py (9-Node Orchestrator)           rag_agent1.py (5-Node Retry Agent)
├─ guardrail                                 ├─ guardrail
├─ intent_router          [Expert routing]   ├─ query_expander
├─ query_expander         [Smart expansion]  ├─ retriever
├─ dual_retriever         [BM25 + semantic]  ├─ reranker
├─ cross_reranker         [Sophisticated]    ├─ generator
├─ layout_rescue          [VLM fallback]     └─ Retries on low confidence
├─ vlm_reread             [Image understanding]
├─ evidence_builder       [Context assembly]
└─ generator              [Answer generation]

rag_agent.py Features:
  ✓ Custom VLM (Nemotron Nano VL-8B)
  ✓ Intent detection (text vs table vs image)
  ✓ Dual retrieval (BM25 + semantic)
  ✓ Multiple rescue strategies

rag_agent1.py Features:
  ✓ Simplified workflow
  ✓ Automatic retries on low confidence
  ✓ Lighter weight, faster startup
  ✓ No VLM dependency
```

**Files**: `rag_agent.py`, `rag_agent1.py`

---

### Phase 5: Multimodal RAG (VLM Integration)
**Goal**: Process images and extract visual content using VLM

**Implementations**:
- `vlm.py` - Pure VLM (no LLM, no embeddings, base64 input)
- `vlm_nvingest.py` - VLM + NV Ingest (full pipeline with 8B model)

**Capability**: Understand images, generate captions, answer visual questions

**Files**: `vlm.py`, `vlm_nvingest.py`

---

### Phase 6: Enterprise Evaluation (Quality Assessment)
**Goal**: Measure RAG system quality with metrics

**Three Evaluation Approaches**:

```
rag_eval/
├─ rag.py              → Custom metrics (Faithfulness, Context Precision, Context Recall)
├─ pipeline.py         → Library mode with OCR + captions
└─ evals/              → Evaluation results

rag_eavl1/
├─ evals.py            → RAGAS framework integration
├─ evals1.py           → Variant evaluation
├─ pipeline1.py        → Clean RAGAS pipeline
└─ milvus.db           → Evaluated embeddings

rag_nemo_sdk/
├─ rag_service/        → RAG wrapped as API endpoint
├─ nemo_eval/          → NeMo evaluator with BYOB, precision, recall
└─ Milvus integration  → Results storage
```

**Metrics Tracked**:
- **Faithfulness**: Does answer match retrieved context?
- **Context Precision**: What % of retrieved chunks are relevant?
- **Context Recall**: Did we retrieve enough relevant chunks?
- **Latency**: Per-node timing breakdown (ms)
- **Confidence**: Model's self-assessment of answer quality

---

### Phase 7: Production-Grade System (NV_Ingest.py)
**Goal**: Create A100-ready deployment with full instrumentation

**Features**:
- Self-hosted NIM services (no caption model requirement)
- Comprehensive timestamp tracking (before/during/after ingestion)
- Per-model latency monitoring (LLM, embedding, reranker)
- Document location tracking in Outputs/
- Environment-based configuration
- Ready for enterprise deployment

**Status**: ✅ All commands ready | ⏳ Pending A100 execution

**Files**: `NV_Ingest.py`, `Outputs/` (results directory)

---

## Architecture Fundamentals

### Two Execution Models Explained

#### Model 1: Library Mode (Local, Current Approach)

```
User Query
    ↓
[rag_agent1.py - Python Script]
    ├─ Imports NV-Ingest as library
    ├─ Launches Ray pipeline (local subprocess)
    ├─ Connects to localhost:7671
    └─ Processes all jobs locally
    ↓
┌─────────────────────────────────────────┐
│ In-Process Execution                    │
│ ┌───────────────────────────────────┐   │
│ │ NV-Ingest Library                 │   │
│ │ • Extract text/images/tables      │   │
│ │ • Chunk efficiently               │   │
│ │ • Generate embeddings (local)     │   │
│ └───────────────────────────────────┘   │
│ ┌───────────────────────────────────┐   │
│ │ Milvus Lite (milvus_rag.db)       │   │
│ │ • Local vector storage            │   │
│ │ • Similarity search               │   │
│ └───────────────────────────────────┘   │
│ ┌───────────────────────────────────┐   │
│ │ LangGraph State Machine           │   │
│ │ • Route queries intelligently     │   │
│ │ • Manage conversation state       │   │
│ └───────────────────────────────────┘   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ External Services (NVIDIA APIs)         │
│ • LLM generation (Llama 3.3)           │
│ • Embeddings (if needed)                │
│ • VLM captions (if enabled)             │
└─────────────────────────────────────────┘
    ↓
Answer + Confidence + Sources
```

**Advantages**:
- ✅ Simple deployment (1 Python script)
- ✅ No network latency between components
- ✅ Easy debugging (single process)
- ✅ Fast startup (~2-5 min for Ray warmup)

**Limitations**:
- ⚠️ Single machine only
- ⚠️ Resource-bound to machine specs
- ⚠️ Ray cluster adds complexity

---

#### Model 2: Microservices Mode (Docker, Scalable)

```
Multiple Clients                    
    ├─ rag_agent1.py
    ├─ rag_agent.py
    └─ Other brokers
    ↓
    └─→ HTTP Ports (8000, 7671, 6379)
            ↓
    ┌───────────────────────────────────────┐
    │ Docker Compose Infrastructure         │
    │ ┌─────────────────┐                   │
    │ │ Redis 6379      │ Caching/Broker    │
    │ ├─────────────────┤                   │
    │ │ Milvus 19530    │ Vector DB         │
    │ ├─────────────────┤                   │
    │ │ NV-Ingest 8000  │ API Server        │
    │ ├─────────────────┤                   │
    │ │ NIM Services    │ Model Inference   │
    │ │ - LLM (8000)    │                   │
    │ │ - Embedding     │                   │
    │ │ - VLM           │                   │
    │ └─────────────────┘                   │
    └───────────────────────────────────────┘
```

**Advantages**:
- ✅ Horizontal scalability (multiple clients)
- ✅ Service isolation & failure containment
- ✅ Can upgrade services independently
- ✅ Shared GPU resources

**Limitations**:
- ⚠️ Network latency between services
- ⚠️ Complex deployment & orchestration
- ⚠️ Debugging harder (distributed system)

---

## Core Approaches

### Approach 1: Library Mode (pipeline.py)

**What It Does**: Ingests documents using NV Ingest as a Python library

```python
# Start Ray pipeline locally
from nv_ingest.framework.orchestration.ray.util.pipeline import run_pipeline
config = PipelineCreationSchema()
run_pipeline(config, block=False, run_in_subprocess=True)

# Connect to local broker
client = NvIngestClient(
    message_client_hostname="localhost",
    message_client_port=7671
)

# Define ingestion pipeline
ingestor = (
    Ingestor(client=client)
    .files("Docs/file.pdf")
    .extract(extract_text=True, extract_images=True)
    .split(chunk_size=512, chunk_overlap=50)
    .caption(endpoint_url="...", model_name="nemotron-nano-vl-8b")
    .embed()
    .vdb_upload(collection_name="docs", milvus_uri="milvus.db")
)

results, failures = ingestor.ingest(show_progress=True)
```

**Key Configuration** (from code):
```
MILVUS_URI        = "milvus.db"           # Local file
COLLECTION        = "multimodal_docs"     # Collection name
DENSE_DIM         = 2048                  # Embedding dimension
CHUNK_SIZE        = 512                   # Characters per chunk
TOKENIZER         = "intfloat/e5-large"   # Chunk boundary detector
EXTRACT_METHOD    = "pdfium"              # PDF extraction engine
```

**Pipeline Stages**:
1. **Load**: Read file from disk
2. **Extract**: Text, tables, charts, images (configurable)
3. **Split**: Smart chunking with overlap
4. **Caption**: Generate image descriptions (optional)
5. **Embed**: Convert text to vectors
6. **VDB Upload**: Store in Milvus

**Time Breakdown** (typical):
- Load: ~100ms
- Extract: ~500ms-2s (depends on doc size)
- Split: ~200ms
- Caption: ~1-2s per image
- Embed: ~500ms-1s
- VDB Upload: ~300ms

---

### Approach 2: Microservices Mode (docker-compose.yml)

**What It Does**: Run NV Ingest as a separate Docker service

```yaml
version: "3.9"

services:
  redis:
    image: redis:7
    ports:
      - "6379:6379"
    # Caching for pipeline
    
  milvus:
    image: milvusdb/milvus:v2.4.4
    ports:
      - "19530:19530"
    # Vector database service
    
  nv-ingest-api:
    image: nvcr.io/nvidia/nemo-retriever/nv-ingest:25.9.0
    ports:
      - "8000:8000"
    environment:
      NVIDIA_API_KEY: ${NVIDIA_API_KEY}
    # Ingestion microservice
```

**How Clients Connect**:
```python
import requests

# For library mode (doesn't use this)
# For microservices, send HTTP requests:
response = requests.post(
    "http://localhost:8000/ingest",
    json={
        "files": ["path/to/doc.pdf"],
        "extract_text": True,
        "chunk_size": 512,
        "collection": "docs"
    }
)
```

---

### Approach 3: Agentic Workflow (rag_agent.py & rag_agent1.py)

**What It Does**: Intelligent routing using state machines to decide when/what to retrieve

#### Comparison: 9-Node vs 5-Node Agent

```
┌─────────────────────────────────────────────────────────────────┐
│ rag_agent.py (9 Nodes) - Expert System                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  guardrail          [Input validation & intent detection]      │
│       ↓                                                          │
│  intent_router      [Route by content type: text/table/image]  │
│       ↓                                                          │
│  query_expander     [Generate search variants]                 │
│       ↓                                                          │
│  dual_retriever     [BM25 + semantic search]                   │
│       ↓                                                          │
│  cross_reranker     [Sophisticated reranking]                  │
│       ↓                                                          │
│  layout_rescue      [Fallback for layout-heavy docs]           │
│       ↓                                                          │
│  vlm_reread         [Custom VLM for images]          ⭐       │
│       ↓                                                          │
│  evidence_builder   [Assemble context intelligently]           │
│       ↓                                                          │
│  generator          [Generate answer + confidence]             │
│                                                                 │
│ Features:                                                       │
│  ✓ Custom Nemotron Nano VL-8B model                           │
│  ✓ Intent-aware routing                                        │
│  ✓ Multiple fallback strategies                                │
│  ✓ Advanced reranking                                          │
│  ✓ Comprehensive latency tracking                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ rag_agent1.py (5 Nodes) - Lightweight Retry Agent             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  guardrail          [Input validation]                         │
│       ↓                                                          │
│  query_expander     [Generate search variants]                 │
│       ↓                                                          │
│  retriever          [BM25 + semantic search]                   │
│       ↓                                                          │
│  reranker           [Local cross-encoder]                      │
│       ↓                                                          │
│  generator          [Generate answer]                          │
│       ↓                                                          │
│  confidence check   → Low? Retry with expanded query ↻         │
│                                                                 │
│ Features:                                                       │
│  ✓ Automatic retry on low confidence                          │
│  ✓ Lighter footprint                                           │
│  ✓ No external VLM dependency                                  │
│  ✓ Faster startup                                              │
│  ✓ Ideal for text-only workloads                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Ingestion Pipeline (Library Mode)

### How It Works: Step-by-Step

**Step 1: Initialize**
```python
from nv_ingest.framework.orchestration.ray.util.pipeline import run_pipeline

# Start Ray cluster + broker on localhost:7671
config = PipelineCreationSchema()
run_pipeline(config, block=False, run_in_subprocess=True)
```

**Result**: Ray cluster running, listening on 7671

---

**Step 2: Connect Client**
```python
from nv_ingest_client.client import NvIngestClient
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient

client = NvIngestClient(
    message_client_allocator=SimpleClient,
    message_client_port=7671,
    message_client_hostname="localhost"
)
```

**Result**: Client connected to local Ray pipeline

---

**Step 3: Build Ingestion Workflow**
```python
ingestor = (
    Ingestor(client=client)
    .files(["Docs/file1.pdf", "Docs/file2.docx"])
    .extract(
        extract_text=True,        # ✓ Always
        extract_tables=True,      # ✓ For data extraction
        extract_charts=True,      # ✓ For visual data
        extract_images=True,      # ✓ For multimodal
        extract_infographics=True, # ✓ For complex visuals
        table_output_format="markdown"
    )
    .split(
        tokenizer="intfloat/e5-large-unsupervised",
        chunk_size=512,
        chunk_overlap=50
    )
    .caption(
        endpoint_url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_name="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        api_key=NVIDIA_API_KEY
    )
    .embed(
        endpoint_url="https://integrate.api.nvidia.com/v1",
        model_name="nvidia/nv-embedqa-e5-v5",
        api_key=NVIDIA_API_KEY
    )
    .vdb_upload(
        collection_name="multimodal_docs",
        milvus_uri="milvus.db",
        sparse=False,
        dense_dim=2048
    )
)
```

---

**Step 4: Execute**
```python
t0 = time.time()
results, failures = ingestor.ingest(
    show_progress=True,
    return_failures=True
)
elapsed = time.time() - t0

print(f"✓ {len(results)} chunks ingested in {elapsed:.1f}s")
print(f"✗ {len(failures)} failures")
```

---

### What Gets Stored in Milvus

Each chunk becomes a record with:

```json
{
  "id": "auto-generated",
  "embedding": [float, float, ...],        // 2048 dimensions
  "text": "The quick brown fox...",
  "chunk_index": 5,
  "source": "Docs/file.pdf",
  "page": 3,
  "created_at": "2026-04-28T10:30:45Z",
  "confidence": 0.95,
  "source_type": "text|table|chart|image_caption|infographic",
  "metadata": {
    "doc_hash": "sha256:abc123",
    "extraction_method": "pdfium",
    "caption_model": "nemotron-nano-vl-8b"
  }
}
```

---

## Agentic Workflows

### High-Level State Machine (rag_agent1.py)

```
Query: "Why did economics adopt open access first?"
     │
     ├─→ [guardrail] Validation
     │   └─ Check for jailbreak attempts
     │   └─ Detect query intent
     │   └─ Normalize text
     │   Output: cleaned_query, flags
     │
     ├─→ [query_expander] Generate Variants
     │   Input: "Why did economics adopt open access first?"
     │   Output: [
     │      "Why did economics adopt open access?",
     │      "Open access adoption in economics",
     │      "Economics early mover open access",
     │      "Reasons for open access in economics"
     │    ]
     │
     ├─→ [retriever] Dual Search
     │   BM25 Search:     Keyword matching on exact terms
     │   Semantic Search: Vector similarity (embedding-based)
     │   Output: Top-40 raw chunks
     │
     ├─→ [reranker] Score & Filter
     │   Input: Query + 40 chunks
     │   Model: cross-encoder/ms-marco-MiniLM-L-12-v2
     │   Output: Top-15 ranked chunks with scores
     │
     └─→ [generator] Generate Answer
         Input: Query + top-6 chunks + conversation history
         Model: meta/llama-3.3-70b-instruct (NVIDIA API)
         Output: Answer + Confidence + Sources

Confidence Check:
  ├─ High (≥ -3.0): Return answer ✓
  ├─ Medium (-3.0 to -8.0): Return with warning ⚠️
  └─ Low (< -8.0): Retry with expanded query ↻
```

### Detailed Node Functions

#### Node: guardrail
**Purpose**: Validate input, detect jailbreak attempts, understand intent

```python
def node_guardrail(state: AgentState):
    query = state["current_query"]
    flags = []
    
    # Check for injection patterns
    BLOCKED_PATTERNS = [
        r"ignore\s+previous\s+instructions",
        r"you\s+are\s+now",
        r"<\s*script",
        r"jailbreak"
    ]
    
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            flags.append("potential_injection")
            break
    
    # Detect intent
    intent = detect_intent(query)  # text|table|image|chart|mixed
    
    return {
        "current_query": query.strip(),
        "detected_intent": intent,
        "guardrail_flags": flags
    }
```

---

#### Node: query_expander
**Purpose**: Generate search variants for better retrieval coverage

```python
def node_query_expander(state: AgentState):
    query = state["current_query"]
    
    # Use LLM to expand query
    variants_prompt = f"""
    Generate 3-4 alternative search queries for: {query}
    Focus on:
    - Different phrasings
    - Related keywords
    - Synonyms
    
    Return as JSON: {{"variants": ["...", "...", ...]}}
    """
    
    variants = llm_generate(variants_prompt)
    all_queries = [query] + variants.get("variants", [])
    
    return {
        "query_variants": all_queries,
        "node_latencies": {...}
    }
```

---

#### Node: retriever
**Purpose**: Fetch relevant chunks using BM25 + semantic search

```python
def node_retriever(state: AgentState):
    queries = state["query_variants"]
    
    # BM25 retrieval (keyword-based)
    bm25_results = []
    for q in queries:
        hits = bm25_index.search(q, top_k=20)
        bm25_results.extend(hits)
    
    # Semantic retrieval (embedding-based)
    query_embedding = embed_texts([state["current_query"]])[0]
    semantic_results = milvus_client.search(
        collection_name="docs",
        data=[query_embedding],
        limit=20
    )
    
    # Deduplicate and combine
    all_chunks = combine_results(bm25_results, semantic_results)
    
    return {
        "raw_chunks": all_chunks[:40],  # Top-40
        "node_latencies": {...}
    }
```

---

#### Node: reranker
**Purpose**: Score and filter chunks to get most relevant

```python
def node_reranker(state: AgentState):
    query = state["current_query"]
    chunks = state["raw_chunks"]
    
    # Use cross-encoder to score
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    scores = ce.predict([(query, chunk["text"]) for chunk in chunks])
    
    # Rank and filter
    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )[:15]  # Keep top-15
    
    return {
        "ranked_chunks": [chunk for chunk, score in ranked],
        "node_latencies": {...}
    }
```

---

#### Node: generator
**Purpose**: Generate answer using context

```python
def node_generator(state: AgentState):
    query = state["current_query"]
    context = "\n\n".join([
        f"[{i}] {chunk['text']}"
        for i, chunk in enumerate(state["ranked_chunks"][:6])
    ])
    
    system_prompt = """You are a helpful assistant. 
    Answer based on the provided context.
    If you cannot answer, say so."""
    
    user_prompt = f"""Context:
    {context}
    
    Question: {query}
    
    Answer:"""
    
    answer = llm_generate(
        user_prompt,
        system_prompt=system_prompt,
        model=PRIMARY_LLM
    )
    
    # Assess confidence
    confidence = assess_confidence(answer, context)
    
    return {
        "answer": answer,
        "overall_confidence": confidence,
        "sources": state["ranked_chunks"][:3],
        "model_used": PRIMARY_LLM,
        "node_latencies": {...}
    }
```

---

## Multimodal RAG (VLM Integration)

### Why VLM? (Vision Language Model)

**Problem**: Text-based retrieval misses visual information

```
Document: report.pdf
  ├─ Page 1: Text about Q1 results
  ├─ Page 2: Chart showing revenue trends ← TEXT SEARCH MISSES THIS
  └─ Page 3: Table with metrics

Query: "What was the revenue trend?"
Result:  Wrong page found (text-only retrieval)
```

**Solution**: VLM extracts meaning from images

```
With VLM:
  ├─ Extract text as before
  ├─ Convert chart → semantic description ← NEW
  │  "Chart shows revenue growth of 45% YoY"
  ├─ Convert image → caption ← NEW
  │  "Pie chart with 4 segments"
  └─ Combine for retrieval
```

---

### Implementation: vlm_nvingest.py

**What It Does**: Uses Nemotron Nano VL-8B (8B parameters, multimodal)

```python
import base64
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

def caption_image(image_path: str) -> str:
    """Convert image to semantic description"""
    
    # Read image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    # Send to VLM
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                }
            ]
        }],
        temperature=0.3,
        max_tokens=1024
    )
    
    return response.choices[0].message.content
```

**Usage in Pipeline**:
```python
ingestor = (
    Ingestor(client=client)
    .files("Docs/report.pdf")
    .extract(extract_images=True)
    .split(chunk_size=512)
    .caption(
        endpoint_url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_name="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        api_key=NVIDIA_API_KEY
    )
    .embed()
    .vdb_upload(collection_name="multimodal_docs", ...)
)
```

---

### Pure VLM: vlm.py

**What It Does**: VLM-only processing (no embeddings, no LLM)

```python
def query_image_vlm(image_base64: str, question: str) -> str:
    """Ask a question about an image directly"""
    
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
            ]
        }]
    )
    
    return response.choices[0].message.content
```

**Use Case**: Direct image Q&A without storing in vector DB

---

## Evaluation Frameworks

### Three Evaluation Approaches

#### Approach 1: Custom Metrics (rag_eval/rag.py)

**Metrics Computed**:

```
Faithfulness (0-1):
  ├─ Does answer match retrieved context?
  ├─ Computed using: Semantic similarity between answer & context
  └─ Formula: avg_similarity(answer_sentences, context_sentences)

Context Precision (0-1):
  ├─ What % of retrieved chunks are relevant?
  ├─ Computed using: LLM judges relevance
  └─ Formula: relevant_chunks / total_retrieved_chunks

Context Recall (0-1):
  ├─ Did we retrieve enough relevant info?
  ├─ Computed by checking if important facts from golden answer appear
  └─ Formula: facts_in_retrieved / facts_in_golden_answer
```

**Code**:
```python
from rag_eval.rag import TextOnlyManualRAG

rag = TextOnlyManualRAG(
    milvus_path="milvus.db",
    collection_name="multimodal_docs",
    top_k=3
)

result = rag.retrieve("Why did economics adopt open access?")

# result contains:
# - answer: Generated response
# - faithfulness: 0.87
# - context_precision: 0.92
# - context_recall: 0.78
# - latency_ms: 1245
```

---

#### Approach 2: RAGAS Framework (rag_eavl1/evals.py)

**What It Does**: Industry-standard RAG evaluation

```
RAGAS Metrics:
  ├─ Faithfulness: Does answer reflect context?
  ├─ Answer Relevance: Is answer relevant to query?
  ├─ Context Precision: % of retrieved docs that support answer
  ├─ Context Recall: % of relevant docs that were retrieved
  ├─ Answer Similarity: Is answer similar to reference?
  └─ Answer Correctness: Does answer match golden truth?
```

**Setup**:
```python
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall
)
from ragas import evaluate

# Your dataset
dataset = {
    "question": ["Why did economics...?"],
    "ground_truth": ["Because economists..."],
    "answer": ["The field of economics..."],
    "contexts": [["Economics adopted open access..."]]
}

# Evaluate
scores = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevance,
        context_precision,
        context_recall
    ]
)

print(scores)
# Output: 
# {
#   "faithfulness": 0.85,
#   "answer_relevance": 0.92,
#   "context_precision": 0.88,
#   "context_recall": 0.80
# }
```

---

#### Approach 3: NeMo Evaluator (rag_nemo_sdk/)

**What It Does**: NVIDIA's evaluation framework with custom metrics

```
Metrics:
  ├─ BYOB (Bring Your Own Benchmark)
  ├─ Precision: True positives / (true positives + false positives)
  ├─ Recall: True positives / (true positives + false negatives)
  └─ F1-Score: Harmonic mean of precision & recall
```

**Setup**:
```python
from nemo_evaluator import NEMoEvaluator

evaluator = NEMoEvaluator(
    model="meta/llama-3.3-70b-instruct",
    metric="byob"
)

# Wrap RAG as service
from rag_service import RAGService

service = RAGService()

# Evaluate
results = evaluator.evaluate(
    service,
    test_dataset="evals/test_questions.jsonl",
    metrics=["precision", "recall", "f1"]
)
```

---

### Metrics Comparison

```
┌──────────────────────┬──────────┬───────┬──────────────┐
│ Metric               │ Formula  │ Range │ Interpretation │
├──────────────────────┼──────────┼───────┼──────────────┤
│ Faithfulness         │ Semantic │ 0-1   │ 0.8+ = Good    │
│ Context Precision    │ Binary   │ 0-1   │ 0.7+ = Good    │
│ Context Recall       │ Binary   │ 0-1   │ 0.6+ = OK      │
│ Answer Relevance     │ LLM      │ 0-1   │ 0.8+ = Good    │
│ Latency              │ Timer    │ ms    │ <2000 = Good   │
│ Confidence           │ Custom   │ 0-1   │ 0.7+ = Trust   │
└──────────────────────┴──────────┴───────┴──────────────┘
```

---

## Deployment Models

### Current: Library Mode (Local)

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install nv-ingest pymilvus ray openai

# Run ingestion
python pipeline.py

# Run agentic RAG
python rag_agent1.py
```

**Pros**: Simple, fast development, no Docker needed
**Cons**: Single machine only, resource limited

---

### Available: Microservices Mode (Docker)

```bash
# Prepare
cp .env.example .env
export NVIDIA_API_KEY="nvapi-..."

# Deploy
docker-compose up -d

# Verify services
curl http://localhost:8000/health    # NV-Ingest
curl http://localhost:19530/health   # Milvus
redis-cli ping                        # Redis (should return PONG)

# Connect client
python rag_agent1.py --broker localhost:7671
```

**Pros**: Scalable, production-ready, managed resources
**Cons**: More complex, requires Docker, network latency

---

### Target: A100 GPU Deployment (NV_Ingest.py)

```
┌────────────────────────────────────────────────────────┐
│ A100 GPU Node                                          │
│                                                        │
│ ┌─────────────────────────────────────────────────┐  │
│ │ NV_Ingest.py                                   │  │
│ │ ├─ Self-hosted NIM Services                   │  │
│ │ │  ├─ LLM (Llama 3.3 70B)                    │  │
│ │ │  ├─ Embedding (nv-embedqa-e5-v5)          │  │
│ │ │  ├─ Reranker (nemotron-rerank-1b)         │  │
│ │ │  └─ VLM (nemotron-nano-vl-8b)             │  │
│ │ ├─ Localhost endpoints (no NVIDIA API calls) │  │
│ │ ├─ Full document tracking                    │  │
│ │ ├─ Per-model latency logging                 │  │
│ │ └─ Timestamp: start → during → end          │  │
│ │                                              │  │
│ │ Results: Outputs/                            │  │
│ │ ├─ metadata.json (timestamps + latencies)   │  │
│ │ ├─ chunks.jsonl (extracted chunks)          │  │
│ │ └─ logs.txt (detailed execution log)        │  │
│ └─────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

**Features**:
- ✅ All commands ready
- ✅ Comprehensive instrumentation
- ✅ Performance tracking
- ⏳ Pending A100 execution

**Key Differences from Library Mode**:
- Uses self-hosted NIM instead of NVIDIA cloud APIs
- Models run locally on A100 GPU
- No API rate limits
- Unlimited throughput
- Full model control

---

## Performance & Monitoring

### Latency Tracking (Per-Node)

From `rag_agent1.py` output:

```
+-- ANSWER --------------------------------------------------+
| HIGH CONFIDENCE  |  llama-3.3-70b  |  1,245ms
| Why did economics and physics...
+-- LATENCIES ---------------------------------------------+
    guardrail   ........          125ms
    expander    ..........         250ms
    retriever   .......................  850ms
    reranker    .......             175ms
    generator   ..............................  1800ms
```

**Breakdown**:
```
Total: 1,245ms
├─ Query Preparation (guardrail): 125ms (10%)
├─ Query Expansion (expander): 250ms (20%)
├─ Retrieval (retriever): 850ms (68%) ⚠️ Slowest
│  ├─ BM25 index search: 100ms
│  ├─ Embedding query: 350ms
│  ├─ Milvus similarity: 150ms
│  └─ Result deduplication: 250ms
├─ Reranking (reranker): 175ms (14%)
└─ Generation (generator): 1,800ms (144%) ⚠️ Longest
```

**Optimization Opportunities**:
- **Retriever**: Parallelize BM25 + semantic search
- **Generator**: Use smaller model or cached context

---

### Confidence Scoring

```python
def assess_confidence(answer: str, context: str) -> str:
    """
    Determines if answer is high/medium/low confidence
    """
    
    factors = {
        "answer_length": len(answer) > 50,          # Has substance
        "context_match": similarity(answer, context) > 0.7,  # Grounded
        "has_sources": sources is not None and len(sources) > 0,  # Cited
        "no_hallucinations": not contains_hallucinations(answer),
        "keyword_coverage": query_terms_in_answer(query, answer)
    }
    
    confidence_score = sum(factors.values()) / len(factors)
    
    if confidence_score >= 0.8:
        return "high"
    elif confidence_score >= 0.5:
        return "medium"
    else:
        return "low"
```

---

### Monitoring Checklist

```
□ Document Ingestion
  □ Total chunks extracted
  □ Failures / retry count
  □ Time per document
  □ Average chunk size
  
□ Retrieval Quality
  □ Average retrieval rank
  □ Reranker score distribution
  □ Context precision %
  □ Relevance feedback

□ Generation Quality
  □ Answer length
  □ Confidence distribution
  □ Hallucination rate
  □ Answer similarity to reference

□ System Performance
  □ Per-node latencies
  □ Total response time
  □ GPU utilization
  □ Memory usage
  □ API rate limits
```

---

## File Structure

```
jaswanth/
├─ README_ENTERPRISE_RAG.md        ← This file
├─ ARCHITECTURE_INTERPRETATION.txt  ← Design decisions
│
├─ INGESTION & LIBRARY MODE
├─ pipeline.py                      ← Full NV Ingest pipeline
├─ pipeline1.py                     ← Variant 1
├─ pipeline2.py                     ← Variant 2
│
├─ AGENTIC WORKFLOWS
├─ rag_agent.py                     ← 9-node expert system
├─ rag_agent1.py                    ← 5-node retry agent
├─ rag_agent.log                    ← Execution logs
│
├─ MULTIMODAL
├─ vlm.py                           ← Pure VLM (no embeddings)
├─ vlm_nvingest.py                  ← VLM + NV Ingest
│
├─ EVALUATION
├─ rag_eval/
│  ├─ rag.py                        ← Custom evaluation metrics
│  ├─ pipeline.py                   ← With OCR + captions
│  └─ evals/                        ← Evaluation results
│
├─ rag_eavl1/
│  ├─ evals.py                      ← RAGAS framework
│  ├─ evals1.py                     ← Variant
│  ├─ pipeline1.py                  ← Clean pipeline
│  └─ milvus.db                     ← Evaluation vectors
│
├─ rag_nemo_sdk/
│  ├─ rag_service/                  ← API endpoint wrapper
│  ├─ nemo_eval/                    ← NeMo evaluator
│  └─ Milvus integration
│
├─ PRODUCTION
├─ NV_Ingest.py                     ← A100-ready system
├─ Outputs/                         ← Results directory
│  ├─ metadata.json                 ← Timestamps + latencies
│  ├─ chunks.jsonl                  ← Extracted content
│  └─ logs.txt                      ← Detailed logs
│
├─ INFRASTRUCTURE
├─ Dockerfile                       ← Container image
├─ docker-compose.yml               ← Microservices setup
├─ milvus-isolated.yml              ← Isolated Milvus config
├─ milvus-clean.yml                 ← Clean Milvus setup
│
├─ CONFIGURATION
├─ .env                             ← Environment variables
├─ .env.example                     ← Template
│
├─ DOCUMENTATION
├─ Docs/                            ← Sample documents
├─ results/                         ← Evaluation results
├─ rag_artifacts/                   ← Intermediate artifacts
└─ retriever/                       ← Retrieval utilities
```

---

## Future Roadmap

### Short Term (This Quarter)

- [ ] **A100 Deployment**: Execute NV_Ingest.py on A100 GPU
- [ ] **Performance Benchmarking**: Compare library vs microservices
- [ ] **Evaluation Pipeline**: Automate RAGAS scoring
- [ ] **Documentation**: API specs for each node

### Medium Term (Next Quarter)

- [ ] **Multi-GPU Support**: Distributed Ray cluster
- [ ] **Real-time Streaming**: Kafka/Kinesis integration
- [ ] **Fine-tuning**: Adapt models for domain
- [ ] **Caching Layer**: Redis integration
- [ ] **API Gateway**: REST/gRPC interface

### Long Term (This Year)

- [ ] **Production SLA**: 99.9% uptime target
- [ ] **Cost Optimization**: Model quantization
- [ ] **Security**: Authentication & encryption
- [ ] **Monitoring Dashboard**: Grafana integration
- [ ] **Auto-scaling**: Dynamic resource allocation

---

## Key Learnings & Best Practices

### What Worked Well ✅

1. **Library Mode for Development**
   - Fast iteration cycle
   - No Docker complexity
   - Easy debugging

2. **LangGraph for State Management**
   - Clear workflow definition
   - Type-safe state transitions
   - Excellent for agentic systems

3. **Milvus Lite for Prototyping**
   - Zero dependencies
   - Fast similarity search
   - File-based persistence

4. **NVIDIA's Hosted Models**
   - Consistent quality
   - No GPU required
   - Simple API

### Challenges Encountered ⚠️

1. **Embedding Dimension Trade-offs**
   - Higher dim (2048) = better quality but slower search
   - Lower dim (512) = faster but accuracy loss

2. **Chunking Strategy**
   - Fixed size doesn't work for all content
   - Table splitting produces fragmented context
   - Need adaptive chunking

3. **Confidence Estimation**
   - Hard to calibrate for different domains
   - Hallucinations sometimes score high
   - Needs domain-specific tuning

4. **Latency Hotspots**
   - Embedding generation (often 30% of time)
   - Reranking on large result sets (20% of time)
   - LLM generation (can be 50% of time)

### Recommendations for Production

```
1. START WITH
   ├─ Library mode (pipeline.py)
   ├─ rag_agent1.py (simpler workflow)
   ├─ RAGAS evaluation
   └─ Milvus Lite

2. THEN SCALE TO
   ├─ Microservices (docker-compose up)
   ├─ rag_agent.py (expert system)
   ├─ NeMo evaluator
   └─ Milvus server

3. FINALLY OPTIMIZE
   ├─ A100 deployment (NV_Ingest.py)
   ├─ Fine-tuned models
   ├─ Auto-scaling Ray cluster
   └─ Performance monitoring
```

---

## Commands Reference

### Running Ingestion

```bash
# Library mode (simple)
python pipeline.py

# With retry agent
python rag_agent1.py

# With expert system
python rag_agent.py

# Multimodal VLM
python vlm_nvingest.py
```

### Evaluation

```bash
# Custom metrics
cd rag_eval && python rag.py

# RAGAS framework
cd rag_eavl1 && python evals.py

# NeMo evaluation
cd rag_nemo_sdk && python scripts/evaluate.py
```

### Docker (Microservices)

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f nv-ingest-api

# Stop services
docker-compose down
```

### A100 Deployment

```bash
# Run on A100 (with all NIM services)
python NV_Ingest.py --gpu --benchmark

# Results stored in:
# Outputs/metadata.json       ← Timestamps + latencies
# Outputs/chunks.jsonl        ← Extracted chunks
# Outputs/logs.txt            ← Execution log
```

---

## Conclusion

This enterprise RAG application demonstrates a complete journey from basic retrieval-augmented generation to a sophisticated, production-ready multimodal system. By leveraging NVIDIA's technology stack (NV Ingest, Nemotron models, NIM services), we've created a flexible platform that can run locally for development or scale to production across multiple GPUs.

**Key Achievements**:
- ✅ Library mode for rapid development
- ✅ Agentic workflows for intelligent routing
- ✅ Multimodal support (text + images + charts)
- ✅ Multiple evaluation frameworks
- ✅ Production instrumentation and monitoring
- ✅ A100-ready deployment system

**Next Steps**:
- Execute NV_Ingest.py on A100 GPU
- Benchmark performance improvements
- Integrate into production environment
- Monitor and optimize based on metrics

---

## Contact & Support

For questions about architecture or implementation:
- Review `ARCHITECTURE_INTERPRETATION.txt` for design decisions
- Check individual `README.md` files in each subdirectory
- Examine logs in respective directories for debugging

---

**Document Version**: 1.0
**Last Updated**: April 28, 2026
**Status**: Ready for Production Deployment
make a ppt should look like clean and without designing way no AI generated stuff at all
make it very simple and provide the downloadable link
add more information arhcitetcure that i given right worfloes diagrams workfloes make it clean stunning ppt with miniaml colours dont add any ai images















CHAPTER 1
 INTRODUCTION
 

1.1 Overview

Speech-based interaction has become a key component in modern AI systems. However, most systems rely on multi-stage processing pipelines, which introduce delays and reduce natural interaction.

This project aims to develop a real-time speech-to-speech system that eliminates intermediate bottlenecks and enables seamless communication.

1.2 Problem Statement

Existing conversational voice systems predominantly rely on a sequential processing pipeline, where speech input is first converted into text using Speech‑to‑Text (STT), then processed by a natural language or reasoning model, and finally transformed back into audio output via Text‑to‑Speech (TTS). While this approach is functionally effective, it introduces significant end‑to‑end latency, as each stage must complete before the next can begin. This batch‑oriented design disrupts conversational flow and results in delayed, unnatural interactions. Furthermore, such systems lack true real‑time audio streaming capabilities, preventing them from processing speech continuously or responding mid‑utterance. As a result, they struggle to support advanced conversational features such as interruption handling, overlapping dialogue, and efficient dynamic tool execution during live interactions. These limitations reduce the usability of traditional pipelines in scenarios that demand low‑latency, natural, and responsive voice‑based communication.

1.3 Objectives

The primary objective of this project is to design and implement a real‑time speech‑to‑speech conversational system that overcomes the latency and interaction constraints of traditional sequential pipelines. The system aims to enable continuous, low‑latency audio streaming by leveraging PCM‑based audio transmission and integrating the Azure Realtime Speech‑to‑Speech (S2S) model for live conversational intelligence. Another key objective is to support dynamic tool invocation during active conversations, allowing the model to call external tools and incorporate their results seamlessly into ongoing responses. The project also focuses on reducing response latency to improve conversational naturalness and ensuring robust support for multi‑turn interactions, enabling the system to maintain context and continuity across conversational exchanges. Together, these objectives seek to deliver a more responsive, human‑like voice interaction experience suitable for real‑world applications.

1.4 Scope

The scope of this project encompasses the development of a foundational real‑time conversational voice platform applicable across multiple domains. It is designed to support voice assistant systems that require immediate and natural responses, as well as AI‑powered customer support solutions where low latency and conversational continuity are critical for user satisfaction. The system is also relevant to healthcare conversational applications, such as virtual patient assistants or triage systems, where responsive voice interaction can enhance accessibility and usability. Additionally, the architecture is suitable for digital avatars and interactive agents, enabling lifelike spoken interaction in virtual environments. While the project focuses on the core speech‑to‑speech interaction pipeline and supporting mechanisms, it provides a scalable and extensible foundation for future enhancements and domain‑specific customization.

 

CHAPTER 2

BACKGROUND & LITERATURE REVIEW

2.1 Traditional Speech Systems

Traditional speech‑based conversational systems typically follow a sequential processing pipeline consisting of Speech Input, Speech‑to‑Text (STT), Text Processing, Text‑to‑Speech (TTS), and Audio Output. In this architecture, spoken input is first fully captured and converted into text before any language understanding or reasoning takes place. The generated textual response is then converted back into spoken audio for playback. While this approach simplifies system design and modularization, it introduces high end‑to‑end latency due to its strictly sequential execution. Each stage must complete before the next begins, resulting in noticeable delays between user input and system response. Additionally, converting audio into text early in the pipeline often leads to a loss of speech‑specific nuances, such as tone, emphasis, pauses, and emotional cues, which can negatively impact conversational quality and naturalness.

2.2 Real-Time Speech Systems

Modern real‑time speech systems aim to overcome the limitations of traditional pipelines by enabling continuous audio processing and streaming interaction. Instead of waiting for complete utterances, these systems process audio directly as it arrives, allowing input and output to be handled simultaneously. This streaming‑first approach significantly reduces perceived latency by enabling the system to begin generating responses while the user is still speaking. Real‑time speech systems support more natural conversational dynamics, including faster turn‑taking and smoother dialogue flow. By minimizing buffering and eliminating rigid sequential boundaries, these systems better approximate human conversational behavior and are particularly suited for applications requiring immediacy and responsiveness.

 

Modern systems:

Process audio directly
Stream input/output simultaneously
Reduce latency
2.3 Related Technologies

The development of conversational speech systems relies on advancements across several key technological areas. Speech recognition has been significantly improved by deep learning–based models such as Whisper, which enable robust transcription across diverse accents and acoustic conditions. Language modeling is primarily driven by transformer‑based architectures, which excel at understanding context, generating coherent responses, and reasoning across multi‑turn conversations. For speech synthesis, neural Text‑to‑Speech models have replaced traditional rule‑based approaches, enabling more natural, expressive, and human‑like audio generation. Together, these technologies form the foundation upon which both traditional and modern conversational speech systems are built.

Speech Recognition

Whisper models
Language Models

Transformer-based models
Speech Synthesis

Neural TTS
 

2.4 Gap Analysis

Despite advances in speech recognition, language modeling, and speech synthesis, many existing conversational systems still fail to deliver truly real‑time interaction. Most systems remain turn‑based rather than fully duplex, meaning they cannot listen and respond simultaneously. This limitation prevents interruption handling and dynamic conversational flow. Furthermore, integration of external tools and services during live conversations is often limited or inefficient, as tool invocation is typically performed after a full input and reasoning cycle completes. These gaps highlight the need for architectures that support continuous streaming, full‑duplex interaction, and seamless tool execution in real time, motivating the design choices explored in this project.

 

Existing systems:

Not real-time
Not full-duplex
Limited tool integration
CHAPTER 3

 SYSTEM ARCHITECTURE

3.1 Overall Architecture

The system architecture is designed to support real‑time, bidirectional speech‑to‑speech interaction by integrating a browser‑based frontend, a streaming backend, an AI realtime processing layer, and an external tool execution layer. The architecture follows a modular design that separates audio capture, communication, reasoning, and tool execution, enabling scalability and maintainability.

The frontend serves as the user interaction layer and communicates with the backend through a persistent WebSocket connection. The backend acts as the central coordination layer, managing audio streaming, session handling, and message routing. It forwards streaming audio input to the AI model through a realtime API and simultaneously handles streaming audio output returned by the model. The AI model processes the incoming audio stream, generates responses in real time, and can dynamically invoke external tools when additional information or computation is required. These tool invocations are handled by a dedicated tool layer, whose results are fed back into the AI model to produce final spoken responses.

This architecture enables low‑latency, continuous interaction by avoiding batch processing and allowing input, processing, and output to occur concurrently.

 

System Architecture Diagram
+-------------+      +-------------+      +----------------+
|  Frontend   | ---> |   Backend   | ---> |  AI Model      |
| (Audio I/O) | <--- | (WebSocket) | <--- | (Realtime API) |
+-------------+      +-------------+      +----------------+
                           |
                           v
                     +-----------+
                     | Tool Layer|
                     +-----------+

3.2 Components

1. Frontend

The frontend component is responsible for all user‑facing audio interactions. It captures microphone input from the user using browser‑based audio APIs and processes the raw audio into a format suitable for real‑time streaming. Specifically, the recorded audio is converted from floating‑point samples into PCM‑16 format, which is widely supported by realtime speech models and optimized for low‑latency transmission. The frontend sends audio chunks continuously to the backend over a WebSocket connection. Additionally, it receives streaming audio responses from the backend, decodes them, and plays them immediately to the user. This enables smooth, uninterrupted playback and contributes to a natural conversational experience.

Captures audio
Converts Float32 → PCM16
Plays output audio
 

2. Backend

The backend acts as the orchestration and communication layer of the system. It maintains persistent WebSocket connections with frontend clients and manages real‑time audio streaming sessions. Incoming audio chunks from the frontend are buffered briefly to maintain packet order and timing consistency before being forwarded to the AI model through the realtime API. The backend also receives streaming audio and text responses from the model and routes them back to the frontend with minimal delay. In addition to audio handling, the backend manages session state, coordinates tool execution requests, and ensures reliable communication between system components. This intermediary role allows the backend to isolate frontend and model logic while enforcing control over data flow and system stability.

Handles WebSocket communication
Buffers audio chunks
Sends data to model
3. AI Model

The AI model is responsible for real‑time speech understanding, reasoning, and response generation. It processes streaming audio input directly and maintains conversational context throughout the session. Unlike traditional systems that require complete utterances, the model operates incrementally, enabling it to begin response generation as soon as sufficient information is available. During a conversation, the model can dynamically request the execution of external tools, such as data retrieval or calculation services, when additional information is needed. Once tool results are returned, the model incorporates them into the ongoing response and continues generating audio output. This capability enables intelligent, context‑aware interactions within a single continuous conversation flow.

Processes streaming audio
Generates responses
Calls tools dynamically
 

 

4. Tool Layer

The tool layer provides access to external services and functions that extend the model’s capabilities beyond natural language reasoning. Examples include weather information retrieval, mathematical calculations, and other application‑specific services. When the AI model determines that a tool is required, the backend invokes the appropriate tool through this layer and returns the result to the model in a structured format. By separating tool execution from the core model logic, the system maintains a clean architecture while enabling flexible integration of new tools. This design allows the conversational agent to provide accurate, up‑to‑date, and actionable responses during live interactions without disrupting the real‑time conversational flow.

Weather API
Calculator
External services
 

CHAPTER 4

METHODOLOGY

4.1 Audio Processing

Audio processing is a critical component of the proposed real‑time speech‑to‑speech system, as it directly affects latency and interaction quality. Audio input captured from the frontend microphone is initially represented in Float32 format, which is commonly used by browser audio APIs due to its high precision. However, this format is not optimal for real‑time transmission or model compatibility. Therefore, the audio data is converted into PCM‑16 format, a lightweight and widely supported representation that balances quality and efficiency. The system supports sampling rates of 16 kHz and 24 kHz, enabling flexibility depending on model requirements and performance considerations. To ensure safe and consistent transmission over WebSocket connections, the processed audio chunks are encoded using Base64 encoding, allowing binary data to be embedded within standard text‑based messages.

Input: Float32
Converted to PCM16
Sample Rate: 16kHz / 24kHz
Encoded using Base64
4.2 Streaming Pipeline

The streaming pipeline is designed to enable continuous, low‑latency audio transmission from the user to the AI model. The pipeline begins with real‑time audio capture at the frontend, where the microphone input is continuously recorded. The captured audio is divided into small, time‑bounded chunks to ensure smooth streaming and minimal buffering delay. Each chunk is then Base64‑encoded and transmitted to the backend through a persistent WebSocket connection. The backend forwards these audio chunks to the AI model in near real time. This streaming‑oriented pipeline avoids batch processing and ensures that audio input, processing, and response generation can occur concurrently, forming the foundation for natural conversational interaction.

Audio Capture → Chunking → Base64 Encoding → WebSocket → Model

4.3 Event Flow

Communication between the backend and the AI model follows a structured event‑driven protocol. As audio chunks arrive, they are sent to the model using input_audio_buffer.append events, allowing the model to process incoming speech incrementally. Once the user finishes speaking, an input_audio_buffer.commit event is issued to signal the completion of the current utterance. This commit triggers the model to finalize its interpretation of the input and begin response generation, which is initiated through a response.create event. The model then streams its output back to the backend as a sequence of incremental updates, including audio.delta events for synthesized speech output and text.delta events for partial or complete textual responses. This event flow enables real‑time feedback while maintaining clear conversational boundaries.

input_audio_buffer.append
input_audio_buffer.commit
response.create
Receive:
audio.delta
text.delta
4.4 Tool Calling Workflow

The system supports dynamic tool invocation as part of the conversational flow, enabling the AI model to access external information or perform computations during live interactions. When a user provides input, the model analyzes the request and determines whether a tool is required to generate an accurate response. If a tool is needed, the model emits a structured tool‑call request identifying the appropriate function and associated parameters. The backend then executes the requested tool through the tool layer and returns the result to the model. The model incorporates this result into its reasoning process and continues generating the final response, which is streamed back to the user in audio form. This workflow allows tools to be integrated seamlessly without interrupting the conversational experience.

User Input → Model → Tool Detection → Tool Execution → Response Generation

4.5 Latency Optimization Techniques

Several targeted optimization techniques were employed to reduce latency and improve real‑time responsiveness. The use of compressed codecs such as WebM was avoided due to their encoding and decoding overhead. Instead, raw PCM‑16 audio was adopted to minimize processing delay. Audio data is streamed in small chunks rather than waiting for complete utterances, enabling early processing and faster response initiation. Additionally, the system maintains persistent sessions with the AI model, eliminating the overhead associated with repeated connection setup and teardown. Together, these optimizations significantly reduce end‑to‑end latency and contribute to a smoother, more natural conversational experience.

Removed WebM codec
Used PCM16 format
Streaming chunks
Persistent session
 

 

 

CHAPTER 5

IMPLEMENTATION

5.1 Backend Implementation

The backend of the system is implemented using a Flask‑based server, which serves as the central coordination layer for real‑time communication, session management, and model interaction. Flask was chosen for its lightweight nature and flexibility in handling asynchronous workflows. The backend establishes and maintains persistent WebSocket connections with frontend clients to support continuous bi‑directional communication required for real‑time audio streaming. Through these WebSocket connections, the backend receives streaming audio input, forwards it to the AI model, and simultaneously streams model responses back to the frontend. Integration with the Azure Realtime SDK enables the backend to manage realtime model sessions, transmit audio events, receive streaming responses, and coordinate tool invocation. This backend implementation acts as the backbone of the system, ensuring low‑latency data flow, session stability, and reliable orchestration across all components.

Flask server
WebSocket handling
Azure SDK integration
 

5.2 Frontend Implementation

The frontend is implemented using HTML and JavaScript, leveraging standard browser technologies to ensure broad compatibility and ease of deployment. The Web Audio API is used to capture microphone input from the user in real time and process the raw audio data for streaming. Audio samples are prepared for transmission by converting them into a model‑compatible format before being streamed to the backend through a WebSocket connection. The frontend is also responsible for real‑time playback of the synthesized audio responses received from the backend. Incoming audio is decoded and played immediately to minimize perceived latency and maintain conversational fluidity. This implementation enables users to interact with the system using natural voice input while receiving uninterrupted spoken responses.

HTML + JavaScript
Web Audio API
Real-time playback
 

5.3 Tool Integration

The system includes a dedicated mechanism for integrating external tools that enhance the conversational capabilities of the AI model. Example tools implemented in this project include a weather information service and a calculator, which allow the system to retrieve real‑world data and perform computations during live conversations. When the AI model determines that external information or processing is required, it issues a tool request that is handled by the backend. The backend executes the requested tool and returns the result to the model in a structured format. This approach enables dynamic tool usage without interrupting the real‑time conversational flow and allows additional tools and services to be integrated in the future with minimal changes to the overall architecture.

Example tools:

Weather API
Calculator
5.4 Key Features

The implemented system provides several key features that distinguish it from traditional conversational voice systems. Most notably, it supports real‑time audio streaming, enabling continuous speech input and output with
