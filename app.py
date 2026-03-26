 head -30 pipeline.py
"""
NV-Ingest 25.9.0 Library Mode RAG Pipeline
===========================================
This is the REAL NV-Ingest pipeline using:
  - run_pipeline()         → starts Ray subprocess
  - SimpleClient broker    → port 7671
  - Ingestor chain         → .load() .extract() .split() .caption() .embed() .vdb_upload()
  - Milvus Lite            → local file DB (milvus_rag.db)
  - NVIDIA cloud NIMs      → no GPU needed

Milvus Lite explanation:
  - milvus_uri="milvus_rag.db" creates a LOCAL FILE database
  - No Docker needed, no separate Milvus server
  - .vdb_upload() in the Ingestor chain stores embeddings INTO this file
  - pymilvus MilvusClient reads FROM this same file for retrieval
  - It's the same Milvus, just embedded mode (like SQLite vs PostgreSQL)

API Keys needed:
  - NVIDIA_API_KEY      → from https://build.nvidia.com (free)
  - NVIDIA_BUILD_API_KEY → SAME key as above (set both to same value)
  Both are needed because different NIM endpoints check different env vars.

Install (on Linux with uv):
  uv venv --python 3.12 nvingest
  source nvingest/bin/activate
  uv pip install nv-ingest==25.9.0 nv-ingest-api==25.9.0 nv-ingest-client==25.9.0 milvus-lite==2.4.12

Run:
  export NVIDIA_API_KEY=nvapi-xxxxx
  export NVIDIA_BUILD_API_KEY=nvapi-xxxxx   # same key
(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ cat .env
NVIDIA_API_KEY=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
NVIDIA_BUILD_API_KEY=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

(myenv) clouduser01@AZRCIDEVNIVIDIA:~/jaswanth$ 
