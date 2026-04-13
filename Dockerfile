Absolutely — I went through NVIDIA’s docs and pulled the strongest proof points you can use.
Your core finding is largely correct, but it needs to be framed precisely:

The Python “library mode” script is mainly an orchestration/client path for small-scale workloads. The full NV Ingest / NeMo Retriever extraction architecture (Page Elements, Graphic Elements, Table Structure, OCR, VLM, etc.) is implemented through NVIDIA NIM microservices, and the self-hosted/production form of that architecture is documented under Docker Compose / Helm with GPU requirements. [docs.nvidia.com], [docs.nvidia.com], [docs.nvidia.com]


1) The single strongest proof for your argument
Library mode is for small-scale use and depends on existing NIM endpoints
NVIDIA’s own Quickstart (Library Mode) says library mode is for “small-scale workloads, such as workloads of fewer than 100 documents” and that it “depends on NIMs that are already self-hosted, or, by default, NIMs that are hosted on build.nvidia.com.” That is the clearest evidence that library mode does not itself provision the full model stack locally; it assumes those NIMs already exist somewhere. [docs.nvidia.com]
Why this matters for your explanation
This directly supports saying:

your Python script is not the same thing as deploying the full architecture locally; it is the pipeline/client layer that submits jobs and uses available NIM endpoints, whether cloud-hosted or self-hosted. [docs.nvidia.com], [docs.nvidia.com]
if the specialized visual-extraction NIMs are not self-hosted on GPUs and you are only using the lightweight library flow, then you should not describe it as a full local deployment of the complete architecture shown in the diagram. [docs.nvidia.com], [docs.nvidia.com], [catalog.ng...nvidia.com]


2) NVIDIA does document that the architecture is built from specialized models/NIMs
NVIDIA’s docs explicitly describe NeMo Retriever / NV Ingest as using specialized NVIDIA NIM microservices to extract text, tables, charts, infographics, and OCR-enhanced content. [docs.nvidia.com], [docs.nvidia.com]
The Support Matrix explicitly names the extraction models/features in the pipeline:

nemotron-page-elements-v3 — detects/classifies page regions like table/chart/infographic. [docs.nvidia.com]
nemotron-table-structure-v1 — preserves table structure by detecting rows/columns/cells. [docs.nvidia.com]
nemotron-graphic-elements-v1 — extracts chart/graphic elements such as legends, axes, titles, values. [docs.nvidia.com]
nemotron-ocr-v1 — OCR text extraction from images. [docs.nvidia.com]
llama-nemotron-embed-1b-v2 — embeddings. [docs.nvidia.com]
VLM — experimental image captioning of unstructured images. [docs.nvidia.com]

So if your lead asks “where exactly are page elements / OCR / table structure / graphic elements documented?”, the cleanest answer is:

They are documented in NVIDIA’s Support Matrix for NeMo Retriever Extraction and in the product overview pages as specialized NIM-backed extraction components. [docs.nvidia.com], [docs.nvidia.com], [docs.nvidia.com]


3) NVIDIA also documents that these features require GPU-backed deployment in self-hosted mode
Core extraction pipeline requires GPU
NVIDIA’s Support Matrix says the core pipeline features run on a single A10G or better GPU, and those core features include page elements, table structure, graphic elements, OCR, embedding, and retrieval. [docs.nvidia.com]
That is important because it means the architecture in your diagram is not just a plain CPU-side Python script capability. The specialized extraction stack is tied to GPU-backed deployment requirements. [docs.nvidia.com], [docs.nvidia.com]
Advanced features require additional dedicated GPU support
The same Support Matrix says advanced features require additional GPU support and disk space, including:

VLM for image captioning. [docs.nvidia.com]
nemotron-parse for advanced visual parsing. [docs.nvidia.com]
audio extraction. [docs.nvidia.com]

For nemotron-parse, NVIDIA is even more explicit in the Advanced Visual Parsing doc: because of VRAM limitations in the current release, it “must run on a dedicated additional GPU.” [docs.nvidia.com]
So if you want a proof sentence for management, this is a strong one:

NVIDIA documents that the visual extraction stack is GPU-based, and some advanced components such as VLM and nemotron-parse require additional dedicated GPU resources in self-hosted deployment. [docs.nvidia.com], [docs.nvidia.com]


4) Docker Compose / Helm is where NVIDIA documents the self-hosted model deployment
NVIDIA’s Deploy With Docker Compose (Self-Hosted) doc says to use docker compose to start the needed services, and it explicitly warns that NIM containers can take 10–15 minutes on first startup to pull and load models. It also says the default configuration might not fit on a single GPU for some hardware targets. [docs.nvidia.com]
NVIDIA’s Helm chart documentation says the NVIDIA nim-operator must be installed and configured so that the NVIDIA NIMs are properly deployed. [catalog.ng...nvidia.com], [github.com]
Those two docs together are your proof that:

the full production/self-hosted architecture is expected to run as a set of NIM microservices in Docker Compose or Helm, not merely inside the short Python library-mode script. [docs.nvidia.com], [catalog.ng...nvidia.com], [github.com]
this setup is where GPU resource management and service deployment are actually handled. [docs.nvidia.com], [catalog.ng...nvidia.com]


5) Your finding about 25.9.0 is also supported by NVIDIA release notes
The 25.9.0 release notes say that release added support for:

nemoretriever-ocr-v1, and it points readers to Deploy With Docker Compose (Self-Hosted) and Helm for details. [github.com]
Llama Nemotron VLM 8B for image captioning. [github.com]
running more than one VLM at a time by using Helm. [github.com]

This helps you argue that the richer multimodal architecture was being added and documented in the self-hosted deployment path, not just in the bare library-mode example. [github.com], [docs.nvidia.com], [catalog.ng...nvidia.com]

6) Why your JPEG/ID-card result may not match expectations
This part is also important for your explanation.
Default extraction paths are not all meant for the same input type
NVIDIA’s Python API docs say:

pdfium is the default PDF extraction method and does not capture text from scanned pages/images. [docs.nvidia.com]
ocr processes pages through the full OCR pipeline for scanned/corrupt text cases. [docs.nvidia.com]
pdfium_hybrid mixes native text extraction and OCR for scanned pages. [docs.nvidia.com]

NVIDIA’s Advanced Visual Parsing doc also says nemotron-parse only supports PDFs, not image files. [docs.nvidia.com]
Why this matters to your demo
If your input was a driving-license-like JPEG rather than a PDF, then some of the higher-accuracy PDF-specific extraction paths are not applicable in the same way. That helps explain why separately running a VLM on the image might work better for that particular case than relying on a generic extraction flow. [docs.nvidia.com], [docs.nvidia.com], [docs.nvidia.com]
So this is a defensible statement:

The architecture supports image-aware extraction, but not every extraction mode is identical across PDFs vs standalone image inputs, and nemotron-parse specifically is documented as PDF-only. [docs.nvidia.com], [docs.nvidia.com]


7) What is explicitly documented vs what is your reasonable inference
Explicitly documented by NVIDIA
These are safe to present as direct facts:

Library mode is for small-scale workloads (<100 docs) and depends on already self-hosted NIMs or build.nvidia.com-hosted NIMs by default. [docs.nvidia.com]
The extraction stack uses specialized NIMs for page elements, table structure, graphic elements, OCR, embeddings, plus advanced VLM/image-captioning support. [docs.nvidia.com], [docs.nvidia.com]
The core pipeline requires GPU and the advanced features need additional GPU support. [docs.nvidia.com]
Self-hosted deployment is documented via Docker Compose and Helm, including model pulling/loading and NIM deployment. [docs.nvidia.com], [catalog.ng...nvidia.com]
nemotron-parse is PDF-only and requires a dedicated additional GPU when run locally. [docs.nvidia.com]

Reasonable inference (say this carefully)
These are not stated in exactly your wording, but they are strongly implied by the docs:

Your Python script is mainly the orchestration/client layer, not the full self-hosted deployment of every NIM in the architecture. That inference follows from library mode depending on pre-existing NIM endpoints and from self-hosted deployment being documented separately in Docker/Helm. [docs.nvidia.com], [docs.nvidia.com], [catalog.ng...nvidia.com]
If you want the architecture to behave like the full production diagram on your own infrastructure, you should deploy the relevant NIMs with GPUs, rather than treating the lightweight library example as equivalent to the full self-hosted architecture. [docs.nvidia.com], [docs.nvidia.com], [catalog.ng...nvidia.com]

What I did not find explicitly documented
I did not find an official NVIDIA sentence that says, word-for-word, “library mode is only the frontend part” or “each of Page Elements / OCR / Table Structure needs 8–12 GB individually.” Instead, NVIDIA documents aggregate GPU requirements for the core/advanced pipeline and documents self-hosted deployment of the NIM services. For OCR, the RAG Blueprint docs do provide a concrete default OCR GPU memory setting (OCR_CUDA_MEMORY_POOL_MB=8192) and describe the OCR service as GPU-accelerated, but NVIDIA does not appear to publish one clean per-model “8–12 GB each” table for all extraction NIMs in the extraction docs I found. [docs.nvidia.com], [docs.nvidia.com]

8) Paste-ready documentation note for your lead
You can send this almost as-is:
Plain TextFindings on NV Ingest / NeMo Retriever (library mode vs full architecture)1. The Python script we demoed is valid, but it represents the library-mode orchestration path, not a full self-hosted deployment of the complete multimodal extraction architecture.   - NVIDIA’s library-mode quickstart states that library mode is intended for small-scale workloads (<100 docs) and depends on NIMs that are already self-hosted or, by default, hosted on build.nvidia.com.2. The architecture shown in NVIDIA documentation (Page Elements, Graphic Elements, Table Structure, OCR, Embed, VLM, etc.) is real and is implemented through specialized NVIDIA NIM microservices.   - NVIDIA’s Support Matrix explicitly lists nemotron-page-elements, nemotron-table-structure, nemotron-graphic-elements, nemotron-ocr, and embedding as pipeline components.3. GPU-backed deployment is required for the self-hosted/production form of this architecture.   - NVIDIA documents that the core extraction pipeline runs on a GPU (A10G or better in the support matrix), and advanced features like VLM and nemotron-parse require additional GPU support.   - Docker Compose / Helm docs are the places where NVIDIA documents self-hosted NIM deployment and GPU-oriented startup.4. Therefore, if we want the behavior to match the full documented architecture on our own infra, we need the relevant NIMs/models deployed with GPU resources (Docker Compose or Helm), rather than assuming the lightweight library-mode script alone is equivalent to the complete architecture.5. Additional nuance:   - For PDFs, NVIDIA documents different extraction methods (pdfium, OCR, nemotron-parse, etc.).   - nemotron-parse is documented as PDF-only, not for standalone image files.   - For image-like inputs such as driving-license JPEGs, separate VLM/OCR handling may produce better results than relying on a generic default extraction path.Show less
Supported by NVIDIA docs: library mode quickstart, support matrix, self-hosted deployment guide, helm docs, advanced visual parsing docs, and 25.9.0 release notes. [docs.nvidia.com], [docs.nvidia.com], [docs.nvidia.com], [catalog.ng...nvidia.com], [docs.nvidia.com], [github.com]

9) Best one-line conclusion you can say in the meeting

The library-mode script is a lightweight client/orchestration path for small-scale use and can call hosted or already-deployed NIMs, but the full Page Elements / Table Structure / Graphic Elements / OCR / VLM architecture is documented as a GPU-backed NIM deployment in the self-hosted Docker/Helm path. [docs.nvidia.com], [docs.nvidia.com], [docs.nvidia.com], [catalog.ng...nvidia.com]
