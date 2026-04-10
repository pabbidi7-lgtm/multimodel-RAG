The Core Problem — Why Your Nemotron Models Are NOT Running
Your code is correct. Your pipeline starts. Your extraction runs. But the Nemotron models (OCR, page-elements, graphic-elements, table-structure) are never called because of one reason:
Those models are not cloud API calls. They are locally running inference servers that your pipeline connects to via HTTP/gRPC on localhost ports.
When you write:
bashexport OCR_HTTP_ENDPOINT="https://ai.api.nvidia.com/v1/cv/nvidia/paddle-ocr"
This does NOT work for nv-ingest library mode. The pipeline internally expects something like:
bashexport OCR_HTTP_ENDPOINT="http://localhost:8009"
Because nv-ingest expects those NIMs to be running as servers on your machine, not as cloud API calls.

What the GitHub Repo Actually Shows
When you look at https://github.com/NVIDIA/NeMo-Retriever/blob/release/25.9.0/README.md, the architecture diagram shows:
Your pipeline.py
      ↓
nv-ingest pipeline (library mode)
      ↓
calls localhost:8009  → PaddleOCR NIM (running locally)
calls localhost:8000  → YOLOX page-elements NIM (running locally)  
calls localhost:8004  → YOLOX graphic-elements NIM (running locally)
calls localhost:8007  → YOLOX table-structure NIM (running locally)
Every one of those services must be downloaded and running on your machine before your pipeline.py can use them.

What Your Lead Means by "Custom Functions from the Repo"
Your lead is saying this:
Inside the repo there are files like:
src/nv_ingest/stages/extractors/
src/nv_ingest/stages/nim/
These files contain the actual code that calls OCR, page-elements, graphic-elements, table-structure NIMs. Your lead wants you to look at how those functions call the NIMs and either:

Run those NIMs locally so the existing code works, OR
Understand the calling pattern and wire it into your pipeline

She already ran this successfully. What she ran was the full stack — NIMs running locally + pipeline.py connecting to them.

The Two Ways to Run These NIMs Locally
Option A — Docker (The Standard Way)
Each NIM is a Docker container. You pull and run each one:
bash# OCR NIM
docker run --gpus all -p 8009:8000 \
  -e NGC_API_KEY=$NVIDIA_API_KEY \
  nvcr.io/nvidia/nv-ingest/paddle-ocr:25.9.0

# Page elements NIM  
docker run --gpus all -p 8000:8000 \
  -e NGC_API_KEY=$NVIDIA_API_KEY \
  nvcr.io/nvidia/nv-ingest/nv-yolox-page-elements-v1:25.9.0

# Graphic elements NIM
docker run --gpus all -p 8004:8000 \
  -e NGC_API_KEY=$NVIDIA_API_KEY \
  nvcr.io/nvidia/nv-ingest/nv-yolox-graphic-elements-v1:25.9.0

# Table structure NIM
docker run --gpus all -p 8007:8000 \
  -e NGC_API_KEY=$NVIDIA_API_KEY \
  nvcr.io/nvidia/nv-ingest/nv-yolox-table-structure-v1:25.9.0
Then your bash exports become:
bashexport OCR_HTTP_ENDPOINT="http://localhost:8009"
export YOLOX_HTTP_ENDPOINT="http://localhost:8000"
export YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT="http://localhost:8004"
export YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT="http://localhost:8007"
export OCR_INFER_PROTOCOL="http"
export YOLOX_INFER_PROTOCOL="http"
export YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL="http"
export YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL="http"
Option B — Docker Compose (What the Repo Uses)
The repo has a docker-compose.yaml file that starts ALL services together including all NIMs. This is the cleanest way. From the repo root:
bashdocker compose up
That starts everything — pipeline service + all NIM containers + all ports configured correctly.

Why build.nvidia.com API Calls Don't Work Here
build.nvidia.com gives you REST API access for prototyping single calls. For example you can send one image and get OCR back.
But nv-ingest library mode internally sends many small requests to these NIMs during pipeline processing — one per page, one per detected region, etc. It expects a low-latency local server, not a cloud API with authentication headers and rate limits on every call.
The pipeline was designed for local NIM servers. That is why your cloud endpoint exports were silently ignored.

What Actually Happened When Your Lead Ran It
Your lead ran either:

docker compose up from the repo — which started all NIM containers locally, then ran pipeline.py against localhost ports, OR
Had access to a server where those NIM containers were already running

She did NOT use build.nvidia.com API endpoints inside the pipeline.

What You Need to Do Now — Clear Steps
Step 1 — Check if Docker is available on your machine:
bashdocker --version
docker compose version
Step 2 — Check if you have GPU and enough VRAM:
Each NIM needs GPU. Rough requirements:

PaddleOCR: ~2GB VRAM
YOLOX page-elements: ~2GB VRAM
YOLOX graphic-elements: ~2GB VRAM
YOLOX table-structure: ~2GB VRAM

bashnvidia-smi
Step 3 — Clone the repo if you haven't:
bashgit clone -b release/25.9.0 https://github.com/NVIDIA/NeMo-Retriever.git
cd NeMo-Retriever
Step 4 — Check the docker-compose file:
bashcat docker-compose.yaml
Look for the NIM service definitions and what ports they use.
Step 5 — Start everything:
bashexport NVIDIA_API_KEY="nvapi-yourkey"
docker compose up
Step 6 — Then in a separate terminal run your pipeline.py with localhost endpoint exports.

Summary in Simple Words
Your pipeline.py code is fine. Your extraction settings are fine. The only missing piece is that the 4 Nemotron NIM models must be running as local Docker containers on specific ports before your pipeline starts. Without those containers running, the pipeline silently falls back to basic text-only extraction. That is why you get output but no real image/chart/table extraction.
Your lead ran the full docker compose stack. That is what you need to replicate.so that means when i will be in github repo and i will do docker compose up the nemotron modesl will start?or should i download the models and steup that with the gpu then do the docker compose up?
so the repo was pushed and it is 25.9.0 version whcich is older version
so by cloning and them and running the docker compose up that means the models will start locally ?i was not getting this casue those models and ports already in coantainer those containers will be not available in my laptop right?
how does those container will start and how does  i can run them?8:08 AMGreat question. Let me explain this clearly.

How Docker + NIM Works — Simple Explanation
When you run docker compose up, Docker does this automatically:
docker compose up
      ↓
reads docker-compose.yaml
      ↓
sees "I need nvcr.io/nvidia/paddle-ocr:25.9.0"
      ↓
checks if image exists locally → NO
      ↓
automatically PULLS (downloads) from NVIDIA's container registry
      ↓
starts the container with GPU access
      ↓
model is now running on localhost:8009
You do NOT manually download models. Docker pulls everything automatically using your NVIDIA API key to authenticate.

But Here Is The Honest Problem For You
These containers are very large and need serious hardware.
NIM ContainerSizeVRAM NeededPaddleOCR~8GB download4GB VRAMYOLOX page-elements~6GB download4GB VRAMYOLOX graphic-elements~6GB download4GB VRAMYOLOX table-structure~6GB download4GB VRAMnv-ingest itself~15GB download-
Total: ~40GB+ download, needs 16GB+ VRAM minimum
This will NOT run on a normal laptop. It needs a proper GPU server.

So Ask Your Lead These 3 Questions Right Now
This is the most important thing you can do:
Question 1:

"When you ran the pipeline successfully, were you running on a GPU server or your laptop?"

Question 2:

"Do we have a shared GPU server I can SSH into where the NIM containers are already running?"

Question 3:

"Can you share the exact docker-compose command or environment setup you used?"


The Three Realistic Scenarios
Scenario A — You have a GPU server available
bash# SSH into server
ssh user@your-gpu-server

# Clone repo
git clone -b release/25.9.0 https://github.com/NVIDIA/NeMo-Retriever.git
cd NeMo-Retriever

# Set key (needed to pull containers from NVIDIA registry)
export NVIDIA_API_KEY="nvapi-yourkey"
export NGC_API_KEY="nvapi-yourkey"  # same key, different variable name

# Pull and start everything
docker compose up
# This will take 30-60 minutes first time (downloading ~40GB)
Scenario B — NIMs already running on a shared server
Your lead may have already started the NIM containers on a team server. In this case you just point your localhost exports to that server's IP:
bashexport OCR_HTTP_ENDPOINT="http://TEAM_SERVER_IP:8009"
export YOLOX_HTTP_ENDPOINT="http://TEAM_SERVER_IP:8000"
export YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT="http://TEAM_SERVER_IP:8004"
export YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT="http://TEAM_SERVER_IP:8007"
Scenario C — You only have a laptop
You cannot run the full stack locally. You need cloud GPU access like:

AWS EC2 with A100/H100
Google Cloud with A100
NVIDIA LaunchPad (free access sometimes available)


What the Repo Clone Actually Contains
When you clone the repo, you get:
NeMo-Retriever/
├── docker-compose.yaml        ← defines all NIM containers + ports
├── docker-compose.override.yaml
├── src/
│   └── nv_ingest/
│       └── stages/
│           ├── nim/           ← code that CALLS the NIM containers
│           └── extractors/    ← extraction logic
├── examples/
│   └── pipeline.py            ← similar to your pipeline.py
└── README.md
The repo contains the code that calls the NIMs, not the NIMs themselves. The NIMs are pulled as Docker containers from nvcr.io (NVIDIA's container registry) when you run docker compose up.
