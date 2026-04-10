You’re asking a good and simple question, so I’ll keep this simple and direct:
Short answer
Claude is partly right, but too absolute.
What Claude got right

If you run the repo’s self-hosted stack with docker compose up, Docker will pull the container images from NVIDIA’s registry (after docker login nvcr.io and setting the right keys) and then start the NIM services locally. The self-hosted quickstart explicitly says to clone the repo, log in to NGC, set NGC_API_KEY / NIM_NGC_API_KEY, and start with docker compose up (or docker compose --profile retrieval up). It also warns that the first startup can take 10–15 minutes to pull and fully load models. [docs.nvidia.com]
You do not manually download model weights one by one in the normal Docker Compose flow; the containers and model assets are fetched as part of the deployment flow using your NVIDIA/NGC keys. The object-detection and OCR “getting started” docs both say you export an NGC key and the container uses it to download the required models/resources. [docs.nvidia.com], [github.com]

What Claude got wrong / overstated

It is not true that “nv-ingest library mode only works with localhost and cloud endpoints don’t work.” The NeMo Retriever docs for library mode explicitly say you can integrate with Nemotron RAG models locally or via NIM endpoints, and the repo’s nemo_retriever README says the two inference options are:

run Nemotron RAG models on your local GPU(s), or
make over-the-network inference calls to build.nvidia.com hosted or locally deployed NeMo Retriever NIM endpoints. [docs.nvidia.com], [build.nvidia.com], [build.nvidia.com]



So the correct statement is:

docker compose up is the standard way to start the full self-hosted local stack, but library mode can also call hosted or other deployed NIM endpoints if the version/output schemas match. [docs.nvidia.com], [build.nvidia.com], [docs.nvidia.com]


So, if you clone the repo and run docker compose up, will the models start locally?
Yes — if these conditions are met

You are on a machine/server with Docker + Docker Compose installed. [docs.nvidia.com]
You have a valid NVIDIA/NGC API key and have authenticated to nvcr.io. The self-hosted guide explicitly requires docker login nvcr.io and a .env containing NGC_API_KEY / NIM_NGC_API_KEY. [docs.nvidia.com]
You have a supported GPU and enough disk space for the core features. The current support matrix says the core extraction features (page-elements-v3, table-structure-v1, graphic-elements-v1, nemotron-ocr-v1, embedding, retrieval) run on a single A10G or better GPU, and the total disk requirement for core features is about ~150GB. [org.ngc.nvidia.com]
But for the older 25.9.0 release, the release notes explicitly say A10G and L40S were not supported in that release. So for 25.9.0 specifically, you should not assume an A10G/L40S setup is valid even though the latest docs support A10G/L40S for newer releases. [docs.nvidia.com], [org.ngc.nvidia.com]

So the honest practical answer is:

Yes, docker compose up starts the local NIM services — but only on a machine with the right Docker/GPU environment and access to NVIDIA’s container registry. [docs.nvidia.com], [docs.nvidia.com]


Do the containers exist on your laptop by default?
No
If you clone the repo, you only get:

the code
the docker-compose.yaml
the client/service logic

You do not get the NIM containers preinstalled. The containers are pulled from NVIDIA’s registry when you run docker compose up (or docker compose pull) after authenticating. The self-hosted docs describe exactly that startup flow. [docs.nvidia.com], [docs.nvidia.com]
So your understanding here is right:

The repo contains the orchestration/configuration, not the full models already sitting on your laptop.
Docker downloads the images when you start the stack. [docs.nvidia.com], [docs.nvidia.com]


Do you need to “download the models and set them up with the GPU” manually before docker compose up?
Usually no — not manually
In the normal Docker Compose path:

you clone the repo,
log in to nvcr.io,
set the .env / shell variables,
run docker compose up,
Docker pulls the images,
the containers fetch their model assets as needed. [docs.nvidia.com], [docs.nvidia.com], [github.com]

So you do not separately pre-download each model the way you would with raw Hugging Face weights.

How much GPU / RAM / disk do you need?
The safest official answer
For the current NeMo Retriever support matrix, the core features — including:

nemotron-page-elements-v3
nemotron-table-structure-v1
nemotron-graphic-elements-v1
nemotron-ocr-v1
embedding / retrieval

run on a single A10G or better GPU, and the docs show about ~150GB disk space for core features. The matrix also lists GPU memory examples such as A10G 24GB, L40S 48GB, A100 40/80GB, H100 80GB, etc. [org.ngc.nvidia.com]
But for 25.9.0 specifically
The 25.9.0 release notes say A10G and L40S were not supported in that release. [docs.nvidia.com]
So for your exact version (25.9.0), the safest assumption is:

use a bigger, officially supported GPU server rather than trying to make it work on a modest laptop/workstation,
and budget for ~150GB disk for the core stack. [docs.nvidia.com], [org.ngc.nvidia.com]

Important note
I would not trust the exact per-container VRAM numbers Claude gave you (2GB / 4GB per model) because I do not have official NVIDIA documentation supporting those exact figures. The official docs give system-level requirements, not those exact container-by-container numbers. [org.ngc.nvidia.com], [docs.nvidia.com]

How should you prepare before you get a costly GPU server?
This is the part that matters for saving money.
Do these things before you start billable GPU time
1) On your normal machine, prepare the repo and config
Shellgit clone -b release/25.9.0 https://github.com/NVIDIA/NeMo-Retriever.gitcd NeMo-Retriever``Show more lines
The self-hosted docs explicitly start with cloning the repo and changing into it. [docs.nvidia.com]
2) Read the compose file and env expectations
Open:

docker-compose.yaml
.env / example env files if present

The self-hosted docs say to create a .env containing at least:

NGC_API_KEY
NIM_NGC_API_KEY [docs.nvidia.com]

3) Prepare your API key usage
Do not paste keys into random scripts.
Use shell env vars or .env on the server. NVIDIA docs explicitly show exporting the key and using it for docker login nvcr.io. [docs.nvidia.com], [docs.nvidia.com]
4) Ask your lead / infra team these exact questions
Because you are right that learning on paid GPU time is expensive.
Ask:

Which exact GPU SKU did you use for 25.9.0?
Did you use the repo’s docker compose up or a pre-existing shared server?
Do we already have a shared machine where the NIM containers are running?
Which .env values / profiles did you use?

This is especially important because 25.9.0 has older support/known-issue constraints. [docs.nvidia.com], [docs.nvidia.com]

What I recommend you actually do on the GPU server
Once you get access to the GPU machine:
Step-by-step
1) SSH in and verify Docker + GPU
Shelldocker --versiondocker compose versionnvidia-smiShow more lines
The self-hosted path requires Docker/Compose and a supported NVIDIA GPU environment. [docs.nvidia.com], [docs.nvidia.com]
2) Clone the repo
Shellgit clone -b release/25.9.0 https://github.com/NVIDIA/NeMo-Retriever.gitcd NeMo-RetrieverShow more lines
 [docs.nvidia.com]
3) Set your keys
Shellexport NGC_API_KEY="YOUR_KEY"export NIM_NGC_API_KEY="$NGC_API_KEY"``Show more lines
That’s the pattern the self-hosted docs describe. [docs.nvidia.com]
4) Log in to NVIDIA’s registry
Shellecho "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin``Show more lines
This is required to pull the containers. [docs.nvidia.com], [docs.nvidia.com]
5) Optionally pre-pull images
If you want to reduce uncertainty during billable time, you can do:
Shelldocker compose pullShow more lines
This is a good practical preparation step because the docs say first startup can spend 10–15 minutes pulling/loading models. [docs.nvidia.com]
6) Start the stack
For default LanceDB/in-process storage:
Shelldocker compose upShow more lines
If you need Milvus/retrieval profile:
Shelldocker compose --profile retrieval upShow more lines
Those are the official self-hosted startup patterns. [docs.nvidia.com]

One very important correction to keep in mind
Because you were testing hosted endpoints earlier:
Your issue was not simply “cloud endpoints never work”
That statement is too strong. The NeMo Retriever docs and repo README indicate that network-hosted NIM endpoints are supported in some library-mode scenarios, and the docs explicitly mention cloud-based inference for nemotron-parse. [build.nvidia.com], [docs.nvidia.com], [docs.nvidia.com]
However, for your actual 25.9.0 pipeline + specific endpoints, you hit a concrete incompatibility on page-elements-v3 ('paragraph'), so self-hosted compose with a version-matched backend is the safer path for reproducing the README architecture. [docs.nvidia.com], [github.com]
So Claude’s “must be localhost” is too absolute, but for your practical goal of matching the README architecture exactly, local Docker Compose is the best route. [docs.nvidia.com], [build.nvidia.com]

My honest recommendation
If you want the closest possible match to what the 25.9.0 README architecture is claiming:
Use a GPU server + clone the release/25.9.0 repo + docker compose up
That is the cleanest way to reproduce the architecture as intended. The self-hosted docs are built around that model. [docs.nvidia.com], [github.com]
If you only have a normal laptop
Then do not expect to run the full self-hosted stack reliably. The official docs point to a real GPU-backed deployment, and the support matrix/disk requirements are not laptop-friendly for the full extraction stack. [org.ngc.nvidia.com], [docs.nvidia.com]

Final answer in one line
Yes:
If you clone the 25.9.0 repo and run docker compose up on a supported GPU machine with Docker and NVIDIA registry access configured, the local NIM services are supposed to start by pulling the required containers and models. [docs.nvidia.com], [docs.nvidia.com]
No:
You do not need to manually download every model first — Docker/NGC handles that as part of the startup flow. [docs.nvidia.com], [docs.nvidia.com]
