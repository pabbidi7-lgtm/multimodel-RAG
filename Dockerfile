# ingest_documents.py
from nv_ingest_client.client import Ingestor, NvIngestClient

FILEPATHS = [
    "data/multimodal/multimodal_test.pdf",
    "data/multimodal/woods_frost.pdf"
]

COLLECTION_NAME = "multimodal_data_nvingest"

MILVUS_URI = "http://localhost:19530"
MINIO_ENDPOINT = "localhost:9010"

# Server Mode (Create NeMo Retriever Library client)
client = NvIngestClient(
    message_client_hostname="localhost",
    message_client_port=7670
)

ingestor = Ingestor(client=client)

ingestor = ingestor.files(FILEPATHS)

ingestor = ingestor.extract(
                extract_text=True,
                extract_tables=True,
                extract_charts=True,
                extract_images=False,
                text_depth="page",
                table_output_format="markdown"
            )
ingestor = ingestor.split(
                tokenizer="intfloat/e5-large-unsupervised",
                chunk_size=51,
                chunk_overlap=15,
                params={"split_source_types": ["PDF", "text", "html", "mp3", "docx", "pptx"]},
            )

ingestor = ingestor.embed(
    # For self-hosted: "http://nemotron-embedding-ms:8000/v1"
    # For cloud (NVIDIA-hosted): "https://integrate.api.nvidia.com/v1"
    endpoint_url="http://nemotron-embedding-ms:8000/v1",
    model_name="nvidia/llama-nemotron-embed-1b-v2"
)

ingestor = ingestor.vdb_upload(
                collection_name=COLLECTION_NAME,
                milvus_uri=MILVUS_URI,
                minio_endpoint=MINIO_ENDPOINT,
                sparse=False,
                enable_images=True,
                recreate=False,
                dense_dim=2048,
                stream=False
            )

results, failures = ingestor.ingest(show_progress=True, return_failures=True)
