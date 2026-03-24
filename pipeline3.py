import os
import logging
from typing import List
import requests
import fitz  # PyMuPDF
from pymilvus import MilvusClient

# ---------- Load .env ----------
from dotenv import load_dotenv
load_dotenv()

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------- Env ----------
if "NVIDIA_API_KEY" not in os.environ:
    raise RuntimeError("Set NVIDIA_API_KEY in .env file")
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]

# ---------- Milvus ----------
MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
COLLECTION = "rag_documents"
DIM = 1024

milvus = MilvusClient(uri=MILVUS_URI)
logger.info(f"Connected to Milvus at {MILVUS_URI}")


def extract_text_from_pdf(file_path: str) -> List[dict]:
    doc = fitz.open(file_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:
            pages.append({
                "text": text,
                "page": page_num + 1,
                "source": os.path.basename(file_path),
                "type": "text",
            })
    doc.close()
    logger.info(f"Extracted {len(pages)} text pages from {file_path}")
    return pages


def extract_tables_from_pdf(file_path: str) -> List[dict]:
    doc = fitz.open(file_path)
    tables = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        try:
            found_tables = page.find_tables()
            for table in found_tables:
                data = table.extract()
                if not data or len(data) < 2:
                    continue
                headers = data[0]
                md_lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
                md_lines.append("| " + " | ".join("---" for _ in headers) + " |")
                for row in data[1:]:
                    md_lines.append("| " + " | ".join(str(c) for c in row) + " |")
                md = "\n".join(md_lines)
                tables.append({
                    "text": md,
                    "page": page_num + 1,
                    "source": os.path.basename(file_path),
                    "type": "table",
                })
        except Exception as e:
            logger.debug(f"Table extraction on page {page_num + 1}: {e}")
    doc.close()
    logger.info(f"Extracted {len(tables)} tables from {file_path}")
    return tables


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks


def embed_texts(texts: List[str], input_type: str = "passage") -> List[List[float]]:
    url = "https://integrate.api.nvidia.com/v1/embeddings"
    all_embeddings = []
    batch_size = 10
    max_chars = 2000
    texts = [t[:max_chars] if len(t) > max_chars else t for t in texts]

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch = [t if t.strip() else "empty" for t in batch]
        payload = {
            "model": "nvidia/nv-embedqa-e5-v5",
            "input": batch,
            "input_type": input_type,
            "encoding_format": "float",
            "truncate": "END",
        }
        headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}"}
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        if r.status_code != 200:
            logger.error(f"Embedding API error {r.status_code}: {r.text}")
            r.raise_for_status()
        batch_embs = [e["embedding"] for e in r.json()["data"]]
        all_embeddings.extend(batch_embs)
        logger.info(f"  Embedded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
    return all_embeddings


def ingest_document(file_paths: List[str]) -> int:
    all_chunks = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            continue
        logger.info(f"Processing: {file_path}")
        pages = extract_text_from_pdf(file_path)
        tables = extract_tables_from_pdf(file_path)
        for page in pages:
            for chunk in chunk_text(page["text"]):
                all_chunks.append({
                    "text": chunk,
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_type": page["type"],
                })
        for table in tables:
            all_chunks.append({
                "text": table["text"],
                "source": table["source"],
                "page": table["page"],
                "chunk_type": table["type"],
            })
    if not all_chunks:
        logger.warning("No chunks extracted")
        return 0
    logger.info(f"Embedding {len(all_chunks)} chunks via NVIDIA NIM...")
    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts, input_type="passage")
    data = []
    for i, chunk in enumerate(all_chunks):
        data.append({
            "vector": embeddings[i],
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_type": chunk["chunk_type"],
        })
    milvus.insert(collection_name=COLLECTION, data=data)
    logger.info(f"Inserted {len(data)} chunks into Milvus")
    return len(data)


def retrieve(query: str, top_k: int = 5) -> List[dict]:
    q_emb = embed_texts([query], input_type="query")[0]
    hits = milvus.search(
        collection_name=COLLECTION,
        data=[q_emb],
        limit=top_k,
        output_fields=["text", "source", "page", "chunk_type"],
    )[0]
    return [
        {
            "text": h["entity"].get("text", ""),
            "source": h["entity"].get("source", ""),
            "page": h["entity"].get("page", 0),
            "score": h["distance"],
        }
        for h in hits if h["entity"].get("text")
    ]


def rag_chatbot(query: str) -> str:
    results = retrieve(query)
    if not results:
        return "No relevant information found."
    context = "\n\n---\n\n".join(r["text"] for r in results)
    prompt = f"""Use the following context to answer the question.
If the answer is not in the context, say so.

Context:
{context}

Question: {query}
Answer:"""
    try:
        r = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            json={
                "model": "meta/llama-3.3-70b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.7,
            },
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"LLM error: {e}"


def ensure_collection():
    if not milvus.has_collection(COLLECTION):
        milvus.create_collection(
            collection_name=COLLECTION,
            dimension=DIM,
            metric_type="L2",
            auto_id=True,
        )
        logger.info(f"Created collection '{COLLECTION}'")
    else:
        logger.info(f"Collection '{COLLECTION}' exists")


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    
    # Drop old invoice collection
    if milvus.has_collection(COLLECTION):
        milvus.drop_collection(COLLECTION)
        print(f"Dropped old collection '{COLLECTION}'")
    
    ensure_collection()

    pdfs = [
        "C:\\Users\\pabbidi\\Downloads\\NV-Ingest\\docks\\PK0016.pdf",
    ]

    for pdf in pdfs:
        if not os.path.exists(pdf):
            print(f"File not found: {pdf}")
        else:
            num = ingest_document([pdf])
            print(f"Ingested {num} chunks from {os.path.basename(pdf)}")

    queries = [
        "What are all the test results that are outside the normal biological reference interval, and by how much?",
        "Based on the kidney function test results and the eGFR classification table, what is the patient's GFR category and what does it signify?",
        "What is the patient's HbA1c value and according to the ADA guidelines in the report, does this patient fall in the non-diabetic, prediabetic, or diabetic range? Is the fasting glucose consistent with this classification?",
        "Summarize all findings from the ultrasound whole abdomen including the impression, and what further tests were advised?",
        "What are the patient's lipid profile results and based on the clinical decision limits table in the report, classify each lipid parameter as optimal, borderline high, or high?",
    ]

    print("\n" + "=" * 60)
    for q in queries:
        print(f"\nQ: {q}")
        print(f"A: {rag_chatbot(q)}")
        print("-" * 60)