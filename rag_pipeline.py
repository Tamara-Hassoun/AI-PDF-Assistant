import os, requests, time
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests

# =========================
# إعدادات Gemini API
# =========================
GEMINI_API_KEY = ""

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent"




# =========================
# Embedding Model
# =========================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# Vector Database (Chroma)
# =========================
chroma_client = chromadb.Client(
    Settings(persist_directory="./chroma_db")
)

collection = chroma_client.get_or_create_collection(
    name="lecture_saver"
)

# =========================
# PDF Ingestion
# =========================
def ingest_pdf(pdf_path, chunk_size=800):
    reader = PdfReader(pdf_path)
    filename = os.path.basename(pdf_path)

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text:
            continue

        chunks = [
            text[i:i + chunk_size]
            for i in range(0, len(text), chunk_size)
        ]

        for idx, chunk in enumerate(chunks):
            embedding = embedding_model.encode(chunk).tolist()

            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[f"{filename}_p{page_number}_c{idx}"],
                metadatas=[{
                    "source": filename,
                    "page": page_number
                }]
            )


def ingest_multiple_pdfs(pdf_paths, chunk_size=800):

    for pdf_path in pdf_paths:
        ingest_pdf(pdf_path, chunk_size=chunk_size)
            

# =========================
# Retrieval
# =========================
def retrieve_chunks(query, k=4):
    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    return documents, metadatas

# =========================
# Generation (Gemini)
# =========================

def generate_answer(context, question):
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY not set."

    prompt_text = f"""
You are a helpful assistant answering questions based ONLY on the provided lecture context.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 512
        }
    }

    try:
        resp = requests.post(
            GEMINI_ENDPOINT,
            headers=headers,
            params=params,
            json=payload,
            timeout=30
        )

        if resp.status_code != 200:
            return f"API error {resp.status_code}: {resp.text}"

        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        return f"Connection error: {e}"


# =========================
# Full RAG Pipeline
# =========================
def ask_question(question):
    docs, metas = retrieve_chunks(question, k=3)  


    context = "\n\n".join(docs)
    answer = generate_answer(context, question)

    sources = []
    for meta in metas:
        source_str = f"{meta['source']} (Page {meta['page']})"
        if source_str not in sources:
            sources.append(source_str)

    return answer, sources
