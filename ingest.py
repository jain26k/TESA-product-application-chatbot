# ============================================================
# ingest.py — TESA Knowledge Base Builder
# Vector DB: Pinecone (cloud) — course recommended for prod
# ============================================================

import os
import re
import glob
import numpy as np
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
import fitz
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

PAGES = [
    {"url": "https://www.actiontesa.com/products/plain-mdf-boards/",           "source": "MDF Board"},
    {"url": "https://www.actiontesa.com/products/plain-particle-boards/",      "source": "Particle Board"},
    {"url": "https://www.actiontesa.com/products/plain-hdhmr-boards/",         "source": "HDHMR Board"},
    {"url": "https://www.actiontesa.com/products/plain-boilo-bwp-hdf-boards/", "source": "BOILO BWP HDF Board"},
    {"url": "https://www.actiontesa.com/products/abraze-board/",               "source": "Abraze Board"},
    {"url": "https://www.actiontesa.com/products/moist-master/",               "source": "Moist Master"},
    {"url": "https://www.actiontesa.com/products/ornamatte/",                  "source": "Ornamatte"},
    {"url": "https://www.actiontesa.com/products/embossed-hdf-boards/",        "source": "Embossed HDF Board"},
    {"url": "https://www.actiontesa.com/products/uv-high-gloss-boards/",       "source": "UV High Gloss Board"},
    {"url": "https://www.actiontesa.com/products/acrylic-high-gloss-boards/",  "source": "Acrylic High Gloss Board"},
    {"url": "https://www.actiontesa.com/products/plain-hdhmr-doors/",          "source": "HDHMR Doors"},
    {"url": "https://www.actiontesa.com/products/moistro-doors/",              "source": "MOISTRO Doors"},
    {"url": "https://www.actiontesa.com/products/hdf-laminated-flooring/",     "source": "HDF Laminated Flooring"},
    {"url": "https://www.actiontesa.com/products/hdf-value-added-flooring/",   "source": "HDF Value Added Flooring"},
    {"url": "https://www.actiontesa.com/about-us/",                            "source": "About Action TESA"},
]

def scrape_page(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        print(f"  ✗ Failed: {url}: {e}")
        return None

def extract_pdf_text(pdf_path):
    try:
        text = ""
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"  ✗ PDF failed {pdf_path}: {e}")
        return None

def extract_pdf_ocr(pdf_path):
    try:
        import pytesseract
        from pdf2image import convert_from_path
        print(f"   → Running OCR...")
        pages = convert_from_path(pdf_path, dpi=200)
        text = ""
        for i, page in enumerate(pages):
            text += pytesseract.image_to_string(page, lang='eng') + "\n"
            print(f"   → OCR page {i+1}/{len(pages)} done")
        return text.strip()
    except Exception as e:
        print(f"  ✗ OCR failed {pdf_path}: {e}")
        return None

def chunk_text(text, chunk_size=200, overlap=20):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def build_vector_db():
    print("🪵 TESA Advisor — Building Knowledge Base (Pinecone)")
    print("=" * 50)

    # ── Connect to Pinecone index ──
    index_name = os.getenv("PINECONE_INDEX", "tesa-knowledge")
    index = pc.Index(index_name)
    
    # Clear existing vectors
    try:
        index.delete(delete_all=True)
        print("♻️  Cleared existing vectors")
    except:
        pass

    vectors_to_upsert = []
    doc_id = 0
    total_chunks = 0

    # ── WEBSITE PAGES ──
    print("\n🌐 Ingesting website pages...")
    for page in PAGES:
        print(f"\n📄 Scraping: {page['source']}")
        text = scrape_page(page["url"])
        if not text:
            continue
        chunks = chunk_text(text)
        print(f"   → {len(chunks)} chunks")
        for chunk in chunks:
            embedding = embed_text(chunk)
            vectors_to_upsert.append({
                "id": f"doc_{doc_id}",
                "values": embedding,
                "metadata": {
                    "source": page["source"],
                    "url": page["url"],
                    "text": chunk
                }
            })
            doc_id += 1
            total_chunks += 1
            # Upsert in batches of 100
            if len(vectors_to_upsert) >= 100:
                index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []

    # ── PDF FILES ──
    print("\n\n📚 Ingesting PDFs from /docs...")
    pdf_files = glob.glob("./docs/*.pdf")
    print(f"   Found {len(pdf_files)} PDFs")
    for pdf_path in pdf_files:
        source_name = os.path.basename(pdf_path).replace(".pdf", "")
        print(f"\n📄 Reading: {source_name}")
        text = extract_pdf_text(pdf_path)
        if not text or len(text) < 100:
            print(f"   → Text extraction failed, trying OCR...")
            text = extract_pdf_ocr(pdf_path)
        if not text:
            continue
        chunks = chunk_text(text)
        print(f"   → {len(chunks)} chunks")
        for chunk in chunks:
            embedding = embed_text(chunk)
            vectors_to_upsert.append({
                "id": f"doc_{doc_id}",
                "values": embedding,
                "metadata": {
                    "source": source_name,
                    "url": "actiontesa.com/brochure",
                    "text": chunk
                }
            })
            doc_id += 1
            total_chunks += 1
            if len(vectors_to_upsert) >= 100:
                index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []

    # Upsert remaining
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)

    print(f"\n✅ Done! Stored {total_chunks} chunks in Pinecone")
    print("\nNow run: streamlit run chatbot_rag.py")

if __name__ == "__main__":
    build_vector_db()