# ============================================================
# ingest.py — TESA Knowledge Base Builder
# Fixed: smaller chunks, proper PDF reading
# ============================================================

import os
import re
import glob
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import fitz  # pymupdf

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

# ── SCRAPE ──
def scrape_page(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        print(f"  ✗ Failed: {url}: {e}")
        return None

# ── READ PDF ──
def extract_pdf_text(pdf_path):
    try:
        import pytesseract
        from pdf2image import convert_from_path
        from PIL import Image
        
        print(f"   → Running OCR (this takes ~30 seconds per PDF)...")
        
        # Convert PDF pages to images
        pages = convert_from_path(pdf_path, dpi=200)
        
        text = ""
        for i, page in enumerate(pages):
            # Run OCR on each page image
            page_text = pytesseract.image_to_string(page, lang='eng')
            text += page_text + "\n"
            print(f"   → OCR page {i+1}/{len(pages)} done")
        
        return text.strip()
    except Exception as e:
        print(f"  ✗ OCR failed for {pdf_path}: {e}")
        return None

# ── CHUNK — fixed size 200 words, 20 word overlap ──
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

# ── EMBED ──
def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# ── BUILD VECTOR DB ──
def build_vector_db():
    print("🪵 TESA Advisor — Building Knowledge Base")
    print("=" * 50)

    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    try:
        chroma_client.delete_collection("tesa_knowledge")
        print("♻️  Cleared existing knowledge base")
    except:
        pass

    collection = chroma_client.create_collection(
        name="tesa_knowledge",
        metadata={"hnsw:space": "cosine"}
    )

    doc_id = 0
    total_chunks = 0

    # ── INGEST WEBSITE PAGES ──
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
            collection.add(
                ids=[f"doc_{doc_id}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": page["source"], "url": page["url"]}]
            )
            doc_id += 1
            total_chunks += 1

    # ── INGEST PDF FILES ──
    print("\n\n📚 Ingesting PDF files from /docs...")
    pdf_files = glob.glob("./docs/*.pdf")
    print(f"   Found {len(pdf_files)} PDF files")

    if len(pdf_files) == 0:
        print("   ⚠️  No PDFs found — make sure PDFs are in the /docs folder")
    
    for pdf_path in pdf_files:
        source_name = os.path.basename(pdf_path).replace(".pdf", "")
        print(f"\n📄 Reading PDF: {source_name}")
        text = extract_pdf_text(pdf_path)
        if not text:
            continue
        chunks = chunk_text(text)
        print(f"   → {len(chunks)} chunks")
        for chunk in chunks:
            embedding = embed_text(chunk)
            collection.add(
                ids=[f"doc_{doc_id}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": source_name, "url": "actiontesa.com/brochure"}]
            )
            doc_id += 1
            total_chunks += 1

    print(f"\n✅ Done! Stored {total_chunks} chunks total")
    print("📁 Saved to ./chroma_db")
    print("\nNow run: streamlit run chatbot_rag.py")

if __name__ == "__main__":
    build_vector_db()