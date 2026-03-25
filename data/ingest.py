"""
data/ingest.py
--------------
Run this ONCE after placing your PDF files in data/raw/
It builds the FAISS vector index used by the retrieval agent.

Usage:
    python data/ingest.py
"""

import os
import json
import sys
from pathlib import Path

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

sys.path.append(".")
from config import FAISS_INDEX_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

# ---------------------------------------------------------------------------
# Map each PDF filename to its metadata
# Add more acts here as needed
# ---------------------------------------------------------------------------
LEGAL_SOURCES = {
    # Central acts (already have these)
    "constitution.pdf":        {"act": "Indian Constitution",         "type": "central"},
    "ipc.pdf":                 {"act": "Indian Penal Code",            "type": "central"},
    "crpc.pdf":                {"act": "Code of Criminal Procedure",   "type": "central"},
    "cpc.pdf":                 {"act": "Civil Procedure Code",         "type": "central"},
    "consumer_protection.pdf": {"act": "Consumer Protection Act 2019", "type": "central"},

    # Add these — download from indiacode.nic.in
    "rent_control.pdf":        {"act": "Delhi Rent Control Act",       "type": "state"},
    "transfer_property.pdf":   {"act": "Transfer of Property Act",     "type": "central"},
    "negotiable_instruments.pdf": {"act": "Negotiable Instruments Act","type": "central"},
    "labour_act.pdf":          {"act": "Industrial Disputes Act",      "type": "central"},
    "domestic_violence.pdf":   {"act": "Protection of Women from DV Act","type": "central"},
}


def extract_text(pdf_path: str) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc  = fitz.open(pdf_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


def build_index():
    raw_dir  = Path("data/raw")
    all_docs = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )

    print("\n========================================")
    print("  AI Legal Assistant — Index Builder")
    print("========================================\n")

    for filename, meta in LEGAL_SOURCES.items():
        path = raw_dir / filename
        if not path.exists():
            print(f"  [SKIP] {filename} — not found in data/raw/")
            continue

        print(f"  [PROCESSING] {filename}")
        raw_text = extract_text(str(path))
        chunks   = splitter.create_documents(
            texts=[raw_text],
            metadatas=[{
                "source":   filename,
                "act":      meta["act"],
                "law_type": meta["type"],
            }],
        )
        all_docs.extend(chunks)
        print(f"    → {len(chunks)} chunks created")

    if not all_docs:
        print("\n  ERROR: No PDFs were processed.")
        print("  Please add PDF files to the data/raw/ folder and try again.")
        sys.exit(1)

    print(f"\n  Total chunks across all documents: {len(all_docs)}")
    print("\n  Generating embeddings using HuggingFace all-MiniLM-L6-v2")
    print("  (Downloads ~80MB on first run, then cached locally. Takes 2–5 min.)\n")

    embeddings  = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(all_docs, embeddings)

    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"\n  FAISS index saved → {FAISS_INDEX_PATH}")

    # Save a human-readable preview of the first 20 chunks for debugging
    os.makedirs("data/processed", exist_ok=True)
    preview = [
        {
            "chunk_id": i,
            "act":      d.metadata["act"],
            "preview":  d.page_content[:200],
        }
        for i, d in enumerate(all_docs[:20])
    ]
    with open("data/processed/chunks_preview.json", "w", encoding="utf-8") as f:
        json.dump(preview, f, indent=2, ensure_ascii=False)

    print("  Chunk preview saved → data/processed/chunks_preview.json")
    print("\n  Index build complete. You can now run: streamlit run ui/app.py\n")


if __name__ == "__main__":
    build_index()
