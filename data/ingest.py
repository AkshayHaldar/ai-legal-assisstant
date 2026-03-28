"""
data/ingest.py
--------------
Run this ONCE after placing your PDF files in data/raw/
It builds the FAISS vector index used by the retrieval agent.
"""

import os
import json
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import fitz  # PyMuPDF
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

sys.path.append(".")
from config import FAISS_INDEX_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP 

# Map known PDF filenames to their metadata.
LEGAL_SOURCES = {
    "constitution.pdf":        {"act": "Indian Constitution",         "type": "central"},
    "ipc.pdf":                 {"act": "Indian Penal Code",            "type": "central"},
    "crpc.pdf":                {"act": "Code of Criminal Procedure",   "type": "central"},
    "cpc.pdf":                 {"act": "Civil Procedure Code",         "type": "central"},
    "consumer_protection.pdf": {"act": "Consumer Protection Act 2019", "type": "central"},
    "rent_control.pdf":        {"act": "Delhi Rent Control Act",       "type": "state"},
    "transfer_property.pdf":   {"act": "Transfer of Property Act",     "type": "central"},
    "negotiable_instruments.pdf": {"act": "Negotiable Instruments Act","type": "central"},
    "labour_act.pdf":          {"act": "Industrial Disputes Act",      "type": "central"},
    "domestic_violence.pdf":   {"act": "Protection of Women from DV Act","type": "central"},
}

def process_single_file(file_path_str: str):
    """Worker function to extract text and create chunks for one file."""
    path = Path(file_path_str)
    filename = path.name
    meta = LEGAL_SOURCES.get(filename, {"act": filename.split('.')[0].replace('_', ' ').title(), "type": "general"})

    try:
        if path.suffix.lower() == ".pdf":
            doc  = fitz.open(file_path_str)
            text = "".join(page.get_text() for page in doc)
            doc.close()
        elif path.suffix.lower() in [".txt", ".md"]:
            with open(file_path_str, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            text = ""
    except Exception as e:
        print(f"\n[Error reading {filename}]: {e}")
        text = ""

    if not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    
    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{
            "source":   filename,
            "act":      meta["act"],
            "law_type": meta["type"],
        }],
    )
    return chunks

def build_index():
    raw_dir  = Path("data/raw")
    all_docs = []

    print("\n========================================")
    print("  AI Legal Assistant - FAST Index Builder")
    print("========================================\n")

    if not raw_dir.exists():
        print(f"Directory {raw_dir} does not exist.")
        sys.exit(1)

    all_files = [str(f) for f in raw_dir.iterdir() if f.suffix.lower() in [".pdf", ".txt", ".md"]]
    
    if not all_files:
        print("\n  ERROR: No PDFs, TXTs, or MDs found in data/raw/.")
        sys.exit(1)

    print(f"Found {len(all_files)} files. Starting concurrent fast extraction...\n")
    
    # Process files concurrently to maximize CPU usage
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_single_file, f): f for f in all_files}
        for future in tqdm(as_completed(futures), total=len(all_files), desc="Chunking Files"):
            result_chunks = future.result()
            all_docs.extend(result_chunks)

    if not all_docs:
        print("\n  ERROR: No text could be extracted from the files.")
        sys.exit(1)

    print(f"\nTotal chunks across all documents: {len(all_docs)}")
    print("Generating embeddings (Shows progress below & might take a few minutes)...\n")

    embeddings  = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        show_progress=True,
        # Batch size doubled for much faster embedding processing!
        encode_kwargs={'batch_size': 128} 
    )
    
    # Embeddings calculation
    vectorstore = FAISS.from_documents(all_docs, embeddings)

    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"\nFAISS index saved -> {FAISS_INDEX_PATH}")

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

    print("Chunk preview saved -> data/processed/chunks_preview.json")       
    print("\nIndex build complete. You can now run: streamlit run ui/app.py\n")

if __name__ == "__main__":
    build_index()
