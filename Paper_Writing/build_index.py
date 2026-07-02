#!/usr/bin/env python3
"""
Build the FAISS index for the RAG chatbot.
This is equivalent to running notebook cells 1-6 (Phase 1).
"""

import os
import json
import re
import fitz  # PyMuPDF
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Check for API key (optional for just building index)
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("Note: ANTHROPIC_API_KEY not set. This is OK for building the index.")
    print("You'll need it later for chatting with Claude.\n")

# Configuration
PAPERS_DIR = Path("./Reference Papers")
VECTOR_STORE_DIR = Path("./vector_store")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

print("=" * 60)
print("Building FAISS Index for Research Papers")
print("=" * 60)
print(f"Papers directory: {PAPERS_DIR}")
print(f"Output directory: {VECTOR_STORE_DIR}")
print(f"Embedding model: {EMBEDDING_MODEL}")
print("=" * 60)

# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(pdf_path: Path) -> dict:
    """Extract text and structural info from a PDF."""
    doc = fitz.open(pdf_path)

    # Extract title from first page
    title = pdf_path.stem
    try:
        page1 = doc[0]
        blocks = page1.get_text("dict", sort=True)["blocks"]
        max_font_size = 0
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["size"] > max_font_size:
                            max_font_size = span["size"]
                            title = span["text"].strip()
    except Exception:
        pass

    # Extract sections
    sections = []
    current_heading = None
    current_text = []
    stop_extraction = False

    for page_num in range(len(doc)):
        if stop_extraction:
            break

        page = doc[page_num]

        try:
            page_dict = page.get_text("dict", sort=True)
            blocks = page_dict["blocks"]
        except Exception:
            text = page.get_text("text", sort=True)
            if text.strip():
                current_text.append(text)
            continue

        # Calculate median font size
        font_sizes = []
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_sizes.append(span["size"])
        median_font = np.median(font_sizes) if font_sizes else 10

        for block in blocks:
            if "lines" not in block:
                continue

            block_text = ""
            block_font_size = 0

            for line in block["lines"]:
                line_text = ""
                for span in line["spans"]:
                    line_text += span["text"]
                    block_font_size = max(block_font_size, span["size"])
                block_text += line_text + " "

            block_text = block_text.strip()
            if not block_text:
                continue

            # Check for References section
            if re.match(r"^(References|Bibliography|REFERENCES|BIBLIOGRAPHY)\s*$", block_text):
                stop_extraction = True
                break

            # Detect headings
            is_heading = False

            if block_font_size > median_font * 1.1:
                is_heading = True

            if len(block_text) < 100 and (
                re.match(r"^[IVX]+\.", block_text) or
                re.match(r"^\d+(\.\d+)*\.?\s+[A-Z]", block_text) or
                block_text.isupper() or
                block_text in ["Abstract", "Introduction", "Conclusion", "Methods", "Results", "Discussion"]
            ):
                is_heading = True

            if is_heading:
                if current_text:
                    sections.append({
                        "heading": current_heading,
                        "text": " ".join(current_text)
                    })
                current_heading = block_text
                current_text = []
            else:
                current_text.append(block_text)

    if current_text:
        sections.append({
            "heading": current_heading,
            "text": " ".join(current_text)
        })

    doc.close()

    return {
        "filename": pdf_path.name,
        "title": title,
        "sections": sections
    }

# ============================================================
# CHUNKING
# ============================================================

def chunk_document(doc: dict, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list:
    """Chunk a document respecting section boundaries."""
    chunks = []
    chunk_size_chars = chunk_size * 4
    overlap_chars = overlap * 4

    for section in doc["sections"]:
        section_text = section["text"]
        section_heading = section["heading"]

        section_text = re.sub(r"\s+", " ", section_text).strip()

        if len(section_text) < 50:
            continue

        if len(section_text) <= chunk_size_chars:
            chunk_text = f"[Title: {doc['title']} | Section: {section_heading or 'N/A'}]\n{section_text}"
            chunks.append({
                "text": chunk_text,
                "filename": doc["filename"],
                "title": doc["title"],
                "section": section_heading
            })
        else:
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', section_text)

            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size_chars:
                    current_chunk += sentence + " "
                else:
                    if current_chunk.strip():
                        chunk_text = f"[Title: {doc['title']} | Section: {section_heading or 'N/A'}]\n{current_chunk.strip()}"
                        chunks.append({
                            "text": chunk_text,
                            "filename": doc["filename"],
                            "title": doc["title"],
                            "section": section_heading
                        })

                    if len(current_chunk) > overlap_chars:
                        current_chunk = current_chunk[-overlap_chars:] + sentence + " "
                    else:
                        current_chunk = sentence + " "

            if current_chunk.strip():
                chunk_text = f"[Title: {doc['title']} | Section: {section_heading or 'N/A'}]\n{current_chunk.strip()}"
                chunks.append({
                    "text": chunk_text,
                    "filename": doc["filename"],
                    "title": doc["title"],
                    "section": section_heading
                })

    return chunks

# ============================================================
# INDEX BUILDING
# ============================================================

def build_index(all_chunks: list):
    """Embed all chunks and build a FAISS index."""
    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Embedding {len(all_chunks)} chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))

    print(f"Index built with {index.ntotal} vectors")
    return index, all_chunks

def save_index(index, chunks, store_dir=VECTOR_STORE_DIR):
    """Save FAISS index and chunk metadata to disk."""
    store_dir.mkdir(exist_ok=True)

    faiss.write_index(index, str(store_dir / "faiss_index.bin"))

    with open(store_dir / "chunks_metadata.json", "w") as f:
        json.dump(chunks, f, indent=2)

    print("=" * 60)
    print(f"✓ Saved {index.ntotal} vectors and metadata to {store_dir}")
    print("=" * 60)

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Find all PDFs
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    print(f"\nFound {len(pdf_files)} PDFs\n")

    if len(pdf_files) == 0:
        print("ERROR: No PDFs found in", PAPERS_DIR)
        print("Make sure the 'Reference Papers' directory exists and contains PDFs.")
        exit(1)

    # Extract and chunk all PDFs
    all_chunks = []
    failed_pdfs = []

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            doc = extract_pdf(pdf_path)
            chunks = chunk_document(doc)
            all_chunks.extend(chunks)
        except Exception as e:
            failed_pdfs.append((pdf_path.name, str(e)))
            print(f"\n  ✗ FAILED {pdf_path.name}: {e}")

    print("\n" + "-" * 60)
    print(f"Total chunks extracted: {len(all_chunks)}")
    if failed_pdfs:
        print(f"\nFailed PDFs ({len(failed_pdfs)}):")
        for name, error in failed_pdfs:
            print(f"  • {name}: {error}")
    print("-" * 60)

    # Build and save index
    if all_chunks:
        index, chunks = build_index(all_chunks)
        save_index(index, chunks)

        print("\n✓ Ingestion complete!")
        print("\nNext steps:")
        print("1. Set your API key: export ANTHROPIC_API_KEY='your-key-here'")
        print("2. Open the notebook: ./venv/bin/jupyter notebook rag_chatbot.ipynb")
        print("3. Run cells 1, 7-10 to start chatting!")
    else:
        print("\n✗ No chunks extracted. Check your PDF files.")
        exit(1)
