#!/usr/bin/env python3
"""Test script to debug RAG chatbot notebook cells"""

import os
os.environ['ANTHROPIC_API_KEY'] = 'test-key-not-needed-for-ingestion'

import json
import re
import fitz  # PyMuPDF
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# Configuration
PAPERS_DIR = Path("./Reference Papers")
VECTOR_STORE_DIR = Path("./vector_store")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

print("="*60)
print("Testing RAG Chatbot Components")
print("="*60)

# Test 1: Check papers directory
print(f"\n1. Checking papers directory: {PAPERS_DIR}")
pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
print(f"   Found {len(pdf_files)} PDFs")
if len(pdf_files) == 0:
    print("   ERROR: No PDFs found!")
    exit(1)
print(f"   First PDF: {pdf_files[0].name}")

# Test 2: PDF extraction function
print(f"\n2. Testing PDF extraction...")

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

# Test on first PDF
try:
    test_pdf = pdf_files[0]
    doc = extract_pdf(test_pdf)
    print(f"   ✓ Extracted: {doc['title']}")
    print(f"   ✓ Sections: {len(doc['sections'])}")
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Chunking function
print(f"\n3. Testing chunking...")

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

try:
    chunks = chunk_document(doc)
    print(f"   ✓ Created {len(chunks)} chunks")
    if len(chunks) > 0:
        print(f"   ✓ First chunk preview: {chunks[0]['text'][:100]}...")
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("✓ All tests passed! Ready to run full ingestion.")
print("="*60)
