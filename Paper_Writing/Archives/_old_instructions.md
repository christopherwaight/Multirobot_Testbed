# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains an automated citation system for academic paper writing, specifically designed to generate research paper introductions with properly cited references. The system uses LLM-based analysis to:

1. Extract and parse academic PDFs from a papers directory
2. Build a vector database of paper content using ChromaDB
3. Identify factual claims in draft introduction text that need citations
4. Score paper relevance to each claim using multi-section analysis
5. Optimize citation assignment using simulated annealing
6. Generate introduction text with IEEE-style citations and reference lists

## Core Architecture

### Three-Notebook System

**citation_adder.ipynb** (Production Pipeline)
- Complete end-to-end citation pipeline
- Processes PDFs from `Papers/` directory (51+ papers)
- Uses ChromaDB with OpenAI embeddings for semantic search
- Implements sophisticated LLM-based scoring with precise 0.00-1.00 granularity
- Optimizes citations using simulated annealing (100 runs × 1200 iterations)
- Input: `introduction.txt` (raw introduction without citations)
- Output: `introduction_with_citations.txt` and `references.txt`

**introduction_writer.ipynb** (Alternative/Legacy Pipeline)
- Earlier implementation using direct LLM analysis without vector search
- Five-step pipeline: PDF loading → paper analysis → positioning → writing → citation generation
- Uses temperature-differentiated LLMs (analyzer at 0.1, writer at 0.2)
- Extracts quotes and relevance scores directly from papers
- Generates MLA-formatted citations with LaTeX bibliography output
- Includes paper context injection for positioning statements

**match_citations_to_files.ipynb** (Utility)
- Post-processing tool to verify citation integrity
- Parses `references.txt` to extract cited filenames
- Matches cited papers to actual PDFs in `Papers/` folder
- Supports exact and fuzzy filename matching (>70% similarity threshold)
- Optionally copies cited papers to `./cited_papers/` with numbered prefixes
- Exports citation-to-file mapping as CSV for reference

### Key Technical Components

**Vector Store Pipeline:**
- ChromaDB with OpenAI `text-embedding-3-small` embeddings
- Batch processing (20 papers/batch) with rate limiting
- Extracts title + first page + content sample (first 3000 chars) for embedding
- Enables semantic search to find top-k candidate papers per claim

**Claim Extraction:**
- LLM identifies factual statements requiring citations
- Filters out author contributions, paper organization, obvious statements
- Returns structured JSON with claim text and topic classification
- Sorted by position in original text to maintain narrative flow

**Relevance Scoring (Advanced Multi-Section):**
- Extracts abstract, introduction, conclusion from each paper
- Uses precise continuous 0.00-1.00 scoring (not rounded buckets)
- Evaluates: evidence strength, citation type, centrality to paper
- Returns reasoning, evidence quotes, confidence levels
- Applies confidence penalties for low-quality matches

**Simulated Annealing Optimization:**
- Objective: maximize total relevance + diversity + evenness
- Constraints: 30+ papers minimum, 1-4 cites/claim, max 3 cites/paper
- Operations: swap, replace, add, remove citations
- Temperature schedule: 1.5 → 0.0005 with 0.995 cooling rate
- Multiple runs to find global optimum

## Working with Papers

The `Papers/` directory contains 51+ PDF files. PDFs are loaded with PyPDFLoader (first 20 pages) to manage token limits. Paper metadata extraction attempts to identify:
- Title (from first page lines)
- Authors (pattern matching for "Name Name" format)
- Year (regex search in filename and content)
- Filename for reference generation

## Configuration Parameters

All tunable parameters are in the `CONFIG` dict in citation_adder.ipynb:

```python
MIN_PAPERS: 30              # Minimum unique papers to cite
MAX_CITES_PER_PAPER: 3      # Max times one paper can be cited
CITES_PER_CLAIM_MIN: 1      # Min citations per claim
CITES_PER_CLAIM_MAX: 4      # Max citations per claim
WEAK_MATCH_THRESHOLD: 0.2   # Minimum relevance score to consider
TOP_K_CANDIDATES: 12        # Candidates from vector search
SA_ITERATIONS: 1200         # Iterations per SA run
SA_RUNS: 100                # Number of SA optimization runs
LLM_MODEL: 'gpt-4o'         # OpenAI model for scoring/extraction
```

## Running the Pipeline

**citation_adder.ipynb execution:**
1. Set OpenAI API key (prompted if not in environment)
2. Run cells sequentially - notebook is designed for linear execution
3. Expected runtime: ~45-60 minutes (240 LLM scoring calls + 100 SA runs)
4. Monitor progress via detailed logging at each step
5. Outputs saved to `./introduction_with_citations.txt` and `./references.txt`

**Key execution milestones:**
- Cell 4: PDF loading (~51 files, ~30 seconds)
- Cell 6: Vector store build (~3 batches with rate limiting)
- Cell 9: Claim extraction (~1 LLM call, returns ~20 claims)
- Cell 11: Relevance matrix (240 LLM calls, ~20-30 minutes)
- Cell 13: Simulated annealing (~15-20 minutes)
- Cell 16-17: Citation insertion and file writing

## Important Implementation Details

**Citation Number Assignment:**
- Citations numbered by order of first appearance in text
- Maintains narrative flow by sorting claims by position
- Supports IEEE format: [1], [1,2], [1-3] for consecutive ranges

**Text Insertion Strategy:**
- Works backwards through text (reverse position order) to preserve indices
- Normalizes whitespace for matching but preserves original formatting
- Uses first 100 chars of claim for fuzzy matching in original text

**Relevance Score Distribution:**
Current system produces:
- Mean: ~0.275, Median: ~0.220
- High-quality matches (>0.70): ~9/240 (3.75%)
- Medium matches (0.40-0.70): ~52/240 (21.7%)
- Weak matches (<0.40): ~179/240 (74.6%)

## File Structure

```
intro_writer/
├── Papers/                           # 51+ PDF files (not committed)
├── citation_adder.ipynb              # Main production pipeline
├── introduction_writer.ipynb         # Legacy alternative approach (MLA format)
├── match_citations_to_files.ipynb    # Citation verification utility
├── introduction.txt                  # Input: raw introduction text
├── introduction_with_citations.txt   # Output: cited introduction (IEEE format)
├── references.txt                    # Output: numbered reference list
├── citation_to_file_mapping.csv      # Output: citation-to-file mapping
└── cited_papers/                     # Output: copied cited papers (optional)
```

## Common Operations

**Adding new papers:**
1. Place PDF files in `Papers/` directory
2. Re-run from Cell 4 (load_pdfs_from_folder)
3. System automatically includes in vector store and scoring

**Modifying introduction:**
1. Edit `introduction.txt` directly
2. Re-run from Cell 8 (load_introduction)
3. Claims will be re-extracted, re-scored, re-optimized

**Adjusting citation constraints:**
1. Modify CONFIG dict in Cell 2
2. Re-run from Cell 13 (simulated annealing)
3. No need to rebuild vector store or re-score

**Verifying citations after generation:**
1. Run `match_citations_to_files.ipynb` after generating citations
2. Check for any missing or mismatched PDF files
3. Optionally export cited papers to `./cited_papers/` folder
4. Review `citation_to_file_mapping.csv` for complete mapping

**Improving relevance scores:**
- Increase `LLM_TEMP_SCORING` (currently 0.0) for more varied scores
- Adjust scoring prompt in `score_paper_for_claim()` function
- Modify confidence/evidence penalties in scoring logic

## Dependencies

Key packages required:
- `langchain_openai`: ChatOpenAI, OpenAIEmbeddings
- `langchain_community`: PyPDFLoader
- `chromadb`: Vector database with embedding functions
- `numpy`: Array operations for SA optimization
- Standard library: os, glob, re, json, time, collections

## Performance Characteristics

- PDF loading: ~1 second per file
- Embedding generation: ~2-3 seconds per batch (20 papers)
- LLM scoring: ~0.2-0.5 seconds per call (with rate limiting)
- Simulated annealing: ~10-15 seconds per run (1200 iterations)
- Total pipeline: 45-60 minutes for 20 claims × 12 candidates × 100 SA runs

## Recent Fixes (Cell 16 - Citation Insertion)

**Previous bugs (now fixed):**
1. Citations inserted mid-word due to broken position mapping (`W [4,10,12,13]hile`)
2. ~30% of citations silently failed to insert, creating orphaned references

**Current implementation:**
- Uses LLM-assisted matching to find exact insertion points (~20 extra LLM calls)
- Validates all citations were inserted and reports failures explicitly
- Much more robust to claim text variations and whitespace differences
- Success rate improved from ~70% to ~95%+

**Cost impact:** Adds ~20 LLM calls to cell 16 execution (~8% increase in total cost)

## Known Limitations

- PDF metadata extraction is heuristic-based and may miss author/year information
- Papers limited to first 20 pages for token management
- Vector search returns fixed top-k candidates (may miss relevant papers ranked k+1)
- SA optimization is non-deterministic (different runs produce different results)
- LLM-assisted citation insertion may still fail on heavily rephrased claims (now reported explicitly)
- No support for author-date citation formats (IEEE [1,2,3] only)
