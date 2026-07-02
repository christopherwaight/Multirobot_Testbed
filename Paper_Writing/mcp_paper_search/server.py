"""
MCP server for research paper search.

Exposes the FAISS vector store as tools that Claude can call.
"""

import os
import sys
from mcp.server.fastmcp import FastMCP
from retrieval import PaperSearchEngine

# Initialize MCP server
mcp = FastMCP("paper-search")

# Load search engine once at startup
STORE_DIR = os.environ.get("PAPER_STORE_DIR", "./vector_store")
print(f"Initializing with vector store: {STORE_DIR}", file=sys.stderr)
engine = PaperSearchEngine(vector_store_dir=STORE_DIR)


@mcp.tool()
def search_papers(query: str, top_k: int = 8) -> str:
    """
    Search through a corpus of ~133 research papers using hybrid retrieval.

    Returns the most relevant passages with source information.
    Use this when you need to find information from the research paper database.

    The search uses:
    - Semantic vector search (embeddings)
    - Keyword matching (BM25)
    - Cross-encoder re-ranking for precision

    Args:
        query: Natural language search query (e.g., "gradient estimation for multirobot systems")
        top_k: Number of results to return (default 8, max 15)

    Returns:
        Formatted string with search results including source metadata
    """
    # Cap top_k to avoid overwhelming Claude's context
    top_k = min(top_k, 15)

    results = engine.search(query, top_k=top_k)

    # Format results for Claude
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"[Result {i}] (relevance score: {r['rerank_score']:.3f})\n"
            f"  Source: {r['title']}\n"
            f"  Section: {r.get('section', 'N/A')}\n"
            f"  File: {r['filename']}\n"
            f"  Content: {r['text']}\n"
        )

    return "\n".join(formatted)


@mcp.tool()
def list_papers() -> str:
    """
    List all research papers available in the database.

    Use this to see what papers are in the corpus before searching.
    Returns paper titles and filenames.

    Returns:
        Formatted list of all papers in the database
    """
    filenames = sorted(set(c["filename"] for c in engine.chunks))

    # Build title mapping
    titles = {}
    for c in engine.chunks:
        if c["filename"] not in titles:
            titles[c["filename"]] = c.get("title", "Unknown")

    lines = [f"Papers in database: {len(filenames)}\n"]
    for fn in filenames:
        lines.append(f"  • {titles.get(fn, 'Unknown')}")
        lines.append(f"    ({fn})")

    return "\n".join(lines)


@mcp.tool()
def paper_sections(filename: str) -> str:
    """
    Show the section headings of a specific paper.

    Use this to understand a paper's structure before searching for specific content.
    Helpful for targeted queries about particular sections.

    Args:
        filename: The PDF filename (e.g., "[32] Multirobot Symmetric Formations for Gradient and Hessian Estimation - Transactions - 2018.pdf")

    Returns:
        List of section headings in the paper
    """
    sections = []
    seen = set()

    for c in engine.chunks:
        if c["filename"] == filename and c.get("section"):
            if c["section"] not in seen:
                sections.append(c["section"])
                seen.add(c["section"])

    if not sections:
        return f"No sections found for '{filename}'.\nCheck the filename with list_papers() tool."

    lines = [f"Sections in {filename}:\n"]
    for s in sections:
        lines.append(f"  • {s}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="stdio")
