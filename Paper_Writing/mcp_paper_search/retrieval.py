"""
Retrieval engine for research paper search.
Extracted from RAG notebook for use by MCP server.
"""

import json
import sys
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


def _log(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


class PaperSearchEngine:
    """
    Loads the FAISS index and provides hybrid retrieval + re-ranking.

    This wraps the same logic as the RAG notebook's retrieve() function,
    but as a class so the MCP server can hold loaded models in memory.
    """

    def __init__(self,
                 vector_store_dir: str = "./vector_store",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """
        Initialize the search engine and load all models.

        Args:
            vector_store_dir: Path to directory containing faiss_index.bin and chunks_metadata.json
            embedding_model: Sentence transformer model name
            reranker_model: Cross-encoder model name for re-ranking
        """
        self.vector_store_dir = Path(vector_store_dir)

        _log("=" * 60)
        _log("Loading Paper Search Engine...")
        _log("=" * 60)

        # Load FAISS index
        _log(f"Loading FAISS index from {self.vector_store_dir}...")
        index_path = self.vector_store_dir / "faiss_index.bin"
        self.index = faiss.read_index(str(index_path))

        # Load chunk metadata
        _log("Loading chunk metadata...")
        metadata_path = self.vector_store_dir / "chunks_metadata.json"
        with open(metadata_path) as f:
            self.chunks = json.load(f)

        _log(f"Loaded {self.index.ntotal} vectors and {len(self.chunks)} chunks")

        # Load embedding model
        _log(f"Loading embedding model: {embedding_model}...")
        self.embed_model = SentenceTransformer(embedding_model)

        # Load re-ranker
        _log(f"Loading re-ranker: {reranker_model}...")
        self.reranker = CrossEncoder(reranker_model)

        # Build BM25 index
        _log("Building BM25 index...")
        tokenized_corpus = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        _log("=" * 60)
        _log("Search engine ready!")
        _log(f"  Papers indexed: {len(set(c['filename'] for c in self.chunks))}")
        _log(f"  Total chunks: {len(self.chunks)}")
        _log("=" * 60)

    def search(self, query: str, top_k: int = 8, pool_size: int = 24) -> list:
        """
        Hybrid retrieval + cross-encoder re-ranking.

        Three-stage pipeline:
        1. Vector + BM25 → RRF merge → pool_size candidates
        2. Cross-encoder re-ranks candidates
        3. Return top_k results

        Args:
            query: Search query (natural language)
            top_k: Number of final results to return
            pool_size: Number of candidates to pull before re-ranking

        Returns:
            List of dicts with keys: text, filename, title, section, rerank_score
        """
        # === Stage 1: Vector arm ===
        query_vec = self.embed_model.encode([query], normalize_embeddings=True).astype("float32")
        vec_scores, vec_indices = self.index.search(query_vec, pool_size)
        vec_results = [(int(idx), float(score)) for score, idx in zip(vec_scores[0], vec_indices[0])]

        # === Stage 1: BM25 arm ===
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top = np.argsort(bm25_scores)[-pool_size:][::-1]
        bm25_results = [(int(idx), float(bm25_scores[idx])) for idx in bm25_top]

        # === Stage 1: Reciprocal Rank Fusion ===
        rrf_k = 60
        fused = {}

        for rank, (idx, _) in enumerate(vec_results):
            fused[idx] = fused.get(idx, 0) + 1 / (rrf_k + rank + 1)

        for rank, (idx, _) in enumerate(bm25_results):
            fused[idx] = fused.get(idx, 0) + 1 / (rrf_k + rank + 1)

        # Take top pool_size candidates for re-ranking
        candidates = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:pool_size]

        # === Stage 2: Cross-encoder re-ranking ===
        pairs = [(query, self.chunks[idx]["text"]) for idx, _ in candidates]
        rerank_scores = self.reranker.predict(pairs)

        # Combine and sort by re-ranker score
        scored = list(zip([idx for idx, _ in candidates], rerank_scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for idx, score in scored[:top_k]:
            chunk = self.chunks[idx].copy()
            chunk["rerank_score"] = float(score)
            results.append(chunk)

        return results


if __name__ == "__main__":
    # Test the search engine
    import sys

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "gradient estimation multirobot formations"

    print(f"\nTest query: {query}\n")

    engine = PaperSearchEngine()
    results = engine.search(query, top_k=3)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for i, r in enumerate(results, 1):
        print(f"\n[Result {i}] (score: {r['rerank_score']:.4f})")
        print(f"  Title: {r['title']}")
        print(f"  Section: {r.get('section', 'N/A')}")
        print(f"  File: {r['filename']}")
        print(f"  Preview: {r['text'][:200]}...")
