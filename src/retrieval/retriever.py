"""
src/retrieval/retriever.py

Responsibility: Retrieve the most relevant chunks for a user query.

Two retrieval modes are implemented:

1. Semantic search (FAISS similarity)
   - Query is embedded → cosine similarity against all chunk vectors
   - Captures meaning/intent even when exact words differ
   - Fast: O(n) with FAISS's flat index, or sub-linear with IVF

2. Hybrid search (Semantic + BM25)
   - BM25 is a classical TF-IDF-based term-frequency ranker
   - Complementary to semantic search: catches exact keyword matches
     that embeddings sometimes miss (e.g. model names, paper IDs)
   - Final score = SEMANTIC_WEIGHT * semantic + (1-SEMANTIC_WEIGHT) * BM25
   - Produces more robust retrieval especially for technical queries

Why hybrid for a fresher project?
  It's a genuine ML system design concept that interviewers love. It shows
  you know the limitations of purely semantic retrieval.
"""

import logging
from typing import List, Tuple

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from config import TOP_K_RETRIEVE, SEMANTIC_WEIGHT
from src.embeddings.embedder import embed_query

logger = logging.getLogger(__name__)


# ── Semantic retrieval ────────────────────────────────────────────────────────

def semantic_search(
    vectorstore: FAISS,
    query: str,
    top_k: int = TOP_K_RETRIEVE,
) -> List[Tuple[Document, float]]:
    """
    Run pure semantic (embedding) search against the FAISS index.

    Args:
        vectorstore : Loaded FAISS index.
        query       : User's question.
        top_k       : Number of results to return.

    Returns:
        List of (Document, similarity_score) tuples, ranked best-first.
    """
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    # FAISS returns L2 distance for inner-product normalised vectors → lower=better
    # Convert to a 0-1 similarity: sim = 1 / (1 + distance)
    return [(doc, float(1 / (1 + score))) for doc, score in results]


# ── BM25 retrieval ────────────────────────────────────────────────────────────

class BM25Retriever:
    """
    Lightweight BM25 retriever wrapping the rank_bm25 library.

    Must be rebuilt whenever the corpus changes (i.e. when new PDFs are added).
    In a production system this would be persisted; here we rebuild from the
    in-memory chunk list passed at construction time.
    """

    def __init__(self, documents: List[Document]):
        """
        Build a BM25 index from the given documents.

        Tokenisation: simple whitespace split (good enough for retrieval).
        """
        self._docs = documents
        tokenised_corpus = [doc.page_content.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenised_corpus)
        logger.info(f"BM25 index built over {len(documents)} chunks.")

    def search(self, query: str, top_k: int = TOP_K_RETRIEVE) -> List[Tuple[Document, float]]:
        """
        Return top-k documents ranked by BM25 score.

        Scores are normalised to [0, 1] by dividing by the max score.
        """
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        # Normalise
        max_score = scores.max() if scores.max() > 0 else 1.0
        norm_scores = scores / max_score

        # Pair each doc with its score and sort descending
        ranked = sorted(
            zip(self._docs, norm_scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]


# ── Hybrid retrieval ──────────────────────────────────────────────────────────

def hybrid_search(
    vectorstore: FAISS,
    bm25_retriever: BM25Retriever,
    query: str,
    top_k: int = TOP_K_RETRIEVE,
) -> List[Tuple[Document, float]]:
    """
    Combine semantic and BM25 scores with a weighted sum (Reciprocal Rank Fusion
    style but score-based for simplicity).

    Steps:
      1. Get top-k results from semantic search (normalised scores).
      2. Get top-k results from BM25 (normalised scores).
      3. Build a union of retrieved chunk IDs.
      4. For each chunk, compute: w*semantic + (1-w)*bm25.
      5. Return top-k by combined score.

    Args:
        vectorstore    : FAISS index.
        bm25_retriever : Pre-built BM25 index.
        query          : User's question.
        top_k          : Number of final results.

    Returns:
        List of (Document, combined_score) tuples, best-first.
    """
    semantic_results = semantic_search(vectorstore, query, top_k=top_k)
    bm25_results     = bm25_retriever.search(query, top_k=top_k)

    # Build score maps keyed by chunk_id
    sem_map  = {doc.metadata["chunk_id"]: (doc, score) for doc, score in semantic_results}
    bm25_map = {doc.metadata["chunk_id"]: (doc, score) for doc, score in bm25_results}

    all_ids = set(sem_map) | set(bm25_map)

    combined: List[Tuple[Document, float]] = []
    for cid in all_ids:
        doc        = (sem_map.get(cid) or bm25_map.get(cid))[0]
        sem_score  = sem_map[cid][1]  if cid in sem_map  else 0.0
        bm25_score = bm25_map[cid][1] if cid in bm25_map else 0.0
        final      = SEMANTIC_WEIGHT * sem_score + (1 - SEMANTIC_WEIGHT) * bm25_score
        combined.append((doc, final))

    combined.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Hybrid search returned {len(combined[:top_k])} candidates")
    return combined[:top_k]
