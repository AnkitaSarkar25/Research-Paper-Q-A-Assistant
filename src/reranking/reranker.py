"""
src/reranking/reranker.py

Responsibility: Rerank the top-k retrieved chunks to surface the most
relevant ones before passing them to the LLM.

Why rerank?
  First-stage retrieval (FAISS / BM25) optimises for coverage — it casts a
  wide net. But the LLM context window is limited and noisy context degrades
  answer quality. Reranking is a precision step: it re-scores only the small
  candidate set with a finer signal.

Two reranking strategies implemented:

1. Cross-encoder (local, default)
   A cross-encoder model takes (query, passage) as a joint input and outputs
   a relevance score. This is far more accurate than bi-encoder (embedding)
   similarity because the model sees both texts together.
   Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~85 MB, runs on CPU)

2. Simple lexical overlap (fallback)
   If the cross-encoder model is unavailable, fall back to a keyword-overlap
   heuristic: count query words found in the chunk (normalised by chunk length).
   Adds zero extra dependencies.
"""

import logging
from typing import List, Tuple

from langchain_core.documents import Document

from config import TOP_K_RERANK

logger = logging.getLogger(__name__)


# ── Cross-encoder reranking ──────────────────────────────────────────────────

_cross_encoder = None  # lazy-loaded singleton

def _get_cross_encoder():
    """Lazy-load the cross-encoder model (downloads once, then cached)."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading cross-encoder model …")
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("Cross-encoder loaded.")
        except Exception as exc:
            logger.warning(f"Cross-encoder unavailable ({exc}). Will use lexical fallback.")
            _cross_encoder = "unavailable"
    return _cross_encoder


def _lexical_overlap_score(query: str, text: str) -> float:
    """
    Fallback reranker: fraction of unique query words found in the chunk.

    Simple but surprisingly effective for keyword-rich technical queries.
    """
    query_words = set(query.lower().split())
    text_words  = set(text.lower().split())
    if not query_words:
        return 0.0
    return len(query_words & text_words) / len(query_words)


def rerank(
    query: str,
    candidates: List[Tuple[Document, float]],
    top_k: int = TOP_K_RERANK,
) -> List[Tuple[Document, float]]:
    """
    Rerank candidate (Document, first-stage-score) pairs.

    Tries the cross-encoder first; falls back to lexical overlap.

    Args:
        query      : Original user question.
        candidates : List of (Document, score) from retrieval.
        top_k      : How many to return after reranking.

    Returns:
        Top-k (Document, rerank_score) tuples, best-first.
    """
    if not candidates:
        return []

    encoder = _get_cross_encoder()

    if encoder != "unavailable":
        # Cross-encoder: score each (query, chunk_text) pair
        pairs  = [(query, doc.page_content) for doc, _ in candidates]
        scores = encoder.predict(pairs)          # returns numpy array
        ranked = sorted(
            zip([doc for doc, _ in candidates], scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
    else:
        # Lexical fallback
        ranked = sorted(
            [(doc, _lexical_overlap_score(query, doc.page_content))
             for doc, _ in candidates],
            key=lambda x: x[1],
            reverse=True,
        )

    top = ranked[:top_k]
    logger.info(f"Reranking: {len(candidates)} candidates → {len(top)} selected")
    return top
