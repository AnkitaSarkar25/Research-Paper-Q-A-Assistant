"""
src/embeddings/embedder.py

Responsibility: Convert text chunks into dense vector representations.

Model choice — sentence-transformers/all-MiniLM-L6-v2:
  - Runs locally → no API calls, no quota, no latency overhead
  - 384-dimensional vectors → compact FAISS index
  - Trained on semantic similarity → great for Q&A retrieval
  - ~22 MB download, cached after first run

Why not Gemini embeddings?
  Gemini embedding API is excellent but adds per-call latency and quota risk
  during demos. For a fresher project, a local model is more reliable and still
  achieves strong semantic quality. The pipeline is abstracted so swapping is
  a one-line config change.

Caching strategy:
  sentence-transformers automatically caches the model in ~/.cache/huggingface.
  We wrap the LangChain HuggingFaceEmbeddings class which handles batching
  internally, keeping memory usage low.
"""

import logging
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

# Module-level singleton — model is loaded once and reused across calls.
# This is important: loading a transformer model takes ~2-3 seconds.
_embedding_model: HuggingFaceEmbeddings | None = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Return (or lazily initialise) the singleton embedding model.

    Using a module-level singleton means Streamlit won't reload the model
    on every re-run. Combined with st.cache_resource this is very fast.
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},         # change to 'cuda' if GPU available
            encode_kwargs={"normalize_embeddings": True},  # cosine similarity ready
        )
        logger.info("Embedding model loaded successfully.")
    return _embedding_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of raw strings and return their vectors.

    Used for embedding the user's query at retrieval time.

    Args:
        texts: Plain strings to embed.

    Returns:
        List of float vectors, one per input string.
    """
    model = get_embedding_model()
    return model.embed_documents(texts)


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string.

    sentence-transformers has a separate embed_query path that may apply
    query-specific prompt prefixes for asymmetric retrieval models.
    """
    model = get_embedding_model()
    return model.embed_query(query)
