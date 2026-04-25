"""
src/vectorstore/faiss_store.py

Responsibility: Build, persist, and load a FAISS vector index.

Why FAISS?
  - Runs 100% locally — no cloud credentials needed
  - Extremely fast similarity search (milliseconds for thousands of vectors)
  - LangChain has first-class FAISS support with metadata handling built in
  - Persists to disk → no re-embedding on app restart (huge time saver)

Index layout on disk (VECTORSTORE_DIR/):
  index.faiss  — raw FAISS binary index
  index.pkl    — Python pickle of the docstore + id mapping

The LangChain FAISS wrapper stores Document objects alongside vectors so
we get text + metadata back on every search hit without a separate lookup.
"""

import logging
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import VECTORSTORE_DIR
from src.embeddings.embedder import get_embedding_model

logger = logging.getLogger(__name__)

_INDEX_PATH = str(VECTORSTORE_DIR)  # FAISS save/load expects a string path


def build_vectorstore(chunks: List[Document]) -> FAISS:
    """
    Create a new FAISS index from a list of chunk Documents.

    Steps:
      1. Extract page_content strings from each Document.
      2. Batch-embed them using the embedding model.
      3. Store vectors + Document objects in a FAISS index.
      4. Persist the index to disk.

    Args:
        chunks: List of chunk Documents (with metadata).

    Returns:
        A ready-to-query FAISS vectorstore.

    Raises:
        ValueError: If chunks list is empty.
    """
    if not chunks:
        raise ValueError("Cannot build a vector store from an empty chunk list.")

    logger.info(f"Building FAISS index from {len(chunks)} chunks …")
    embedding_model = get_embedding_model()

    # FAISS.from_documents handles batching internally
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )

    # Persist so subsequent app runs skip re-embedding
    vectorstore.save_local(_INDEX_PATH)
    logger.info(f"FAISS index saved to {_INDEX_PATH}")
    return vectorstore


def load_vectorstore() -> FAISS | None:
    """
    Load a previously saved FAISS index from disk.

    Returns:
        FAISS vectorstore if the index exists, else None.
    """
    index_file = Path(_INDEX_PATH) / "index.faiss"
    if not index_file.exists():
        logger.info("No existing FAISS index found on disk.")
        return None

    logger.info(f"Loading FAISS index from {_INDEX_PATH}")
    embedding_model = get_embedding_model()
    vectorstore = FAISS.load_local(
        _INDEX_PATH,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,  # safe: we wrote this file ourselves
    )
    logger.info("FAISS index loaded successfully.")
    return vectorstore


def add_documents_to_store(
    vectorstore: FAISS, new_chunks: List[Document]
) -> FAISS:
    """
    Add new chunks to an existing FAISS index (incremental ingestion).

    Useful when the user uploads additional PDFs without clearing the store.

    Args:
        vectorstore : Existing FAISS index.
        new_chunks  : New chunk Documents to add.

    Returns:
        Updated FAISS vectorstore (same object, mutated in place).
    """
    if not new_chunks:
        return vectorstore

    logger.info(f"Adding {len(new_chunks)} new chunks to existing index …")
    vectorstore.add_documents(new_chunks)
    vectorstore.save_local(_INDEX_PATH)
    logger.info("Updated FAISS index saved.")
    return vectorstore


def clear_vectorstore() -> None:
    """
    Delete the persisted FAISS index files from disk.

    Called automatically when the pipeline detects a corrupted index
    (e.g. one built from HTML-contaminated text).
    Also called by the "Clear Knowledge Base" button in the UI.
    """
    import shutil
    index_dir = Path(_INDEX_PATH)
    if index_dir.exists():
        shutil.rmtree(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"FAISS index cleared: {_INDEX_PATH}")
    else:
        logger.info("No index directory found — nothing to clear.")