"""
src/pipeline.py

Responsibility: Orchestrate the full RAG pipeline end-to-end.

This module is the single entry point that Streamlit calls.
It wires together all the individual modules:

  PDF upload
     ↓
  Ingestion (PyPDF)
     ↓
  Preprocessing (cleaning)
     ↓
  Chunking (section-aware + fixed-size fallback)
     ↓
  Embedding (sentence-transformers)
     ↓
  FAISS Vector Store
     ↓
  [At query time]
  Hybrid Retrieval (FAISS semantic + BM25)
     ↓
  Cross-encoder Reranking
     ↓
  Context Building
     ↓
  Gemini Flash 2.5 Generation
     ↓
  Evaluation metrics
     ↓
  Structured response → Streamlit UI

Design principle: Each step is a pure function/class call. No business logic
lives in this file — it just composes the other modules.
"""

import logging
from pathlib import Path
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import TOP_K_RETRIEVE, TOP_K_RERANK, RAW_DIR
from src.ingestion.pdf_loader import load_multiple_pdfs
from src.preprocessing.cleaner import clean_documents
from src.chunking.chunker import chunk_documents
from src.vectorstore.faiss_store import build_vectorstore, load_vectorstore, add_documents_to_store
from src.retrieval.retriever import hybrid_search, semantic_search, BM25Retriever
from src.reranking.reranker import rerank
from src.generation.generator import generate_answer
from src.evaluation.evaluator import evaluate_response
from src.utils.helpers import get_paper_stats

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Stateful RAG pipeline that holds the vector store and BM25 index in memory.

    Streamlit stores this object in st.session_state so it survives re-runs
    without rebuilding the index from scratch.

    Attributes:
        vectorstore     : FAISS index (None until PDFs are ingested).
        bm25_retriever  : BM25Retriever instance (None until PDFs are ingested).
        all_chunks      : All chunk Documents currently in the index.
        ingested_papers : Set of paper names already processed.
    """

    def __init__(self):
        self.vectorstore: FAISS | None       = None
        self.bm25_retriever: BM25Retriever | None = None
        self.all_chunks: List[Document]      = []
        self.ingested_papers: set[str]       = set()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_pdfs(self, pdf_paths: List[Path]) -> dict:
        """
        Process a list of PDF files and add them to the knowledge base.

        Skips papers that have already been ingested (idempotent).

        Args:
            pdf_paths : Paths to PDF files saved on disk.

        Returns:
            Dict with ingestion statistics.
        """
        # Filter to only new papers
        new_paths = [p for p in pdf_paths if Path(p).stem not in self.ingested_papers]

        if not new_paths:
            logger.info("All uploaded PDFs already ingested — skipping.")
            return {"new_chunks": 0, "message": "Already indexed."}

        logger.info(f"Ingesting {len(new_paths)} new PDF(s) …")

        # ── Step 1: Load ──────────────────────────────────────────────────────
        pages = load_multiple_pdfs(new_paths)

        # ── Step 2: Clean ─────────────────────────────────────────────────────
        clean_pages = clean_documents(pages)

        # ── Step 3: Chunk ─────────────────────────────────────────────────────
        new_chunks = chunk_documents(clean_pages)

        if not new_chunks:
            return {"new_chunks": 0, "message": "No text could be extracted."}

        # ── Step 4: Embed + Store ─────────────────────────────────────────────
        if self.vectorstore is None:
            self.vectorstore = build_vectorstore(new_chunks)
        else:
            self.vectorstore = add_documents_to_store(self.vectorstore, new_chunks)

        # ── Step 5: Update BM25 (rebuild over full corpus) ───────────────────
        self.all_chunks.extend(new_chunks)
        self.bm25_retriever = BM25Retriever(self.all_chunks)

        # Track which papers are indexed
        for path in new_paths:
            self.ingested_papers.add(Path(path).stem)

        stats = get_paper_stats(self.all_chunks)
        logger.info(f"Ingestion complete. {stats}")
        return {"new_chunks": len(new_chunks), "stats": stats}

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, user_question: str) -> dict:
        """
        Answer a question using the full RAG pipeline.

        Args:
            user_question : The user's raw question string.

        Returns:
            Dict with keys: answer, sources, evaluation, context_str.

        Raises:
            RuntimeError: If no papers have been ingested yet.
        """
        if self.vectorstore is None or self.bm25_retriever is None:
            raise RuntimeError(
                "No documents ingested yet. Please upload at least one PDF."
            )

        # ── Step 1: Hybrid retrieval ──────────────────────────────────────────
        candidates: List[Tuple[Document, float]] = hybrid_search(
            vectorstore=self.vectorstore,
            bm25_retriever=self.bm25_retriever,
            query=user_question,
            top_k=TOP_K_RETRIEVE,
        )

        # ── Step 2: Rerank ────────────────────────────────────────────────────
        top_chunks = rerank(
            query=user_question,
            candidates=candidates,
            top_k=TOP_K_RERANK,
        )

        # ── Step 3: Generate ──────────────────────────────────────────────────
        result = generate_answer(
            query=user_question,
            reranked_chunks=top_chunks,
        )

        # ── Step 4: Evaluate ──────────────────────────────────────────────────
        eval_metrics = evaluate_response(
            query=user_question,
            answer=result["answer"],
            context_str=result["context_str"],
            sources=result["sources"],
        )

        return {
            "answer":     result["answer"],
            "sources":    result["sources"],
            "evaluation": eval_metrics,
            "context_str": result["context_str"],
        }

    # ── Status ────────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        """Return True if the pipeline has at least one paper indexed."""
        return self.vectorstore is not None and len(self.all_chunks) > 0

    def get_stats(self) -> dict:
        """Return current knowledge base statistics."""
        return get_paper_stats(self.all_chunks) if self.all_chunks else {}

    def reset(self) -> None:
        """Clear the knowledge base and start fresh."""
        self.vectorstore      = None
        self.bm25_retriever   = None
        self.all_chunks       = []
        self.ingested_papers  = set()
        logger.info("Pipeline reset.")


def _index_is_corrupted(chunks: List[Document]) -> bool:
    """
    Scan a sample of loaded chunks for HTML contamination.

    If any chunk's page_content contains 2+ HTML tags, the index was built
    from unclean text and must be discarded and rebuilt.

    We sample up to 50 chunks (not all, to keep startup fast).
    Returns True if the index is corrupted.
    """
    import re
    sample = chunks[:50]
    tag_re = re.compile(r"<[a-zA-Z/][^>]{0,80}>")
    for doc in sample:
        content = doc.page_content or ""
        if len(tag_re.findall(content)) >= 2:
            logger.warning(
                f"Corrupted chunk detected in saved index — "
                f"paper='{doc.metadata.get('paper_name')}' "
                f"page={doc.metadata.get('page_number')}: "
                f"content starts with: {content[:120]!r}"
            )
            return True
    return False


def try_load_existing_index() -> "RAGPipeline":
    """
    Attempt to restore a pipeline from a persisted FAISS index.

    Includes an integrity check: if the saved index contains HTML-
    contaminated chunks (built before the cleaner was fixed), the index
    is automatically discarded and the user is prompted to re-index.
    This prevents the HTML-in-citations bug from persisting across sessions.

    Returns:
        Initialised RAGPipeline (may be empty if no valid saved index found).
    """
    from src.vectorstore.faiss_store import clear_vectorstore  # imported lazily

    pipeline = RAGPipeline()
    existing_store = load_vectorstore()

    if existing_store is None:
        logger.info("No saved index found — starting fresh.")
        return pipeline

    # Reconstruct chunk list for validation + BM25
    all_chunks = list(existing_store.docstore._dict.values())

    # ── Integrity check ───────────────────────────────────────────────────────
    if _index_is_corrupted(all_chunks):
        logger.warning(
            "Saved FAISS index contains HTML-contaminated chunks. "
            "Discarding stale index — user must re-upload PDFs."
        )
        try:
            clear_vectorstore()
        except Exception as exc:
            logger.error(f"Failed to clear corrupted index: {exc}")
        # Return empty pipeline — the Streamlit UI will show the upload prompt
        pipeline._index_was_corrupt = True   # flag so UI can show a notice
        return pipeline

    # ── Healthy index — restore normally ──────────────────────────────────────
    pipeline.vectorstore = existing_store
    pipeline.all_chunks  = all_chunks
    pipeline.bm25_retriever = BM25Retriever(all_chunks)
    pipeline.ingested_papers = {
        doc.metadata.get("paper_name", "") for doc in all_chunks
    }
    logger.info(
        f"Restored pipeline: {len(all_chunks)} chunks "
        f"from {len(pipeline.ingested_papers)} paper(s) — index is clean."
    )
    return pipeline