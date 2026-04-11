"""
src/ingestion/pdf_loader.py

Responsibility: Load raw PDF files and extract (page_text, metadata) pairs.

Why PyPDFLoader?
  - Part of LangChain's document loaders → output is List[Document]
  - Each Document already carries page_number in its metadata
  - Simple, battle-tested, handles most research PDFs

Design decision: We return LangChain Document objects (not plain dicts) so
the rest of the pipeline can stay idiomatic LangChain code.
"""

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf(pdf_path: str | Path) -> List[Document]:
    """
    Load a single PDF and return one Document per page.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        List of LangChain Documents. Each Document has:
            - page_content : raw text of that page
            - metadata     : { 'source': str, 'page': int, 'paper_name': str }

    Raises:
        FileNotFoundError: If the PDF does not exist.
        RuntimeError    : If PyPDF cannot parse the file.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Loading PDF: {pdf_path.name}")

    try:
        loader = PyPDFLoader(str(pdf_path))
        pages: List[Document] = loader.load()
    except Exception as exc:
        raise RuntimeError(f"Failed to parse {pdf_path.name}: {exc}") from exc

    # Enrich metadata with a human-readable paper_name (stem = filename minus ext)
    paper_name = pdf_path.stem
    for doc in pages:
        doc.metadata["paper_name"] = paper_name
        # PyPDFLoader sets metadata['page'] as 0-indexed → convert to 1-indexed
        doc.metadata["page_number"] = doc.metadata.get("page", 0) + 1

    logger.info(f"  → {len(pages)} pages extracted from '{paper_name}'")
    return pages


def load_multiple_pdfs(pdf_paths: List[str | Path]) -> List[Document]:
    """
    Convenience wrapper: load and concatenate pages from multiple PDFs.

    Args:
        pdf_paths: List of paths to PDF files.

    Returns:
        Flat list of all Documents across all PDFs.
    """
    all_docs: List[Document] = []
    for path in pdf_paths:
        try:
            docs = load_pdf(path)
            all_docs.extend(docs)
        except (FileNotFoundError, RuntimeError) as exc:
            # Log the error but keep processing remaining PDFs
            logger.error(f"Skipping {path}: {exc}")
    logger.info(f"Total pages loaded: {len(all_docs)}")
    return all_docs
