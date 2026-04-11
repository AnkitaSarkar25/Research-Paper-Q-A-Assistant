"""
src/chunking/chunker.py

Responsibility: Split cleaned Document pages into smaller, overlapping chunks.

Why chunk?
  LLMs have limited context windows, and embeddings work best on focused pieces
  of text (~200-800 tokens). Chunking lets us retrieve only the relevant slice
  of a long paper instead of the whole thing.

Strategy used (two-tier):
  1. Section-aware chunking (preferred)
     Split on section headings first (e.g. "1. Introduction", "Abstract",
     "Conclusion"). This keeps semantically coherent text together.

  2. Fixed-size fallback (RecursiveCharacterTextSplitter)
     If a section is still too large, recursively split on paragraph, sentence,
     and word boundaries to keep chunks within CHUNK_SIZE characters.

Each chunk carries rich metadata so citations are precise:
  { paper_name, chunk_id, page_number, section (if detected) }
"""

import re
import logging
import uuid
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# ── Section heading detection ─────────────────────────────────────────────────
# Matches common research paper section headers:
#   "1. Introduction", "2.3 Related Work", "Abstract", "Conclusion", etc.
_SECTION_RE = re.compile(
    r"^(?:\d+\.?\d*\.?\s+)?[A-Z][A-Za-z\s\-]{2,50}$",
    re.MULTILINE,
)


def _detect_sections(text: str) -> List[tuple[str, str]]:
    """
    Split text into (section_title, section_body) pairs.

    Falls back to [('', full_text)] if no headings are detected.
    """
    matches = list(_SECTION_RE.finditer(text))
    if len(matches) < 2:
        # Not enough section markers → treat whole text as one unnamed section
        return [("", text)]

    sections = []
    for i, match in enumerate(matches):
        title = match.group().strip()
        start = match.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body  = text[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


def _make_chunk(
    text: str,
    paper_name: str,
    page_number: int,
    section: str,
    index: int,
) -> Document:
    """
    Wrap a text snippet in a LangChain Document with full metadata.

    chunk_id is a deterministic string combining paper, page, section, and
    index so it can be used as a stable key in the vector store.
    """
    chunk_id = f"{paper_name}__p{page_number}__s{section[:20]}__c{index}"
    return Document(
        page_content=text,
        metadata={
            "paper_name":  paper_name,
            "page_number": page_number,
            "section":     section,
            "chunk_id":    chunk_id,
        },
    )


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Convert a list of page-level Documents into a list of chunk-level Documents.

    Pipeline:
      For each page:
        1. Attempt section-aware splitting.
        2. For each section, apply RecursiveCharacterTextSplitter.
        3. Attach metadata to every resulting chunk.

    Args:
        documents: Cleaned, page-level Documents.

    Returns:
        Flat list of chunk Documents ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # priority order
    )

    all_chunks: List[Document] = []
    global_chunk_index = 0

    for doc in documents:
        paper_name  = doc.metadata.get("paper_name", "unknown")
        page_number = doc.metadata.get("page_number", 0)
        text        = doc.page_content

        # Step 1: section-aware split
        sections = _detect_sections(text)

        for section_title, section_body in sections:
            # Step 2: fixed-size split within the section
            sub_texts = splitter.split_text(section_body)

            for sub_text in sub_texts:
                if not sub_text.strip():
                    continue
                chunk = _make_chunk(
                    text=sub_text,
                    paper_name=paper_name,
                    page_number=page_number,
                    section=section_title,
                    index=global_chunk_index,
                )
                all_chunks.append(chunk)
                global_chunk_index += 1

    logger.info(
        f"Chunking complete: {len(documents)} pages → {len(all_chunks)} chunks"
    )
    return all_chunks
