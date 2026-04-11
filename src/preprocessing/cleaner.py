"""
src/preprocessing/cleaner.py

Responsibility: Clean raw page text before it is chunked.

Research PDFs often contain:
  - Running headers / footers  (page numbers, journal names)
  - Excessive whitespace / hyphenated line breaks
  - Reference-list noise at the end of papers
  - Ligature artefacts from PDF fonts (ﬁ → fi)

Cleaning here improves embedding quality and reduces hallucination caused by
the LLM reading garbled or redundant text.
"""

import re
import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ── Regex patterns ────────────────────────────────────────────────────────────

# Matches standalone page numbers on their own line (e.g. "  42  ")
_PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*$", re.MULTILINE)

# Merges hyphenated line-breaks common in PDF text extraction (e.g. "meth-\nod")
_HYPHEN_BREAK_RE = re.compile(r"-\n(\w)")

# Collapses multiple blank lines into a single blank line
_MULTI_BLANK_RE = re.compile(r"\n{3,}")

# Replaces common PDF ligature characters with their ASCII equivalents
_LIGATURES = {
    "\ufb01": "fi",  # ﬁ
    "\ufb02": "fl",  # ﬂ
    "\ufb00": "ff",  # ﬀ
    "\ufb03": "ffi", # ﬃ
    "\ufb04": "ffl", # ﬄ
}


def _fix_ligatures(text: str) -> str:
    """Replace Unicode ligature characters with plain ASCII equivalents."""
    for ligature, replacement in _LIGATURES.items():
        text = text.replace(ligature, replacement)
    return text


def _remove_page_numbers(text: str) -> str:
    """Strip lone numeric lines that are almost certainly page numbers."""
    return _PAGE_NUMBER_RE.sub("", text)


def _fix_hyphenated_breaks(text: str) -> str:
    """
    Merge words split across lines by PDF line-wrapping.
    Example: 'meth-\nod' → 'method'
    """
    return _HYPHEN_BREAK_RE.sub(r"\1", text)


def _collapse_whitespace(text: str) -> str:
    """Reduce multiple consecutive blank lines to one."""
    return _MULTI_BLANK_RE.sub("\n\n", text).strip()


def clean_text(text: str) -> str:
    """
    Apply the full cleaning pipeline to a single string.

    Pipeline order matters:
      1. Fix encoding artefacts (ligatures) first so regexes work correctly.
      2. Remove spurious page numbers.
      3. Re-join hyphenated words.
      4. Normalise whitespace.

    Args:
        text: Raw extracted page text.

    Returns:
        Cleaned text string.
    """
    text = _fix_ligatures(text)
    text = _remove_page_numbers(text)
    text = _fix_hyphenated_breaks(text)
    text = _collapse_whitespace(text)
    return text


def clean_documents(documents: List[Document]) -> List[Document]:
    """
    Clean the page_content of every Document in-place (returns same list).

    Skips and logs any Document where cleaning produces empty text
    (e.g. cover pages, blank pages).

    Args:
        documents: List of LangChain Documents with raw page_content.

    Returns:
        List of Documents with cleaned page_content (blanks removed).
    """
    cleaned = []
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        if doc.page_content.strip():          # keep non-empty pages only
            cleaned.append(doc)
        else:
            logger.debug(
                f"Skipping blank page {doc.metadata.get('page_number')} "
                f"from '{doc.metadata.get('paper_name')}'"
            )
    logger.info(f"Cleaned documents: {len(cleaned)} / {len(documents)} pages retained")
    return cleaned
