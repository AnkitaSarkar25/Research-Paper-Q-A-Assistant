"""
src/preprocessing/cleaner.py

Responsibility: Clean raw page text BEFORE it is chunked or embedded.

This is the PRIMARY defence against HTML contamination in the vector store.
Cleaning must happen here — at ingestion time — not at display time, because
once corrupted text is embedded into FAISS it cannot be fixed without
rebuilding the entire index.

Root cause of the HTML-in-citations bug:
  PyPDF sometimes extracts text that contains HTML markup sequences when a PDF
  was generated from an HTML source (e.g. browser-printed web pages, Jupyter
  notebooks exported to PDF, or papers downloaded via some academic portals).
  If this raw text reaches the FAISS index, every excerpt retrieved will
  contain raw HTML tags visible to the user.

Defence layers applied in order:
  1. HTML stripping       — remove ALL tags before anything else
  2. Ligature repair      — Unicode → ASCII
  3. Hyphen-break merge   — re-join split words
  4. Noise removal        — URLs, DOIs, ISSNs, watermarks
  5. References cutoff    — stop at the reference list
  6. Whitespace collapse  — normalise spacing
"""

import re
import logging
from html.parser import HTMLParser
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 1 — HTML STRIPPER  (runs first, always)
# ═══════════════════════════════════════════════════════════════════════════════

class _VisibleTextExtractor(HTMLParser):
    """
    stdlib html.parser subclass that collects only visible text nodes,
    discarding all tags, attributes, scripts, and style blocks.

    Why stdlib and not regex?
      - Handles malformed / truncated HTML correctly (tolerant parser)
      - No external dependencies
      - Correctly decodes HTML entities (&nbsp; → space, &amp; → &, etc.)
      - Handles nested tags and partial inputs safely
    """

    _DISCARD_TAGS = {"script", "style", "head", "meta", "link"}

    def __init__(self):
        super().__init__(convert_charrefs=True)  # auto-decode &nbsp; etc.
        self._parts: list[str] = []
        self._discard_depth: int = 0

    def handle_starttag(self, tag: str, attrs):
        if tag in self._DISCARD_TAGS:
            self._discard_depth += 1

    def handle_endtag(self, tag: str):
        if tag in self._DISCARD_TAGS and self._discard_depth > 0:
            self._discard_depth -= 1

    def handle_data(self, data: str):
        if self._discard_depth == 0 and data.strip():
            self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _strip_html(text: str) -> str:
    """
    Remove all HTML/XML markup from `text` and return plain prose.

    Always runs — even on text that "looks clean" — because HTML tags
    embedded in PDF-extracted text have no angle-bracket guarantee
    (some PDF renderers encode < as &#60; or &lt;).

    Three-pass approach for robustness:
      Pass 1: stdlib HTMLParser  — handles well-formed and malformed HTML
      Pass 2: Regex mop-up      — catches any fragment the parser missed
      Pass 3: Entity decode      — handle any surviving &amp; &nbsp; etc.
    """
    if not text:
        return text

    # Pass 1 — HTMLParser
    parser = _VisibleTextExtractor()
    try:
        parser.feed(text)
        text = parser.get_text()
    except Exception:
        # Parser raised (extremely rare) — fall through to regex
        pass

    # Pass 2 — Regex mop-up for any surviving tag fragments
    text = re.sub(r"<[^>]{0,200}>?", " ", text)

    # Pass 3 — Named and numeric HTML entities that survived
    text = re.sub(r"&[a-zA-Z]{2,8};", " ", text)
    text = re.sub(r"&#\d{1,6};",      " ", text)

    return text


def _is_html_contaminated(text: str) -> bool:
    """
    Heuristic: returns True if text looks like it contains significant
    HTML markup (not just a stray < from math notation).

    Used as a fast pre-check to log warnings during ingestion.
    """
    tag_count = len(re.findall(r"<[a-zA-Z/][^>]{0,80}>", text))
    return tag_count >= 2


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYERS 2-6 — PDF ARTEFACT CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

# Unicode ligature → ASCII
_LIGATURES: dict[str, str] = {
    "\ufb00": "ff",  "\ufb01": "fi",  "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl", "\ufb06": "st",
    "\u00e6": "ae",  "\u0153": "oe",
}

# Standalone page numbers on their own line
_PAGE_NUM_RE    = re.compile(r"^\s*\d+\s*$", re.MULTILINE)

# Hyphenated line-break  e.g. "meth-\nod" → "method"
_HYPHEN_BREAK_RE = re.compile(r"-\n(\w)")

# Multiple blank lines → single blank line
_MULTI_BLANK_RE  = re.compile(r"\n{3,}")


def _fix_ligatures(text: str) -> str:
    for lig, rep in _LIGATURES.items():
        text = text.replace(lig, rep)
    return text


def _remove_noise(text: str) -> str:
    """Remove URLs, DOIs, ISSNs, watermarks, and reference sections."""
    # URLs
    text = re.sub(r"https?://\S+", "", text)

    # DOI patterns  e.g.  doi: 10.1162/neco.2006...  or  10.1162/neco
    text = re.sub(r"\b(?:doi\s*[:.]?\s*)?10\.\d{4,}/\S+", "", text, flags=re.I)

    # ISSN / eISSN lines
    text = re.sub(r"e?-?issn\b.*", "", text, flags=re.I)

    # arXiv IDs  e.g.  arXiv:1412.3555
    text = re.sub(r"\barxiv\s*:\s*\d{4}\.\d+\b", "", text, flags=re.I)

    # "Downloaded from …" and "Terms and conditions" watermarks
    text = re.sub(r"downloaded from\b.*",      "", text, flags=re.I)
    text = re.sub(r"terms and conditions\b.*", "", text, flags=re.I)
    text = re.sub(r"all rights reserved\b.*",  "", text, flags=re.I)
    text = re.sub(r"how to cite this\b.*",     "", text, flags=re.I)

    # Cut at the References / Bibliography section
    # (everything after this is citation noise, not content)
    text = re.split(r"\n\s*(?:references|bibliography)\s*\n", text, flags=re.I)[0]

    return text


def clean_text(text: str) -> str:
    """
    Apply the full 6-layer cleaning pipeline to a single page string.

    Order is critical:
      1. Strip HTML first  — so later regexes operate on plain text
      2. Fix ligatures     — Unicode artefacts from font encoding
      3. Merge hyphen-breaks — re-join PDF line-split words
      4. Remove noise      — URLs, DOIs, watermarks, reference sections
      5. Remove lone page numbers
      6. Collapse whitespace

    Args:
        text: Raw text extracted from a PDF page.

    Returns:
        Cleaned plain-text string ready for chunking.
    """
    if not text:
        return text

    # Warn if we're about to strip significant HTML
    if _is_html_contaminated(text):
        logger.warning(
            "HTML markup detected in extracted page text — stripping. "
            "This usually means the PDF was generated from a web page. "
            "If you see garbled output, re-check the source PDF."
        )

    # Layer 1 — HTML stripping (always runs)
    text = _strip_html(text)

    # Layer 2 — Ligature repair
    text = _fix_ligatures(text)

    # Layer 3 — Hyphen-break merge
    text = _HYPHEN_BREAK_RE.sub(r"\1", text)

    # Layer 4 — Noise removal
    text = _remove_noise(text)

    # Layer 5 — Lone page-number lines
    text = _PAGE_NUM_RE.sub("", text)

    # Layer 6 — Whitespace collapse
    text = _MULTI_BLANK_RE.sub("\n\n", text).strip()

    return text


# ═══════════════════════════════════════════════════════════════════════════════
#  DOCUMENT-LEVEL ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def clean_documents(documents: List[Document]) -> List[Document]:
    """
    Clean the page_content of every Document and return only non-empty pages.

    Also rejects any document whose page_content is still HTML-like after
    cleaning (belt-and-braces guard against truly unparseable content).

    Args:
        documents: Page-level Documents from the PDF loader.

    Returns:
        List of Documents with clean plain-text page_content.
    """
    cleaned: List[Document] = []
    html_warned = 0

    for doc in documents:
        original = doc.page_content or ""

        if _is_html_contaminated(original):
            html_warned += 1

        doc.page_content = clean_text(original)

        if not doc.page_content.strip():
            logger.debug(
                f"Skipping blank page {doc.metadata.get('page_number')} "
                f"of '{doc.metadata.get('paper_name')}'"
            )
            continue

        # Final guard: if HTML tags still remain after cleaning, the page
        # is likely a template / UI frame — discard it entirely.
        if _is_html_contaminated(doc.page_content):
            logger.warning(
                f"Page {doc.metadata.get('page_number')} of "
                f"'{doc.metadata.get('paper_name')}' still contains HTML "
                f"after cleaning — skipping to protect index quality."
            )
            continue

        cleaned.append(doc)

    if html_warned:
        logger.warning(
            f"{html_warned} page(s) contained HTML markup. "
            "They have been stripped. If results look wrong, check the source PDFs."
        )

    logger.info(
        f"Cleaning complete: {len(cleaned)} / {len(documents)} pages retained."
    )
    return cleaned