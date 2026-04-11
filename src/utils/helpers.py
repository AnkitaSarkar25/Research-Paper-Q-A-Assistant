"""
src/utils/helpers.py

Shared utility functions used across multiple modules.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import List

from langchain_core.documents import Document


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a clean format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def save_uploaded_file(uploaded_file, destination_dir: Path) -> Path:
    """
    Save a Streamlit UploadedFile object to disk and return its path.

    Streamlit's UploadedFile is a file-like object; we read its bytes
    and write them to the destination directory.

    Args:
        uploaded_file   : st.file_uploader result object.
        destination_dir : Directory to save the file into.

    Returns:
        Path to the saved file.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    dest_path = destination_dir / uploaded_file.name
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest_path


def clear_directory(directory: Path) -> None:
    """Remove and recreate a directory (used to reset the knowledge base)."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def count_tokens_approx(text: str) -> int:
    """
    Rough token count estimate: 1 token ≈ 4 characters (OpenAI/Google rule of thumb).
    Used for display purposes only.
    """
    return len(text) // 4


def format_sources_for_display(sources: List[dict]) -> str:
    """
    Format the list of source dicts into a readable markdown string.

    Args:
        sources : List of source dicts from generate_answer().

    Returns:
        Markdown-formatted string.
    """
    lines = []
    for i, s in enumerate(sources, 1):
        lines.append(
            f"**[{i}] {s['paper_name']}** — Page {s['page_number']}  \n"
            f"*Section: {s['section'] or 'General'}*  \n"
            f"Relevance score: `{s['score']}`  \n"
            f"> {s['excerpt']}"
        )
    return "\n\n---\n\n".join(lines)


def get_paper_stats(documents: List[Document]) -> dict:
    """
    Compute summary statistics over a list of chunk Documents.

    Returns dict with total chunks, unique papers, and per-paper chunk counts.
    """
    paper_counts: dict[str, int] = {}
    for doc in documents:
        name = doc.metadata.get("paper_name", "unknown")
        paper_counts[name] = paper_counts.get(name, 0) + 1

    return {
        "total_chunks":  len(documents),
        "unique_papers": len(paper_counts),
        "per_paper":     paper_counts,
    }
