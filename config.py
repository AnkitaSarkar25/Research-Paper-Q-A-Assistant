"""
config.py — Central configuration for the Research Paper Q&A Assistant.
All paths, model names, and hyperparameters live here so every module
imports from ONE place. Changing a value here propagates everywhere.
"""

import os
from pathlib import Path

# ── Project root (this file's parent directory) ─────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()

# ── Data directories ─────────────────────────────────────────────────────────
DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"          # uploaded PDFs land here
PROCESSED_DIR  = DATA_DIR / "processed"   # cleaned text / chunk JSONs

# ── FAISS vector store persistence ──────────────────────────────────────────
VECTORSTORE_DIR = PROCESSED_DIR / "vectorstore"

# ── Model settings ───────────────────────────────────────────────────────────
# LLM — Gemini 2.5 Flash (via google-generativeai / LangChain)
LLM_MODEL_NAME = "gemini-2.5-flash-preview-05-20"

# Embeddings — lightweight local model (no extra API quota needed)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # sentence-transformers

# ── Chunking hyperparameters ─────────────────────────────────────────────────
CHUNK_SIZE    = 800    # characters per chunk
CHUNK_OVERLAP = 150    # overlap between consecutive chunks

# ── Retrieval hyperparameters ─────────────────────────────────────────────────
TOP_K_RETRIEVE = 10   # how many chunks to fetch from FAISS
TOP_K_RERANK   = 5    # how many chunks to pass to the LLM after reranking

# ── BM25 hybrid search weight ─────────────────────────────────────────────────
# final_score = SEMANTIC_WEIGHT * semantic + (1 - SEMANTIC_WEIGHT) * bm25
SEMANTIC_WEIGHT = 0.7

# ── Generation settings ───────────────────────────────────────────────────────
MAX_OUTPUT_TOKENS = 1024
TEMPERATURE       = 0.2   # low → less hallucination

# ── Ensure directories exist at import time ───────────────────────────────────
for _d in [RAW_DIR, PROCESSED_DIR, VECTORSTORE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
