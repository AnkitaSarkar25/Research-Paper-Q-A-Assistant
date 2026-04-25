"""
src/generation/generator.py

Responsibility: Build the RAG prompt and call Gemini Flash 2.5 to generate
a grounded, citation-backed answer.

Key design decisions to minimise hallucination:

1. Strict context-only instruction
   The system prompt explicitly forbids the LLM from using prior knowledge.
   Any claim must be traceable to the supplied context blocks.

2. Structured context blocks
   Each retrieved chunk is formatted as:
     [SOURCE: paper_name | Page: N | Chunk: id]
     <text>
   This makes it easy for the LLM to write precise citations.

3. Low temperature (0.2)
   Less creativity → more faithful reproduction of retrieved facts.

4. Explicit citation format
   The prompt asks for citations in the format [paper_name, p.N] so the
   parser in this module can extract them reliably.

5. "Answer ONLY if context is sufficient" instruction
   If the context doesn't contain the answer, the LLM says "I don't know"
   rather than hallucinating. This is the single most important anti-hallucination
   technique in RAG.
"""

import logging
import os
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import LLM_MODEL_NAME, MAX_OUTPUT_TOKENS, TEMPERATURE

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Research Paper Q&A Assistant. Your ONLY job is to answer
questions using the provided context excerpts from research papers.

RULES — follow them strictly:
1. Base your answer SOLELY on the [CONTEXT] provided below. Do NOT use any prior knowledge.
2. If the context does not contain enough information to answer, respond with:
   "I don't have enough information in the provided papers to answer this question."
3. Cite every factual claim using the format: [PaperName, p.PageNumber]
4. Be concise and precise. Avoid filler sentences.
5. If multiple papers discuss the same topic, synthesise their views and note agreements or disagreements.
6. Never fabricate statistics, names, or claims that are not in the context.

Output format:
**Answer:**
<your detailed answer with inline citations like [PaperName, p.3]>

**Key Citations:**
- [PaperName, p.N]: <one-line quote or paraphrase that directly supports the answer>
"""


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(chunks: List[Tuple[Document, float]]) -> str:
    """
    Format reranked chunks into a numbered context block for the prompt.

    Example output:
      [SOURCE 1: attention_is_all_you_need | Page: 3 | Score: 0.92]
      The model uses multi-head attention to …

    Args:
        chunks: List of (Document, rerank_score) from the reranker.

    Returns:
        Formatted context string to inject into the prompt.
    """
    blocks = []
    for i, (doc, score) in enumerate(chunks, start=1):
        meta    = doc.metadata
        header  = (
            f"[SOURCE {i}: {meta.get('paper_name', 'unknown')} | "
            f"Page: {meta.get('page_number', '?')} | "
            f"Score: {score:.2f}]"
        )
        blocks.append(f"{header}\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(blocks)


# ── LLM singleton ─────────────────────────────────────────────────────────────

_llm: ChatGoogleGenerativeAI | None = None

def get_llm() -> ChatGoogleGenerativeAI:
    """Lazily initialise and cache the Gemini LLM client."""
    global _llm
    if _llm is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY environment variable is not set. "
                "Get one at https://aistudio.google.com/app/apikey"
            )
        logger.info(f"Initialising LLM: {LLM_MODEL_NAME}")
        _llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            google_api_key=api_key,
            # max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE
        )
    return _llm


# ── Answer generation ─────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    reranked_chunks: List[Tuple[Document, float]],
) -> dict:
    """
    Run the full RAG generation step.

    Flow:
      1. Build structured context from reranked chunks.
      2. Compose system + human messages.
      3. Call Gemini Flash 2.5.
      4. Parse and return a structured response dict.

    Args:
        query           : User's original question.
        reranked_chunks : Top-k (Document, score) from the reranker.

    Returns:
        dict with keys:
          - answer      : str  — LLM's full response text
          - context_str : str  — the context block that was sent
          - sources     : List[dict] — citation metadata for the UI
    """
    llm = get_llm()

    context_str = build_context(reranked_chunks)

    human_message_content = (
        f"[CONTEXT]\n{context_str}\n\n"
        f"[QUESTION]\n{query}"
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_message_content),
    ]

    logger.info("Sending request to Gemini …")
    response = llm.invoke(messages)
    answer_text = response.content
    logger.info("Response received.")

    # Build structured source list for the UI
    sources = []
    seen_ids = set()
    for doc, score in reranked_chunks:
        cid = doc.metadata.get("chunk_id", "")
        if cid not in seen_ids:
            sources.append({
                "paper_name":  doc.metadata.get("paper_name", "unknown"),
                "page_number": doc.metadata.get("page_number", "?"),
                "section":     doc.metadata.get("section", ""),
                "excerpt":     doc.page_content[:300] + "…" if len(doc.page_content) > 300 else doc.page_content,
                "score":       round(score, 3),
            })
            seen_ids.add(cid)

    return {
        "answer":      answer_text,
        "context_str": context_str,
        "sources":     sources,
    }
