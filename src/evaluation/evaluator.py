"""
src/evaluation/evaluator.py

Responsibility: Basic quality evaluation of the RAG pipeline outputs.

Why evaluation matters (even for a fresher project):
  RAG systems can silently degrade when:
    - Retrieved chunks are irrelevant (bad retrieval)
    - The LLM ignores the context (hallucination)
    - The same chunk dominates every answer (diversity failure)

  Showing an evaluator — even a simple one — signals ML engineering maturity.

Metrics implemented (all heuristic, no ground-truth labels needed):

1. Context Utilisation Score
   Checks what fraction of the answer's sentences contain evidence from
   the retrieved context. High score → LLM is grounded.

2. Retrieval Relevance Score
   Measures lexical overlap between the query and each retrieved chunk.
   Proxy for "did we retrieve topically relevant chunks?"

3. Source Diversity Score
   Checks how many different papers appear in the top-k sources.
   Low diversity (all from one paper) may indicate retrieval bias.

4. Answer Confidence Flags
   Detects hedging language ("I think", "possibly", "I don't know") that
   suggests the LLM was uncertain — often a sign of under-retrieval.
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# Hedging phrases that indicate LLM uncertainty
_HEDGES = [
    "i think", "i believe", "possibly", "perhaps", "might be",
    "i'm not sure", "i don't know", "i cannot determine", "unclear",
    "not mentioned", "not provided", "don't have enough",
]


def context_utilisation_score(answer: str, context_str: str) -> float:
    """
    Estimate how much of the answer is grounded in the retrieved context.

    Method:
      For each sentence in the answer, check if any 5-word n-gram from it
      appears in the context string. Sentences with a hit are "grounded".

    Score = grounded_sentences / total_sentences (0.0 – 1.0)

    Args:
        answer      : LLM's answer text.
        context_str : The context block that was passed to the LLM.

    Returns:
        Float between 0 (no grounding) and 1 (fully grounded).
    """
    sentences = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 10]
    if not sentences:
        return 0.0

    context_lower = context_str.lower()
    grounded = 0

    for sentence in sentences:
        words = sentence.lower().split()
        # Check 5-word windows
        found = False
        for i in range(len(words) - 4):
            ngram = " ".join(words[i:i+5])
            if ngram in context_lower:
                found = True
                break
        if found:
            grounded += 1

    score = grounded / len(sentences)
    return round(score, 3)


def retrieval_relevance_score(query: str, sources: List[dict]) -> float:
    """
    Measure average keyword overlap between the query and retrieved excerpts.

    Higher score means retrieved chunks are more topically relevant.

    Args:
        query   : User's question.
        sources : List of source dicts from generate_answer().

    Returns:
        Float between 0 and 1.
    """
    if not sources:
        return 0.0

    query_words = set(query.lower().split())
    # Remove common stop words so they don't inflate the score
    stop_words  = {"the", "a", "an", "is", "in", "of", "and", "or", "to", "what", "how", "why"}
    query_words -= stop_words

    if not query_words:
        return 0.0

    scores = []
    for source in sources:
        excerpt_words = set(source["excerpt"].lower().split())
        overlap = len(query_words & excerpt_words) / len(query_words)
        scores.append(overlap)

    return round(sum(scores) / len(scores), 3)


def source_diversity_score(sources: List[dict]) -> float:
    """
    Measure how many distinct papers appear in the retrieved sources.

    Score = unique_papers / total_sources
    Score of 1.0 means every source is from a different paper (ideal for
    multi-paper synthesis). Score of 0.2 means all sources are from one paper.

    Args:
        sources : List of source dicts.

    Returns:
        Float between 0 and 1.
    """
    if not sources:
        return 0.0
    unique_papers = len({s["paper_name"] for s in sources})
    return round(unique_papers / len(sources), 3)


def detect_uncertainty_flags(answer: str) -> List[str]:
    """
    Find hedging phrases in the answer that suggest low LLM confidence.

    Args:
        answer : LLM's answer text.

    Returns:
        List of hedging phrases found (empty list = confident answer).
    """
    answer_lower = answer.lower()
    return [h for h in _HEDGES if h in answer_lower]


def evaluate_response(
    query: str,
    answer: str,
    context_str: str,
    sources: List[dict],
) -> dict:
    """
    Run all evaluation metrics and return a summary dict.

    Args:
        query       : User's original question.
        answer      : LLM's answer.
        context_str : Full context block passed to the LLM.
        sources     : Retrieved source dicts.

    Returns:
        Dict with metric names → values, suitable for displaying in the UI.
    """
    ctx_score       = context_utilisation_score(answer, context_str)
    rel_score       = retrieval_relevance_score(query, sources)
    div_score       = source_diversity_score(sources)
    uncertainty     = detect_uncertainty_flags(answer)

    # Overall quality signal (simple average of the three numeric metrics)
    overall = round((ctx_score + rel_score + div_score) / 3, 3)

    evaluation = {
        "context_utilisation":  ctx_score,
        "retrieval_relevance":  rel_score,
        "source_diversity":     div_score,
        "overall_quality":      overall,
        "uncertainty_flags":    uncertainty,
        "is_uncertain":         len(uncertainty) > 0,
        "num_sources_used":     len(sources),
        "num_unique_papers":    len({s["paper_name"] for s in sources}),
    }

    logger.info(f"Evaluation: {evaluation}")
    return evaluation
