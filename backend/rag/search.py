"""
RAG Search — query ChromaDB for LeetCode problems.
Uses top-5 retrieval + title-based re-ranking to improve accuracy.
"""

import os
import json
import re
from backend.rag.chromadb_setup import get_collection

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))


def _title_similarity_boost(query: str, title: str) -> float:
    """
    Give a boost score if the query closely matches the problem title.
    This re-ranks results to prefer exact or near-exact title matches.
    """
    query_lower = query.lower().strip()
    title_lower = title.lower().strip()

    # Exact match
    if query_lower == title_lower:
        return 0.5

    # Query is a substring of title or vice versa
    if query_lower in title_lower or title_lower in query_lower:
        return 0.3

    # Word overlap ratio
    query_words = set(re.findall(r'\w+', query_lower))
    title_words = set(re.findall(r'\w+', title_lower))
    if not query_words or not title_words:
        return 0.0
    overlap = len(query_words & title_words)
    total = len(query_words | title_words)
    return (overlap / total) * 0.25


def search_problem(query: str) -> tuple[dict | None, float]:
    """
    Search ChromaDB for a LeetCode problem matching the query.
    Retrieves top 5 results and re-ranks using title similarity.

    Args:
        query: Clean search string (2-5 words)

    Returns:
        Tuple of (problem_metadata, confidence_score).
        If no match, returns (None, 0.0).
    """
    collection = get_collection()

    # Retrieve top 5 candidates for re-ranking
    results = collection.query(
        query_texts=[query],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    if not results["ids"][0]:
        return None, 0.0

    # Re-rank: combine L2 similarity with title match boost
    best_problem = None
    best_score = 0.0

    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        base_score = 1 / (1 + distance)
        title = results["metadatas"][0][i].get("title", "")
        boost = _title_similarity_boost(query, title)
        combined_score = base_score + boost

        if combined_score > best_score:
            best_score = combined_score
            best_problem = results["metadatas"][0][i]

    if not best_problem:
        return None, 0.0

    # Deserialize JSON string fields back to their original types
    for field in ["topics", "examples", "constraints", "hints"]:
        if field in best_problem and isinstance(best_problem[field], str):
            try:
                best_problem[field] = json.loads(best_problem[field])
            except (json.JSONDecodeError, TypeError):
                pass

    return best_problem, best_score
