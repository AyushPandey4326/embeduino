"""BM25 retrieval and Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenization for BM25."""
    import re
    return [t.lower() for t in re.findall(r"\w+", text) if len(t) > 1]


def bm25_rank(
    question: str,
    chunks: List[Dict[str, Any]],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Dict[str, Any]]:
    """
    Rank chunks by BM25 score.
    
    Returns list of dicts with 'id', 'text', 'metadata', 'bm25_score'.
    """
    if not chunks:
        return []
    
    query_tokens = tokenize(question)
    if not query_tokens:
        return []
    
    corpus = [tokenize(c.get("text", "")) for c in chunks]
    N = len(corpus)
    avgdl = sum(len(d) for d in corpus) / N if N > 0 else 0
    
    df: Dict[str, int] = {}
    for doc in corpus:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    
    idf: Dict[str, float] = {}
    for term in query_tokens:
        n = df.get(term, 0)
        idf[term] = math.log((N - n + 0.5) / (n + 0.5) + 1.0)
    
    scores = []
    for i, doc_tokens in enumerate(corpus):
        doc_len = len(doc_tokens)
        freq = Counter(doc_tokens)
        score = 0.0
        for term in query_tokens:
            if term not in freq:
                continue
            tf = freq[term]
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avgdl)) if avgdl > 0 else 1.0
            score += idf.get(term, 0.0) * (numerator / denominator)
        
        result = dict(chunks[i])
        result["bm25_score"] = score
        scores.append(result)
    
    scores.sort(key=lambda x: x["bm25_score"], reverse=True)
    return scores


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    id_key: str = "id",
) -> List[Dict[str, Any]]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.
    
    Each list should be ordered by relevance (best first).
    
    RRF score for an item = sum over all lists of 1/(k + rank_in_list).
    
    Returns: merged list ordered by RRF score, with 'rrf_score' added.
    """
    item_map: Dict[str, Dict[str, Any]] = {}
    rrf_scores: Dict[str, float] = {}
    
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            item_id = item.get(id_key)
            if item_id is None:
                continue
            
            if item_id not in item_map:
                item_map[item_id] = dict(item)
                rrf_scores[item_id] = 0.0
            
            rrf_scores[item_id] += 1.0 / (k + rank)
    
    fused = []
    for item_id, item in item_map.items():
        item["rrf_score"] = rrf_scores[item_id]
        fused.append(item)
    
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused
