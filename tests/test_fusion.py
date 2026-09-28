"""Tests for BM25 and Reciprocal Rank Fusion."""

import pytest

from embeduino.fusion import bm25_rank, reciprocal_rank_fusion


def test_bm25_rank_basic():
    """BM25 should rank documents by keyword relevance."""
    chunks = [
        {
            "id": "chunk1",
            "text": "digitalWrite sets a pin HIGH or LOW",
            "metadata": {},
        },
        {
            "id": "chunk2",
            "text": "analogRead reads analog input values",
            "metadata": {},
        },
        {
            "id": "chunk3",
            "text": "digitalWrite is used for digital output",
            "metadata": {},
        },
    ]
    
    question = "How do I use digitalWrite?"
    ranked = bm25_rank(question, chunks)
    
    assert len(ranked) == 3
    assert all("bm25_score" in r for r in ranked)
    
    top_ids = [r["id"] for r in ranked[:2]]
    assert "chunk1" in top_ids or "chunk3" in top_ids


def test_bm25_rank_empty():
    """BM25 should handle empty chunks gracefully."""
    assert bm25_rank("test", []) == []
    assert bm25_rank("", [{"id": "x", "text": "foo"}]) == []


def test_reciprocal_rank_fusion_basic():
    """RRF should merge ranked lists by reciprocal rank."""
    list1 = [
        {"id": "a", "score": 0.9},
        {"id": "b", "score": 0.7},
        {"id": "c", "score": 0.5},
    ]
    list2 = [
        {"id": "c", "score": 0.95},
        {"id": "a", "score": 0.8},
        {"id": "d", "score": 0.6},
    ]
    
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    
    assert len(fused) == 4
    assert all("rrf_score" in item for item in fused)
    
    assert fused[0]["id"] in ["a", "c"]
    
    ids = [item["id"] for item in fused]
    assert set(ids) == {"a", "b", "c", "d"}


def test_reciprocal_rank_fusion_single_list():
    """RRF should work with a single list."""
    list1 = [
        {"id": "x", "score": 1.0},
        {"id": "y", "score": 0.5},
    ]
    
    fused = reciprocal_rank_fusion([list1], k=60)
    
    assert len(fused) == 2
    assert fused[0]["id"] == "x"
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


def test_reciprocal_rank_fusion_empty():
    """RRF should handle empty lists."""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_reciprocal_rank_fusion_k_parameter():
    """RRF k parameter should affect scores."""
    list1 = [{"id": "a"}, {"id": "b"}]
    
    fused_k10 = reciprocal_rank_fusion([list1], k=10)
    fused_k100 = reciprocal_rank_fusion([list1], k=100)
    
    assert fused_k10[0]["rrf_score"] > fused_k100[0]["rrf_score"]
