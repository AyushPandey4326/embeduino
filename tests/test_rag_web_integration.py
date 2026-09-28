"""Integration tests for RAG pipeline with realistic web search fusion."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from embeduino.config import Settings
from embeduino.rag import ask
from embeduino.web_search import WebHit


@pytest.fixture
def mock_local_chunks():
    """Realistic local chunks that would be in vector and BM25 results."""
    return [
        {
            "id": "analog_read_desc_1",
            "text": "Reads the value from an analog pin. Arduino boards contain analog inputs which can read voltage from 0 to 5 volts.",
            "metadata": {
                "source": "analog_read.md",
                "heading_path": "analogRead() > Description",
                "chunk_type": "local",
            },
            "score": 0.42,
        },
        {
            "id": "digital_write_desc_1",
            "text": "Write a HIGH or LOW value to a digital pin. If the pin has been configured as OUTPUT with pinMode(), its voltage will be set to the corresponding value.",
            "metadata": {
                "source": "digital_write.md",
                "heading_path": "digitalWrite() > Description",
                "chunk_type": "local",
            },
            "score": 0.38,
        },
        {
            "id": "pin_mode_desc_1",
            "text": "Configures the specified pin to behave either as an input or an output. Pins can be configured as INPUT, OUTPUT, or INPUT_PULLUP.",
            "metadata": {
                "source": "pin_mode.md",
                "heading_path": "pinMode() > Description",
                "chunk_type": "local",
            },
            "score": 0.35,
        },
        {
            "id": "pwm_analog_write_1",
            "text": "Writes an analog value (PWM wave) to a pin. Can be used to light an LED at varying brightnesses or drive a motor at various speeds.",
            "metadata": {
                "source": "pwm_analog_write.md",
                "heading_path": "analogWrite() > Description",
                "chunk_type": "local",
            },
            "score": 0.33,
        },
        {
            "id": "serial_begin_1",
            "text": "Sets the data rate in bits per second (baud) for serial data transmission. For communicating with Serial Monitor, use 9600.",
            "metadata": {
                "source": "serial.md",
                "heading_path": "Serial.begin() > Description",
                "chunk_type": "local",
            },
            "score": 0.30,
        },
    ]


@pytest.fixture
def mock_web_hits():
    """Web hits from SerpApi fixture as WebHit objects."""
    fixture_path = Path(__file__).parent / "fixtures" / "serpapi_uno_r4_response.json"
    fixture_data = json.loads(fixture_path.read_text())
    
    from embeduino.web_search import SerpApiSearcher
    import tempfile
    
    searcher = SerpApiSearcher(
        api_key="test",
        cache_dir=Path(tempfile.mkdtemp()),
        max_calls_per_run=10,
        timeout_s=90,
    )
    hits = searcher._parse_results(fixture_data, num_results=5)
    return hits  # Return WebHit objects, not dicts


@pytest.fixture
def mock_settings(tmp_path):
    """Test settings without API keys."""
    return Settings(
        openai_api_key="",
        serpapi_api_key="test_key",
        generation_backend="extractive",
        top_k=4,
        min_score=0.3,
        web_mode="auto",
        web_num=5,
        web_sites=["docs.arduino.cc", "github.com", "forum.arduino.cc"],
        serp_cache=tmp_path / ".serp_cache",
        serp_max_calls=10,
        serp_timeout_s=90,
        rrf_k=60,
    )


def test_web_chunks_reach_citations_realistic_fusion(
    mock_local_chunks, mock_web_hits, mock_settings
):
    """
    Regression test: Web chunks must survive RRF fusion with realistic local lists.
    
    Root cause: Web chunks only appeared in web_ranked list (1/(k+rank) ≈ 0.016),
    while local chunks appeared in both vector_ranked and bm25_ranked (2/(k+rank) ≈ 0.032).
    This structural bias meant web chunks never made top_k=4.
    
    Fix: Run rerank and BM25 over combined local+web pool so web chunks appear in both lists.
    """
    question = "What are the pin specifications for Arduino UNO R4 WiFi?"
    
    # Mock VectorStore
    mock_store = MagicMock()
    mock_store.query.return_value = mock_local_chunks
    mock_store.get_all_chunks.return_value = mock_local_chunks
    
    # Mock SerpApiSearcher to return web hits
    with patch("embeduino.rag.SerpApiSearcher") as mock_searcher_class:
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = (
            mock_web_hits,  # Already WebHit objects
            True,  # credit_used
            "api_success"
        )
        mock_searcher_class.return_value = mock_searcher
        
        result = ask(question, mock_store, mock_settings)
    
    # Assertions
    assert result.answer != "I don't know", "Should answer with web chunks available"
    assert not result.weak_retrieval, "Should not be weak with web results"
    assert result.web_search_used, "Web search should have been triggered"
    
    # At least one web chunk must be in citations
    web_citations = [c for c in result.citations if c["origin"] == "web"]
    assert len(web_citations) >= 1, f"Expected ≥1 web citation, got {len(web_citations)}"
    
    # Web citations must have URLs
    for cite in web_citations:
        assert cite["url"], f"Web citation {cite['id']} missing URL"
        assert "forum.arduino.cc" in cite["url"], "URL should be from trusted site"
    
    # At least one web chunk in retrieved results
    web_retrieved = [h for h in result.retrieved if h.get("metadata", {}).get("chunk_type") == "web"]
    assert len(web_retrieved) >= 1, f"Expected ≥1 web chunk in retrieved, got {len(web_retrieved)}"
    
    # Web chunks should have competitive RRF scores now (not stuck at 0.016)
    for h in web_retrieved:
        rrf = h.get("rrf_score", 0.0)
        assert rrf > 0.02, f"Web chunk {h['id']} has low RRF={rrf:.4f}, structural bias not fixed"


def test_local_only_question_still_works(mock_local_chunks, mock_settings):
    """
    Local questions like 'What does digitalWrite do?' should still answer from local docs.
    """
    question = "What does digitalWrite do?"
    
    # Mock VectorStore with digitalWrite chunks ranked high
    digital_write_chunks = [
        {
            "id": "digital_write_desc_1",
            "text": "Write a HIGH or LOW value to a digital pin. If the pin has been configured as OUTPUT with pinMode(), its voltage will be set to the corresponding value: 5V (or 3.3V) for HIGH, 0V for LOW.",
            "metadata": {
                "source": "digital_write.md",
                "heading_path": "digitalWrite() > Description",
                "chunk_type": "local",
            },
            "score": 0.85,
        },
        {
            "id": "digital_write_syntax_1",
            "text": "digitalWrite(pin, value)\npin: the Arduino pin number.\nvalue: HIGH or LOW.",
            "metadata": {
                "source": "digital_write.md",
                "heading_path": "digitalWrite() > Syntax",
                "chunk_type": "local",
            },
            "score": 0.72,
        },
    ] + mock_local_chunks[:3]
    
    mock_store = MagicMock()
    mock_store.query.return_value = digital_write_chunks
    mock_store.get_all_chunks.return_value = digital_write_chunks
    
    # Web search should not trigger for well-covered local question
    with patch("embeduino.rag.should_trigger_web_search", return_value=False):
        result = ask(question, mock_store, mock_settings)
    
    # Assertions
    assert result.answer != "I don't know", "Should answer from local docs"
    assert not result.weak_retrieval, "Should not be weak"
    assert not result.web_search_used, "Web search should not trigger"
    
    # All citations should be local
    local_citations = [c for c in result.citations if c["origin"] == "local"]
    assert len(local_citations) >= 1, "Should have local citations"
    
    # Answer should mention digitalWrite
    assert "digital" in result.answer.lower() or "write" in result.answer.lower()


def test_web_chunks_pass_weak_gate_with_lexical_overlap(
    mock_local_chunks, mock_web_hits, mock_settings
):
    """
    Web chunks with decent lexical overlap should pass the weak gate.
    """
    question = "What are the technical specifications for the Arduino UNO R4 WiFi board?"
    
    mock_store = MagicMock()
    mock_store.query.return_value = mock_local_chunks
    mock_store.get_all_chunks.return_value = mock_local_chunks
    
    with patch("embeduino.rag.SerpApiSearcher") as mock_searcher_class:
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = (
            mock_web_hits,
            True,
            "api_success"
        )
        mock_searcher_class.return_value = mock_searcher
        
        result = ask(question, mock_store, mock_settings)
    
    # Should not be weak with web results that have lexical overlap
    assert not result.weak_retrieval, "Web chunks with lexical overlap should pass weak gate"
    assert result.web_search_used
    
    # Check that web chunks are in top results
    top_web = [h for h in result.retrieved[:4] if h.get("metadata", {}).get("chunk_type") == "web"]
    assert len(top_web) >= 1, "At least one web chunk should be in top 4"


def test_extractive_answer_uses_web_chunk_as_primary(
    mock_local_chunks, mock_web_hits, mock_settings
):
    """
    Extractive answer should be able to use a web chunk as primary source.
    """
    question = "Where can I find the pinout diagram for Arduino UNO R4 WiFi?"
    
    mock_store = MagicMock()
    # Return low-relevance local chunks
    mock_store.query.return_value = [
        {**chunk, "score": 0.25} for chunk in mock_local_chunks
    ]
    mock_store.get_all_chunks.return_value = mock_local_chunks
    
    with patch("embeduino.rag.SerpApiSearcher") as mock_searcher_class:
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = (
            mock_web_hits,
            True,
            "api_success"
        )
        mock_searcher_class.return_value = mock_searcher
        
        result = ask(question, mock_store, mock_settings)
    
    # Answer should reference web source
    assert "forum.arduino.cc" in result.answer or "web" in result.answer.lower()
    
    # Top citation should be web
    assert result.citations[0]["origin"] == "web", "Top citation should be from web when web is most relevant"
    assert result.citations[0]["url"], "Top web citation must have URL"


def test_zero_web_results_handled_gracefully(mock_local_chunks, mock_settings):
    """
    When web search returns 0 results, should still work with local chunks.
    """
    question = "What does analogRead do?"
    
    mock_store = MagicMock()
    mock_store.query.return_value = mock_local_chunks
    mock_store.get_all_chunks.return_value = mock_local_chunks
    
    with patch("embeduino.rag.SerpApiSearcher") as mock_searcher_class:
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = ([], True, "api_success")  # 0 results
        mock_searcher_class.return_value = mock_searcher
        
        result = ask(question, mock_store, mock_settings, web_mode_override="always")
    
    # Should still answer from local
    assert result.answer != "I don't know"
    assert result.web_search_used
    
    # All citations should be local
    web_citations = [c for c in result.citations if c["origin"] == "web"]
    assert len(web_citations) == 0, "Should have no web citations when 0 web results"
