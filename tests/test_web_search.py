"""Tests for SerpApi web search integration."""

import json
from pathlib import Path

import pytest

from embeduino.web_search import SerpApiSearcher, should_trigger_web_search


@pytest.fixture
def mock_serp_response():
    """Sample SerpApi organic_results response."""
    return {
        "organic_results": [
            {
                "position": 1,
                "title": "Arduino UNO R4 WiFi Specs",
                "link": "https://docs.arduino.cc/hardware/uno-r4-wifi",
                "snippet": "The UNO R4 WiFi features a Renesas RA4M1 chip and ESP32-S3 WiFi module.",
            },
            {
                "position": 2,
                "title": "Getting Started with UNO R4",
                "link": "https://docs.arduino.cc/tutorials/uno-r4-wifi/",
                "snippet": "Learn how to set up your Arduino UNO R4 WiFi board.",
            },
            {
                "position": 3,
                "title": "UNO R4 WiFi GitHub",
                "link": "https://github.com/arduino/ArduinoCore-renesas",
                "snippet": "Arduino core for Renesas RA4M1 (UNO R4)",
            },
        ]
    }


def test_serpapi_parse_results(mock_serp_response, tmp_path):
    """SerpApi should parse organic results into WebHits."""
    searcher = SerpApiSearcher(
        api_key="test_key",
        cache_dir=tmp_path,
        max_calls_per_run=10,
    )
    
    hits = searcher._parse_results(mock_serp_response)
    
    assert len(hits) == 3
    assert hits[0].title == "Arduino UNO R4 WiFi Specs"
    assert hits[0].url == "https://docs.arduino.cc/hardware/uno-r4-wifi"
    assert "Renesas RA4M1" in hits[0].snippet
    assert hits[0].id.startswith("web:")


def test_serpapi_cache_read_write(mock_serp_response, tmp_path):
    """Cache should store and retrieve results."""
    searcher = SerpApiSearcher(
        api_key="",
        cache_dir=tmp_path,
        max_calls_per_run=10,
    )
    
    cache_key = searcher._cache_key("test query", {})
    
    searcher._write_cache(cache_key, mock_serp_response)
    
    cached = searcher._read_cache(cache_key)
    assert cached == mock_serp_response
    
    missing = searcher._read_cache("nonexistent_key")
    assert missing is None


def test_serpapi_no_api_key(tmp_path):
    """Search without API key should return empty results."""
    searcher = SerpApiSearcher(
        api_key="",
        cache_dir=tmp_path,
        max_calls_per_run=10,
    )
    
    hits, credit_used = searcher.search(
        "test question",
        trusted_sites=["docs.arduino.cc"],
        num_results=5,
    )
    
    assert hits == []
    assert credit_used is False


def test_should_trigger_web_search_modes():
    """Web search trigger should respect mode settings."""
    local_hits = [{"id": "test", "score": 0.8}]
    
    assert should_trigger_web_search("test", local_hits, False, "always") is True
    
    assert should_trigger_web_search("test", local_hits, False, "off") is False
    
    assert should_trigger_web_search("test", local_hits, True, "auto") is True
    
    assert should_trigger_web_search("test", local_hits, False, "auto") is False


def test_should_trigger_web_search_new_topics():
    """Web search should trigger for new board/library mentions."""
    local_hits = [{"id": "test", "score": 0.8}]
    
    questions_should_trigger = [
        "How do I use Arduino UNO R4 WiFi?",
        "What pins are on the Nano ESP32?",
        "How to use the Nano 33 BLE Sense?",
        "Where can I find the MKR WiFi library?",
    ]
    
    for q in questions_should_trigger:
        assert should_trigger_web_search(q, local_hits, False, "auto") is True
    
    assert should_trigger_web_search(
        "What does digitalWrite do?", local_hits, False, "auto"
    ) is False


def test_webhit_to_dict():
    """WebHit should convert to retrievable dict format."""
    from embeduino.web_search import WebHit
    
    hit = WebHit(
        id="web:abc123",
        text="Arduino UNO R4\\n\\nGreat board",
        url="https://docs.arduino.cc/uno-r4",
        title="Arduino UNO R4",
        position=1,
        snippet="Great board",
    )
    
    d = hit.to_dict()
    
    assert d["id"] == "web:abc123"
    assert d["text"] == "Arduino UNO R4\\n\\nGreat board"
    assert d["metadata"]["url"] == "https://docs.arduino.cc/uno-r4"
    assert d["metadata"]["chunk_type"] == "web"
    assert d["metadata"]["title"] == "Arduino UNO R4"


def test_serpapi_max_calls(mock_serp_response, tmp_path, monkeypatch):
    """Searcher should respect max_calls limit."""
    searcher = SerpApiSearcher(
        api_key="fake_key",
        cache_dir=tmp_path,
        max_calls_per_run=2,
    )
    
    searcher.calls_made = 2
    
    hits, credit_used = searcher.search(
        "test", ["docs.arduino.cc"], num_results=5
    )
    
    assert hits == []
    assert credit_used is False
