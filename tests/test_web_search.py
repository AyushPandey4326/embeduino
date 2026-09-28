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


def test_serpapi_parse_results(mock_serp_response, tmp_path, monkeypatch):
    """SerpApi should parse organic results into WebHits, slicing locally."""
    # Mock _fetch_page to return None (use snippets only for this test)
    def mock_fetch(self, url):
        return None
    monkeypatch.setattr("embeduino.web_search.SerpApiSearcher._fetch_page", mock_fetch)
    
    searcher = SerpApiSearcher(
        api_key="test_key",
        cache_dir=tmp_path,
        max_calls_per_run=10,
        timeout_s=90,
    )
    
    question = "What are the pin specifications for Arduino UNO R4 WiFi?"
    hits = searcher._parse_results(mock_serp_response, num_results=2, question=question)
    
    assert len(hits) == 2
    assert hits[0].title == "Arduino UNO R4 WiFi Specs"
    assert hits[0].url == "https://docs.arduino.cc/hardware/uno-r4-wifi"
    assert "Renesas RA4M1" in hits[0].snippet
    assert hits[0].id.startswith("web:")
    
    hits_all = searcher._parse_results(mock_serp_response, num_results=100, question=question)
    assert len(hits_all) == 3


def test_serpapi_cache_read_write(mock_serp_response, tmp_path):
    """Cache should store and retrieve results."""
    searcher = SerpApiSearcher(
        api_key="",
        cache_dir=tmp_path,
        max_calls_per_run=10,
        timeout_s=90,
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
        timeout_s=90,
    )
    
    hits, credits_used, status = searcher.search(
        "test question",
        trusted_sites=["docs.arduino.cc"],
        num_results=5,
    )
    
    assert hits == []
    assert credits_used == 0
    assert status == "no_key"


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
    assert d["score"] == 0.5  # default score


def test_serpapi_max_calls(mock_serp_response, tmp_path, monkeypatch):
    """Searcher should respect max_calls limit."""
    searcher = SerpApiSearcher(
        api_key="fake_key",
        cache_dir=tmp_path,
        max_calls_per_run=2,
        timeout_s=90,
    )
    
    searcher.calls_made = 2
    
    hits, credits_used, status = searcher.search(
        "test", ["docs.arduino.cc"], num_results=5
    )
    
    assert hits == []
    assert credits_used == 0
    assert status == "max_calls"


def test_web_chunks_survive_to_answer_context(tmp_path, monkeypatch):
    """Regression test: web chunks with URLs must reach final answer context."""
    import json
    from pathlib import Path
    
    fixture_path = Path(__file__).parent / "fixtures" / "serpapi_uno_r4_response.json"
    fixture_data = json.loads(fixture_path.read_text())
    
    # Mock HTTP to return fixture
    def mock_get(url, params=None, timeout=None, **kwargs):
        class MockResponse:
            def raise_for_status(self):
                pass
            def json(self):
                return fixture_data
        return MockResponse()
    
    monkeypatch.setattr("httpx.get", mock_get)
    
    # Mock _fetch_page
    def mock_fetch(self, url):
        return None
    monkeypatch.setattr("embeduino.web_search.SerpApiSearcher._fetch_page", mock_fetch)
    
    searcher = SerpApiSearcher(
        api_key="test_key",
        cache_dir=tmp_path,
        max_calls_per_run=10,
        timeout_s=90,
    )
    
    hits, credits_used, status = searcher.search(
        "What are the pin specifications for Arduino UNO R4 WiFi?",
        trusted_sites=["docs.arduino.cc", "github.com", "forum.arduino.cc"],
        num_results=5,
        use_async=False,  # Sync to simplify mocking
    )
    
    assert status == "api_success"
    assert credits_used >= 1
    assert len(hits) >= 1
    assert all(hit.url for hit in hits)
    assert all(hit.title for hit in hits)
    
    # All should be forum.arduino.cc from the fixture (after domain filtering)
    for hit in hits:
        assert "forum.arduino.cc" in hit.url


def test_keyword_query_building():
    """Keyword extraction should be simple for product queries."""
    from embeduino.web_search import _build_keyword_query
    
    # Product + spec question → simple query
    q1 = "What are the pin specifications for Arduino UNO R4 WiFi?"
    k1 = _build_keyword_query(q1)
    assert k1 == '"UNO R4 WiFi" specifications'
    
    # Product without spec → just product
    q2 = "How do I use digitalWrite with ESP32?"
    k2 = _build_keyword_query(q2)
    assert k2 == '"ESP32"'
    
    # Product + datasheet → specifications
    q3 = "Nano 33 BLE datasheet"
    k3 = _build_keyword_query(q3)
    assert '"Nano 33 BLE"' in k3
    assert 'specifications' in k3
    
    # No product → keyword extraction
    q4 = "How to use analogRead function?"
    k4 = _build_keyword_query(q4)
    assert "analogRead" in k4 or "analog" in k4


def test_no_num_parameter_in_request(tmp_path, monkeypatch):
    """Request should not include num parameter."""
    request_params = []
    
    def mock_get(url, params=None, timeout=None):
        request_params.append(params.copy() if params else {})
        
        class MockResponse:
            def raise_for_status(self):
                pass
            def json(self):
                return {
                    "search_metadata": {"id": "test_id", "status": "Success"},
                    "organic_results": []
                }
        
        return MockResponse()
    
    monkeypatch.setattr("httpx.get", mock_get)
    
    searcher = SerpApiSearcher(
        api_key="fake_key",
        cache_dir=tmp_path,
        max_calls_per_run=10,
        timeout_s=90,
    )
    
    hits, credits_used, status = searcher.search(
        "test query",
        trusted_sites=["docs.arduino.cc"],
        num_results=5,
        use_async=False,
    )
    
    assert len(request_params) > 0
    for params in request_params:
        assert "num" not in params


def test_async_polling_success(tmp_path, monkeypatch):
    """Async mode should poll until Success status."""
    call_count = [0]
    
    def mock_get(url, params=None, timeout=None, **kwargs):
        call_count[0] += 1
        
        class MockResponse:
            def raise_for_status(self):
                pass
            def json(self):
                if "searches/" in url and call_count[0] == 2:
                    return {
                        "search_metadata": {"status": "Processing"},
                        "organic_results": []
                    }
                elif "searches/" in url:
                    return {
                        "search_metadata": {"status": "Success"},
                        "organic_results": [
                            {"position": 1, "title": "Test", "link": "https://docs.arduino.cc/test", "snippet": "snippet"}
                        ]
                    }
                else:
                    return {
                        "search_metadata": {"id": "test_search_id"},
                    }
        
        return MockResponse()
    
    monkeypatch.setattr("httpx.get", mock_get)
    monkeypatch.setattr("time.sleep", lambda x: None)
    
    # Mock _fetch_page to return None (use snippets only)
    def mock_fetch(self, url):
        return None
    monkeypatch.setattr("embeduino.web_search.SerpApiSearcher._fetch_page", mock_fetch)
    
    searcher = SerpApiSearcher(
        api_key="fake_key",
        cache_dir=tmp_path,
        max_calls_per_run=10,
        timeout_s=90,
    )
    
    hits, credits_used, status = searcher.search(
        "test query",
        trusted_sites=["docs.arduino.cc"],
        num_results=5,
        use_async=True,
    )
    
    assert status == "api_success"
    assert credits_used >= 1
    assert len(hits) == 1
    assert call_count[0] >= 3


def test_async_overall_timeout(tmp_path, monkeypatch):
    """Async polling should timeout and return timeout status."""
    def mock_get(url, params=None, timeout=None):
        class MockResponse:
            def raise_for_status(self):
                pass
            def json(self):
                if "searches/" in url:
                    return {"search_metadata": {"status": "Processing"}}
                else:
                    return {"search_metadata": {"id": "test_id"}}
        return MockResponse()
    
    monkeypatch.setattr("httpx.get", mock_get)
    monkeypatch.setattr("time.sleep", lambda x: None)
    
    def mock_time():
        mock_time.count += 1
        return mock_time.count * 50
    mock_time.count = 0
    
    monkeypatch.setattr("time.time", mock_time)
    
    searcher = SerpApiSearcher(
        api_key="fake_key",
        cache_dir=tmp_path,
        max_calls_per_run=10,
        timeout_s=5,
    )
    
    hits, credits_used, status = searcher.search(
        "test query",
        trusted_sites=["docs.arduino.cc"],
        num_results=5,
        use_async=True,
    )
    
    assert status == "timeout"
    assert credits_used >= 1
    assert hits == []


def test_api_failure_status(tmp_path, monkeypatch):
    """Failed API call should return api_failed status."""
    def mock_get(*args, **kwargs):
        import httpx
        raise httpx.TimeoutException("Mock timeout")
    
    monkeypatch.setattr("httpx.get", mock_get)
    
    searcher = SerpApiSearcher(
        api_key="fake_key",
        cache_dir=tmp_path,
        max_calls_per_run=10,
        timeout_s=90,
    )
    
    hits, credits_used, status = searcher.search(
        "test", ["docs.arduino.cc"], num_results=5, use_async=False
    )
    
    assert hits == []
    assert credits_used == 0
    assert status == "api_failed"


def test_domain_filtering_rejects_untrusted(tmp_path, monkeypatch):
    """Results from untrusted domains should be filtered out."""
    mock_response = {
        "organic_results": [
            {"position": 1, "title": "Reddit Post", "link": "https://www.reddit.com/r/arduino/...", "snippet": "Community discussion"},
            {"position": 2, "title": "Arduino Docs", "link": "https://docs.arduino.cc/hardware/uno-r4", "snippet": "Official docs"},
            {"position": 3, "title": "Random Blog", "link": "https://random-blog.com/arduino", "snippet": "Tutorial"},
        ]
    }
    
    # Mock _fetch_page to return None
    def mock_fetch(self, url):
        return None
    monkeypatch.setattr("embeduino.web_search.SerpApiSearcher._fetch_page", mock_fetch)
    
    searcher = SerpApiSearcher(
        api_key="test",
        cache_dir=tmp_path,
        max_calls_per_run=10,
        timeout_s=90,
        trusted_domains=["docs.arduino.cc", "arduino.cc", "github.com", "forum.arduino.cc"],
    )
    
    question = "test query"
    hits = searcher._parse_results(mock_response, num_results=10, question=question)
    
    # Only the docs.arduino.cc result should remain
    assert len(hits) == 1
    assert hits[0].url == "https://docs.arduino.cc/hardware/uno-r4"
    assert "reddit" not in hits[0].url.lower()
    assert "random-blog" not in hits[0].url.lower()


def test_fallback_counts_second_credit(tmp_path, monkeypatch):
    """Fallback search should add a second credit to the count."""
    call_count = [0]
    
    def mock_get(url, params=None, timeout=None, **kwargs):
        call_count[0] += 1
        
        class MockResponse:
            def raise_for_status(self):
                pass
            def json(self):
                # First call (async): return 0 organic results
                if call_count[0] == 1:
                    return {"search_metadata": {"id": "test_id"}}
                # Polling: return success with 0 results
                elif "searches/" in url and call_count[0] == 2:
                    return {
                        "search_metadata": {"status": "Success"},
                        "organic_results": []  # 0 results triggers fallback
                    }
                # Fallback call: return some results
                else:
                    return {
                        "organic_results": [
                            {"position": 1, "title": "Test", "link": "https://docs.arduino.cc/test", "snippet": "snippet"}
                        ]
                    }
        return MockResponse()
    
    monkeypatch.setattr("httpx.get", mock_get)
    monkeypatch.setattr("time.sleep", lambda x: None)
    
    # Mock _fetch_page
    def mock_fetch(self, url):
        return None
    monkeypatch.setattr("embeduino.web_search.SerpApiSearcher._fetch_page", mock_fetch)
    
    searcher = SerpApiSearcher(
        api_key="fake_key",
        cache_dir=tmp_path,
        max_calls_per_run=10,
        timeout_s=90,
    )
    
    hits, credits_used, status = searcher.search(
        "test query",
        trusted_sites=["docs.arduino.cc", "github.com"],
        num_results=5,
        prefer_docs=True,
        use_async=True,
    )
    
    assert status == "api_success"
    assert credits_used == 2, f"Expected 2 credits (initial + fallback), got {credits_used}"
    assert len(hits) >= 1


def test_page_fetching_extracts_main_content(tmp_path, monkeypatch):
    """Page fetching should extract main content and create chunks."""
    from pathlib import Path
    
    # Load the HTML fixture
    fixture_path = Path(__file__).parent / "fixtures" / "arduino_uno_r4_wifi.html"
    html_content = fixture_path.read_text()
    
    def mock_httpx_get(url, headers=None, timeout=None, follow_redirects=None):
        class MockResponse:
            def raise_for_status(self):
                pass
            headers = {"content-type": "text/html; charset=utf-8"}
            content = html_content.encode()
        return MockResponse()
    
    monkeypatch.setattr("httpx.get", mock_httpx_get)
    
    searcher = SerpApiSearcher(
        api_key="test",
        cache_dir=tmp_path,
        max_calls_per_run=10,
        timeout_s=90,
    )
    
    url = "https://docs.arduino.cc/hardware/uno-r4-wifi"
    content = searcher._fetch_page(url)
    
    assert content is not None
    assert "Pin Specifications" in content
    assert "14 (of which 6 provide PWM output)" in content
    assert "Renesas RA4M1" in content
    # Analytics script should be removed
    assert "analytics.js" not in content
    # Footer should be removed
    assert "All rights reserved" not in content or len(content) < 5000


def test_page_fetching_creates_multiple_chunks(tmp_path, monkeypatch):
    """Fetched pages should be chunked into multiple retrievable pieces."""
    from pathlib import Path
    
    fixture_path = Path(__file__).parent / "fixtures" / "arduino_uno_r4_wifi.html"
    html_content = fixture_path.read_text()
    
    def mock_httpx_get(url, headers=None, timeout=None, follow_redirects=None):
        class MockResponse:
            def raise_for_status(self):
                pass
            headers = {"content-type": "text/html; charset=utf-8"}
            content = html_content.encode()
        return MockResponse()
    
    monkeypatch.setattr("httpx.get", mock_httpx_get)
    
    mock_serp_response = {
        "organic_results": [
            {
                "position": 1,
                "title": "Arduino UNO R4 WiFi",
                "link": "https://docs.arduino.cc/hardware/uno-r4-wifi",
                "snippet": "Official documentation"
            }
        ]
    }
    
    searcher = SerpApiSearcher(
        api_key="test",
        cache_dir=tmp_path,
        max_calls_per_run=10,
        timeout_s=90,
    )
    
    question = "What are the pin specifications for Arduino UNO R4 WiFi?"
    hits = searcher._parse_results(mock_serp_response, num_results=5, question=question)
    
    # Should create multiple chunks from the page
    assert len(hits) >= 1
    assert all(hit.url == "https://docs.arduino.cc/hardware/uno-r4-wifi" for hit in hits)
    
    # At least one chunk should contain pin specs
    pin_spec_chunks = [h for h in hits if "Pin Specifications" in h.text or "Digital I/O" in h.text]
    assert len(pin_spec_chunks) >= 1
    
    # Check that the chunk has useful content (not just snippet)
    assert any("14 (of which 6 provide PWM output)" in h.text for h in hits)


def test_dynamic_scoring_beats_flat_score(tmp_path, monkeypatch):
    """Pages with better lexical overlap should get higher scores."""
    # Mock _fetch_page to return None (use snippets)
    def mock_fetch(self, url):
        return None
    monkeypatch.setattr("embeduino.web_search.SerpApiSearcher._fetch_page", mock_fetch)
    
    searcher = SerpApiSearcher(
        api_key="test",
        cache_dir=tmp_path,
        max_calls_per_run=10,
        timeout_s=90,
    )
    
    question = "What are the pin specifications for Arduino UNO R4 WiFi?"
    
    # Text with high overlap
    relevant_text = "Arduino UNO R4 WiFi pin specifications include 14 digital I/O pins"
    score_high = searcher._compute_score(question, relevant_text)
    
    # Text with low overlap
    generic_text = "Download the Arduino software from our website"
    score_low = searcher._compute_score(question, generic_text)
    
    assert score_high > score_low
    assert score_high > 0.5
    assert score_low < 0.7
