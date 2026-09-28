"""SerpApi Google Search integration with async mode, caching and credit management."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class WebHit:
    """A web search result converted to a retrievable chunk."""
    id: str
    text: str
    url: str
    title: str
    position: int
    snippet: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": {
                "url": self.url,
                "title": self.title,
                "source": self.url,
                "heading_path": self.title,
                "chunk_type": "web",
                "position": self.position,
            },
            "score": 0.5,
            "lex_overlap": 0.0,
            "rerank_score": 0.5,
        }


class SerpApiSearcher:
    """SerpApi Google Search client with async mode, disk caching and credit tracking."""
    
    def __init__(
        self,
        api_key: str,
        cache_dir: Path,
        max_calls_per_run: int = 10,
        timeout_s: int = 90,
    ):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_calls = max_calls_per_run
        self.calls_made = 0
        self.timeout_s = timeout_s
    
    def _cache_key(self, query: str, params: Dict[str, Any]) -> str:
        """Generate cache key from query + params."""
        key_data = json.dumps({"q": query, **params}, sort_keys=True)
        return hashlib.sha1(key_data.encode()).hexdigest()
    
    def _read_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Cache read error for %s: %s", cache_key, e)
        return None
    
    def _write_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Cache write error for %s: %s", cache_key, e)
    
    def _poll_search(self, search_id: str) -> tuple[Optional[Dict[str, Any]], str]:
        """
        Poll SerpApi for async search results.
        
        Returns (data, status) where status is 'success', 'timeout', or 'error'.
        Polling does not cost credits.
        """
        url = f"https://serpapi.com/searches/{search_id}.json"
        params = {"api_key": self.api_key}
        
        start_time = time.time()
        poll_interval = 2.5
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > self.timeout_s:
                logger.warning("SerpApi async search timed out after %.1fs", elapsed)
                return None, "timeout"
            
            try:
                resp = httpx.get(url, params=params, timeout=10.0)
                resp.raise_for_status()
                data = resp.json()
                
                status = data.get("search_metadata", {}).get("status", "Unknown")
                logger.debug("SerpApi poll: status=%s, elapsed=%.1fs", status, elapsed)
                
                if status == "Success":
                    logger.info("SerpApi async search completed in %.1fs", elapsed)
                    return data, "success"
                elif status == "Error":
                    logger.error("SerpApi async search failed: %s", data.get("error"))
                    return None, "error"
                elif status in ("Processing", "Queued"):
                    time.sleep(poll_interval)
                else:
                    logger.warning("Unknown SerpApi status: %s", status)
                    return None, "error"
            except Exception as e:
                logger.error("SerpApi poll failed: %s", e)
                return None, "error"
    
    def search(
        self,
        question: str,
        trusted_sites: List[str],
        num_results: int = 5,
        prefer_docs: bool = True,
        use_async: bool = True,
    ) -> tuple[List[WebHit], bool, str]:
        """
        Run SerpApi Google search with site filter.
        
        Returns (hits, used_credit, status).
        status is one of: "cache_hit", "api_success", "api_failed", "no_key", "max_calls", "timeout"
        
        When use_async=True, initiates async search (async=true) and polls for results.
        Polling does not cost credits, only the initial async search request costs 1 credit.
        """
        query = _build_keyword_query(question)
        site_filter = f"site:{trusted_sites[0]}" if prefer_docs and trusted_sites else " OR ".join(f"site:{s}" for s in trusted_sites)
        full_query = f"{query} {site_filter}"
        
        params = {
            "engine": "google",
            "hl": "en",
            "gl": "us",
        }
        
        cache_key = self._cache_key(full_query, params)
        cached = self._read_cache(cache_key)
        
        if cached is not None:
            logger.info("SerpApi: served from cache (key=%s)", cache_key[:8])
            hits = self._parse_results(cached, num_results)
            return hits, False, "cache_hit"
        
        if self.calls_made >= self.max_calls:
            logger.warning(
                "SerpApi: max calls reached (%d/%d), skipping",
                self.calls_made, self.max_calls
            )
            return [], False, "max_calls"
        
        if not self.api_key:
            logger.warning("SerpApi: no API key set, skipping")
            return [], False, "no_key"
        
        url = "https://serpapi.com/search"
        params["api_key"] = self.api_key
        params["q"] = full_query
        
        if use_async:
            params["async"] = "true"
        
        try:
            logger.info("SerpApi: %s search (query=%r)", "async" if use_async else "sync", full_query[:80])
            if use_async:
                print("  Searching the web...")
            
            resp = httpx.get(url, params=params, timeout=15.0)
            resp.raise_for_status()
            data = resp.json()
            self.calls_made += 1
            
            if use_async:
                search_id = data.get("search_metadata", {}).get("id")
                if not search_id:
                    logger.error("SerpApi async response missing search_metadata.id")
                    return [], True, "api_failed"
                
                logger.debug("SerpApi: async search queued, id=%s", search_id)
                result_data, poll_status = self._poll_search(search_id)
                
                if poll_status == "success" and result_data:
                    self._write_cache(cache_key, result_data)
                    hits = self._parse_results(result_data, num_results)
                    organic_count = len(result_data.get("organic_results", []))
                    logger.debug("SerpApi: received %d organic_results", organic_count)
                    logger.info("SerpApi: retrieved %d results (1 credit used)", len(hits))
                    
                    if organic_count == 0 and prefer_docs and self.calls_made < self.max_calls:
                        return self._fallback_search(query, params, num_results)
                    
                    return hits, True, "api_success"
                elif poll_status == "timeout":
                    return [], True, "timeout"
                else:
                    return [], True, "api_failed"
            else:
                self._write_cache(cache_key, data)
                hits = self._parse_results(data, num_results)
                organic_count = len(data.get("organic_results", []))
                logger.debug("SerpApi: received %d organic_results", organic_count)
                logger.info("SerpApi: retrieved %d results (1 credit used)", len(hits))
                
                if organic_count == 0 and prefer_docs and self.calls_made < self.max_calls:
                    return self._fallback_search(query, params, num_results)
                
                return hits, True, "api_success"
        except Exception as e:
            logger.error("SerpApi call failed: %s", e)
            return [], False, "api_failed"
    
    def _fallback_search(
        self, query: str, base_params: Dict[str, Any], num_results: int
    ) -> tuple[List[WebHit], bool, str]:
        """Fallback search without site filter."""
        logger.info("SerpApi: 0 results with site filter, retrying without")
        
        fallback_params = {k: v for k, v in base_params.items() if k != "async"}
        fallback_params["q"] = query
        fallback_key = self._cache_key(query, {k: v for k, v in fallback_params.items() if k != "api_key"})
        
        fallback_cached = self._read_cache(fallback_key)
        if fallback_cached:
            logger.info("SerpApi fallback: served from cache")
            hits = self._parse_results(fallback_cached, num_results)
            return hits, False, "cache_hit"
        
        url = "https://serpapi.com/search"
        
        try:
            resp = httpx.get(url, params=fallback_params, timeout=15.0)
            resp.raise_for_status()
            data = resp.json()
            self.calls_made += 1
            self._write_cache(fallback_key, data)
            hits = self._parse_results(data, num_results)
            logger.debug("SerpApi fallback: received %d organic_results", len(data.get("organic_results", [])))
            logger.info("SerpApi fallback: retrieved %d results (2nd credit used)", len(hits))
            return hits, True, "api_success"
        except Exception as e:
            logger.error("SerpApi fallback call failed: %s", e)
            return [], True, "api_failed"
    
    def _parse_results(self, data: Dict[str, Any], num_results: int) -> List[WebHit]:
        """Parse organic_results from SerpApi JSON, slicing to num_results locally."""
        results = data.get("organic_results", [])[:num_results]
        hits = []
        for r in results:
            position = r.get("position", 0)
            title = r.get("title", "")
            link = r.get("link", "")
            snippet = r.get("snippet", "")
            
            if not link:
                continue
            
            hit_id = f"web:{hashlib.sha1(link.encode()).hexdigest()[:12]}"
            text = f"{title}\n\n{snippet}" if snippet else title
            
            hits.append(
                WebHit(
                    id=hit_id,
                    text=text,
                    url=link,
                    title=title,
                    position=position,
                    snippet=snippet,
                )
            )
        return hits


def _build_keyword_query(question: str) -> str:
    """Extract keywords from natural language question for better search results."""
    import re
    
    stop_words = {
        "what", "are", "the", "is", "how", "do", "does", "did", "can", "could",
        "should", "would", "where", "when", "why", "which", "who", "for", "with",
        "from", "to", "of", "in", "on", "at", "by", "a", "an", "i", "you"
    }
    
    words = re.findall(r'\w+', question)
    
    keywords = []
    for w in words:
        w_lower = w.lower()
        if w_lower in stop_words:
            continue
        if any(c.isdigit() for c in w):
            keywords.append(w)
        elif len(w) > 2:
            keywords.append(w_lower)
    
    return " ".join(keywords[:10]) if keywords else question


def should_trigger_web_search(
    question: str,
    local_hits: List[Dict[str, Any]],
    is_weak: bool,
    mode: str,
) -> bool:
    """
    Decide whether to trigger web search.
    
    - mode "always": always trigger
    - mode "off": never trigger
    - mode "auto": trigger if local is weak OR question mentions uncovered topics
    """
    if mode == "off":
        return False
    if mode == "always":
        return True
    
    if is_weak:
        return True
    
    q_lower = question.lower()
    new_topics = [
        "uno r4", "r4 wifi", "nano esp32", "esp32", "nano 33", "mkr", "portenta",
        "giga", "nicla", "opta", "version 2", "v2.", "v1.", "library",
    ]
    
    for topic in new_topics:
        if topic in q_lower:
            logger.info("Web trigger: question mentions '%s'", topic)
            return True
    
    return False
