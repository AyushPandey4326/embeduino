"""SerpApi Google Search integration with caching and credit management."""

from __future__ import annotations

import hashlib
import json
import logging
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
    """SerpApi Google Search client with disk caching and credit tracking."""
    
    def __init__(
        self,
        api_key: str,
        cache_dir: Path,
        max_calls_per_run: int = 10,
    ):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_calls = max_calls_per_run
        self.calls_made = 0
    
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
    
    def search(
        self,
        question: str,
        trusted_sites: List[str],
        num_results: int = 5,
        prefer_docs: bool = True,
    ) -> tuple[List[WebHit], bool, str]:
        """
        Run SerpApi Google search with site filter.
        
        Returns (hits, used_credit, status).
        status is one of: "cache_hit", "api_success", "api_failed", "no_key", "max_calls"
        """
        query = _build_keyword_query(question)
        site_filter = f"site:{trusted_sites[0]}" if prefer_docs and trusted_sites else " OR ".join(f"site:{s}" for s in trusted_sites)
        full_query = f"{query} {site_filter}"
        
        params = {
            "engine": "google",
            "hl": "en",
            "gl": "us",
            "num": num_results,
        }
        
        cache_key = self._cache_key(full_query, params)
        cached = self._read_cache(cache_key)
        
        if cached is not None:
            logger.info("SerpApi: served from cache (key=%s)", cache_key[:8])
            hits = self._parse_results(cached)
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
        
        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                logger.info("SerpApi: calling API (query=%r) attempt %d", full_query[:80], attempt + 1)
                resp = httpx.get(url, params=params, timeout=30.0)
                resp.raise_for_status()
                data = resp.json()
                self._write_cache(cache_key, data)
                self.calls_made += 1
                hits = self._parse_results(data)
                organic_count = len(data.get("organic_results", []))
                logger.debug("SerpApi: received %d organic_results", organic_count)
                logger.info("SerpApi: retrieved %d results (credit used)", len(hits))
                
                if organic_count == 0 and not prefer_docs:
                    return hits, True, "api_success"
                elif organic_count == 0 and prefer_docs and self.calls_made < self.max_calls:
                    logger.info("SerpApi: 0 results with docs preference, retrying without site filter")
                    fallback_query = query
                    fallback_params = dict(params)
                    fallback_params["q"] = fallback_query
                    fallback_key = self._cache_key(fallback_query, {k: v for k, v in fallback_params.items() if k != "api_key"})
                    
                    fallback_cached = self._read_cache(fallback_key)
                    if fallback_cached:
                        logger.info("SerpApi fallback: served from cache")
                        hits = self._parse_results(fallback_cached)
                        return hits, False, "cache_hit"
                    
                    try:
                        resp2 = httpx.get(url, params=fallback_params, timeout=30.0)
                        resp2.raise_for_status()
                        data2 = resp2.json()
                        self._write_cache(fallback_key, data2)
                        self.calls_made += 1
                        hits = self._parse_results(data2)
                        logger.debug("SerpApi fallback: received %d organic_results", len(data2.get("organic_results", [])))
                        logger.info("SerpApi fallback: retrieved %d results (2nd credit used)", len(hits))
                        return hits, True, "api_success"
                    except Exception as e2:
                        logger.error("SerpApi fallback call failed: %s", e2)
                        return [], True, "api_failed"
                
                return hits, True, "api_success"
            except Exception as e:
                logger.error("SerpApi call failed (attempt %d/%d): %s", attempt + 1, max_retries + 1, e)
                if attempt < max_retries:
                    import time
                    backoff = 4 * (2 ** attempt)
                    logger.info("Retrying in %ds...", backoff)
                    time.sleep(backoff)
                else:
                    return [], False, "api_failed"
        
        return [], False, "api_failed"
    
    def _parse_results(self, data: Dict[str, Any]) -> List[WebHit]:
        """Parse organic_results from SerpApi JSON."""
        results = data.get("organic_results", [])
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
    
    words = re.findall(r'\w+', question.lower())
    
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    return " ".join(keywords[:8]) if keywords else question


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
