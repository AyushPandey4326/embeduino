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
            "score": 0.0,
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
    ) -> tuple[List[WebHit], bool]:
        """
        Run SerpApi Google search with site filter.
        
        Returns (hits, used_credit).
        used_credit is False if served from cache.
        """
        site_filter = " OR ".join(f"site:{s}" for s in trusted_sites)
        query = f"{question} ({site_filter})"
        
        params = {
            "engine": "google",
            "hl": "en",
            "gl": "us",
            "num": num_results,
        }
        
        cache_key = self._cache_key(query, params)
        cached = self._read_cache(cache_key)
        
        if cached is not None:
            logger.info("SerpApi: served from cache (key=%s)", cache_key[:8])
            hits = self._parse_results(cached)
            return hits, False
        
        if self.calls_made >= self.max_calls:
            logger.warning(
                "SerpApi: max calls reached (%d/%d), skipping",
                self.calls_made, self.max_calls
            )
            return [], False
        
        if not self.api_key:
            logger.warning("SerpApi: no API key set, skipping")
            return [], False
        
        url = "https://serpapi.com/search"
        params["api_key"] = self.api_key
        params["q"] = query
        
        try:
            logger.info("SerpApi: calling API (query=%r)", query[:80])
            resp = httpx.get(url, params=params, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            self._write_cache(cache_key, data)
            self.calls_made += 1
            hits = self._parse_results(data)
            logger.info("SerpApi: retrieved %d results (credit used)", len(hits))
            return hits, True
        except Exception as e:
            logger.error("SerpApi call failed: %s", e)
            return [], False
    
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
            text = f"{title}\n\n{snippet}"
            
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
