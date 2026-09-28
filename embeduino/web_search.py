"""SerpApi Google Search integration with async mode, page fetching, and domain filtering."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

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
    fetched_content: str = ""  # Full page content if fetched
    
    def to_dict(self, score: float = 0.5) -> Dict[str, Any]:
        """Convert to dict with computed score."""
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
            "score": score,
            "lex_overlap": 0.0,
            "rerank_score": score,
        }


class SerpApiSearcher:
    """SerpApi Google Search client with async mode, page fetching, disk caching and credit tracking."""
    
    def __init__(
        self,
        api_key: str,
        cache_dir: Path,
        max_calls_per_run: int = 10,
        timeout_s: int = 90,
        trusted_domains: List[str] = None,
    ):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.page_cache_dir = self.cache_dir.parent / ".page_cache"
        self.page_cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_calls = max_calls_per_run
        self.calls_made = 0
        self.timeout_s = timeout_s
        self.trusted_domains = trusted_domains or [
            "docs.arduino.cc",
            "arduino.cc",
            "www.arduino.cc",
            "github.com",
            "forum.arduino.cc",
        ]
    
    def _is_trusted_domain(self, url: str) -> bool:
        """Check if URL is from a trusted domain."""
        try:
            parsed = urlparse(url)
            host = parsed.netloc.lower()
            # Remove www. prefix for comparison
            host_normalized = host.replace("www.", "")
            
            for domain in self.trusted_domains:
                domain_normalized = domain.replace("www.", "")
                if host == domain or host_normalized == domain_normalized:
                    return True
                # Allow subdomains
                if host.endswith(f".{domain}") or host_normalized.endswith(f".{domain_normalized}"):
                    return True
            return False
        except Exception as e:
            logger.warning("Failed to parse URL %s: %s", url, e)
            return False
    
    def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch and extract main text from a web page."""
        # Check cache first
        url_hash = hashlib.sha1(url.encode()).hexdigest()[:16]
        cache_file = self.page_cache_dir / f"{url_hash}.txt"
        
        if cache_file.exists():
            try:
                return cache_file.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("Page cache read error for %s: %s", url, e)
        
        # Fetch page
        try:
            logger.debug("Fetching page: %s", url)
            resp = httpx.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; EmbeduinoBot/1.0)"},
                timeout=15.0,
                follow_redirects=True,
            )
            resp.raise_for_status()
            
            content_type = resp.headers.get("content-type", "").lower()
            if "pdf" in content_type:
                logger.debug("Skipping PDF: %s", url)
                return None
            
            if "html" not in content_type and "text" not in content_type:
                logger.debug("Skipping non-HTML content: %s", url)
                return None
            
            # Extract text
            soup = BeautifulSoup(resp.content, "lxml")
            
            # Remove script, style, nav, footer, and other non-content elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
                tag.decompose()
            
            # Get main content (try common content containers first)
            main_content = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile(r"content|main|article|body", re.I))
            
            if main_content:
                text = main_content.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)
            
            # Clean up whitespace
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            text = "\n\n".join(lines)
            
            # Limit size
            if len(text) > 10000:
                text = text[:10000]
            
            # Cache it
            try:
                cache_file.write_text(text, encoding="utf-8")
            except Exception as e:
                logger.warning("Page cache write error for %s: %s", url, e)
            
            logger.info("Fetched %d chars from %s", len(text), url)
            return text
            
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", url, e)
            return None
    
    def _chunk_page_content(self, content: str, url: str, title: str, max_chunk_size: int = 800) -> List[str]:
        """Chunk fetched page content into manageable pieces."""
        if not content:
            return []
        
        # Split by paragraphs
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            if current_size + para_size > max_chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(para)
            current_size += para_size
        
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        # Limit to top 3 chunks
        return chunks[:3]
    
    def _compute_score(self, question: str, text: str) -> float:
        """Compute relevance score based on lexical overlap."""
        q_terms = set(re.findall(r'\w+', question.lower()))
        q_terms = {t for t in q_terms if len(t) > 2}
        
        if not q_terms:
            return 0.5
        
        text_terms = set(re.findall(r'\w+', text.lower()))
        overlap = len(q_terms & text_terms)
        score = 0.3 + (0.7 * overlap / len(q_terms))
        
        return min(score, 1.0)
    
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
    ) -> tuple[List[WebHit], int, str]:
        """
        Run SerpApi Google search with site filter.
        
        Returns (hits, credits_used, status).
        credits_used is the total number of credits consumed (0, 1, or 2).
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
            hits = self._parse_results(cached, num_results, question)
            return hits, 0, "cache_hit"
        
        if self.calls_made >= self.max_calls:
            logger.warning(
                "SerpApi: max calls reached (%d/%d), skipping",
                self.calls_made, self.max_calls
            )
            return [], 0, "max_calls"
        
        if not self.api_key:
            logger.warning("SerpApi: no API key set, skipping")
            return [], 0, "no_key"
        
        url = "https://serpapi.com/search"
        params["api_key"] = self.api_key
        params["q"] = full_query
        
        if use_async:
            params["async"] = "true"
        
        credits_used = 0
        
        try:
            logger.info("SerpApi: %s search (query=%r)", "async" if use_async else "sync", full_query[:80])
            if use_async:
                print("  Searching the web...")
            
            resp = httpx.get(url, params=params, timeout=15.0)
            resp.raise_for_status()
            data = resp.json()
            self.calls_made += 1
            credits_used = 1
            
            if use_async:
                search_id = data.get("search_metadata", {}).get("id")
                if not search_id:
                    logger.error("SerpApi async response missing search_metadata.id")
                    return [], credits_used, "api_failed"
                
                logger.debug("SerpApi: async search queued, id=%s", search_id)
                result_data, poll_status = self._poll_search(search_id)
                
                if poll_status == "success" and result_data:
                    self._write_cache(cache_key, result_data)
                    hits = self._parse_results(result_data, num_results, question)
                    organic_count = len(result_data.get("organic_results", []))
                    logger.debug("SerpApi: received %d organic_results", organic_count)
                    logger.info("SerpApi: retrieved %d web chunks (%d credit used)", len(hits), credits_used)
                    
                    if organic_count == 0 and prefer_docs and self.calls_made < self.max_calls:
                        fallback_hits, fallback_credits, fallback_status = self._fallback_search(query, trusted_sites, num_results, question)
                        credits_used += fallback_credits
                        if fallback_hits:
                            return fallback_hits, credits_used, "api_success"
                    
                    return hits, credits_used, "api_success"
                elif poll_status == "timeout":
                    return [], credits_used, "timeout"
                else:
                    return [], credits_used, "api_failed"
            else:
                self._write_cache(cache_key, data)
                hits = self._parse_results(data, num_results, question)
                organic_count = len(data.get("organic_results", []))
                logger.debug("SerpApi: received %d organic_results", organic_count)
                logger.info("SerpApi: retrieved %d web chunks (%d credit used)", len(hits), credits_used)
                
                if organic_count == 0 and prefer_docs and self.calls_made < self.max_calls:
                    fallback_hits, fallback_credits, fallback_status = self._fallback_search(query, trusted_sites, num_results, question)
                    credits_used += fallback_credits
                    if fallback_hits:
                        return fallback_hits, credits_used, "api_success"
                
                return hits, credits_used, "api_success"
        except Exception as e:
            logger.error("SerpApi call failed: %s", e)
            return [], credits_used, "api_failed"
    
    def _fallback_search(
        self, query: str, trusted_sites: List[str], num_results: int, question: str
    ) -> tuple[List[WebHit], int, str]:
        """Fallback search with OR of all trusted site filters (not unfiltered)."""
        logger.info("SerpApi: 0 results with docs preference, retrying with all trusted sites")
        
        # Use OR of all trusted sites, never unfiltered
        site_filter = " OR ".join(f"site:{s}" for s in trusted_sites)
        fallback_query = f"{query} {site_filter}"
        
        fallback_params = {
            "engine": "google",
            "hl": "en",
            "gl": "us",
            "api_key": self.api_key,
            "q": fallback_query,
        }
        
        fallback_key = self._cache_key(fallback_query, {k: v for k, v in fallback_params.items() if k != "api_key"})
        
        fallback_cached = self._read_cache(fallback_key)
        if fallback_cached:
            logger.info("SerpApi fallback: served from cache")
            hits = self._parse_results(fallback_cached, num_results, question)
            return hits, 0, "cache_hit"
        
        url = "https://serpapi.com/search"
        
        try:
            resp = httpx.get(url, params=fallback_params, timeout=15.0)
            resp.raise_for_status()
            data = resp.json()
            self.calls_made += 1
            self._write_cache(fallback_key, data)
            hits = self._parse_results(data, num_results, question)
            logger.debug("SerpApi fallback: received %d organic_results", len(data.get("organic_results", [])))
            logger.info("SerpApi fallback: retrieved %d web chunks (2nd credit used)", len(hits))
            return hits, 1, "api_success"
        except Exception as e:
            logger.error("SerpApi fallback call failed: %s", e)
            return [], 1, "api_failed"
    
    def _parse_results(self, data: Dict[str, Any], num_results: int, question: str) -> List[WebHit]:
        """Parse organic_results from SerpApi JSON, filter by trusted domains, fetch pages."""
        results = data.get("organic_results", [])
        hits = []
        
        for r in results[:num_results * 2]:  # Get more to account for filtering
            position = r.get("position", 0)
            title = r.get("title", "")
            link = r.get("link", "")
            snippet = r.get("snippet", "")
            
            if not link:
                continue
            
            # Filter by trusted domain
            if not self._is_trusted_domain(link):
                logger.debug("Filtered untrusted domain: %s", link)
                continue
            
            # Try to fetch full page content
            page_content = self._fetch_page(link)
            
            if page_content:
                # Chunk the page content
                chunks_text = self._chunk_page_content(page_content, link, title)
                
                for i, chunk_text in enumerate(chunks_text):
                    hit_id = f"web:{hashlib.sha1(f'{link}:{i}'.encode()).hexdigest()[:12]}"
                    text = f"{title}\n\n{chunk_text}"
                    score = self._compute_score(question, text)
                    
                    hits.append(
                        WebHit(
                            id=hit_id,
                            text=text,
                            url=link,
                            title=title,
                            position=position + i * 0.1,  # Sub-position for chunks from same page
                            snippet=snippet,
                            fetched_content=chunk_text,
                        )
                    )
            else:
                # Fallback to snippet if fetch failed
                hit_id = f"web:{hashlib.sha1(link.encode()).hexdigest()[:12]}"
                text = f"{title}\n\n{snippet}" if snippet else title
                score = self._compute_score(question, text)
                
                hits.append(
                    WebHit(
                        id=hit_id,
                        text=text,
                        url=link,
                        title=title,
                        position=position,
                        snippet=snippet,
                        fetched_content="",
                    )
                )
            
            if len(hits) >= num_results:
                break
        
        logger.info("Parsed %d web chunks from %d organic results", len(hits), len(results))
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
