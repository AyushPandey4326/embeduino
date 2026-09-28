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


ARDUINO_PRODUCT_SLUGS = {
    "uno r4 wifi": "uno-r4-wifi",
    "uno r4 minima": "uno-r4-minima",
    "uno r3": "uno-rev3",
    "uno rev3": "uno-rev3",
    "nano": "nano",
    "nano 33 ble": "nano-33-ble",
    "nano esp32": "nano-esp32",
    "nano every": "nano-every",
    "mega 2560": "mega-2560",
    "leonardo": "leonardo",
}


def _detect_arduino_product(question: str) -> Optional[str]:
    """Detect Arduino product name in question and return hardware slug."""
    question_lower = question.lower()
    for product_name, slug in ARDUINO_PRODUCT_SLUGS.items():
        if product_name in question_lower:
            return slug
    return None


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
        """Fetch and extract main text from a web page, with Gatsby page-data.json support."""
        url_hash = hashlib.sha1(url.encode()).hexdigest()[:16]
        cache_file = self.page_cache_dir / f"{url_hash}.txt"
        
        # Special handling for docs.arduino.cc Gatsby pages (check BEFORE cache)
        if "docs.arduino.cc/hardware/" in url:
            content = self._fetch_gatsby_page_data(url)
            if content:
                try:
                    cache_file.write_text(content, encoding="utf-8")
                except Exception as e:
                    logger.warning("Page cache write error for %s: %s", url, e)
                return content
            # If Gatsby fetch fails, fall through to HTML fetch with cache
        
        # Check cache for non-Gatsby or failed Gatsby pages
        if cache_file.exists():
            try:
                return cache_file.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("Page cache read error for %s: %s", url, e)
        
        # Fallback to HTML fetch
        try:
            logger.debug("Fetching HTML page: %s", url)
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
            
            # Remove non-content elements (aggressive for GitHub/Discourse)
            for tag in soup([
                "script", "style", "nav", "footer", "header", "aside", "iframe", "noscript",
                "button",  # GitHub buttons
            ]):
                tag.decompose()
            
            # Remove by class/id patterns
            ui_patterns = [
                "sign-in", "signin", "notification", "subscribe", "login",
                "avatar", "timestamp", "username", "reply", "comment-meta",
                "sidebar", "menu", "breadcrumb", "pagination",
                "fork", "star", "watch", "sponsor",
                "header", "footer", "navbar",
            ]
            
            for pattern in ui_patterns:
                for tag in soup.find_all(class_=lambda x: x and pattern in x.lower()):
                    tag.decompose()
                for tag in soup.find_all(id=lambda x: x and pattern in x.lower()):
                    tag.decompose()
            
            # Get main content
            main_content = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile(r"content|main|article|body", re.I))
            
            if main_content:
                text = main_content.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)
            
            # Clean up whitespace
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            # Filter out common UI text fragments
            ui_fragments = {
                "you must be signed in", "fork", "star", "watch", "notification settings",
                "back to top", "© 20", "all rights reserved", "terms of service",
                "privacy policy", "cookie", "sign in", "sign up", "register",
            }
            lines = [
                line for line in lines
                if not any(frag in line.lower() for frag in ui_fragments) or len(line) > 80
            ]
            
            text = "\n\n".join(lines)
            
            # Limit size
            if len(text) > 10000:
                text = text[:10000]
            
            # Generic Gatsby fallback: if HTML extraction yields <150 chars and it's docs.arduino.cc, try page-data
            if len(text) < 150 and "docs.arduino.cc" in url:
                # Skip non-English paths
                parsed = urlparse(url)
                if "/pt/" not in parsed.path and "fun%C3%A7%C3%B5es" not in parsed.path and "/es/" not in parsed.path:
                    logger.debug("Short HTML (%d chars), trying Gatsby page-data fallback for %s", len(text), url)
                    gatsby_content = self._fetch_gatsby_page_data(url)
                    if gatsby_content and len(gatsby_content) > len(text):
                        logger.info("Gatsby fallback successful: %d chars vs %d HTML chars", len(gatsby_content), len(text))
                        text = gatsby_content
            
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
    
    def _fetch_gatsby_page_data(self, url: str) -> Optional[str]:
        """
        Fetch page-data.json for Gatsby sites (docs.arduino.cc).
        
        Extracts techspecs from result.data.techspecs.fields.content
        and converts YAML-like structure to readable "Section - Key: Value" lines.
        Also works for any docs.arduino.cc page with mdx content.
        """
        try:
            # Extract path from URL
            parsed = urlparse(url)
            path = parsed.path.rstrip('/').lstrip('/')
            
            # Skip non-English paths
            if '/pt/' in path or '/es/' in path or 'fun%C3%A7%C3%B5es' in path:
                return None
            
            page_data_url = f"https://docs.arduino.cc/page-data/{path}/page-data.json"
            
            logger.debug("Fetching Gatsby page-data: %s", page_data_url)
            resp = httpx.get(
                page_data_url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10.0,
            )
            resp.raise_for_status()
            
            data = resp.json()
            result_data = data.get("result", {}).get("data", {})
            
            # Extract overview/description
            product = result_data.get("product", {})
            child_mdx = product.get("childMdx", {})
            excerpt = child_mdx.get("excerpt", "")
            
            # Extract techspecs (for hardware pages)
            techspecs = result_data.get("techspecs", {})
            techspecs_content = techspecs.get("fields", {}).get("content", "")
            
            if not techspecs_content and not excerpt:
                # Try to get any content field
                logger.debug("No techspecs or excerpt found in page-data for %s", url)
                return None
            
            # Parse YAML-like techspecs into "Section - Key: Value" lines
            lines = []
            if excerpt:
                lines.append(excerpt)
                lines.append("")
            
            if techspecs_content:
                current_section = ""
                for line in techspecs_content.split("\n"):
                    line = line.rstrip()
                    if not line:
                        continue
                    
                    # Section header (no leading spaces, ends with :)
                    if not line.startswith(" ") and line.endswith(":"):
                        current_section = line.rstrip(":")
                    # Key-value pair (leading spaces)
                    elif line.startswith(" ") and ":" in line:
                        # Parse "  Key: Value"
                        key_value = line.strip()
                        if ":" in key_value:
                            key, value = key_value.split(":", 1)
                            key = key.strip()
                            value = value.strip()
                            # Make self-contained: "Section - Key: Value"
                            if current_section:
                                lines.append(f"{current_section} - {key}: {value}")
                            else:
                                lines.append(f"{key}: {value}")
            
            content = "\n".join(lines)
            if content:
                logger.info("Fetched %d chars from Gatsby page-data for %s", len(content), url)
            return content if content else None
            
        except Exception as e:
            logger.warning("Failed to fetch Gatsby page-data for %s: %s", url, e)
            return None
    
    def _chunk_page_content(self, content: str, url: str, title: str, max_chunk_size: int = 800, min_chunk_size: int = 150) -> List[str]:
        """Chunk fetched page content into manageable pieces, dropping short chunks."""
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
                chunk_text = "\n\n".join(current_chunk)
                # Only keep chunks with enough real content
                if len(chunk_text) >= min_chunk_size:
                    chunks.append(chunk_text)
                current_chunk = []
                current_size = 0
            
            current_chunk.append(para)
            current_size += para_size
        
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            if len(chunk_text) >= min_chunk_size:
                chunks.append(chunk_text)
        
        # Limit to top 3 chunks
        return chunks[:3]
    
    def _compute_score(self, question: str, text: str, url: str = "") -> float:
        """
        Compute relevance score based on lexical overlap and source priority.
        
        Source priority:
        1. docs.arduino.cc/hardware/* (official product pages): +0.3
        2. Other docs.arduino.cc: +0.2
        3. github.com/arduino/*: +0.15
        4. forum.arduino.cc: +0.1
        5. Other github.com: +0.05
        """
        q_terms = set(re.findall(r'\w+', question.lower()))
        q_terms = {t for t in q_terms if len(t) > 2}
        
        if not q_terms:
            base_score = 0.5
        else:
            text_terms = set(re.findall(r'\w+', text.lower()))
            overlap = len(q_terms & text_terms)
            base_score = 0.3 + (0.7 * overlap / len(q_terms))
            base_score = min(base_score, 1.0)
        
        # Add source priority bonus
        bonus = 0.0
        url_lower = url.lower()
        
        if "docs.arduino.cc/hardware/" in url_lower:
            bonus = 0.3  # Highest priority: official product pages
        elif "docs.arduino.cc" in url_lower:
            bonus = 0.2  # Other docs pages
        elif "github.com/arduino/" in url_lower:
            bonus = 0.15  # Official Arduino GitHub
        elif "forum.arduino.cc" in url_lower:
            bonus = 0.1  # Forum
        elif "github.com" in url_lower:
            bonus = 0.05  # Other GitHub
        
        return min(base_score + bonus, 1.0)
    
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
        
        Uses as_sitesearch parameter for reliable site filtering.
        Checks search_information for query rewrites.
        """
        # Direct product-page resolution for known Arduino boards
        direct_hits = []
        product_slug = _detect_arduino_product(question)
        if product_slug:
            hardware_url = f"https://docs.arduino.cc/hardware/{product_slug}/"
            logger.info("Detected Arduino product '%s', fetching page-data directly", product_slug)
            content = self._fetch_gatsby_page_data(hardware_url)
            if content:
                chunks = self._chunk_page_content(content, hardware_url, f"Arduino {product_slug}")
                for i, chunk_text in enumerate(chunks):
                    hit_id = f"web:direct:{hashlib.sha1(f'{hardware_url}:{i}'.encode()).hexdigest()[:12]}"
                    text = f"Arduino {product_slug}\n\n{chunk_text}"
                    score = self._compute_score(question, text, hardware_url)
                    direct_hits.append(
                        WebHit(
                            id=hit_id,
                            text=text,
                            url=hardware_url,
                            title=f"Arduino {product_slug}",
                            position=0.5 + i * 0.1,
                            snippet=f"Direct hardware page for {product_slug}",
                            fetched_content=chunk_text,
                        )
                    )
                logger.info("Direct product fetch: retrieved %d chunks from %s", len(direct_hits), hardware_url)
        
        query = _build_keyword_query(question)
        
        # For spec/pinout/datasheet questions, use path-scoped docs.arduino.cc/hardware
        is_spec_question = any(kw in question.lower() for kw in ["spec", "pinout", "datasheet", "pin", "voltage", "current"])
        preferred_site = None
        if prefer_docs and trusted_sites:
            if is_spec_question and any("docs.arduino.cc" in site for site in trusted_sites):
                preferred_site = "docs.arduino.cc/hardware"
            else:
                preferred_site = trusted_sites[0]
        
        params = {
            "engine": "google",
            "hl": "en",
            "gl": "us",
            "q": query,  # No site: in query string
        }
        
        if preferred_site:
            params["as_sitesearch"] = preferred_site
        
        cache_key = self._cache_key(query + (f"_site:{preferred_site}" if preferred_site else ""), params)
        cached = self._read_cache(cache_key)
        
        if cached is not None:
            logger.info("SerpApi: served from cache (key=%s)", cache_key[:8])
            hits = self._parse_results(cached, num_results, question)
            # Combine with direct hits
            return direct_hits + hits, 0, "cache_hit"
        
        if self.calls_made >= self.max_calls:
            logger.warning(
                "SerpApi: max calls reached (%d/%d), skipping",
                self.calls_made, self.max_calls
            )
            return direct_hits, 0, "max_calls"
        
        if not self.api_key:
            logger.warning("SerpApi: no API key set, skipping")
            return direct_hits, 0, "no_key"
        
        url = "https://serpapi.com/search"
        params["api_key"] = self.api_key
        
        if use_async:
            params["async"] = "true"
        
        credits_used = 0
        
        try:
            logger.info("SerpApi: %s search (query=%r, as_sitesearch=%s)", 
                       "async" if use_async else "sync", query[:80], preferred_site or "none")
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
                    # Check if Google rewrote the query
                    search_info = result_data.get("search_information", {})
                    if search_info.get("spelling_fix") or search_info.get("showing_results_for"):
                        logger.warning("Google rewrote query: %s", search_info.get("query_displayed"))
                    
                    self._write_cache(cache_key, result_data)
                    hits = self._parse_results(result_data, num_results, question)
                    organic_count = len(result_data.get("organic_results", []))
                    trusted_count = len(hits)
                    
                    logger.debug("SerpApi: received %d organic_results, %d on trusted domains", organic_count, trusted_count)
                    logger.info("SerpApi: retrieved %d web chunks (%d credit used)", len(hits), credits_used)
                    
                    # If fewer than 2 results on trusted domains, try fallback
                    if trusted_count < 2 and prefer_docs and self.calls_made < self.max_calls:
                        logger.info("SerpApi: only %d trusted results, running fallback", trusted_count)
                        fallback_hits, fallback_credits, fallback_status = self._fallback_search(query, trusted_sites, num_results, question)
                        credits_used += fallback_credits
                        if len(fallback_hits) > len(hits):
                            return direct_hits + fallback_hits, credits_used, "api_success"
                    
                    return direct_hits + hits, credits_used, "api_success"
                elif poll_status == "timeout":
                    return [], credits_used, "timeout"
                else:
                    return [], credits_used, "api_failed"
            else:
                # Sync mode (similar logic)
                search_info = data.get("search_information", {})
                if search_info.get("spelling_fix") or search_info.get("showing_results_for"):
                    logger.warning("Google rewrote query: %s", search_info.get("query_displayed"))
                
                self._write_cache(cache_key, data)
                hits = self._parse_results(data, num_results, question)
                organic_count = len(data.get("organic_results", []))
                trusted_count = len(hits)
                
                logger.debug("SerpApi: received %d organic_results, %d on trusted domains", organic_count, trusted_count)
                logger.info("SerpApi: retrieved %d web chunks (%d credit used)", len(hits), credits_used)
                
                if trusted_count < 2 and prefer_docs and self.calls_made < self.max_calls:
                    logger.info("SerpApi: only %d trusted results, running fallback", trusted_count)
                    fallback_hits, fallback_credits, fallback_status = self._fallback_search(query, trusted_sites, num_results, question)
                    credits_used += fallback_credits
                    if len(fallback_hits) > len(hits):
                        return direct_hits + fallback_hits, credits_used, "api_success"
                
                return direct_hits + hits, credits_used, "api_success"
        except Exception as e:
            logger.error("SerpApi call failed: %s", e)
            return direct_hits, credits_used, "api_failed"
    
    def _fallback_search(
        self, query: str, trusted_sites: List[str], num_results: int, question: str
    ) -> tuple[List[WebHit], int, str]:
        """Fallback search with OR of all trusted site filters in query."""
        logger.info("SerpApi: fallback with all trusted sites")
        
        # Use OR of all trusted sites in the query
        site_filter = " OR ".join(f"site:{s}" for s in trusted_sites)
        fallback_query = f"{query} ({site_filter})"
        
        fallback_params = {
            "engine": "google",
            "hl": "en",
            "gl": "us",
            "api_key": self.api_key,
            "q": fallback_query,
            # Don't use as_sitesearch here since we have multiple sites
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
            organic_count = len(data.get("organic_results", []))
            trusted_count = len(hits)
            logger.debug("SerpApi fallback: received %d organic_results, %d on trusted domains", organic_count, trusted_count)
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
                    score = self._compute_score(question, text, link)
                    
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
                score = self._compute_score(question, text, link)
                
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
    """
    Extract keywords from natural language question for better search results.
    
    - When product detected: quote it + max 1 extra word ("UNO R4 WiFi" specifications)
    - Maps spec-type terms: pin/pins/specifications → specifications/pinout/datasheet
    - Preserves original casing for product identifiers
    """
    import re
    
    # Detect product names (board names with model numbers)
    product_patterns = [
        r'\b(UNO\s+R\d+(?:\s+\w+)?)\b',  # UNO R4, UNO R4 WiFi
        r'\b(Nano\s+(?:33|ESP32)(?:\s+\w+)?)\b',  # Nano 33 BLE, Nano ESP32
        r'\b(ESP32(?:-\w+)?)\b',  # ESP32, ESP32-S3
        r'\b(Mega\s+\d+)\b',  # Mega 2560
        r'\b(Due|Leonardo|Micro|Yun)\b',  # Other boards
    ]
    
    product_match = None
    for pattern in product_patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            product_match = match.group(1)
            break
    
    # If product detected, use simple query: "Product" + one spec term
    if product_match:
        # Check for spec-type question
        question_lower = question.lower()
        if any(term in question_lower for term in ['pin', 'pinout', 'specification', 'spec', 'datasheet']):
            return f'"{product_match}" specifications'
        else:
            # Generic product query
            return f'"{product_match}"'
    
    # Fall back to keyword extraction if no product detected
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
        # Keep tokens with digits or longer than 2 chars
        if any(c.isdigit() for c in w) or len(w) > 2:
            keywords.append(w)
    
    return " ".join(keywords[:6]) if keywords else question


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
