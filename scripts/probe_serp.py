#!/usr/bin/env python3
"""Probe SerpApi search without running full RAG pipeline.

Usage:
    python scripts/probe_serp.py "What are the pin specifications for Arduino UNO R4 WiFi?"
    python scripts/probe_serp.py "How to use digitalWrite with ESP32?" --async

Shows:
- Exact query params sent
- Organic results count
- Domain distribution
- search_information (query rewrites, etc.)
- Chosen pages (trusted only)
- Costs 1 credit per run
"""

import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from embeduino.web_search import _build_keyword_query


def probe_search(question: str, use_async: bool = True):
    """Probe SerpApi with a question and show diagnostic info."""
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        print("❌ SERPAPI_API_KEY not set")
        return
    
    # Build query
    query = _build_keyword_query(question)
    
    # Params
    params = {
        "engine": "google",
        "hl": "en",
        "gl": "us",
        "q": query,
        "as_sitesearch": "docs.arduino.cc",
        "api_key": api_key,
    }
    
    if use_async:
        params["async"] = "true"
    
    print("=" * 80)
    print("PROBE SERPAPI SEARCH")
    print("=" * 80)
    print(f"\n📝 Question: {question}")
    print(f"\n🔍 Built query: {query}")
    print(f"\n📤 Params sent:")
    safe_params = {k: v for k, v in params.items() if k != "api_key"}
    safe_params["api_key"] = f"{api_key[:8]}..."
    print(json.dumps(safe_params, indent=2))
    
    # Make request
    url = "https://serpapi.com/search"
    
    try:
        resp = httpx.get(url, params=params, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
        
        if use_async:
            search_id = data.get("search_metadata", {}).get("id")
            print(f"\n⏳ Async search queued: {search_id}")
            print("   Polling for results...")
            
            # Poll
            import time
            poll_url = f"https://serpapi.com/searches/{search_id}.json"
            poll_params = {"api_key": api_key}
            
            for i in range(30):
                time.sleep(2)
                poll_resp = httpx.get(poll_url, params=poll_params, timeout=10.0)
                poll_resp.raise_for_status()
                data = poll_resp.json()
                
                status = data.get("search_metadata", {}).get("status")
                if status == "Success":
                    print(f"   ✅ Completed after {(i+1)*2}s")
                    break
                elif status == "Error":
                    print(f"   ❌ Failed: {data.get('error')}")
                    return
        
        # Analyze results
        search_info = data.get("search_information", {})
        organic = data.get("organic_results", [])
        
        print(f"\n📊 Organic results: {len(organic)}")
        
        # Check for query rewrites
        if search_info.get("spelling_fix"):
            print(f"   ⚠️  Spelling corrected to: {search_info.get('spelling_fix')}")
        if search_info.get("showing_results_for"):
            print(f"   ⚠️  Showing results for: {search_info.get('showing_results_for')}")
        if search_info.get("query_displayed"):
            print(f"   📝 Query displayed: {search_info.get('query_displayed')}")
        
        # Domain distribution
        domain_counts = {}
        trusted_domains = ["docs.arduino.cc", "arduino.cc", "www.arduino.cc", "github.com", "forum.arduino.cc"]
        
        print(f"\n🌐 Domain distribution:")
        for result in organic:
            link = result.get("link", "")
            if not link:
                continue
            
            try:
                parsed = urlparse(link)
                domain = parsed.netloc.lower()
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            except:
                pass
        
        for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
            is_trusted = any(
                domain == td or domain.replace("www.", "") == td.replace("www.", "")
                or domain.endswith(f".{td}")
                for td in trusted_domains
            )
            marker = "✅" if is_trusted else "❌"
            print(f"   {marker} {domain}: {count}")
        
        # Show chosen pages (trusted only)
        print(f"\n✅ Trusted pages (would be fetched):")
        trusted_count = 0
        for i, result in enumerate(organic[:10], 1):
            link = result.get("link", "")
            title = result.get("title", "")
            
            if not link:
                continue
            
            try:
                parsed = urlparse(link)
                domain = parsed.netloc.lower()
                is_trusted = any(
                    domain == td or domain.replace("www.", "") == td.replace("www.", "")
                    or domain.endswith(f".{td}")
                    for td in trusted_domains
                )
                
                if is_trusted:
                    trusted_count += 1
                    print(f"   {trusted_count}. {title}")
                    print(f"      {link}")
            except:
                pass
        
        if trusted_count == 0:
            print("   ⚠️  No trusted results! Fallback would run.")
        elif trusted_count < 2:
            print(f"   ⚠️  Only {trusted_count} trusted result(s). Fallback would run.")
        
        print(f"\n💰 Cost: 1 SerpApi credit")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Probe SerpApi search")
    parser.add_argument("question", help="Question to search")
    parser.add_argument("--async", dest="use_async", action="store_true", help="Use async mode")
    parser.add_argument("--sync", dest="use_async", action="store_false", help="Use sync mode")
    parser.set_defaults(use_async=True)
    
    args = parser.parse_args()
    
    probe_search(args.question, args.use_async)
