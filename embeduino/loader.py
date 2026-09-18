"""Load offline sample corpus and optionally fetch from docs.arduino.cc."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

DocPair = Tuple[str, str]  # (source_label, text)


def load_local_corpus(data_dir: Path) -> List[DocPair]:
    """Load markdown/HTML docs from data_dir (respects manifest.json if present)."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {data_dir}")

    files: List[Path] = []
    manifest = data_dir / "manifest.json"
    if manifest.exists():
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        for name in meta.get("documents", []):
            p = data_dir / name
            if p.exists():
                files.append(p)
            else:
                logger.warning("Manifest entry missing: %s", p)
    else:
        files = sorted(
            p
            for p in data_dir.iterdir()
            if p.suffix.lower() in {".md", ".markdown", ".html", ".htm", ".txt"}
        )

    docs: List[DocPair] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        label = f"local:{path.name}"
        docs.append((label, text))
        logger.info("Loaded %s (%d chars)", label, len(text))
    return docs


def fetch_arduino_pages(
    urls: Optional[List[str]] = None,
    timeout: float = 15.0,
) -> List[DocPair]:
    """
    Optionally fetch live pages from docs.arduino.cc.
    Returns empty list if network/HTTP fails (offline-safe).
    """
    default_urls = [
        "https://docs.arduino.cc/language-reference/en/functions/digital-io/digitalwrite/",
        "https://docs.arduino.cc/language-reference/en/functions/digital-io/pinMode/",
        "https://docs.arduino.cc/language-reference/en/functions/analog-io/analogRead/",
    ]
    urls = urls or default_urls
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed; skipping live fetch")
        return []

    docs: List[DocPair] = []
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            for url in urls:
                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                    label = f"web:{url}"
                    docs.append((label, resp.text))
                    logger.info("Fetched %s (%d chars)", label, len(resp.text))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to fetch %s: %s", url, exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Live fetch unavailable: %s", exc)
    return docs
