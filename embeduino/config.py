"""Configuration loaded from environment / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root: parent of the embeduino package
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    embedding_backend: str = "local"
    local_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_api_key: str = ""
    openai_embed_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    generation_backend: str = "local"
    top_k: int = 4
    min_score: float = 0.30
    chroma_path: Path = PROJECT_ROOT / ".chroma"
    collection: str = "embeduino_docs"
    data_dir: Path = PROJECT_ROOT / "data" / "arduino_docs"
    
    serpapi_api_key: str = ""
    web_mode: str = "auto"
    web_num: int = 5
    web_sites: list[str] = None
    serp_cache: Path = PROJECT_ROOT / ".serp_cache"
    serp_max_calls: int = 10
    rrf_k: int = 60

    @classmethod
    def from_env(cls) -> "Settings":
        chroma = os.getenv("EMBEDUINO_CHROMA_PATH", ".chroma")
        chroma_path = Path(chroma)
        if not chroma_path.is_absolute():
            chroma_path = PROJECT_ROOT / chroma_path
        
        serp_cache = os.getenv("EMBEDUINO_SERP_CACHE", ".serp_cache")
        serp_cache_path = Path(serp_cache)
        if not serp_cache_path.is_absolute():
            serp_cache_path = PROJECT_ROOT / serp_cache_path
        
        web_sites_str = os.getenv(
            "EMBEDUINO_WEB_SITES",
            "docs.arduino.cc github.com forum.arduino.cc"
        )
        web_sites = [s.strip() for s in web_sites_str.split() if s.strip()]
        
        return cls(
            embedding_backend=os.getenv("EMBEDUINO_EMBEDDING_BACKEND", "local").lower(),
            local_model=os.getenv(
                "EMBEDUINO_LOCAL_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_embed_model=os.getenv(
                "EMBEDUINO_OPENAI_EMBED_MODEL", "text-embedding-3-small"
            ),
            openai_chat_model=os.getenv("EMBEDUINO_OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            generation_backend=os.getenv("EMBEDUINO_GENERATION_BACKEND", "local").lower(),
            top_k=int(os.getenv("EMBEDUINO_TOP_K", "4")),
            min_score=float(os.getenv("EMBEDUINO_MIN_SCORE", "0.30")),
            chroma_path=chroma_path,
            collection=os.getenv("EMBEDUINO_COLLECTION", "embeduino_docs"),
            data_dir=PROJECT_ROOT / "data" / "arduino_docs",
            serpapi_api_key=os.getenv("SERPAPI_API_KEY", ""),
            web_mode=os.getenv("EMBEDUINO_WEB_MODE", "auto").lower(),
            web_num=int(os.getenv("EMBEDUINO_WEB_NUM", "5")),
            web_sites=web_sites,
            serp_cache=serp_cache_path,
            serp_max_calls=int(os.getenv("EMBEDUINO_SERP_MAX_CALLS", "10")),
            rrf_k=int(os.getenv("EMBEDUINO_RRF_K", "60")),
        )


def get_settings() -> Settings:
    return Settings.from_env()
