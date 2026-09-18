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

    @classmethod
    def from_env(cls) -> "Settings":
        chroma = os.getenv("EMBEDUINO_CHROMA_PATH", ".chroma")
        chroma_path = Path(chroma)
        if not chroma_path.is_absolute():
            chroma_path = PROJECT_ROOT / chroma_path
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
        )


def get_settings() -> Settings:
    return Settings.from_env()
