"""Embedding backends: local sentence-transformers or OpenAI."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Sequence

from embeduino.config import Settings

logger = logging.getLogger(__name__)


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        ...

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


class LocalEmbedder(Embedder):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        logger.info("Loading local embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._model.encode(
            list(texts),
            show_progress_bar=len(texts) > 8,
            normalize_embeddings=True,
        )
        return [v.tolist() for v in vectors]


class OpenAIEmbedder(Embedder):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        # batch in chunks of 64
        out: List[List[float]] = []
        batch_size = 64
        items = list(texts)
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            ordered = sorted(resp.data, key=lambda d: d.index)
            out.extend([d.embedding for d in ordered])
        return out


def get_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "openai":
        return OpenAIEmbedder(settings.openai_api_key, settings.openai_embed_model)
    return LocalEmbedder(settings.local_model)
