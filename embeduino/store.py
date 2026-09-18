"""Chroma local vector store persistence."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from embeduino.chunker import Chunk
from embeduino.embeddings import Embedder

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        persist_path: Path,
        collection_name: str,
        embedder: Embedder,
    ):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._client = chromadb.PersistentClient(
            path=str(self.persist_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection_name = collection_name

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Reset collection %s", self.collection_name)

    def upsert_chunks(self, chunks: Sequence[Chunk], batch_size: int = 32) -> int:
        if not chunks:
            return 0
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = list(chunks[i : i + batch_size])
            texts = [c.text for c in batch]
            embeddings = self.embedder.embed(texts)
            ids = [c.id for c in batch]
            metadatas = [
                {
                    "source": c.source,
                    "heading_path": c.heading_path or "",
                    "chunk_type": c.chunk_type,
                    "preview": c.preview(160),
                }
                for c in batch
            ]
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            total += len(batch)
            logger.info("Upserted batch %d–%d", i, i + len(batch) - 1)
        return total

    def query(
        self,
        question: str,
        top_k: int = 4,
    ) -> List[Dict[str, Any]]:
        q_emb = self.embedder.embed_one(question)
        result = self._collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits: List[Dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for i, cid in enumerate(ids):
            # cosine distance → similarity
            dist = float(dists[i]) if dists else 1.0
            score = 1.0 - dist
            hits.append(
                {
                    "id": cid,
                    "text": docs[i] if docs else "",
                    "metadata": metas[i] if metas else {},
                    "score": score,
                    "distance": dist,
                }
            )
        return hits

    def count(self) -> int:
        return self._collection.count()
