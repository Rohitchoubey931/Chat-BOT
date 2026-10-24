"""Embeddings + persistent ChromaDB vector store."""
from typing import Dict, List

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import CHROMA_DIR, EMBEDDING_MODEL


class VectorStore:
    def __init__(self) -> None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name="documents", metadata={"hnsw:space": "cosine"}
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def add_chunks(self, chunks: List[Dict], batch_size: int = 256) -> int:
        """Embed and upsert chunks. Re-ingesting the same file does not create duplicates."""
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            self.collection.upsert(
                ids=[c["id"] for c in batch],
                documents=texts,
                embeddings=self._embed(texts),
                metadatas=[c["metadata"] for c in batch],
            )
        return len(chunks)

    def search(self, query: str, top_k: int) -> List[Dict]:
        """Return the top_k most similar chunks with a similarity score (1.0 = identical)."""
        total = self.collection.count()
        if total == 0:
            return []
        result = self.collection.query(
            query_embeddings=self._embed([query]),
            n_results=min(top_k, total),
        )
        hits = []
        for text, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append({"text": text, "metadata": meta, "score": round(1 - dist, 4)})
        return hits

    def list_sources(self) -> List[str]:
        if self.collection.count() == 0:
            return []
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in metas})

    def delete_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})

    def reset(self) -> None:
        self.client.delete_collection("documents")
        self.collection = self.client.get_or_create_collection(
            name="documents", metadata={"hnsw:space": "cosine"}
        )
