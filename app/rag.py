"""The RAG pipeline: ingest -> retrieve -> generate."""
from pathlib import Path
from typing import Dict, List, Optional

from app.config import ALLOWED_EXTENSIONS, DOCS_DIR, MIN_SCORE, TOP_K
from app.llm import generate_answer
from app.loader import build_chunks
from app.vectorstore import VectorStore

NOT_FOUND_MESSAGE = (
    "I couldn't find anything relevant to that in the uploaded documents. "
    "Try rephrasing the question or adding more documents."
)


class RAGPipeline:
    def __init__(self) -> None:
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        self.store = VectorStore()

    # ---------- ingestion ----------
    def ingest_file(self, path: Path) -> int:
        """Chunk, embed and store one file. Returns the number of chunks stored."""
        return self.store.add_chunks(build_chunks(Path(path)))

    def ingest_folder(self, folder: Optional[Path] = None) -> Dict[str, int]:
        folder = Path(folder or DOCS_DIR)
        files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in ALLOWED_EXTENSIONS]
        total_chunks = sum(self.ingest_file(p) for p in files)
        return {"files": len(files), "chunks": total_chunks}

    # ---------- question answering ----------
    def ask(self, question: str, history: Optional[List[Dict]] = None, top_k: int = TOP_K) -> Dict:
        hits = [h for h in self.store.search(question, top_k) if h["score"] >= MIN_SCORE]
        if not hits:
            return {"answer": NOT_FOUND_MESSAGE, "sources": []}

        answer = generate_answer(question, hits, history)
        sources = [
            {
                "source": h["metadata"]["source"],
                "page": h["metadata"].get("page") or None,
                "score": h["score"],
                "snippet": h["text"][:300],
            }
            for h in hits
        ]
        return {"answer": answer, "sources": sources}

    # ---------- management ----------
    def list_documents(self) -> List[str]:
        return self.store.list_sources()

    def delete_document(self, name: str) -> None:
        self.store.delete_source(name)
        target = DOCS_DIR / name
        if target.exists():
            target.unlink()
