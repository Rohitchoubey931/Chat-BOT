"""Document loading and chunking."""
import hashlib
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader

from app.config import ALLOWED_EXTENSIONS, CHUNK_OVERLAP, CHUNK_SIZE


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks, preferring to cut at a space instead of mid-word."""
    text = " ".join(text.split())
    if not text:
        return []
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # always moves forward
    return chunks


def load_file(path: Path) -> List[Dict]:
    """Read a file and return a list of {text, source, page}. page is 0 for files without pages."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [
            {"text": page.extract_text() or "", "source": path.name, "page": i + 1}
            for i, page in enumerate(reader.pages)
        ]
    return [{"text": path.read_text(encoding="utf-8", errors="ignore"), "source": path.name, "page": 0}]


def build_chunks(path: Path) -> List[Dict]:
    """Load a file and return chunks ready for the vector store: {id, text, metadata}."""
    chunks: List[Dict] = []
    for page in load_file(path):
        for idx, piece in enumerate(chunk_text(page["text"])):
            raw_id = f"{page['source']}:{page['page']}:{idx}:{piece}"
            chunks.append(
                {
                    "id": hashlib.sha1(raw_id.encode("utf-8")).hexdigest(),
                    "text": piece,
                    "metadata": {"source": page["source"], "page": page["page"], "chunk": idx},
                }
            )
    return chunks
