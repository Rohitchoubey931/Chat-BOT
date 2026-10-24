"""Central configuration. Every value can be overridden through environment variables or a .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Storage locations
DOCS_DIR = Path(os.getenv("DOCS_DIR", BASE_DIR / "data" / "docs"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "data" / "chroma"))
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", BASE_DIR / "data" / "audio"))

# Embeddings (runs locally, no API key needed)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking and retrieval
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))        # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))  # characters shared between chunks
TOP_K = int(os.getenv("TOP_K", 4))                    # chunks passed to the LLM
MIN_SCORE = float(os.getenv("MIN_SCORE", 0.2))        # cosine similarity below this = "not found"

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()  # "anthropic" or "openai"
LLM_MODEL = os.getenv("LLM_MODEL", "")  # empty = use the provider default below
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1000))
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", 6))    # previous messages kept for follow-up questions

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-4o-mini",
}

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
