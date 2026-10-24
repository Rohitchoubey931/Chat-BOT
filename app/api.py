"""FastAPI application exposing the RAG chatbot."""
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import __version__
from app.config import ALLOWED_EXTENSIONS, AUDIO_DIR, DOCS_DIR, TOP_K
from app.llm import LLMError

app = FastAPI(
    title="RAG Chatbot API",
    description="Ask questions about your own documents using retrieval-augmented generation.",
    version=__version__,
)

_pipeline = None


def get_pipeline():
    """Create the pipeline on first use so the app starts fast and tests don't load models."""
    global _pipeline
    if _pipeline is None:
        from app.rag import RAGPipeline

        _pipeline = RAGPipeline()
    return _pipeline


class Message(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    history: List[Message] = []
    top_k: int = Field(default=TOP_K, ge=1, le=10)
    voice: bool = False


class Source(BaseModel):
    source: str
    page: Optional[int] = None
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    audio_url: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.post("/ingest")
def ingest():
    """Index every supported file currently in the documents folder."""
    return get_pipeline().ingest_folder()


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Upload a PDF, TXT or MD file and index it immediately."""
    filename = Path(file.filename or "").name  # strips any path components
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    destination = DOCS_DIR / filename
    destination.write_bytes(await file.read())
    chunks = get_pipeline().ingest_file(destination)
    return {"filename": filename, "chunks": chunks}


@app.get("/documents")
def documents():
    return {"documents": get_pipeline().list_documents()}


@app.delete("/documents/{name}")
def delete_document(name: str):
    get_pipeline().delete_document(Path(name).name)
    return {"deleted": name}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    pipeline = get_pipeline()
    history = [m.model_dump() for m in request.history]
    try:
        result = pipeline.ask(request.question, history=history, top_k=request.top_k)
    except LLMError as exc:
        raise HTTPException(502, str(exc))

    audio_url = None
    if request.voice and result["sources"]:
        from app.tts import text_to_speech

        try:
            audio_url = f"/audio/{text_to_speech(result['answer']).name}"
        except Exception as exc:  # gTTS needs internet access
            raise HTTPException(502, f"Text-to-speech failed: {exc}")

    return AskResponse(answer=result["answer"], sources=result["sources"], audio_url=audio_url)


@app.get("/audio/{filename}")
def audio(filename: str):
    path = AUDIO_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(404, "Audio file not found")
    return FileResponse(path, media_type="audio/mpeg")
