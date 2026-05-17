"""
FastAPI entrypoint for the CV-RAG service.

Routes:
    GET  /             — chat-style web UI (static/index.html)
    GET  /health       — liveness check
    POST /ask          — JSON answer (optionally with retrieved sources)
    POST /sources      — only the retrieved sources (no LLM call)
    POST /ask-stream   — streamed answer tokens (for the UI)
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag import get_sources, init_rag, run_query, run_query_stream


# ── Request / Response models ────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    show_context: bool = False


class Source(BaseModel):
    student_name: Optional[str] = None
    doc_name: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    question: str
    answer: str
    context: Optional[str] = None
    sources: Optional[List[Source]] = None


class SourcesResponse(BaseModel):
    question: str
    sources: List[Source]


# ── Lifespan: runs init_rag() ONCE on startup ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_rag()
    yield


app = FastAPI(
    title="CV RAG API",
    description="Ask HR questions against a pool of CVs stored in Pinecone.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the UI (and curl from any origin) to call the API freely.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static UI ────────────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def root():
    """Serve the chat UI."""
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return {"message": "UI not found. Visit /docs for the API."}
    return FileResponse(index)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Quick status check — confirms the app is running."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Run the full RAG pipeline and return a JSON answer.

    Body:
        {"question": "Who has Python experience?", "show_context": false}

    When show_context=true the response also includes:
        - context: labelled string passed to the LLM (debug)
        - sources: structured list of CV chunks for the UI
    """
    try:
        result = run_query(request.question, request.show_context)
        return AskResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sources", response_model=SourcesResponse)
def sources(request: AskRequest):
    """Return the retrieved CV chunks for a question — no LLM call."""
    try:
        return SourcesResponse(
            question=request.question,
            sources=get_sources(request.question),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask-stream")
def ask_stream(request: AskRequest):
    """
    Same as /ask but streams answer tokens as they are generated.
    Used by the chat UI for the typewriter effect.
    """
    try:
        return StreamingResponse(
            run_query_stream(request.question),
            media_type="text/plain",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
