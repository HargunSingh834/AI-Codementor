"""
FastAPI application — routes, SSE streaming, startup events.
"""

import os
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Load .env from backend/ directory before any other imports that use env vars
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

from backend.pipeline import process_message
from backend.session.store import clear_session, get_session
from backend.rag.chromadb_setup import get_collection, get_client


# ── Lifespan ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: verify ChromaDB and Groq are accessible
    print("🚀 Starting CodeMentor AI...")
    collection = get_collection()
    print(f"📚 ChromaDB collection '{collection.name}' has {collection.count()} problems")
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key or groq_key == "your_groq_api_key_here":
        print("⚠️  WARNING: GROQ_API_KEY not set! LLM calls will fail.")
    else:
        print("✅ Groq API key configured")
    yield
    # Shutdown
    print("👋 Shutting down CodeMentor AI")


# ── App ─────────────────────────────────────────────
app = FastAPI(
    title="CodeMentor AI",
    description="Socratic LeetCode coding tutor — never reveals answers",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ─────────────────────────
class ChatRequest(BaseModel):
    conversation_id: str
    message: str


class ResetRequest(BaseModel):
    conversation_id: str


# ── Routes ──────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Main endpoint. Runs full pipeline. Streams response via SSE.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    async def event_stream():
        async for token in process_message(req.conversation_id, req.message):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/session/reset")
async def reset_session(req: ResetRequest):
    """Clear session. Starts a fresh conversation."""
    clear_session(req.conversation_id)
    return {"success": True}


@app.get("/session/{conversation_id}")
async def get_session_info(conversation_id: str):
    """Returns current session state (for debugging / frontend display)."""
    session = get_session(conversation_id)
    problem = session.get("problem")
    return {
        "problem": {
            "title": problem["title"],
            "id": problem.get("id"),
            "difficulty": problem.get("difficulty"),
        } if problem else None,
        "history_length": len(session.get("history", [])),
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        client = get_client()
        client.heartbeat()
        chromadb_status = "connected"
    except Exception:
        chromadb_status = "disconnected"

    groq_key = os.getenv("GROQ_API_KEY", "")
    groq_status = "configured" if groq_key and groq_key != "your_groq_api_key_here" else "not_configured"

    return {
        "status": "ok",
        "chromadb": chromadb_status,
        "groq": groq_status,
    }
