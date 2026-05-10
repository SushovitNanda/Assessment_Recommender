"""
main.py
FastAPI application entry point.

Endpoints:
  GET  /health  — readiness check (returns {"status": "ok"})
  POST /chat    — stateless conversational agent endpoint

All heavy state (catalog, FAISS index, LLM client) is loaded
once during startup via the lifespan context manager so the
first real /chat call is fast.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from openai import OpenAI

from agent import run_agent, _make_client, get_gemini_model_id
from catalog import load_catalog, build_catalog_index
from retriever import SHLRetriever
from schemas import ChatRequest, ChatResponse, HealthResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application state container
# ---------------------------------------------------------------------------
class AppState:
    catalog: list[dict] = []
    name_index: dict[str, dict] = {}
    retriever: SHLRetriever = None
    llm_client: OpenAI = None


state = AppState()


# ---------------------------------------------------------------------------
# Lifespan — runs once on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all heavy resources at startup:
      1. Catalog JSON
      2. FAISS index (build or load from cache)
      3. LLM client

    This runs during the cold-start window (up to 2 min allowed by evaluator)
    so that the first /chat call is fast.
    """
    logger.info("=== SHL Recommender startup ===")

    # 1. Load catalog
    logger.info("Loading catalog...")
    state.catalog = load_catalog("data/catalog.json")
    state.name_index = build_catalog_index(state.catalog)
    logger.info(f"Catalog ready: {len(state.catalog)} assessments.")

    # 2. Build / load FAISS index
    logger.info("Initializing retriever...")
    state.retriever = SHLRetriever()
    state.retriever.build_index(
        catalog=state.catalog,
        name_index=state.name_index,
        force_rebuild=False,   # uses cached index if available
    )
    logger.info("Retriever ready.")

    # 3. LLM client
    logger.info("Initializing LLM client...")
    state.llm_client = _make_client()
    logger.info(
        "Gemini 3.1 Flash Lite chat-completions model id (this string is sent on every LLM request): "
        f"{get_gemini_model_id()!r}"
    )
    logger.info("LLM client ready.")

    logger.info("=== Startup complete. Service is ready. ===")
    yield
    # Shutdown (nothing to clean up for this stack)
    logger.info("=== SHL Recommender shutdown ===")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for SHL Individual Test Solutions.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "reply": "An unexpected error occurred. Please try again.",
            "recommendations": [],
            "end_of_conversation": False,
        },
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, summary="Readiness check")
async def health():
    """
    Returns {"status": "ok"} when the service is ready.
    The evaluator allows up to 2 minutes on first call for cold start.
    """
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse, summary="Conversational agent")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Stateless conversational endpoint.
    Every call must include the full conversation history in `messages`.
    Returns the agent's next reply plus, when appropriate, a shortlist of recommendations.

    Request body:
        messages: list of {"role": "user"|"assistant", "content": "..."}

    Response:
        reply: str
        recommendations: list of {name, url, test_type}  — test_type lists all applicable codes (comma-separated); empty when clarifying
        end_of_conversation: bool
    """
    # Basic validation
    if not request.messages:
        raise HTTPException(status_code=422, detail="messages list cannot be empty.")

    # Ensure last message is from user
    if request.messages[-1].role != "user":
        raise HTTPException(
            status_code=422,
            detail="Last message in history must be from 'user'.",
        )

    # Turn cap enforcement: max 8 turns = 4 user + 4 assistant
    # We don't reject the call but we track it for the force-recommend logic in agent.py
    if len(request.messages) > 16:
        # More than 16 messages is almost certainly a bug in the caller
        raise HTTPException(
            status_code=422,
            detail="Conversation history exceeds maximum length.",
        )

    # Run agent
    response = run_agent(
        messages=request.messages,
        retriever=state.retriever,
        catalog=state.catalog,
        client=state.llm_client,
    )

    return response
