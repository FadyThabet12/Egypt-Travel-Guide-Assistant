"""FastAPI application: CORS, one-time startup loading (lifespan), routes."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import get_settings
from app.services.generation import GenerationService
from app.services.retrieval import RetrievalService, build_ollama_client
from app.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the vector store and the Ollama connection ONCE, not on every request."""
    settings = get_settings()
    setup_logging(settings.LOG_LEVEL)
    app.state.settings = settings
    app.state.retrieval = None
    app.state.generation = None
    try:
        client = build_ollama_client(settings)
        app.state.retrieval = RetrievalService(settings, client)
        app.state.generation = GenerationService(settings, client)
        logger.info("Startup complete: LLM=%s, Ollama=%s", settings.LLM_MODEL, settings.OLLAMA_HOST)
    except Exception:  # keep the API up so /health can explain what is wrong
        logger.exception("Startup failed: the API will answer 503 on /query until this is fixed")
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Egypt Travel Guide RAG API", version="1.0.0", lifespan=lifespan,
                  description="Grounded, cited answers about Egypt travel from four official tourism guides.")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["*"])
    app.include_router(query_router)
    return app


app = create_app()
