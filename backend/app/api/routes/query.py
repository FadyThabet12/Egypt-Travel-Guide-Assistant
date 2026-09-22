import logging

from fastapi import APIRouter, HTTPException, Request

from app.schemas.query import HealthResponse, QueryRequest, QueryResponse
from app.services.retrieval import UpstreamError

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    """Liveness + readiness: is the vector store loaded and is Ollama reachable?"""
    retrieval = getattr(request.app.state, "retrieval", None)
    generation = getattr(request.app.state, "generation", None)
    reachable = generation.ping() if generation else False
    return HealthResponse(
        status="ok" if (retrieval is not None and reachable) else "degraded",
        vector_store_loaded=retrieval is not None,
        chunks=retrieval.n_chunks if retrieval is not None else 0,
        llm_model=generation.model if generation else request.app.state.settings.LLM_MODEL,
        ollama_reachable=reachable,
    )


@router.post("/query", response_model=QueryResponse, tags=["rag"])
def query(payload: QueryRequest, request: Request) -> QueryResponse:
    """retrieve -> build prompt -> call LLM -> return a grounded, cited answer."""
    retrieval = getattr(request.app.state, "retrieval", None)
    generation = getattr(request.app.state, "generation", None)
    if retrieval is None or generation is None:
        raise HTTPException(status_code=503, detail="Vector store not loaded. Run the notebook export step, then restart the API.")
    try:
        chunks = retrieval.retrieve(payload.question)
        result = generation.generate(payload.question, chunks, max_distance=retrieval.max_distance)
    except UpstreamError as exc:
        logger.error("Upstream error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    logger.info("Q=%r -> %d chunks (best %.3f), gated=%s, sources=%s", payload.question, len(chunks),
                chunks[0].distance if chunks else -1, result.gated, result.sources)
    return QueryResponse(answer=result.answer, sources=result.sources)
