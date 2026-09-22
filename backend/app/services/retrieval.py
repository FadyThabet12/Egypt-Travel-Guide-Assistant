"""Load the persisted Chroma vector store once and retrieve the chunks closest to a question."""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb
import httpx
import ollama
from chromadb.config import Settings as ChromaSettings

from app.core.config import Settings

logger = logging.getLogger(__name__)


class UpstreamError(RuntimeError):
    """Ollama could not be reached or rejected the request (mapped to HTTP 503 by the API layer)."""


@dataclass
class RetrievedChunk:
    text: str
    source: str
    doc_id: str
    page: int
    distance: float


def build_ollama_client(settings: Settings) -> ollama.Client:
    return ollama.Client(host=settings.OLLAMA_HOST, timeout=settings.LLM_TIMEOUT)


def wrap_upstream(exc: Exception, model: str) -> UpstreamError:
    if isinstance(exc, ollama.ResponseError) and getattr(exc, "status_code", None) == 404:
        return UpstreamError(f"Model '{model}' is not installed in Ollama. Run: ollama pull {model}")
    return UpstreamError(f"The language-model service (Ollama) is unavailable: {exc}")


class RetrievalService:
    """Opens the vector store exported by the notebook. Nothing is rebuilt at request time."""

    def __init__(self, settings: Settings, ollama_client: ollama.Client | None = None):
        store: Path = settings.VECTOR_STORE_DIR
        cfg_file = store / "config.json"
        if not cfg_file.exists():
            raise FileNotFoundError(
                f"{cfg_file} not found. Run notebooks/rag_pipeline.ipynb (section 2.7 Export) to create the vector store.")
        self.config = json.loads(cfg_file.read_text(encoding="utf-8"))
        self._client = chromadb.PersistentClient(path=str(store), settings=ChromaSettings(anonymized_telemetry=False))
        self._collection = self._client.get_collection(self.config["collection_name"])
        self.embed_model: str = settings.EMBED_MODEL or self.config["embedding_model"]
        self.query_prefix: str = self.config.get("query_prefix", "")
        self.top_k: int = settings.TOP_K or int(self.config.get("top_k", 5))
        self.max_distance: float | None = (settings.MAX_DISTANCE if settings.MAX_DISTANCE is not None
                                           else self.config.get("max_distance"))
        self._ollama = ollama_client or build_ollama_client(settings)
        logger.info("Vector store loaded: %d chunks, embedding model %s, top_k=%d, max_distance=%s",
                    self.n_chunks, self.embed_model, self.top_k, self.max_distance)

    @property
    def n_chunks(self) -> int:
        return self._collection.count()

    def embed_query(self, question: str) -> list[float]:
        try:
            resp = self._ollama.embed(model=self.embed_model, input=[self.query_prefix + question])
        except (ConnectionError, httpx.HTTPError, ollama.ResponseError) as exc:
            raise wrap_upstream(exc, self.embed_model) from exc
        return list(getattr(resp, "embeddings", None) or resp["embeddings"])[0]

    def retrieve(self, question: str, k: int | None = None) -> list[RetrievedChunk]:
        res = self._collection.query(query_embeddings=[self.embed_query(question)], n_results=k or self.top_k,
                                     include=["documents", "metadatas", "distances"])
        return [RetrievedChunk(text=doc, source=meta["source"], doc_id=meta["doc_id"], page=int(meta["page"]),
                               distance=float(dist))
                for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]
