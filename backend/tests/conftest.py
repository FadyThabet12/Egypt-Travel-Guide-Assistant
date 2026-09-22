"""Shared test doubles: no real Ollama and no real vector store are needed to run the test-suite."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `import app` works from any cwd

from app.services.generation import GenerationResult  # noqa: E402
from app.services.retrieval import RetrievedChunk, UpstreamError  # noqa: E402

CHUNK = RetrievedChunk(text="Ticket prices: Visitor 450 EGP, Student 230 EGP.", source="Aswan Guide – Temple of Edfu (p. 4)",
                       doc_id="aswan-guide", page=4, distance=0.31)


class FakeRetrieval:
    max_distance = 0.7
    n_chunks = 234
    fail = None

    def __init__(self, *a, **k):
        pass

    def retrieve(self, question, k=None):
        if FakeRetrieval.fail:
            raise FakeRetrieval.fail
        return [CHUNK]


class FakeGeneration:
    model = "fake-llm"

    def __init__(self, *a, **k):
        pass

    def ping(self):
        return True

    def generate(self, question, chunks, max_distance=None):
        return GenerationResult(answer="A visitor ticket costs 450 EGP [1].", sources=[CHUNK.source])


@pytest.fixture
def client(monkeypatch):
    import app.main as main
    FakeRetrieval.fail = None
    monkeypatch.setattr(main, "RetrievalService", FakeRetrieval)
    monkeypatch.setattr(main, "GenerationService", FakeGeneration)
    monkeypatch.setattr(main, "build_ollama_client", lambda settings: object())
    with TestClient(main.create_app()) as c:      # entering the context runs the lifespan (startup loading)
        yield c


@pytest.fixture
def client_without_store(monkeypatch):
    import app.main as main

    def boom(*a, **k):
        raise FileNotFoundError("vector store missing")

    monkeypatch.setattr(main, "RetrievalService", boom)
    monkeypatch.setattr(main, "build_ollama_client", lambda settings: object())
    with TestClient(main.create_app()) as c:
        yield c
