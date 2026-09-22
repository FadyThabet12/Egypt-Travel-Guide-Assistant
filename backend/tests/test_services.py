"""Tests of the real RetrievalService / GenerationService against a tiny real Chroma store and a fake Ollama."""
import json
import re

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from app.core.config import Settings
from app.services.generation import NOT_FOUND, GenerationService, build_messages, extract_sources
from app.services.retrieval import RetrievalService, RetrievedChunk

VOCAB = ["edfu", "ticket", "alexandria", "distance", "student", "museum", "temperature", "oasis"]


def vec(text: str) -> list[float]:
    words = set(re.findall(r"[a-z]+", text.lower()))
    v = [1.0 if w in words else 0.0 for w in VOCAB]
    return v + [0.0 if any(v) else 1.0]          # extra "unknown topic" dimension: orthogonal to every stored chunk


class FakeOllama:
    def __init__(self):
        self.chat_calls = 0

    def embed(self, model, input):
        return {"embeddings": [vec(t.replace("search_query: ", "")) for t in input]}

    def chat(self, model, messages, options=None):
        self.chat_calls += 1
        return {"message": {"content": "A visitor ticket to Edfu costs 450 EGP [2]."}}

    def list(self):
        return {"models": []}


@pytest.fixture
def store(tmp_path):
    docs = [
        ("c1", "Alexandria museum opening hours are 9am-4:30pm.", "Alexandria Guide (p. 29)"),
        ("c2", "Temple of Edfu ticket prices: Visitor 450 EGP, Student 230 EGP.", "Aswan Guide – Temple of Edfu (p. 4)"),
        ("c3", "Average temperature in the oasis in August.", "Egypt Map & Guide (p. 2)"),
    ]
    client = chromadb.PersistentClient(path=str(tmp_path), settings=ChromaSettings(anonymized_telemetry=False))
    col = client.create_collection("test_col", metadata={"hnsw:space": "cosine"}, embedding_function=None)
    col.add(ids=[d[0] for d in docs], documents=[d[1] for d in docs], embeddings=[vec(d[1]) for d in docs],
            metadatas=[{"doc_id": "x", "doc_short": "x", "page": 1, "section": "", "kind": "prose", "method": "t", "source": d[2]} for d in docs])
    (tmp_path / "config.json").write_text(json.dumps(dict(
        collection_name="test_col", embedding_model="fake-embed", query_prefix="search_query: ", top_k=2, max_distance=0.5)))
    return tmp_path


def make(store):
    settings = Settings(VECTOR_STORE_DIR=store, _env_file=None)
    fake = FakeOllama()
    return RetrievalService(settings, fake), GenerationService(settings, fake), fake


def test_retrieval_returns_closest_chunk_first(store):
    retrieval, _, _ = make(store)
    hits = retrieval.retrieve("What is the ticket price for Edfu?")
    assert len(hits) == 2 and hits[0].source.endswith("Temple of Edfu (p. 4)")
    assert hits[0].distance <= hits[1].distance
    assert retrieval.n_chunks == 3 and retrieval.max_distance == 0.5 and retrieval.embed_model == "fake-embed"


def test_missing_store_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Export"):
        RetrievalService(Settings(VECTOR_STORE_DIR=tmp_path, _env_file=None), FakeOllama())


def test_generation_cites_sources_and_maps_numbers(store):
    retrieval, generation, _ = make(store)
    hits = retrieval.retrieve("Edfu ticket price")
    result = generation.generate("Edfu ticket price", hits, max_distance=retrieval.max_distance)
    assert not result.gated and "450 EGP" in result.answer
    assert result.sources == [hits[1].source]       # the model cited passage [2] ...
    assert "[1]" in result.answer and "[2]" not in result.answer    # ... which is renumbered to [1] to match `sources`


def test_out_of_scope_question_is_refused_without_calling_the_llm(store):
    retrieval, generation, fake = make(store)
    hits = retrieval.retrieve("Who won the football match?")      # no vocabulary overlap -> far from everything
    result = generation.generate("Who won the football match?", hits, max_distance=retrieval.max_distance)
    assert result.gated and result.answer == NOT_FOUND and result.sources == [] and fake.chat_calls == 0


def test_extract_sources_rules():
    chunks = [RetrievedChunk("a", "S1", "d", 1, 0.1), RetrievedChunk("b", "S2", "d", 2, 0.2)]
    assert extract_sources("Fact [2] and again [2][1].", chunks) == ("Fact [1] and again [1][2].", ["S2", "S1"])
    assert extract_sources("No citation here.", chunks) == ("No citation here.", ["S1"])     # falls back to top passage
    assert extract_sources(NOT_FOUND, chunks) == (NOT_FOUND, [])                            # refusal has no sources
    assert extract_sources("Bogus [9].", chunks) == ("Bogus.", ["S1"])                     # out-of-range citation dropped


def test_prompt_contains_numbered_passages_and_question():
    chunks = [RetrievedChunk("Text one", "S1", "d", 1, 0.1)]
    msgs = build_messages("Q?", chunks)
    assert msgs[0]["role"] == "system" and "ONLY" in msgs[0]["content"]
    assert "[1] (S1)\nText one" in msgs[1]["content"] and "Question: Q?" in msgs[1]["content"]
