from tests.conftest import FakeRetrieval, UpstreamError


def test_query_happy_path(client):
    r = client.post("/query", json={"question": "How much is a visitor ticket to the Temple of Edfu?"})
    assert r.status_code == 200
    body = r.json()
    assert "450 EGP" in body["answer"]
    assert body["sources"] == ["Aswan Guide – Temple of Edfu (p. 4)"]


def test_query_invalid_input_returns_422(client):
    assert client.post("/query", json={}).status_code == 422                       # missing field
    assert client.post("/query", json={"question": "   "}).status_code == 422      # blank
    assert client.post("/query", json={"question": "hi"}).status_code == 422       # too short
    assert client.post("/query", json={"question": "x" * 1001}).status_code == 422  # too long
    assert client.post("/query", json={"question": 123}).status_code == 422        # wrong type


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["vector_store_loaded"] is True and body["chunks"] == 234


def test_health_degraded_and_query_503_when_store_missing(client_without_store):
    h = client_without_store.get("/health").json()
    assert h["status"] == "degraded" and h["vector_store_loaded"] is False
    assert client_without_store.post("/query", json={"question": "anything at all"}).status_code == 503


def test_query_returns_503_when_ollama_is_down(client):
    FakeRetrieval.fail = UpstreamError("Ollama is not running")
    r = client.post("/query", json={"question": "How far is Luxor from Cairo?"})
    assert r.status_code == 503
    assert "Ollama" in r.json()["detail"]


def test_cors_header_for_allowed_origin(client):
    r = client.get("/health", headers={"Origin": "http://localhost:8501"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:8501"
