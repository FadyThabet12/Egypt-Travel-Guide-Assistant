"""Thin wrapper around the backend REST API. The base URL always comes from the API_BASE_URL environment variable."""
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


class APIError(Exception):
    """An error whose message is safe to show to the end user."""


def get_base_url() -> str:
    url = os.getenv("API_BASE_URL", "").strip().rstrip("/")
    if not url:
        raise APIError("API_BASE_URL is not set. Copy frontend/.env.example to frontend/.env "
                       "(for example API_BASE_URL=http://localhost:8000) and restart the app.")
    return url


class ApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 300.0):
        self.base_url = (base_url or get_base_url()).rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            raise APIError(f"Cannot reach the backend at {self.base_url}.") from exc

    def ask(self, question: str) -> dict:
        """POST /query -> {"answer": str, "sources": list[str]}; raises APIError with a friendly message."""
        try:
            r = requests.post(f"{self.base_url}/query", json={"question": question}, timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise APIError(f"I can't reach the assistant service at {self.base_url}. "
                           "Please check that the backend is running.") from exc
        except requests.Timeout as exc:
            raise APIError("The assistant took too long to answer. The language model may still be loading - "
                           "please try again in a moment.") from exc
        if r.status_code == 422:
            raise APIError("Please type a question of at least 3 characters (and under 1000).")
        if r.status_code == 503:
            detail = r.json().get("detail", "") if r.headers.get("content-type", "").startswith("application/json") else ""
            raise APIError(f"The assistant is not ready yet. {detail}".strip())
        if not r.ok:
            raise APIError(f"Something went wrong on the server (HTTP {r.status_code}). Please try again.")
        data = r.json()
        return {"answer": data["answer"], "sources": data.get("sources", [])}
