"""Application settings, read from environment variables / backend/.env (see .env.example)."""
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- Ollama (LLM + query embeddings) ---
    OLLAMA_HOST: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.2"
    LLM_TIMEOUT: float = 180.0          # seconds; local models can be slow on CPU
    NUM_CTX: int = 4096
    TEMPERATURE: float = 0.1

    # --- vector store (produced by notebooks/rag_pipeline.ipynb) ---
    VECTOR_STORE_DIR: Path = BACKEND_ROOT / "data" / "vector_store"
    EMBED_MODEL: Optional[str] = None   # default: the model recorded in vector_store/config.json
    TOP_K: Optional[int] = None         # default: value in config.json
    MAX_DISTANCE: Optional[float] = None  # refusal threshold; default: calibrated value in config.json

    # --- web ---
    CORS_ORIGINS: str = "http://localhost:8501,http://127.0.0.1:8501,http://localhost:7860"
    LOG_LEVEL: str = "INFO"

    @field_validator("VECTOR_STORE_DIR")
    @classmethod
    def _resolve_dir(cls, v: Path) -> Path:
        return v if v.is_absolute() else (BACKEND_ROOT / v).resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
