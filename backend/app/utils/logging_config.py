import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once (idempotent)."""
    root = logging.getLogger()
    if getattr(root, "_rag_configured", False):
        root.setLevel(level.upper())
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
    root.handlers = [handler]
    root.setLevel(level.upper())
    root._rag_configured = True  # type: ignore[attr-defined]
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
