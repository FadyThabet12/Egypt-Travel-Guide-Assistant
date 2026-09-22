"""Build the grounded prompt, call the local Ollama LLM and extract the cited sources."""
import logging
import re
from dataclasses import dataclass, field

import httpx
import ollama

from app.core.config import Settings
from app.services.retrieval import RetrievedChunk, build_ollama_client, wrap_upstream

logger = logging.getLogger(__name__)

# Keep in sync with the notebook (section 2.4): the prompt is what the evaluation measured.
NOT_FOUND = "I couldn't find that in the provided documents."

SYSTEM_PROMPT = (
    "You answer questions about travel in Egypt using ONLY the numbered context passages you are given.\n"
    "Rules:\n"
    "1. Use only facts stated in the passages. Never add outside knowledge.\n"
    "2. After every fact, cite the passage number in square brackets, for example [1] or [2][3].\n"
    "3. Copy prices, times, distances and numbers exactly as written in the passages.\n"
    "4. If passages give different values, mention both and cite each one.\n"
    f"5. If the passages do not contain the answer, reply exactly: {NOT_FOUND}\n"
    "Keep the answer short (1-4 sentences)."
)


@dataclass
class GenerationResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    gated: bool = False          # True when no LLM call was made because nothing relevant was retrieved


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    context = "\n\n".join(f"[{i}] ({c.source})\n{c.text}" for i, c in enumerate(chunks, 1))
    user = f"Context passages:\n{context}\n\nQuestion: {question}\nAnswer:"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def extract_sources(answer: str, chunks: list[RetrievedChunk]) -> tuple[str, list[str]]:
    """Map the model's [n] citations (position in the retrieved list) to source labels.

    Returns (answer, sources) where the citations inside the answer are renumbered so that
    [1], [2]... match the order of the returned `sources` list. Refusal -> no sources; no citation -> top passage.
    """
    if NOT_FOUND.lower()[:20] in answer.lower():
        return answer, []
    used: list[int] = []
    for n in (int(x) for x in re.findall(r"\[(\d+)\]", answer)):
        if 1 <= n <= len(chunks) and n not in used:
            used.append(n)
    if not used:                                   # no valid citation: drop stray markers, fall back to the top passage
        return re.sub(r"\s*\[\d+\]", "", answer).strip(), ([chunks[0].source] if chunks else [])
    sources: list[str] = []
    number: dict[int, int] = {}
    for n in used:
        label = chunks[n - 1].source
        if label not in sources:
            sources.append(label)
        number[n] = sources.index(label) + 1
    renumbered = re.sub(r"\[(\d+)\]", lambda m: f"[{number[int(m.group(1))]}]" if int(m.group(1)) in number else "", answer)
    return re.sub(r"\s{2,}", " ", renumbered).strip(), sources


class GenerationService:
    def __init__(self, settings: Settings, ollama_client: ollama.Client | None = None):
        self.model = settings.LLM_MODEL
        self._options = {"temperature": settings.TEMPERATURE, "num_ctx": settings.NUM_CTX}
        self._ollama = ollama_client or build_ollama_client(settings)

    def ping(self) -> bool:
        try:
            self._ollama.list()
            return True
        except Exception:  # noqa: BLE001 - any failure means "not reachable"
            return False

    def generate(self, question: str, chunks: list[RetrievedChunk], max_distance: float | None = None) -> GenerationResult:
        if not chunks or (max_distance is not None and chunks[0].distance > max_distance):
            logger.info("Refusing without LLM call (best distance %s > %s)", chunks[0].distance if chunks else None, max_distance)
            return GenerationResult(answer=NOT_FOUND, sources=[], gated=True)
        try:
            resp = self._ollama.chat(model=self.model, messages=build_messages(question, chunks), options=self._options)
        except (ConnectionError, httpx.HTTPError, ollama.ResponseError) as exc:
            raise wrap_upstream(exc, self.model) from exc
        message = getattr(resp, "message", None)
        answer = (getattr(message, "content", None) or resp["message"]["content"]).strip()
        answer, sources = extract_sources(answer, chunks)
        return GenerationResult(answer=answer, sources=sources)
