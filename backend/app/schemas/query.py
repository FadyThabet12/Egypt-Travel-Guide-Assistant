from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="The user's question about Egypt travel.",
                          examples=["How much is a visitor ticket to the Temple of Edfu?"])

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("question must contain at least 3 non-space characters")
        return v


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Answer grounded in the retrieved passages, with [n] citations.")
    sources: list[str] = Field(default_factory=list, description="Sources (document, section, page) the answer cites.")


class HealthResponse(BaseModel):
    status: str
    vector_store_loaded: bool
    chunks: int = 0
    llm_model: str
    ollama_reachable: bool
