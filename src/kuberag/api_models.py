from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str = Field(min_length=1, description="The question to answer.")
    dense_only: bool = Field(
        default=False,
        description="If true, skip sparse retrieval and RRF (debug / A-B toggle).",
    )
    k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Per-retriever candidate count before fusion.",
    )
    top_n: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Final reranked top-N returned to the generator.",
    )
