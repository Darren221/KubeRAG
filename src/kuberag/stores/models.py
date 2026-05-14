from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Hit(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    text: str
    source: str
    score: float
    rank: int
    section: str | None = None
    chunking_strategy: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
