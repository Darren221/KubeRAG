from pathlib import Path
from typing import Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KUBERAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    openai_api_key: SecretStr = Field(validation_alias="OPENAI_API_KEY")

    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o"
    judge_model: str = "gpt-4o-mini"

    chroma_path: Path = Path("./data/chroma")
    bm25_path: Path = Path("./data/bm25.pkl")
    raw_docs_path: Path = Path("./data/raw")

    retrieval_k: int = Field(default=10, ge=1, le=100)
    rrf_dense_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    rerank_top_n: int = Field(default=5, ge=1, le=50)

    dedupe_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    chunk_size: int = Field(default=800, ge=100, le=8000)
    chunk_overlap: int = Field(default=120, ge=0, le=2000)

    confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    api_url: str = "http://localhost:8000"

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self
