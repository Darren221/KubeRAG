from pathlib import Path

import pytest
from pydantic import ValidationError

from kuberag.config import Settings


def test_loads_with_explicit_values() -> None:
    s = Settings(
        OPENAI_API_KEY="sk-test",
        chroma_path=Path("/tmp/chroma"),
        bm25_path=Path("/tmp/bm25.pkl"),
        raw_docs_path=Path("/tmp/raw"),
    )
    assert s.openai_api_key.get_secret_value() == "sk-test"
    assert s.chroma_path == Path("/tmp/chroma")
    assert s.bm25_path == Path("/tmp/bm25.pkl")
    assert s.raw_docs_path == Path("/tmp/raw")


def test_applies_defaults() -> None:
    s = Settings(OPENAI_API_KEY="sk-test")
    assert s.embedding_model == "text-embedding-3-small"
    assert s.generation_model == "gpt-4o"
    assert s.judge_model == "gpt-4o-mini"
    assert s.retrieval_k == 10
    assert s.rrf_dense_weight == pytest.approx(0.7)
    assert s.rerank_top_n == 5
    assert s.dedupe_threshold == pytest.approx(0.95)
    assert s.chunk_size == 800
    assert s.chunk_overlap == 120
    assert s.confidence_threshold == pytest.approx(0.4)


def test_loads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    monkeypatch.setenv("KUBERAG_RETRIEVAL_K", "25")
    monkeypatch.setenv("KUBERAG_GENERATION_MODEL", "gpt-4o-2024-11-20")
    s = Settings()
    assert s.openai_api_key.get_secret_value() == "sk-from-env"
    assert s.retrieval_k == 25
    assert s.generation_model == "gpt-4o-2024-11-20"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings()


def test_rrf_weight_validated_to_unit_interval() -> None:
    with pytest.raises(ValidationError):
        Settings(OPENAI_API_KEY="sk-test", rrf_dense_weight=1.5)
    with pytest.raises(ValidationError):
        Settings(OPENAI_API_KEY="sk-test", rrf_dense_weight=-0.1)


def test_dedupe_threshold_validated_to_unit_interval() -> None:
    with pytest.raises(ValidationError):
        Settings(OPENAI_API_KEY="sk-test", dedupe_threshold=1.5)


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError):
        Settings(OPENAI_API_KEY="sk-test", chunk_size=100, chunk_overlap=200)
