from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from kuberag.api import create_app, get_hybrid_search, get_orchestrator
from kuberag.config import Settings
from kuberag.generation.citations import VerifiedCitation
from kuberag.generation.confidence import ConfidenceBreakdown
from kuberag.generation.orchestrator import GroundedAnswer, InsufficientAnswer
from kuberag.retrieval.fusion import FusedHit

pytestmark = pytest.mark.integration


def make_fused(chunk_id: str = "a", rank: int = 0) -> FusedHit:
    return FusedHit(
        chunk_id=chunk_id,
        text=f"text {chunk_id}",
        source=f"/test/{chunk_id}.md",
        section=None,
        chunking_strategy="fixed",
        rrf_score=0.016,
        rank=rank,
        dense_rank=rank,
        sparse_rank=rank,
    )


def make_verified(marker: int = 1, supported: bool = True) -> VerifiedCitation:
    return VerifiedCitation(
        marker=marker,
        claim_span=f"claim {marker}",
        chunk_id=f"c{marker}",
        source=f"/test/c{marker}.md",
        section=None,
        chunk_text=f"text {marker}",
        supported=supported,
        reason="ok",
    )


def grounded_sample() -> GroundedAnswer:
    return GroundedAnswer(
        text="Pods are units [1].",
        citations=[make_verified(marker=1, supported=True)],
        confidence=ConfidenceBreakdown(
            retrieval=0.9, citation=1.0, completeness=0.95, composite=0.93
        ),
        retrieved_chunks=[make_fused("a")],
    )


def insufficient_sample() -> InsufficientAnswer:
    return InsufficientAnswer(
        reason="retrieval confidence 0.18 below threshold 0.40",
        retrieved_chunks=[make_fused("a")],
        suggested_documents=["/test/a.md"],
    )


def build_app_with_mocks(
    test_settings: Settings,
    *,
    hybrid_result: list[FusedHit],
    answer_result: GroundedAnswer | InsufficientAnswer,
) -> tuple[object, AsyncMock, AsyncMock]:
    app = create_app(test_settings)
    fake_hybrid = AsyncMock()
    fake_hybrid.search = AsyncMock(return_value=hybrid_result)
    fake_orch = AsyncMock()
    fake_orch.answer = AsyncMock(return_value=answer_result)

    app.dependency_overrides[get_hybrid_search] = lambda: fake_hybrid
    app.dependency_overrides[get_orchestrator] = lambda: fake_orch
    return app, fake_hybrid, fake_orch


def test_grounded_answer_round_trip(test_settings: Settings) -> None:
    app, _, _ = build_app_with_mocks(
        test_settings,
        hybrid_result=[make_fused("a")],
        answer_result=grounded_sample(),
    )
    with TestClient(app) as client:
        response = client.post("/v1/ask", json={"question": "what is a pod?"})
        assert response.status_code == 200
        body = response.json()
        assert body["kind"] == "grounded"
        assert body["text"] == "Pods are units [1]."
        assert len(body["citations"]) == 1
        assert body["confidence"]["composite"] == pytest.approx(0.93)


def test_insufficient_answer_round_trip(test_settings: Settings) -> None:
    app, _, _ = build_app_with_mocks(
        test_settings,
        hybrid_result=[],
        answer_result=insufficient_sample(),
    )
    with TestClient(app) as client:
        response = client.post("/v1/ask", json={"question": "moon weather?"})
        assert response.status_code == 200
        body = response.json()
        assert body["kind"] == "insufficient"
        assert "below threshold" in body["reason"]
        assert body["suggested_documents"] == ["/test/a.md"]


def test_hybrid_search_called_with_question(test_settings: Settings) -> None:
    app, fake_hybrid, _ = build_app_with_mocks(
        test_settings,
        hybrid_result=[make_fused("a")],
        answer_result=grounded_sample(),
    )
    with TestClient(app) as client:
        client.post("/v1/ask", json={"question": "what is a pod?"})
        fake_hybrid.search.assert_called_once()
        args, kwargs = fake_hybrid.search.call_args
        assert args[0] == "what is a pod?" or kwargs.get("query") == "what is a pod?"


def test_dense_only_flag_passed_through(test_settings: Settings) -> None:
    app, fake_hybrid, _ = build_app_with_mocks(
        test_settings,
        hybrid_result=[make_fused("a")],
        answer_result=grounded_sample(),
    )
    with TestClient(app) as client:
        client.post("/v1/ask", json={"question": "q", "dense_only": True})
        assert fake_hybrid.search.call_args.kwargs["dense_only"] is True


def test_k_and_top_n_passed_through(test_settings: Settings) -> None:
    app, fake_hybrid, _ = build_app_with_mocks(
        test_settings,
        hybrid_result=[make_fused("a")],
        answer_result=grounded_sample(),
    )
    with TestClient(app) as client:
        client.post(
            "/v1/ask", json={"question": "q", "k": 20, "top_n": 8}
        )
        assert fake_hybrid.search.call_args.kwargs["k"] == 20
        assert fake_hybrid.search.call_args.kwargs["top_n"] == 8


def test_orchestrator_called_with_question_and_chunks(test_settings: Settings) -> None:
    app, _, fake_orch = build_app_with_mocks(
        test_settings,
        hybrid_result=[make_fused("a"), make_fused("b", rank=1)],
        answer_result=grounded_sample(),
    )
    with TestClient(app) as client:
        client.post("/v1/ask", json={"question": "what?"})
    fake_orch.answer.assert_called_once()
    args, kwargs = fake_orch.answer.call_args
    # Either positional or keyword
    question_arg = args[0] if args else kwargs.get("question")
    chunks_arg = args[1] if len(args) > 1 else kwargs.get("chunks")
    assert question_arg == "what?"
    assert len(chunks_arg) == 2


def test_empty_question_returns_422(test_settings: Settings) -> None:
    app, _, _ = build_app_with_mocks(
        test_settings,
        hybrid_result=[],
        answer_result=grounded_sample(),
    )
    with TestClient(app) as client:
        response = client.post("/v1/ask", json={"question": ""})
        assert response.status_code == 422


def test_missing_question_returns_422(test_settings: Settings) -> None:
    app, _, _ = build_app_with_mocks(
        test_settings,
        hybrid_result=[],
        answer_result=grounded_sample(),
    )
    with TestClient(app) as client:
        response = client.post("/v1/ask", json={})
        assert response.status_code == 422


def test_ask_route_in_openapi(test_settings: Settings) -> None:
    app = create_app(test_settings)
    with TestClient(app) as client:
        spec = client.get("/openapi.json").json()
        assert "/v1/ask" in spec["paths"]
        assert "post" in spec["paths"]["/v1/ask"]


def test_grounded_answer_includes_confidence_breakdown(test_settings: Settings) -> None:
    app, _, _ = build_app_with_mocks(
        test_settings,
        hybrid_result=[make_fused("a")],
        answer_result=grounded_sample(),
    )
    with TestClient(app) as client:
        body = client.post("/v1/ask", json={"question": "q"}).json()
        conf = body["confidence"]
        assert "retrieval" in conf
        assert "citation" in conf
        assert "completeness" in conf
        assert "composite" in conf


def test_grounded_answer_includes_retrieved_chunks(test_settings: Settings) -> None:
    app, _, _ = build_app_with_mocks(
        test_settings,
        hybrid_result=[make_fused("a"), make_fused("b", rank=1)],
        answer_result=grounded_sample(),
    )
    with TestClient(app) as client:
        body = client.post("/v1/ask", json={"question": "q"}).json()
        assert len(body["retrieved_chunks"]) >= 1
