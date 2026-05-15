import httpx
import pytest
from dashboard.api_client import KubeRAGClient
from pydantic import TypeAdapter

from kuberag.generation.citations import VerifiedCitation
from kuberag.generation.confidence import ConfidenceBreakdown
from kuberag.generation.orchestrator import GroundedAnswer, InsufficientAnswer
from kuberag.retrieval.fusion import FusedHit


def _grounded_payload() -> dict:  # type: ignore[type-arg]
    answer = GroundedAnswer(
        text="Pods are units [1].",
        citations=[
            VerifiedCitation(
                marker=1,
                claim_span="Pods are units",
                chunk_id="c1",
                source="/test/a.md",
                section=None,
                chunk_text="pod info",
                supported=True,
                reason="ok",
            )
        ],
        confidence=ConfidenceBreakdown(
            retrieval=0.9, citation=1.0, completeness=0.9, composite=0.92
        ),
        retrieved_chunks=[
            FusedHit(
                chunk_id="c1",
                text="pod info",
                source="/test/a.md",
                section=None,
                chunking_strategy="fixed",
                rrf_score=0.016,
                rank=0,
                dense_rank=0,
                sparse_rank=0,
            )
        ],
    )
    return answer.model_dump(mode="json")


def _insufficient_payload() -> dict:  # type: ignore[type-arg]
    return InsufficientAnswer(
        reason="retrieval confidence 0.18 below threshold 0.40",
        retrieved_chunks=[],
        suggested_documents=["/docs/foo.md"],
        generated_text=None,
    ).model_dump(mode="json")


def _make_client(handler):  # type: ignore[no-untyped-def]
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport, base_url="http://test")
    return KubeRAGClient(base_url="http://test", http_client=http_client)


def test_ask_returns_grounded_answer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/ask"
        return httpx.Response(200, json=_grounded_payload())

    client = _make_client(handler)
    result = client.ask("what is a pod?")
    assert isinstance(result, GroundedAnswer)
    assert result.text == "Pods are units [1]."


def test_ask_returns_insufficient_answer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_insufficient_payload())

    client = _make_client(handler)
    result = client.ask("moon weather?")
    assert isinstance(result, InsufficientAnswer)
    assert result.suggested_documents == ["/docs/foo.md"]


def test_ask_sends_dense_only_flag() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=_grounded_payload())

    client = _make_client(handler)
    client.ask("q", dense_only=True)
    assert captured["body"]["dense_only"] is True


def test_ask_default_does_not_set_dense_only_true() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=_grounded_payload())

    client = _make_client(handler)
    client.ask("q")
    assert captured["body"].get("dense_only") is False


def test_ask_raises_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "kaboom"})

    client = _make_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        client.ask("q")


def test_health_returns_typed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "chroma_count": 42,
                "bm25_count": 42,
                "models": {
                    "embedding": "text-embedding-3-small",
                    "generation": "gpt-4o",
                    "judge": "gpt-4o-mini",
                },
            },
        )

    client = _make_client(handler)
    health = client.health()
    assert health.status == "ok"
    assert health.chroma_count == 42
    assert health.models.embedding == "text-embedding-3-small"


def test_list_documents_returns_typed_list() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"source": "/docs/a.md", "chunking_strategy": "fixed", "chunk_count": 3},
                {"source": "/docs/b.md", "chunking_strategy": "fixed", "chunk_count": 1},
            ],
        )

    client = _make_client(handler)
    docs = client.list_documents()
    assert len(docs) == 2
    assert docs[0].source == "/docs/a.md"
    assert docs[0].chunk_count == 3


def test_default_base_url_is_used_when_no_client_provided() -> None:
    # Just verify the constructor accepts a base_url without raising
    client = KubeRAGClient(base_url="http://example.com")
    assert client.base_url == "http://example.com"


def test_type_adapter_handles_either_response_kind() -> None:
    from kuberag.generation.orchestrator import AnswerResult

    adapter = TypeAdapter(AnswerResult)
    grounded = adapter.validate_python(_grounded_payload())
    insufficient = adapter.validate_python(_insufficient_payload())
    assert isinstance(grounded, GroundedAnswer)
    assert isinstance(insufficient, InsufficientAnswer)
