from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kuberag.generation.citations import (
    CitationVerifier,
    ParsedCitation,
    VerificationResponse,
    VerifiedCitation,
)

pytestmark = pytest.mark.integration


def make_parsed(marker: int, claim: str, chunk_text: str, *, chunk_id: str | None = None) -> ParsedCitation:
    return ParsedCitation(
        marker=marker,
        claim_span=claim,
        chunk_id=chunk_id or f"c{marker}",
        source=f"/test/{chunk_id or 'c'}.md",
        section=None,
        chunk_text=chunk_text,
    )


def fake_judge_response(supported: bool, reason: str = "") -> Any:
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(
                parsed=VerificationResponse(supported=supported, reason=reason or "ok")
            )
        )
    ]
    return response


def make_judge_by_claim(verdicts: dict[str, bool]) -> Any:
    async def parse(**kwargs: Any) -> Any:
        user_content = kwargs["messages"][1]["content"]
        for claim, supported in verdicts.items():
            if claim in user_content:
                return fake_judge_response(supported, reason=f"judged {claim!r}")
        raise AssertionError(f"unexpected claim text: {user_content!r}")

    return parse


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.beta.chat.completions.parse = AsyncMock()
    return client


@pytest.fixture
def verifier(mock_client: AsyncMock) -> CitationVerifier:
    return CitationVerifier(client=mock_client, model="gpt-4o-mini")


async def test_verify_returns_supported_true(
    verifier: CitationVerifier, mock_client: AsyncMock
) -> None:
    mock_client.beta.chat.completions.parse.return_value = fake_judge_response(True, "match")
    result = await verifier.verify("Pods are units", "Pods are the smallest deployable unit.")
    assert result.supported is True
    assert result.reason == "match"


async def test_verify_returns_supported_false_for_hallucination(
    verifier: CitationVerifier, mock_client: AsyncMock
) -> None:
    mock_client.beta.chat.completions.parse.return_value = fake_judge_response(False, "no match")
    result = await verifier.verify("Kubernetes runs on Mars", "Pods are the smallest deployable unit.")
    assert result.supported is False


async def test_verify_passes_claim_and_chunk_to_llm(
    verifier: CitationVerifier, mock_client: AsyncMock
) -> None:
    mock_client.beta.chat.completions.parse.return_value = fake_judge_response(True)
    await verifier.verify("the claim text", "the chunk text")
    call = mock_client.beta.chat.completions.parse.call_args
    user_content = call.kwargs["messages"][1]["content"]
    assert "the claim text" in user_content
    assert "the chunk text" in user_content


async def test_verify_uses_response_format(
    verifier: CitationVerifier, mock_client: AsyncMock
) -> None:
    mock_client.beta.chat.completions.parse.return_value = fake_judge_response(True)
    await verifier.verify("claim", "chunk")
    call = mock_client.beta.chat.completions.parse.call_args
    assert call.kwargs["response_format"] is VerificationResponse


async def test_verify_uses_configured_model(mock_client: AsyncMock) -> None:
    verifier = CitationVerifier(client=mock_client, model="my-judge-model")
    mock_client.beta.chat.completions.parse.return_value = fake_judge_response(True)
    await verifier.verify("c", "x")
    call = mock_client.beta.chat.completions.parse.call_args
    assert call.kwargs["model"] == "my-judge-model"


async def test_verify_none_parsed_returns_unsupported(
    verifier: CitationVerifier, mock_client: AsyncMock
) -> None:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(parsed=None))]
    mock_client.beta.chat.completions.parse.return_value = response
    result = await verifier.verify("claim", "chunk")
    assert result.supported is False


async def test_verify_all_empty_makes_no_api_call(
    verifier: CitationVerifier, mock_client: AsyncMock
) -> None:
    result = await verifier.verify_all([])
    assert result == []
    mock_client.beta.chat.completions.parse.assert_not_called()


async def test_verify_all_calls_judge_per_citation(
    verifier: CitationVerifier, mock_client: AsyncMock
) -> None:
    mock_client.beta.chat.completions.parse.return_value = fake_judge_response(True)
    citations = [
        make_parsed(1, "claim one", "chunk one"),
        make_parsed(2, "claim two", "chunk two"),
        make_parsed(3, "claim three", "chunk three"),
    ]
    await verifier.verify_all(citations)
    assert mock_client.beta.chat.completions.parse.call_count == 3


async def test_verify_all_preserves_input_order(
    verifier: CitationVerifier, mock_client: AsyncMock
) -> None:
    mock_client.beta.chat.completions.parse.side_effect = make_judge_by_claim(
        {"alpha": True, "beta": False, "gamma": True}
    )
    citations = [
        make_parsed(1, "alpha", "chunk alpha"),
        make_parsed(2, "beta", "chunk beta"),
        make_parsed(3, "gamma", "chunk gamma"),
    ]
    result = await verifier.verify_all(citations)
    assert [v.claim_span for v in result] == ["alpha", "beta", "gamma"]
    assert [v.supported for v in result] == [True, False, True]


async def test_verify_all_returns_verified_citations(
    verifier: CitationVerifier, mock_client: AsyncMock
) -> None:
    mock_client.beta.chat.completions.parse.return_value = fake_judge_response(True, "ok")
    citations = [make_parsed(1, "claim", "chunk", chunk_id="ck-1")]
    result = await verifier.verify_all(citations)
    assert isinstance(result[0], VerifiedCitation)
    assert result[0].supported is True
    assert result[0].reason == "ok"
    assert result[0].chunk_id == "ck-1"
    assert result[0].marker == 1


async def test_verify_all_carries_through_metadata(
    verifier: CitationVerifier, mock_client: AsyncMock
) -> None:
    mock_client.beta.chat.completions.parse.return_value = fake_judge_response(True)
    citations = [
        ParsedCitation(
            marker=1,
            claim_span="x",
            chunk_id="ck-1",
            source="/docs/intro.md",
            section="Pods",
            chunk_text="...",
        )
    ]
    result = await verifier.verify_all(citations)
    assert result[0].source == "/docs/intro.md"
    assert result[0].section == "Pods"


async def test_max_concurrency_must_be_positive(mock_client: AsyncMock) -> None:
    with pytest.raises(ValueError):
        CitationVerifier(client=mock_client, model="m", max_concurrency=0)
    with pytest.raises(ValueError):
        CitationVerifier(client=mock_client, model="m", max_concurrency=-1)


async def test_verify_all_bounded_concurrency(mock_client: AsyncMock) -> None:
    import asyncio

    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    async def slow_parse(**kwargs: Any) -> Any:
        nonlocal in_flight, peak
        async with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        async with lock:
            in_flight -= 1
        return fake_judge_response(True)

    mock_client.beta.chat.completions.parse.side_effect = slow_parse
    verifier = CitationVerifier(client=mock_client, model="m", max_concurrency=2)
    citations = [make_parsed(i + 1, f"claim {i}", f"chunk {i}") for i in range(8)]
    await verifier.verify_all(citations)
    assert peak <= 2
