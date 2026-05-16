from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kuberag.eval.metrics import (
    Faithfulness,
    FaithfulnessClaim,
    FaithfulnessVerdict,
)


def fake_response(claims: list[tuple[str, bool, str]]) -> Any:
    parsed = FaithfulnessVerdict(
        claims=[
            FaithfulnessClaim(claim=text, supported=supported, reason=reason)
            for text, supported, reason in claims
        ]
    )
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(parsed=parsed))]
    return response


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.beta.chat.completions.parse = AsyncMock()
    return client


@pytest.fixture
def judge(mock_client: AsyncMock) -> Faithfulness:
    return Faithfulness(client=mock_client, model="gpt-4o-mini")


async def test_all_claims_supported_yields_one(
    judge: Faithfulness, mock_client: AsyncMock
) -> None:
    mock_client.beta.chat.completions.parse.return_value = fake_response(
        [
            ("Pods are units.", True, "in chunk 1"),
            ("Pods host containers.", True, "in chunk 1"),
        ]
    )
    verdict = await judge.score(
        "Pods are units. Pods host containers.",
        ["Pods are the smallest units of Kubernetes and host containers."],
    )
    assert verdict.score == 1.0


async def test_partial_support_yields_fraction(
    judge: Faithfulness, mock_client: AsyncMock
) -> None:
    mock_client.beta.chat.completions.parse.return_value = fake_response(
        [
            ("Pods are units.", True, "ok"),
            ("Pods run on Mars.", False, "not in chunks"),
            ("Pods host containers.", True, "ok"),
        ]
    )
    verdict = await judge.score("Pods are units. Pods run on Mars. Pods host containers.", ["..."])
    assert verdict.score == pytest.approx(2 / 3)


async def test_no_claims_supported_yields_zero(
    judge: Faithfulness, mock_client: AsyncMock
) -> None:
    mock_client.beta.chat.completions.parse.return_value = fake_response(
        [
            ("Hallucination 1.", False, "no"),
            ("Hallucination 2.", False, "no"),
        ]
    )
    verdict = await judge.score("Made up answer.", ["unrelated chunk"])
    assert verdict.score == 0.0


async def test_empty_claims_list_yields_one_vacuously(
    judge: Faithfulness, mock_client: AsyncMock
) -> None:
    mock_client.beta.chat.completions.parse.return_value = fake_response([])
    verdict = await judge.score("Insufficient context.", ["..."])
    assert verdict.score == 1.0


async def test_passes_answer_and_chunks_to_llm(
    judge: Faithfulness, mock_client: AsyncMock
) -> None:
    mock_client.beta.chat.completions.parse.return_value = fake_response(
        [("c1", True, "ok")]
    )
    await judge.score("ANSWER TEXT", ["CHUNK ONE", "CHUNK TWO"])
    user_content = mock_client.beta.chat.completions.parse.call_args.kwargs[
        "messages"
    ][1]["content"]
    assert "ANSWER TEXT" in user_content
    assert "CHUNK ONE" in user_content
    assert "CHUNK TWO" in user_content


async def test_uses_response_format(
    judge: Faithfulness, mock_client: AsyncMock
) -> None:
    mock_client.beta.chat.completions.parse.return_value = fake_response(
        [("c1", True, "ok")]
    )
    await judge.score("a", ["c"])
    assert (
        mock_client.beta.chat.completions.parse.call_args.kwargs["response_format"]
        is FaithfulnessVerdict
    )


async def test_uses_configured_model(mock_client: AsyncMock) -> None:
    judge = Faithfulness(client=mock_client, model="custom-model")
    mock_client.beta.chat.completions.parse.return_value = fake_response(
        [("c", True, "")]
    )
    await judge.score("a", ["c"])
    assert (
        mock_client.beta.chat.completions.parse.call_args.kwargs["model"]
        == "custom-model"
    )


async def test_none_parsed_returns_zero(
    judge: Faithfulness, mock_client: AsyncMock
) -> None:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(parsed=None))]
    mock_client.beta.chat.completions.parse.return_value = response
    verdict = await judge.score("a", ["c"])
    assert verdict.score == 0.0


async def test_empty_chunks_still_calls_llm(
    judge: Faithfulness, mock_client: AsyncMock
) -> None:
    mock_client.beta.chat.completions.parse.return_value = fake_response([])
    verdict = await judge.score("a", [])
    # No chunks → likely no claims can be supported, but we still consult the judge.
    assert isinstance(verdict, FaithfulnessVerdict)


async def test_score_computed_from_claims_not_from_llm(
    judge: Faithfulness, mock_client: AsyncMock
) -> None:
    # Even if we somehow got a verdict with mismatched counts, the score is from the claims.
    mock_client.beta.chat.completions.parse.return_value = fake_response(
        [("a", True, ""), ("b", False, "")]
    )
    verdict = await judge.score("a", ["c"])
    assert verdict.score == 0.5


def test_verdict_score_is_a_property() -> None:
    v = FaithfulnessVerdict(
        claims=[
            FaithfulnessClaim(claim="x", supported=True, reason=""),
            FaithfulnessClaim(claim="y", supported=False, reason=""),
            FaithfulnessClaim(claim="z", supported=True, reason=""),
        ]
    )
    assert v.score == pytest.approx(2 / 3)


def test_verdict_score_for_empty_claims_is_one() -> None:
    v = FaithfulnessVerdict(claims=[])
    assert v.score == 1.0
