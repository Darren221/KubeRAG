from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kuberag.eval.metrics import AnswerCorrectness, CorrectnessVerdict


def fake_response(score: float, rationale: str = "ok") -> Any:
    response = MagicMock()
    response.choices = [
        MagicMock(
            message=MagicMock(
                parsed=CorrectnessVerdict(score=score, rationale=rationale)
            )
        )
    ]
    return response


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.beta.chat.completions.parse = AsyncMock()
    return client


@pytest.fixture
def judge(mock_client: AsyncMock) -> AnswerCorrectness:
    return AnswerCorrectness(client=mock_client, model="gpt-4o-mini")


async def test_returns_high_score_for_match(
    judge: AnswerCorrectness, mock_client: AsyncMock
) -> None:
    mock_client.beta.chat.completions.parse.return_value = fake_response(1.0, "exact")
    verdict = await judge.score(
        "What is a Pod?",
        "A Pod is the smallest deployable unit.",
        "A Pod is the smallest deployable unit in Kubernetes.",
    )
    assert verdict.score == 1.0


async def test_returns_low_score_for_contradiction(
    judge: AnswerCorrectness, mock_client: AsyncMock
) -> None:
    mock_client.beta.chat.completions.parse.return_value = fake_response(0.1, "wrong")
    verdict = await judge.score(
        "What is a Pod?",
        "A Pod is the smallest deployable unit.",
        "A Pod is a type of fish that swims in groups.",
    )
    assert verdict.score < 0.3


async def test_passes_question_golden_and_predicted_to_llm(
    judge: AnswerCorrectness, mock_client: AsyncMock
) -> None:
    mock_client.beta.chat.completions.parse.return_value = fake_response(1.0)
    await judge.score("Q text", "GOLDEN text", "PREDICTED text")
    call = mock_client.beta.chat.completions.parse.call_args
    user_content = call.kwargs["messages"][1]["content"]
    assert "Q text" in user_content
    assert "GOLDEN text" in user_content
    assert "PREDICTED text" in user_content


async def test_uses_response_format(
    judge: AnswerCorrectness, mock_client: AsyncMock
) -> None:
    mock_client.beta.chat.completions.parse.return_value = fake_response(0.5)
    await judge.score("q", "g", "p")
    assert (
        mock_client.beta.chat.completions.parse.call_args.kwargs["response_format"]
        is CorrectnessVerdict
    )


async def test_uses_configured_model(mock_client: AsyncMock) -> None:
    judge = AnswerCorrectness(client=mock_client, model="custom-judge")
    mock_client.beta.chat.completions.parse.return_value = fake_response(1.0)
    await judge.score("q", "g", "p")
    assert (
        mock_client.beta.chat.completions.parse.call_args.kwargs["model"]
        == "custom-judge"
    )


async def test_none_parsed_returns_zero_score(
    judge: AnswerCorrectness, mock_client: AsyncMock
) -> None:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(parsed=None))]
    mock_client.beta.chat.completions.parse.return_value = response
    verdict = await judge.score("q", "g", "p")
    assert verdict.score == 0.0


async def test_returns_typed_verdict(
    judge: AnswerCorrectness, mock_client: AsyncMock
) -> None:
    mock_client.beta.chat.completions.parse.return_value = fake_response(0.75, "partial")
    verdict = await judge.score("q", "g", "p")
    assert isinstance(verdict, CorrectnessVerdict)
    assert verdict.rationale == "partial"


def test_verdict_score_validated_to_unit_interval() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CorrectnessVerdict(score=1.5, rationale="too big")
    with pytest.raises(ValidationError):
        CorrectnessVerdict(score=-0.1, rationale="too small")


async def test_empty_inputs_still_call_llm(
    judge: AnswerCorrectness, mock_client: AsyncMock
) -> None:
    # Empty predicted should still get a verdict (likely score 0)
    mock_client.beta.chat.completions.parse.return_value = fake_response(0.0, "no answer")
    verdict = await judge.score("q", "golden", "")
    assert verdict.score == 0.0
