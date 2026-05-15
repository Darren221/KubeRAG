from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from kuberag.generation.generator import GenerationError, Generator
from kuberag.generation.models import Answer, Citation
from kuberag.retrieval.fusion import FusedHit

pytestmark = pytest.mark.integration


def make_chunk(chunk_id: str, text: str, *, rank: int = 0) -> FusedHit:
    return FusedHit(
        chunk_id=chunk_id,
        text=text,
        source=f"/test/{chunk_id}.md",
        section=None,
        chunking_strategy="fixed",
        rrf_score=0.5,
        rank=rank,
        dense_rank=rank,
        sparse_rank=rank,
    )


def fake_parse_response(answer: Answer | None) -> Any:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(parsed=answer))]
    return response


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.beta.chat.completions.parse = AsyncMock()
    return client


@pytest.fixture
def generator(mock_client: AsyncMock) -> Generator:
    return Generator(client=mock_client, model="gpt-4o")


async def test_returns_parsed_answer(generator: Generator, mock_client: AsyncMock) -> None:
    answer = Answer(
        text="Pods are units of work [1].",
        citations=[Citation(marker=1, claim_span="Pods are units of work")],
        insufficient_context=False,
    )
    mock_client.beta.chat.completions.parse.return_value = fake_parse_response(answer)

    result = await generator.generate("what is a pod?", [make_chunk("a", "pods are units")])
    assert result == answer


async def test_passes_grounded_prompt_to_llm(
    generator: Generator, mock_client: AsyncMock
) -> None:
    answer = Answer(
        text="Answer [1].",
        citations=[Citation(marker=1, claim_span="Answer")],
        insufficient_context=False,
    )
    mock_client.beta.chat.completions.parse.return_value = fake_parse_response(answer)

    await generator.generate("what is a pod?", [make_chunk("a", "pods are units")])

    call = mock_client.beta.chat.completions.parse.call_args
    messages = call.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "pods are units" in messages[1]["content"]
    assert "what is a pod?" in messages[1]["content"]


async def test_uses_answer_schema_as_response_format(
    generator: Generator, mock_client: AsyncMock
) -> None:
    answer = Answer(text="x", citations=[Citation(marker=1, claim_span="x")], insufficient_context=False)
    mock_client.beta.chat.completions.parse.return_value = fake_parse_response(answer)

    await generator.generate("q", [make_chunk("a", "text")])
    call = mock_client.beta.chat.completions.parse.call_args
    assert call.kwargs["response_format"] is Answer


async def test_uses_configured_model(mock_client: AsyncMock) -> None:
    generator = Generator(client=mock_client, model="gpt-4o-2024-11-20")
    answer = Answer(text="x", citations=[Citation(marker=1, claim_span="x")], insufficient_context=False)
    mock_client.beta.chat.completions.parse.return_value = fake_parse_response(answer)

    await generator.generate("q", [make_chunk("a", "text")])
    call = mock_client.beta.chat.completions.parse.call_args
    assert call.kwargs["model"] == "gpt-4o-2024-11-20"


async def test_raises_when_parsed_is_none(
    generator: Generator, mock_client: AsyncMock
) -> None:
    mock_client.beta.chat.completions.parse.return_value = fake_parse_response(None)
    with pytest.raises(GenerationError):
        await generator.generate("q", [make_chunk("a", "text")])


async def test_raises_when_grounded_answer_has_no_citations(
    generator: Generator, mock_client: AsyncMock
) -> None:
    # Model claims it answered (insufficient_context=False) but cited nothing
    bad = Answer(text="Pods are units.", citations=[], insufficient_context=False)
    mock_client.beta.chat.completions.parse.return_value = fake_parse_response(bad)
    with pytest.raises(GenerationError):
        await generator.generate("q", [make_chunk("a", "text")])


async def test_insufficient_context_with_empty_citations_is_ok(
    generator: Generator, mock_client: AsyncMock
) -> None:
    # Model says it can't answer → empty citations is fine
    refusal = Answer(
        text="The context does not cover this topic.",
        citations=[],
        insufficient_context=True,
    )
    mock_client.beta.chat.completions.parse.return_value = fake_parse_response(refusal)
    result = await generator.generate("q", [make_chunk("a", "text")])
    assert result.insufficient_context is True
    assert result.citations == []


async def test_empty_question_raises(generator: Generator, mock_client: AsyncMock) -> None:
    with pytest.raises(ValueError):
        await generator.generate("", [make_chunk("a", "text")])
    mock_client.beta.chat.completions.parse.assert_not_called()


async def test_empty_chunks_still_calls_llm(
    generator: Generator, mock_client: AsyncMock
) -> None:
    refusal = Answer(text="No context.", citations=[], insufficient_context=True)
    mock_client.beta.chat.completions.parse.return_value = fake_parse_response(refusal)

    result = await generator.generate("q", [])
    assert result.insufficient_context is True
    mock_client.beta.chat.completions.parse.assert_called_once()


async def test_single_call_per_generate(
    generator: Generator, mock_client: AsyncMock
) -> None:
    answer = Answer(text="x [1].", citations=[Citation(marker=1, claim_span="x")], insufficient_context=False)
    mock_client.beta.chat.completions.parse.return_value = fake_parse_response(answer)
    await generator.generate("q", [make_chunk("a", "text")])
    assert mock_client.beta.chat.completions.parse.call_count == 1
