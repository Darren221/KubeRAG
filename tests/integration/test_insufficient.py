from unittest.mock import AsyncMock, MagicMock

import pytest

from kuberag.generation.citations import VerifiedCitation
from kuberag.generation.generator import GenerationError
from kuberag.generation.models import Answer, Citation
from kuberag.generation.orchestrator import (
    CompletenessJudgment,
    GenerationOrchestrator,
    GroundedAnswer,
    InsufficientAnswer,
)
from kuberag.retrieval.fusion import FusedHit

pytestmark = pytest.mark.integration


def make_chunk(
    chunk_id: str = "a",
    *,
    rank: int = 0,
    rrf_score: float = 0.016,
    source: str = "/test/a.md",
) -> FusedHit:
    return FusedHit(
        chunk_id=chunk_id,
        text=f"text {chunk_id}",
        source=source,
        section=None,
        chunking_strategy="fixed",
        rrf_score=rrf_score,
        rank=rank,
        dense_rank=rank,
        sparse_rank=rank,
    )


def make_answer(text: str, citations: list[tuple[int, str]], *, insufficient: bool = False) -> Answer:
    return Answer(
        text=text,
        citations=[Citation(marker=m, claim_span=s) for m, s in citations],
        insufficient_context=insufficient,
    )


def make_verified(*, marker: int = 1, supported: bool = True) -> VerifiedCitation:
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


@pytest.fixture
def mock_generator() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_verifier() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def mock_completeness() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def orchestrator(
    mock_generator: AsyncMock,
    mock_verifier: AsyncMock,
    mock_completeness: AsyncMock,
) -> GenerationOrchestrator:
    return GenerationOrchestrator(
        generator=mock_generator,
        verifier=mock_verifier,
        completeness_judge=mock_completeness,
        confidence_threshold=0.4,
    )


async def test_empty_chunks_returns_insufficient(
    orchestrator: GenerationOrchestrator,
    mock_generator: AsyncMock,
) -> None:
    result = await orchestrator.answer("what is a pod?", [])
    assert isinstance(result, InsufficientAnswer)
    assert result.kind == "insufficient"
    mock_generator.generate.assert_not_called()


async def test_weak_retrieval_returns_insufficient_without_generating(
    orchestrator: GenerationOrchestrator,
    mock_generator: AsyncMock,
) -> None:
    # rrf_score very low → retrieval_confidence below 0.4 threshold
    weak_chunks = [make_chunk(rrf_score=0.001)]
    result = await orchestrator.answer("what?", weak_chunks)
    assert isinstance(result, InsufficientAnswer)
    mock_generator.generate.assert_not_called()


async def test_weak_retrieval_lists_suggested_documents(
    orchestrator: GenerationOrchestrator,
) -> None:
    chunks = [
        make_chunk("a", rrf_score=0.001, source="/docs/foo.md"),
        make_chunk("b", rrf_score=0.001, source="/docs/bar.md"),
    ]
    result = await orchestrator.answer("?", chunks)
    assert isinstance(result, InsufficientAnswer)
    assert set(result.suggested_documents) == {"/docs/foo.md", "/docs/bar.md"}


async def test_weak_retrieval_includes_retrieved_chunks(
    orchestrator: GenerationOrchestrator,
) -> None:
    chunks = [make_chunk("a", rrf_score=0.001), make_chunk("b", rrf_score=0.001)]
    result = await orchestrator.answer("?", chunks)
    assert isinstance(result, InsufficientAnswer)
    assert len(result.retrieved_chunks) == 2


async def test_model_refusal_returns_insufficient(
    orchestrator: GenerationOrchestrator,
    mock_generator: AsyncMock,
) -> None:
    refusal = make_answer("Insufficient context.", [], insufficient=True)
    mock_generator.generate.return_value = refusal

    chunks = [make_chunk(rrf_score=0.016)]
    result = await orchestrator.answer("q", chunks)
    assert isinstance(result, InsufficientAnswer)
    assert result.generated_text == "Insufficient context."


async def test_generation_error_returns_insufficient(
    orchestrator: GenerationOrchestrator,
    mock_generator: AsyncMock,
) -> None:
    mock_generator.generate.side_effect = GenerationError("model misbehaved")
    chunks = [make_chunk(rrf_score=0.016)]
    result = await orchestrator.answer("q", chunks)
    assert isinstance(result, InsufficientAnswer)


async def test_citation_parse_error_returns_insufficient(
    orchestrator: GenerationOrchestrator,
    mock_generator: AsyncMock,
) -> None:
    # Model claims marker [5] but only 1 chunk supplied
    bad = make_answer("Pods [5].", [(5, "Pods")])
    mock_generator.generate.return_value = bad
    chunks = [make_chunk(rrf_score=0.016)]
    result = await orchestrator.answer("q", chunks)
    assert isinstance(result, InsufficientAnswer)
    assert result.generated_text == "Pods [5]."


async def test_low_composite_confidence_returns_insufficient(
    orchestrator: GenerationOrchestrator,
    mock_generator: AsyncMock,
    mock_verifier: AsyncMock,
    mock_completeness: AsyncMock,
) -> None:
    answer = make_answer("Pods are units [1].", [(1, "Pods are units")])
    mock_generator.generate.return_value = answer
    mock_verifier.verify_all.return_value = [make_verified(marker=1, supported=False)]
    mock_completeness.score.return_value = CompletenessJudgment(score=0.0, reason="no")

    chunks = [make_chunk(rrf_score=0.001)]  # weak retrieval too
    result = await orchestrator.answer("q", chunks)
    assert isinstance(result, InsufficientAnswer)


async def test_high_confidence_returns_grounded_answer(
    orchestrator: GenerationOrchestrator,
    mock_generator: AsyncMock,
    mock_verifier: AsyncMock,
    mock_completeness: AsyncMock,
) -> None:
    answer = make_answer("Pods are units [1].", [(1, "Pods are units")])
    mock_generator.generate.return_value = answer
    mock_verifier.verify_all.return_value = [make_verified(marker=1, supported=True)]
    mock_completeness.score.return_value = CompletenessJudgment(score=1.0, reason="full")

    chunks = [make_chunk(rrf_score=0.016)]
    result = await orchestrator.answer("q", chunks)
    assert isinstance(result, GroundedAnswer)
    assert result.kind == "grounded"
    assert result.text == "Pods are units [1]."
    assert result.confidence.composite >= 0.4


async def test_grounded_answer_includes_verifications(
    orchestrator: GenerationOrchestrator,
    mock_generator: AsyncMock,
    mock_verifier: AsyncMock,
    mock_completeness: AsyncMock,
) -> None:
    answer = make_answer("Pods are units [1].", [(1, "Pods are units")])
    mock_generator.generate.return_value = answer
    verified = [make_verified(marker=1, supported=True)]
    mock_verifier.verify_all.return_value = verified
    mock_completeness.score.return_value = CompletenessJudgment(score=1.0, reason="full")

    result = await orchestrator.answer("q", [make_chunk(rrf_score=0.016)])
    assert isinstance(result, GroundedAnswer)
    assert result.citations == verified


async def test_grounded_answer_includes_confidence_breakdown(
    orchestrator: GenerationOrchestrator,
    mock_generator: AsyncMock,
    mock_verifier: AsyncMock,
    mock_completeness: AsyncMock,
) -> None:
    answer = make_answer("Pods [1].", [(1, "Pods")])
    mock_generator.generate.return_value = answer
    mock_verifier.verify_all.return_value = [make_verified(marker=1, supported=True)]
    mock_completeness.score.return_value = CompletenessJudgment(score=0.9, reason="ok")

    result = await orchestrator.answer("q", [make_chunk(rrf_score=0.016)])
    assert isinstance(result, GroundedAnswer)
    assert result.confidence.completeness == pytest.approx(0.9)
    assert result.confidence.citation == 1.0


async def test_completeness_judge_called_with_question_and_answer(
    orchestrator: GenerationOrchestrator,
    mock_generator: AsyncMock,
    mock_verifier: AsyncMock,
    mock_completeness: AsyncMock,
) -> None:
    answer = make_answer("Answer text [1].", [(1, "Answer text")])
    mock_generator.generate.return_value = answer
    mock_verifier.verify_all.return_value = [make_verified(marker=1, supported=True)]
    mock_completeness.score.return_value = CompletenessJudgment(score=1.0, reason="ok")

    await orchestrator.answer("the question?", [make_chunk(rrf_score=0.016)])
    mock_completeness.score.assert_called_once()
    args = mock_completeness.score.call_args
    assert "the question?" == args.args[0] or args.kwargs.get("question") == "the question?"


async def test_threshold_is_configurable(
    mock_generator: AsyncMock,
    mock_verifier: AsyncMock,
    mock_completeness: AsyncMock,
) -> None:
    # With very strict threshold, even good retrieval is insufficient
    strict = GenerationOrchestrator(
        generator=mock_generator,
        verifier=mock_verifier,
        completeness_judge=mock_completeness,
        confidence_threshold=0.99,
    )
    chunks = [make_chunk(rrf_score=0.016)]
    result = await strict.answer("?", chunks)
    assert isinstance(result, InsufficientAnswer)
    mock_generator.generate.assert_not_called()


async def test_empty_question_raises(orchestrator: GenerationOrchestrator) -> None:
    with pytest.raises(ValueError):
        await orchestrator.answer("", [make_chunk()])


class TestCompletenessJudge:
    async def test_score_returns_judgment(self) -> None:
        from kuberag.generation.orchestrator import CompletenessJudge

        mock_client = AsyncMock()
        mock_client.beta.chat.completions.parse = AsyncMock()
        response = MagicMock()
        response.choices = [
            MagicMock(message=MagicMock(parsed=CompletenessJudgment(score=0.8, reason="mostly")))
        ]
        mock_client.beta.chat.completions.parse.return_value = response

        judge = CompletenessJudge(client=mock_client, model="gpt-4o-mini")
        result = await judge.score("Q?", "A.")
        assert result.score == 0.8

    async def test_score_returns_default_on_none_parsed(self) -> None:
        from kuberag.generation.orchestrator import CompletenessJudge

        mock_client = AsyncMock()
        mock_client.beta.chat.completions.parse = AsyncMock()
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(parsed=None))]
        mock_client.beta.chat.completions.parse.return_value = response

        judge = CompletenessJudge(client=mock_client, model="gpt-4o-mini")
        result = await judge.score("Q?", "A.")
        # Degrades gracefully — neutral midpoint, not crash
        assert 0.0 <= result.score <= 1.0
