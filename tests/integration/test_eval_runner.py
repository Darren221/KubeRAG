from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from kuberag.eval.cache import EvalCache
from kuberag.eval.golden import GoldenEntry
from kuberag.eval.metrics import (
    CorrectnessVerdict,
    FaithfulnessClaim,
    FaithfulnessVerdict,
)
from kuberag.eval.run import (
    AggregateScores,
    EvalReport,
    EvalRunner,
    PerEntryResult,
    render_markdown,
    write_reports,
)
from kuberag.generation.citations import VerifiedCitation
from kuberag.generation.confidence import ConfidenceBreakdown
from kuberag.generation.orchestrator import GroundedAnswer, InsufficientAnswer
from kuberag.retrieval.fusion import FusedHit

pytestmark = pytest.mark.integration


def _golden(
    id_: str = "lookup-001",
    type_: str = "lookup",
    expected: list[str] | None = None,
) -> GoldenEntry:
    return GoldenEntry(
        id=id_,
        question=f"what is {id_}?",
        golden_answer="the golden",
        expected_source_files=expected if expected is not None else ["docs/a.md"],
        type=type_,  # type: ignore[arg-type]
    )


def _fused(chunk_id: str = "c1", text: str = "txt", source: str = "docs/a.md") -> FusedHit:
    return FusedHit(
        chunk_id=chunk_id,
        text=text,
        source=source,
        section=None,
        chunking_strategy="fixed",
        rrf_score=0.016,
        rank=0,
        dense_rank=0,
        sparse_rank=0,
    )


def _verified(supported: bool = True) -> VerifiedCitation:
    return VerifiedCitation(
        marker=1,
        claim_span="claim",
        chunk_id="c1",
        source="docs/a.md",
        section=None,
        chunk_text="txt",
        supported=supported,
        reason="ok",
    )


def _grounded(text: str = "answer text [1].") -> GroundedAnswer:
    return GroundedAnswer(
        text=text,
        citations=[_verified(True)],
        confidence=ConfidenceBreakdown(
            retrieval=0.9, citation=1.0, completeness=0.9, composite=0.92
        ),
        retrieved_chunks=[_fused()],
    )


def _build_runner(
    tmp_path: Path,
    *,
    chunks: list[FusedHit] | None = None,
    answer: GroundedAnswer | InsufficientAnswer | None = None,
    correctness_score: float = 1.0,
    faithfulness_supported: list[bool] | None = None,
) -> tuple[EvalRunner, AsyncMock, AsyncMock, AsyncMock, AsyncMock]:
    hybrid = AsyncMock()
    hybrid.search = AsyncMock(return_value=chunks if chunks is not None else [_fused()])
    orch = AsyncMock()
    orch.answer = AsyncMock(return_value=answer if answer is not None else _grounded())
    correctness = AsyncMock()
    correctness.score = AsyncMock(
        return_value=CorrectnessVerdict(score=correctness_score, rationale="ok")
    )
    faithfulness = AsyncMock()
    fclaims = [
        FaithfulnessClaim(claim=f"c{i}", supported=s, reason="")
        for i, s in enumerate(faithfulness_supported or [True])
    ]
    faithfulness.score = AsyncMock(
        return_value=FaithfulnessVerdict(claims=fclaims)
    )
    cache = EvalCache(tmp_path / "cache")
    runner = EvalRunner(
        hybrid_search=hybrid,
        orchestrator=orch,
        correctness=correctness,
        faithfulness=faithfulness,
        cache=cache,
        corpus_version="cv1",
        config_hash="cfg1",
        retrieval_k=5,
    )
    return runner, hybrid, orch, correctness, faithfulness


async def test_runs_each_entry_through_pipeline(tmp_path: Path) -> None:
    runner, hybrid, orch, correctness, faithfulness = _build_runner(tmp_path)
    report = await runner.run([_golden("a"), _golden("b")])
    assert report.n_questions == 2
    assert hybrid.search.call_count == 2
    assert orch.answer.call_count == 2
    assert correctness.score.call_count == 2
    assert faithfulness.score.call_count == 2


async def test_second_run_uses_cache(tmp_path: Path) -> None:
    runner, hybrid, orch, _, _ = _build_runner(tmp_path)
    entry = _golden("a")
    await runner.run([entry])
    await runner.run([entry])  # second run
    # Only the first run should have called downstream components
    assert hybrid.search.call_count == 1
    assert orch.answer.call_count == 1


async def test_no_cache_flag_bypasses_cache(tmp_path: Path) -> None:
    runner, hybrid, _, _, _ = _build_runner(tmp_path)
    entry = _golden("a")
    await runner.run([entry])
    await runner.run([entry], no_cache=True)
    # Both runs should hit downstream
    assert hybrid.search.call_count == 2


async def test_cache_invalidates_when_corpus_version_changes(tmp_path: Path) -> None:
    cache = EvalCache(tmp_path / "cache")
    runner_a = EvalRunner(
        hybrid_search=AsyncMock(search=AsyncMock(return_value=[_fused()])),
        orchestrator=AsyncMock(answer=AsyncMock(return_value=_grounded())),
        correctness=AsyncMock(
            score=AsyncMock(return_value=CorrectnessVerdict(score=1.0, rationale=""))
        ),
        faithfulness=AsyncMock(
            score=AsyncMock(
                return_value=FaithfulnessVerdict(
                    claims=[FaithfulnessClaim(claim="x", supported=True, reason="")]
                )
            )
        ),
        cache=cache,
        corpus_version="cv1",
        config_hash="cfg1",
    )
    await runner_a.run([_golden("a")])

    runner_b = EvalRunner(
        hybrid_search=AsyncMock(search=AsyncMock(return_value=[_fused()])),
        orchestrator=AsyncMock(answer=AsyncMock(return_value=_grounded())),
        correctness=AsyncMock(
            score=AsyncMock(return_value=CorrectnessVerdict(score=1.0, rationale=""))
        ),
        faithfulness=AsyncMock(
            score=AsyncMock(
                return_value=FaithfulnessVerdict(
                    claims=[FaithfulnessClaim(claim="x", supported=True, reason="")]
                )
            )
        ),
        cache=cache,
        corpus_version="cv2",  # ← changed
        config_hash="cfg1",
    )
    await runner_b.run([_golden("a")])
    # Different corpus → cache miss → downstream called again
    assert runner_b.hybrid_search.search.call_count == 1


async def test_aggregate_averages_per_metric(tmp_path: Path) -> None:
    runner, _, _, _, _ = _build_runner(
        tmp_path,
        correctness_score=0.8,
        faithfulness_supported=[True, False],  # → 0.5
    )
    report = await runner.run([_golden("a"), _golden("b")])
    agg = report.aggregate
    assert agg.correctness == pytest.approx(0.8)
    assert agg.faithfulness == pytest.approx(0.5)


async def test_aggregate_by_type(tmp_path: Path) -> None:
    runner, _, _, _, _ = _build_runner(tmp_path, correctness_score=0.9)
    report = await runner.run(
        [_golden("a", type_="lookup"), _golden("b", type_="multi_hop")]
    )
    assert set(report.by_type.keys()) == {"lookup", "multi_hop"}
    assert all(isinstance(s, AggregateScores) for s in report.by_type.values())


async def test_handles_insufficient_answer(tmp_path: Path) -> None:
    insufficient = InsufficientAnswer(
        reason="low confidence",
        retrieved_chunks=[_fused()],
        suggested_documents=["docs/a.md"],
        generated_text="I couldn't find this.",
    )
    runner, _, _, _, _ = _build_runner(tmp_path, answer=insufficient)
    report = await runner.run([_golden("a")])
    entry = report.entries[0]
    assert entry.answer_kind == "insufficient"
    assert entry.citation_count == 0
    # Citation accuracy on zero citations = 1.0 (vacuous)
    assert entry.citation_accuracy == 1.0


async def test_recall_uses_retrieved_sources(tmp_path: Path) -> None:
    chunks = [_fused(source="docs/foo.md"), _fused(source="docs/bar.md")]
    runner, _, _, _, _ = _build_runner(tmp_path, chunks=chunks)
    report = await runner.run([_golden("a", expected=["docs/foo.md"])])
    # foo.md was retrieved → recall = 1.0
    assert report.entries[0].recall_at_k == 1.0


async def test_writes_json_and_markdown_reports(tmp_path: Path) -> None:
    runner, _, _, _, _ = _build_runner(tmp_path)
    report = await runner.run([_golden("a")])

    output_dir = tmp_path / "results"
    json_path, md_path = write_reports(report, output_dir)
    assert json_path.exists()
    assert md_path.exists()
    assert json_path.suffix == ".json"
    assert md_path.suffix == ".md"


async def test_markdown_contains_headline_numbers(tmp_path: Path) -> None:
    runner, _, _, _, _ = _build_runner(tmp_path, correctness_score=0.85)
    report = await runner.run([_golden("a")])
    md = render_markdown(report)
    assert "Headline scores" in md
    assert "0.85" in md  # correctness shows up
    assert "Answer correctness" in md
    assert "Faithfulness" in md
    assert "Retrieval recall" in md
    assert "Citation accuracy" in md


def test_per_entry_result_is_frozen() -> None:
    r = PerEntryResult(
        golden_id="x",
        question="q",
        type="lookup",
        answer_kind="grounded",
        answer_text="a",
        correctness_score=1.0,
        correctness_rationale="",
        faithfulness_score=1.0,
        recall_at_k=1.0,
        citation_accuracy=1.0,
        citation_count=1,
    )
    with pytest.raises(Exception):
        r.correctness_score = 0.0  # type: ignore[misc]


async def test_report_has_timestamp_and_versions(tmp_path: Path) -> None:
    runner, _, _, _, _ = _build_runner(tmp_path)
    report = await runner.run([_golden("a")])
    assert report.timestamp
    assert report.corpus_version == "cv1"
    assert report.config_hash == "cfg1"


def test_eval_report_aggregate_on_empty() -> None:
    report = EvalReport(
        timestamp="t",
        n_questions=0,
        corpus_version="cv",
        config_hash="cfg",
        retrieval_k=5,
        entries=[],
    )
    agg = report.aggregate
    assert agg.correctness == 0.0
    assert agg.faithfulness == 0.0
