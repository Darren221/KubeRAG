from pathlib import Path

import pytest

from kuberag.eval.compare_chunking import (
    ComparisonReport,
    compare_chunkers,
    render_comparison_markdown,
    write_comparison_report,
)
from kuberag.eval.run import EvalReport, PerEntryResult


def _entry(
    *,
    correctness: float = 1.0,
    faithfulness: float = 1.0,
    recall: float = 1.0,
    citation: float = 1.0,
) -> PerEntryResult:
    return PerEntryResult(
        golden_id="x",
        question="q",
        type="lookup",
        answer_kind="grounded",
        answer_text="a",
        correctness_score=correctness,
        correctness_rationale="ok",
        faithfulness_score=faithfulness,
        recall_at_k=recall,
        citation_accuracy=citation,
        citation_count=1,
    )


def _report(
    *, correctness: float = 1.0, faithfulness: float = 1.0
) -> EvalReport:
    return EvalReport(
        timestamp="t",
        n_questions=1,
        corpus_version="cv",
        config_hash="cfg",
        retrieval_k=5,
        entries=[_entry(correctness=correctness, faithfulness=faithfulness)],
    )


async def test_compare_calls_callable_once_per_chunker() -> None:
    calls: list[str] = []

    async def run_eval(name: str) -> EvalReport:
        calls.append(name)
        return _report()

    await compare_chunkers(
        chunker_names=["fixed", "recursive"], run_eval_for_chunker=run_eval
    )
    assert calls == ["fixed", "recursive"]


async def test_compare_returns_typed_report() -> None:
    async def run_eval(name: str) -> EvalReport:
        return _report()

    result = await compare_chunkers(
        chunker_names=["fixed", "recursive"], run_eval_for_chunker=run_eval
    )
    assert isinstance(result, ComparisonReport)
    assert set(result.reports.keys()) == {"fixed", "recursive"}


async def test_compare_preserves_chunker_order() -> None:
    async def run_eval(name: str) -> EvalReport:
        return _report()

    result = await compare_chunkers(
        chunker_names=["recursive", "fixed"], run_eval_for_chunker=run_eval
    )
    assert result.chunker_names == ["recursive", "fixed"]


async def test_markdown_includes_each_chunker_column() -> None:
    async def run_eval(name: str) -> EvalReport:
        return _report(correctness=0.8 if name == "fixed" else 0.9)

    result = await compare_chunkers(
        chunker_names=["fixed", "recursive"], run_eval_for_chunker=run_eval
    )
    md = render_comparison_markdown(result)
    assert "fixed" in md
    assert "recursive" in md
    assert "0.800" in md
    assert "0.900" in md


async def test_markdown_includes_delta_relative_to_first() -> None:
    async def run_eval(name: str) -> EvalReport:
        return _report(faithfulness=0.81 if name == "fixed" else 0.87)

    result = await compare_chunkers(
        chunker_names=["fixed", "recursive"], run_eval_for_chunker=run_eval
    )
    md = render_comparison_markdown(result)
    # Δ column: recursive - fixed = +0.06
    assert "+0.060" in md or "+0.06" in md


async def test_markdown_shows_negative_delta_when_chunker_worse() -> None:
    async def run_eval(name: str) -> EvalReport:
        return _report(correctness=0.9 if name == "fixed" else 0.7)

    result = await compare_chunkers(
        chunker_names=["fixed", "recursive"], run_eval_for_chunker=run_eval
    )
    md = render_comparison_markdown(result)
    assert "-0.200" in md or "-0.20" in md


async def test_markdown_includes_all_four_metric_rows() -> None:
    async def run_eval(name: str) -> EvalReport:
        return _report()

    result = await compare_chunkers(
        chunker_names=["fixed", "recursive"], run_eval_for_chunker=run_eval
    )
    md = render_comparison_markdown(result)
    assert "Correctness" in md
    assert "Faithfulness" in md
    assert "Recall" in md
    assert "Citation accuracy" in md


async def test_write_comparison_report_writes_both_files(tmp_path: Path) -> None:
    async def run_eval(name: str) -> EvalReport:
        return _report()

    result = await compare_chunkers(
        chunker_names=["fixed", "recursive"], run_eval_for_chunker=run_eval
    )
    json_path, md_path = write_comparison_report(result, tmp_path)
    assert json_path.exists() and md_path.exists()
    assert "chunking_compare" in md_path.name


async def test_single_chunker_has_no_delta_column() -> None:
    async def run_eval(name: str) -> EvalReport:
        return _report()

    result = await compare_chunkers(
        chunker_names=["fixed"], run_eval_for_chunker=run_eval
    )
    md = render_comparison_markdown(result)
    # Only baseline; no delta
    assert "Δ" not in md or "+0.000" not in md


async def test_empty_chunker_list_raises() -> None:
    async def run_eval(name: str) -> EvalReport:
        return _report()

    with pytest.raises(ValueError):
        await compare_chunkers(
            chunker_names=[], run_eval_for_chunker=run_eval
        )
