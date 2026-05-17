from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from kuberag.eval.run import AggregateScores, EvalReport

RunEvalFn = Callable[[str], Awaitable[EvalReport]]


class ComparisonReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: str
    chunker_names: list[str] = Field(min_length=1)
    reports: dict[str, EvalReport]

    @property
    def baseline(self) -> str:
        return self.chunker_names[0]


async def compare_chunkers(
    *,
    chunker_names: list[str],
    run_eval_for_chunker: RunEvalFn,
) -> ComparisonReport:
    if not chunker_names:
        raise ValueError("chunker_names must be non-empty")

    reports: dict[str, EvalReport] = {}
    for name in chunker_names:
        reports[name] = await run_eval_for_chunker(name)

    return ComparisonReport(
        timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ"),
        chunker_names=list(chunker_names),
        reports=reports,
    )


_METRICS: list[tuple[str, str]] = [
    ("Correctness", "correctness"),
    ("Faithfulness", "faithfulness"),
    ("Recall@k", "recall_at_k"),
    ("Citation accuracy", "citation_accuracy"),
]


def _agg_value(scores: AggregateScores, attr: str) -> float:
    return float(getattr(scores, attr))


def render_comparison_markdown(report: ComparisonReport) -> str:
    baseline = report.baseline
    lines: list[str] = [
        "# KubeRAG Chunking-Strategy Comparison",
        "",
        f"- Timestamp: `{report.timestamp}`",
        f"- Baseline chunker: `{baseline}`",
        f"- Compared chunkers: {', '.join(f'`{n}`' for n in report.chunker_names)}",
        "",
        "## Aggregate scores",
        "",
    ]

    has_delta = len(report.chunker_names) > 1

    # Build header
    header_cells = ["Metric"] + [f"`{n}`" for n in report.chunker_names]
    if has_delta:
        for n in report.chunker_names[1:]:
            header_cells.append(f"Δ ({n} - {baseline})")
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cells)) + " |")

    baseline_agg = report.reports[baseline].aggregate
    for metric_label, attr in _METRICS:
        row: list[str] = [metric_label]
        for name in report.chunker_names:
            value = _agg_value(report.reports[name].aggregate, attr)
            row.append(f"{value:.3f}")
        if has_delta:
            base_val = _agg_value(baseline_agg, attr)
            for name in report.chunker_names[1:]:
                this_val = _agg_value(report.reports[name].aggregate, attr)
                delta = this_val - base_val
                row.append(f"{delta:+.3f}")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Per-chunker by-type breakdown", ""])
    for name in report.chunker_names:
        rpt = report.reports[name]
        lines.append(f"### `{name}` — by question type")
        lines.append("")
        lines.append(
            "| Type | Correctness | Faithfulness | Recall@k | Citation accuracy |"
        )
        lines.append("|---|---|---|---|---|")
        for t in sorted(rpt.by_type):
            s = rpt.by_type[t]
            lines.append(
                f"| {t} | {s.correctness:.3f} | {s.faithfulness:.3f} | "
                f"{s.recall_at_k:.3f} | {s.citation_accuracy:.3f} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def write_comparison_report(
    report: ComparisonReport, output_dir: Path
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_ts = report.timestamp.replace(":", "-")
    base = output_dir / f"chunking_compare_{safe_ts}"
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_comparison_markdown(report), encoding="utf-8")
    return json_path, md_path


_DEFAULT_GOLDEN = Path("eval/golden_qa.jsonl")
_DEFAULT_OUTPUT = Path("eval/results")


def _build_per_chunker_runner(chunker_name: str) -> RunEvalFn:
    """Closure that runs an eval for one chunker with isolated stores."""

    async def run(name: str) -> EvalReport:
        from openai import AsyncOpenAI

        from kuberag.config import Settings
        from kuberag.eval.cache import (
            EvalCache,
            build_config_hash,
            build_corpus_version,
        )
        from kuberag.eval.golden import load_golden_set
        from kuberag.eval.metrics import AnswerCorrectness, Faithfulness
        from kuberag.eval.run import EvalRunner
        from kuberag.generation.citations import CitationVerifier
        from kuberag.generation.generator import Generator
        from kuberag.generation.orchestrator import (
            CompletenessJudge,
            GenerationOrchestrator,
        )
        from kuberag.ingest.embedder import Embedder, EmbeddingCache
        from kuberag.retrieval.dense import DenseRetriever
        from kuberag.retrieval.hybrid import HybridSearch
        from kuberag.retrieval.reranker import Reranker
        from kuberag.retrieval.sparse import SparseRetriever
        from kuberag.stores import BM25Store, ChromaStore

        settings = Settings()  # type: ignore[call-arg]
        chroma_path = settings.chroma_path.parent / f"chroma_{name}"
        bm25_path = settings.bm25_path.parent / f"bm25_{name}.pkl"

        if not chroma_path.exists() or not list(chroma_path.iterdir()):
            raise SystemExit(
                f"No corpus indexed at {chroma_path}. Run ingest first:\n"
                f"  KUBERAG_CHROMA_PATH={chroma_path} "
                f"KUBERAG_BM25_PATH={bm25_path} "
                f"uv run python -m kuberag.ingest --path <docs> --chunker {name}"
            )

        client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        chroma = ChromaStore(chroma_path)
        bm25 = BM25Store(bm25_path)
        embedder = Embedder(
            model=settings.embedding_model,
            client=client,
            cache=EmbeddingCache(
                settings.chroma_path.parent / "embedding_cache"
            ),
        )
        hybrid = HybridSearch(
            dense=DenseRetriever(embedder=embedder, store=chroma),
            sparse=SparseRetriever(store=bm25),
            reranker=Reranker(
                client=client,
                model=settings.judge_model,
                top_n=settings.rerank_top_n,
            ),
            dense_weight=settings.rrf_dense_weight,
        )
        orchestrator = GenerationOrchestrator(
            generator=Generator(client=client, model=settings.generation_model),
            verifier=CitationVerifier(client=client, model=settings.judge_model),
            completeness_judge=CompletenessJudge(
                client=client, model=settings.judge_model
            ),
            confidence_threshold=settings.confidence_threshold,
        )
        relevant_config = {
            "embedding_model": settings.embedding_model,
            "generation_model": settings.generation_model,
            "judge_model": settings.judge_model,
            "retrieval_k": settings.retrieval_k,
            "rerank_top_n": settings.rerank_top_n,
            "rrf_dense_weight": settings.rrf_dense_weight,
            "confidence_threshold": settings.confidence_threshold,
            "chunker": name,
        }
        runner = EvalRunner(
            hybrid_search=hybrid,
            orchestrator=orchestrator,
            correctness=AnswerCorrectness(
                client=client, model=settings.judge_model
            ),
            faithfulness=Faithfulness(
                client=client, model=settings.judge_model
            ),
            cache=EvalCache(Path("eval/.cache")),
            corpus_version=build_corpus_version(
                chroma_count=chroma.count(), bm25_count=bm25.count()
            ),
            config_hash=build_config_hash(relevant_config),
            retrieval_k=settings.retrieval_k,
        )
        entries = load_golden_set(_DEFAULT_GOLDEN)
        return await runner.run(entries)

    return run


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="kuberag.eval.compare_chunking",
        description="Compare chunking strategies side-by-side over the golden set.",
    )
    parser.add_argument(
        "--chunkers",
        nargs="+",
        default=["fixed", "recursive"],
        help="Chunker names to compare (must each have an ingested corpus).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Where to write the comparison report.",
    )
    args = parser.parse_args(argv)

    async def _main() -> int:
        report = await compare_chunkers(
            chunker_names=args.chunkers,
            run_eval_for_chunker=_build_per_chunker_runner(args.chunkers[0]),
        )
        json_path, md_path = write_comparison_report(report, args.output_dir)
        print(f"\nComparison written:\n  {json_path}\n  {md_path}")
        return 0

    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())
