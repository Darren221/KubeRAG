from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from kuberag.eval.cache import EvalCache, build_config_hash, build_corpus_version
from kuberag.eval.golden import GoldenEntry, load_golden_set
from kuberag.eval.metrics import (
    AnswerCorrectness,
    Faithfulness,
    citation_accuracy,
    recall_at_k,
)
from kuberag.generation.orchestrator import GenerationOrchestrator, GroundedAnswer
from kuberag.retrieval.hybrid import HybridSearch


class PerEntryResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    golden_id: str
    question: str
    type: str
    answer_kind: Literal["grounded", "insufficient"]
    answer_text: str
    correctness_score: float = Field(ge=0.0, le=1.0)
    correctness_rationale: str
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    citation_accuracy: float = Field(ge=0.0, le=1.0)
    citation_count: int = Field(ge=0)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


class AggregateScores(BaseModel):
    model_config = ConfigDict(frozen=True)

    correctness: float
    faithfulness: float
    recall_at_k: float
    citation_accuracy: float


class EvalReport(BaseModel):
    timestamp: str
    n_questions: int
    corpus_version: str
    config_hash: str
    retrieval_k: int
    entries: list[PerEntryResult]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def aggregate(self) -> AggregateScores:
        return AggregateScores(
            correctness=_mean([e.correctness_score for e in self.entries]),
            faithfulness=_mean([e.faithfulness_score for e in self.entries]),
            recall_at_k=_mean([e.recall_at_k for e in self.entries]),
            citation_accuracy=_mean([e.citation_accuracy for e in self.entries]),
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def by_type(self) -> dict[str, AggregateScores]:
        by_type: dict[str, list[PerEntryResult]] = {}
        for entry in self.entries:
            by_type.setdefault(entry.type, []).append(entry)
        return {
            t: AggregateScores(
                correctness=_mean([e.correctness_score for e in es]),
                faithfulness=_mean([e.faithfulness_score for e in es]),
                recall_at_k=_mean([e.recall_at_k for e in es]),
                citation_accuracy=_mean([e.citation_accuracy for e in es]),
            )
            for t, es in by_type.items()
        }


def render_markdown(report: EvalReport) -> str:
    agg = report.aggregate
    lines = [
        "# KubeRAG Eval Report",
        "",
        f"- Timestamp: `{report.timestamp}`",
        f"- Questions: {report.n_questions}",
        f"- Corpus version: `{report.corpus_version}`",
        f"- Config hash: `{report.config_hash}`",
        f"- Retrieval k: {report.retrieval_k}",
        "",
        "## Headline scores",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| Answer correctness | {agg.correctness:.3f} |",
        f"| Faithfulness | {agg.faithfulness:.3f} |",
        f"| Retrieval recall@k | {agg.recall_at_k:.3f} |",
        f"| Citation accuracy | {agg.citation_accuracy:.3f} |",
        "",
        "## By question type",
        "",
        "| Type | Correctness | Faithfulness | Recall@k | Citation acc. | N |",
        "|---|---|---|---|---|---|",
    ]
    by_type = report.by_type
    counts: dict[str, int] = {}
    for entry in report.entries:
        counts[entry.type] = counts.get(entry.type, 0) + 1
    for t in sorted(by_type):
        s = by_type[t]
        lines.append(
            f"| {t} | {s.correctness:.3f} | {s.faithfulness:.3f} | "
            f"{s.recall_at_k:.3f} | {s.citation_accuracy:.3f} | {counts[t]} |"
        )
    lines.extend(
        [
            "",
            "## Per-entry detail",
            "",
            "| ID | Type | Correctness | Faithfulness | Recall | Cite acc. | Kind |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for e in report.entries:
        lines.append(
            f"| {e.golden_id} | {e.type} | {e.correctness_score:.2f} | "
            f"{e.faithfulness_score:.2f} | {e.recall_at_k:.2f} | "
            f"{e.citation_accuracy:.2f} | {e.answer_kind} |"
        )
    return "\n".join(lines) + "\n"


def write_reports(report: EvalReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_ts = report.timestamp.replace(":", "-")
    json_path = output_dir / f"{safe_ts}.json"
    md_path = output_dir / f"{safe_ts}.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


class EvalRunner:
    def __init__(
        self,
        *,
        hybrid_search: HybridSearch,
        orchestrator: GenerationOrchestrator,
        correctness: AnswerCorrectness,
        faithfulness: Faithfulness,
        cache: EvalCache,
        corpus_version: str,
        config_hash: str,
        retrieval_k: int = 10,
    ) -> None:
        self.hybrid_search = hybrid_search
        self.orchestrator = orchestrator
        self.correctness = correctness
        self.faithfulness = faithfulness
        self.cache = cache
        self.corpus_version = corpus_version
        self.config_hash = config_hash
        self.retrieval_k = retrieval_k

    async def run(
        self,
        entries: Sequence[GoldenEntry],
        *,
        no_cache: bool = False,
    ) -> EvalReport:
        results: list[PerEntryResult] = []
        for entry in entries:
            cache_key = self.cache.key(
                question=entry.question,
                corpus_version=self.corpus_version,
                config_hash=self.config_hash,
            )
            cached_payload: dict[str, Any] | None = None
            if not no_cache:
                cached_payload = self.cache.get(cache_key)

            if cached_payload is not None:
                result = PerEntryResult.model_validate(cached_payload)
            else:
                result = await self._eval_one(entry)
                self.cache.put(cache_key, result.model_dump())

            results.append(result)

        return EvalReport(
            timestamp=datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ"),
            n_questions=len(results),
            corpus_version=self.corpus_version,
            config_hash=self.config_hash,
            retrieval_k=self.retrieval_k,
            entries=results,
        )

    async def _eval_one(self, entry: GoldenEntry) -> PerEntryResult:
        chunks = await self.hybrid_search.search(
            entry.question, k=self.retrieval_k
        )
        answer = await self.orchestrator.answer(entry.question, chunks)

        if isinstance(answer, GroundedAnswer):
            answer_text = answer.text
            citations = list(answer.citations)
        else:
            answer_text = answer.generated_text or ""
            citations = []

        correctness_verdict = await self.correctness.score(
            entry.question, entry.golden_answer, answer_text
        )
        faithfulness_verdict = await self.faithfulness.score(
            answer_text, [c.text for c in chunks]
        )
        recall = recall_at_k(
            [c.source for c in chunks],
            entry.expected_source_files,
            self.retrieval_k,
        )
        cite_acc = citation_accuracy(citations)

        return PerEntryResult(
            golden_id=entry.id,
            question=entry.question,
            type=entry.type,
            answer_kind=answer.kind,
            answer_text=answer_text,
            correctness_score=correctness_verdict.score,
            correctness_rationale=correctness_verdict.rationale,
            faithfulness_score=faithfulness_verdict.score,
            recall_at_k=recall,
            citation_accuracy=cite_acc,
            citation_count=len(citations),
        )


_DEFAULT_GOLDEN = Path("eval/golden_qa.jsonl")
_DEFAULT_OUTPUT = Path("eval/results")
_DEFAULT_CACHE = Path("eval/.cache")


def _build_cli_runner() -> tuple[EvalRunner, Path]:
    """Construct an EvalRunner from real Settings + production components."""
    from openai import AsyncOpenAI

    from kuberag.config import Settings
    from kuberag.generation.citations import CitationVerifier
    from kuberag.generation.generator import Generator
    from kuberag.generation.orchestrator import CompletenessJudge
    from kuberag.ingest.embedder import Embedder, EmbeddingCache
    from kuberag.retrieval.dense import DenseRetriever
    from kuberag.retrieval.reranker import Reranker
    from kuberag.retrieval.sparse import SparseRetriever
    from kuberag.stores import BM25Store, ChromaStore

    settings = Settings()  # type: ignore[call-arg]
    client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    chroma = ChromaStore(settings.chroma_path)
    bm25 = BM25Store(settings.bm25_path)
    embedder = Embedder(
        model=settings.embedding_model,
        client=client,
        cache=EmbeddingCache(settings.chroma_path.parent / "embedding_cache"),
    )
    hybrid = HybridSearch(
        dense=DenseRetriever(embedder=embedder, store=chroma),
        sparse=SparseRetriever(store=bm25),
        reranker=Reranker(
            client=client, model=settings.judge_model, top_n=settings.rerank_top_n
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
    correctness = AnswerCorrectness(client=client, model=settings.judge_model)
    faithfulness = Faithfulness(client=client, model=settings.judge_model)

    relevant_config = {
        "embedding_model": settings.embedding_model,
        "generation_model": settings.generation_model,
        "judge_model": settings.judge_model,
        "retrieval_k": settings.retrieval_k,
        "rerank_top_n": settings.rerank_top_n,
        "rrf_dense_weight": settings.rrf_dense_weight,
        "confidence_threshold": settings.confidence_threshold,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    runner = EvalRunner(
        hybrid_search=hybrid,
        orchestrator=orchestrator,
        correctness=correctness,
        faithfulness=faithfulness,
        cache=EvalCache(_DEFAULT_CACHE),
        corpus_version=build_corpus_version(
            chroma_count=chroma.count(), bm25_count=bm25.count()
        ),
        config_hash=build_config_hash(relevant_config),
        retrieval_k=settings.retrieval_k,
    )
    return runner, _DEFAULT_OUTPUT


async def _main_async(
    golden_path: Path, output_dir: Path, no_cache: bool, limit: int | None
) -> int:
    runner, _ = _build_cli_runner()
    entries = load_golden_set(golden_path)
    if limit is not None:
        entries = entries[:limit]

    report = await runner.run(entries, no_cache=no_cache)
    json_path, md_path = write_reports(report, output_dir)

    print("\n=== KubeRAG Eval Report ===")
    print(f"Questions: {report.n_questions}")
    print(f"Corpus version: {report.corpus_version}")
    agg = report.aggregate
    print(f"Correctness:        {agg.correctness:.3f}")
    print(f"Faithfulness:       {agg.faithfulness:.3f}")
    print(f"Retrieval recall@k: {agg.recall_at_k:.3f}")
    print(f"Citation accuracy:  {agg.citation_accuracy:.3f}")
    print(f"\nReports written:\n  {json_path}\n  {md_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="kuberag.eval.run",
        description="Run KubeRAG eval against a golden Q&A dataset.",
    )
    parser.add_argument(
        "--golden", type=Path, default=_DEFAULT_GOLDEN, help="Path to golden JSONL."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Where to write JSON + Markdown reports.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Bypass eval cache.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N entries (for smoke tests).",
    )
    args = parser.parse_args(argv)
    return asyncio.run(
        _main_async(args.golden, args.output_dir, args.no_cache, args.limit)
    )


if __name__ == "__main__":
    raise SystemExit(main())
