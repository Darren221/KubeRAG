from pathlib import Path

import pytest

from kuberag.eval.golden import (
    GoldenEntry,
    InvalidGoldenSetError,
    load_golden_set,
    summarize_distribution,
)


def _write_jsonl(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _entry_line(
    *,
    entry_id: str = "lookup-001",
    question: str = "what is a pod?",
    golden_answer: str = "A Pod is the smallest deployable unit.",
    expected: list[str] | None = None,
    type_: str = "lookup",
) -> str:
    import json

    payload = {
        "id": entry_id,
        "question": question,
        "golden_answer": golden_answer,
        "expected_source_files": expected if expected is not None else ["a.md"],
        "type": type_,
    }
    return json.dumps(payload)


def test_loads_single_entry(tmp_path: Path) -> None:
    p = tmp_path / "gold.jsonl"
    _write_jsonl(p, [_entry_line()])
    entries = load_golden_set(p)
    assert len(entries) == 1
    assert entries[0].id == "lookup-001"
    assert entries[0].type == "lookup"


def test_returns_typed_pydantic_models(tmp_path: Path) -> None:
    p = tmp_path / "gold.jsonl"
    _write_jsonl(p, [_entry_line()])
    entries = load_golden_set(p)
    assert isinstance(entries[0], GoldenEntry)


def test_skips_blank_lines_and_comments(tmp_path: Path) -> None:
    p = tmp_path / "gold.jsonl"
    _write_jsonl(
        p,
        [
            "# top-level comment",
            "",
            _entry_line(entry_id="a"),
            "  ",
            "# trailing comment",
            _entry_line(entry_id="b"),
        ],
    )
    entries = load_golden_set(p)
    assert [e.id for e in entries] == ["a", "b"]


def test_invalid_type_raises(tmp_path: Path) -> None:
    p = tmp_path / "gold.jsonl"
    _write_jsonl(p, [_entry_line(type_="trick_question")])
    with pytest.raises(InvalidGoldenSetError):
        load_golden_set(p)


def test_empty_question_raises(tmp_path: Path) -> None:
    p = tmp_path / "gold.jsonl"
    _write_jsonl(p, [_entry_line(question="")])
    with pytest.raises(InvalidGoldenSetError):
        load_golden_set(p)


def test_duplicate_ids_raises(tmp_path: Path) -> None:
    p = tmp_path / "gold.jsonl"
    _write_jsonl(p, [_entry_line(entry_id="dup"), _entry_line(entry_id="dup")])
    with pytest.raises(InvalidGoldenSetError):
        load_golden_set(p)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_golden_set(tmp_path / "nope.jsonl")


def test_summarize_distribution_counts_by_type(tmp_path: Path) -> None:
    p = tmp_path / "gold.jsonl"
    _write_jsonl(
        p,
        [
            _entry_line(entry_id="a", type_="lookup"),
            _entry_line(entry_id="b", type_="lookup"),
            _entry_line(entry_id="c", type_="multi_hop"),
            _entry_line(entry_id="d", type_="unanswerable"),
            _entry_line(entry_id="e", type_="ambiguous"),
        ],
    )
    entries = load_golden_set(p)
    counts = summarize_distribution(entries)
    assert counts["lookup"] == 2
    assert counts["multi_hop"] == 1
    assert counts["unanswerable"] == 1
    assert counts["ambiguous"] == 1


def test_unanswerable_entries_allow_empty_expected_sources(tmp_path: Path) -> None:
    p = tmp_path / "gold.jsonl"
    _write_jsonl(
        p,
        [
            _entry_line(
                entry_id="u-1",
                type_="unanswerable",
                expected=[],
            )
        ],
    )
    entries = load_golden_set(p)
    assert entries[0].expected_source_files == []


def test_real_golden_set_loads_cleanly() -> None:
    repo_root = Path(__file__).parent.parent.parent
    golden_path = repo_root / "eval" / "golden_qa.jsonl"
    entries = load_golden_set(golden_path)
    # Should have substantive entries
    assert len(entries) >= 1
    # All entries have unique ids
    assert len({e.id for e in entries}) == len(entries)
