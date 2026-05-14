from pathlib import Path

import pytest

from kuberag.config import Settings
from kuberag.ingest.cli import parse_args, resolve_paths


def test_parse_args_requires_path_or_source() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--chunker", "fixed"])


def test_parse_args_with_path(tmp_path: Path) -> None:
    args = parse_args(["--path", str(tmp_path)])
    assert args.path == tmp_path
    assert args.source is None
    assert args.chunker == "fixed"


def test_parse_args_with_recursive_chunker(tmp_path: Path) -> None:
    args = parse_args(["--path", str(tmp_path), "--chunker", "recursive"])
    assert args.chunker == "recursive"


def test_parse_args_with_source_k8s() -> None:
    args = parse_args(["--source", "k8s"])
    assert args.source == "k8s"
    assert args.path is None


def test_parse_args_path_and_source_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--path", str(tmp_path), "--source", "k8s"])


def test_parse_args_rejects_unknown_chunker(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--path", str(tmp_path), "--chunker", "semantic"])


def test_resolve_paths_single_file(tmp_path: Path, test_settings: Settings) -> None:
    f = tmp_path / "doc.md"
    f.write_text("hello")
    args = parse_args(["--path", str(f)])
    assert resolve_paths(args, test_settings) == [f]


def test_resolve_paths_directory_finds_supported_extensions(
    tmp_path: Path, test_settings: Settings
) -> None:
    (tmp_path / "a.md").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "c.html").write_text("<p>c</p>")
    (tmp_path / "d.unknown").write_text("d")
    args = parse_args(["--path", str(tmp_path)])
    names = {p.name for p in resolve_paths(args, test_settings)}
    assert names == {"a.md", "b.txt", "c.html"}


def test_resolve_paths_directory_recurses(tmp_path: Path, test_settings: Settings) -> None:
    nested = tmp_path / "sub" / "deeper"
    nested.mkdir(parents=True)
    (nested / "doc.md").write_text("hi")
    args = parse_args(["--path", str(tmp_path)])
    paths = resolve_paths(args, test_settings)
    assert any(p.name == "doc.md" for p in paths)


def test_resolve_paths_nonexistent_raises(tmp_path: Path, test_settings: Settings) -> None:
    args = parse_args(["--path", str(tmp_path / "missing")])
    with pytest.raises(FileNotFoundError):
        resolve_paths(args, test_settings)


def test_resolve_paths_returns_sorted(tmp_path: Path, test_settings: Settings) -> None:
    (tmp_path / "z.md").write_text("z")
    (tmp_path / "a.md").write_text("a")
    args = parse_args(["--path", str(tmp_path)])
    paths = resolve_paths(args, test_settings)
    assert paths == sorted(paths)
