from pathlib import Path

from kuberag.ingest.k8s_source import find_markdown_files


def make_repo_layout(repo_root: Path) -> None:
    """Construct a fake kubernetes/website-shaped directory."""
    docs = repo_root / "content/en/docs"
    (docs / "concepts").mkdir(parents=True)
    (docs / "tasks").mkdir(parents=True)
    (docs / "reference").mkdir(parents=True)
    (repo_root / "content/en/blog").mkdir(parents=True)
    (repo_root / "content/zh-cn/docs/concepts").mkdir(parents=True)

    (docs / "concepts/overview.md").write_text("# Overview")
    (docs / "tasks/install.md").write_text("# Install")
    (docs / "reference/api.md").write_text("# API")
    (repo_root / "content/en/blog/post.md").write_text("# Blog Post")
    (repo_root / "content/zh-cn/docs/concepts/overview.md").write_text("# Overview ZH")
    (docs / "concepts/image.png").write_bytes(b"")


def test_finds_files_in_each_subtree(tmp_path: Path) -> None:
    make_repo_layout(tmp_path)
    files = find_markdown_files(tmp_path)
    names = {f.name for f in files}
    assert {"overview.md", "install.md", "api.md"} <= names


def test_excludes_other_subtrees(tmp_path: Path) -> None:
    make_repo_layout(tmp_path)
    files = find_markdown_files(tmp_path)
    assert all("blog" not in str(f) for f in files)
    assert all("zh-cn" not in str(f) for f in files)


def test_excludes_non_markdown(tmp_path: Path) -> None:
    make_repo_layout(tmp_path)
    files = find_markdown_files(tmp_path)
    assert all(f.suffix == ".md" for f in files)


def test_handles_missing_subtree(tmp_path: Path) -> None:
    (tmp_path / "content/en/docs/concepts").mkdir(parents=True)
    (tmp_path / "content/en/docs/concepts/overview.md").write_text("# Overview")
    files = find_markdown_files(tmp_path)
    assert len(files) == 1


def test_empty_repo_returns_empty(tmp_path: Path) -> None:
    assert find_markdown_files(tmp_path) == []


def test_returns_sorted(tmp_path: Path) -> None:
    make_repo_layout(tmp_path)
    files = find_markdown_files(tmp_path)
    assert files == sorted(files)


def test_finds_files_nested_deeply(tmp_path: Path) -> None:
    nested = tmp_path / "content/en/docs/concepts/architecture/control-plane"
    nested.mkdir(parents=True)
    (nested / "api-server.md").write_text("# API Server")
    files = find_markdown_files(tmp_path)
    assert any(f.name == "api-server.md" for f in files)


def test_custom_subtrees(tmp_path: Path) -> None:
    (tmp_path / "alt/path").mkdir(parents=True)
    (tmp_path / "alt/path/doc.md").write_text("# Doc")
    files = find_markdown_files(tmp_path, subtrees=["alt/path"])
    assert len(files) == 1
    assert files[0].name == "doc.md"
