import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts/ to sys.path so we can import seed
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import seed  # noqa: E402


def test_should_skip_seeding_when_both_stores_populated_and_match() -> None:
    chroma = MagicMock()
    chroma.count.return_value = 42
    bm25 = MagicMock()
    bm25.count.return_value = 42
    assert seed.should_skip_seeding(chroma, bm25) is True


def test_should_seed_when_chroma_empty() -> None:
    chroma = MagicMock()
    chroma.count.return_value = 0
    bm25 = MagicMock()
    bm25.count.return_value = 0
    assert seed.should_skip_seeding(chroma, bm25) is False


def test_should_reseed_when_stores_disagree(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Simulates a prior seed that crashed between chroma write and bm25 write.
    # Without the consistency check, hybrid search would silently degrade to
    # dense-only on next boot.
    chroma = MagicMock()
    chroma.count.return_value = 42
    bm25 = MagicMock()
    bm25.count.return_value = 0
    assert seed.should_skip_seeding(chroma, bm25) is False
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "out of sync" in out


def test_seed_docs_list_is_substantive() -> None:
    # Sanity check the curated list is non-trivial but under our budget
    assert 20 <= len(seed.SEED_DOCS) <= 30


def test_seed_docs_all_point_to_english() -> None:
    # Defense against accidentally including a translated path
    assert all(p.startswith("content/en/docs/") for p in seed.SEED_DOCS)


def test_seed_docs_all_markdown() -> None:
    assert all(p.endswith(".md") for p in seed.SEED_DOCS)


def test_seed_docs_no_duplicates() -> None:
    assert len(seed.SEED_DOCS) == len(set(seed.SEED_DOCS))


def test_download_doc_writes_under_target_dir(tmp_path: Path) -> None:
    fake_content = b"# Pods\nA pod is..."

    def fake_urlopen(url: str, timeout: int = 30) -> MagicMock:
        response = MagicMock()
        response.__enter__ = lambda self: response
        response.__exit__ = lambda self, *args: None
        response.read.return_value = fake_content
        return response

    with patch.object(seed.urllib.request, "urlopen", side_effect=fake_urlopen):
        out = seed.download_doc(
            "content/en/docs/concepts/workloads/pods/_index.md", tmp_path
        )
    assert out.exists()
    assert out.read_bytes() == fake_content
    # Path mirrors the source structure under target_dir
    assert "content/en/docs/concepts/workloads/pods/_index.md" in str(out)


def test_download_doc_creates_parent_dirs(tmp_path: Path) -> None:
    def fake_urlopen(url: str, timeout: int = 30) -> MagicMock:
        response = MagicMock()
        response.__enter__ = lambda self: response
        response.__exit__ = lambda self, *args: None
        response.read.return_value = b"x"
        return response

    deep_path = "content/en/docs/concepts/workloads/pods/_index.md"
    with patch.object(seed.urllib.request, "urlopen", side_effect=fake_urlopen):
        out = seed.download_doc(deep_path, tmp_path)
    assert out.parent.is_dir()


def test_download_doc_uses_ref_in_url(tmp_path: Path) -> None:
    captured: dict[str, str] = {}

    def fake_urlopen(url: str, timeout: int = 30) -> MagicMock:
        captured["url"] = url
        response = MagicMock()
        response.__enter__ = lambda self: response
        response.__exit__ = lambda self, *args: None
        response.read.return_value = b"x"
        return response

    with patch.object(seed.urllib.request, "urlopen", side_effect=fake_urlopen):
        seed.download_doc("path/to/doc.md", tmp_path, ref="v1.30.0")
    assert "v1.30.0" in captured["url"]
    assert "path/to/doc.md" in captured["url"]
