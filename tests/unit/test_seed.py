import importlib
import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add scripts/ to sys.path so we can import seed
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import seed  # noqa: E402


def _store(count: int) -> MagicMock:
    store = MagicMock()
    store.count.return_value = count
    return store


# ─── decide_seed_action ────────────────────────────────────────────────────


def test_decide_seed_action_skip_when_both_match_and_nonzero() -> None:
    assert seed.decide_seed_action(_store(42), _store(42)) is seed.SeedAction.SKIP


def test_decide_seed_action_seed_when_both_empty() -> None:
    assert seed.decide_seed_action(_store(0), _store(0)) is seed.SeedAction.SEED


def test_decide_seed_action_reseed_when_chroma_populated_bm25_empty() -> None:
    assert seed.decide_seed_action(_store(42), _store(0)) is seed.SeedAction.RESEED


def test_decide_seed_action_reseed_when_chroma_empty_bm25_populated() -> None:
    # Symmetric to the populated/empty case — both directions are partial-seed
    # artifacts and both deserve the same recovery path. Prior version
    # short-circuited on chroma==0 and silently skipped the re-seed.
    assert seed.decide_seed_action(_store(0), _store(42)) is seed.SeedAction.RESEED


def test_decide_seed_action_reseed_when_both_populated_but_differ() -> None:
    # General mismatch (e.g. ingest crashed mid-batch).
    assert seed.decide_seed_action(_store(42), _store(100)) is seed.SeedAction.RESEED


# ─── _validate_seed_ref ────────────────────────────────────────────────────


def test_validate_seed_ref_accepts_sha() -> None:
    sha = "06a3cd92aed8ca35c9fd966bb153dd46e21306e2"
    assert seed._validate_seed_ref(sha) == sha


def test_validate_seed_ref_accepts_tag() -> None:
    assert seed._validate_seed_ref("v1.30.0") == "v1.30.0"


def test_validate_seed_ref_accepts_branch() -> None:
    assert seed._validate_seed_ref("main") == "main"


def test_validate_seed_ref_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="invalid KUBERAG_SEED_REF"):
        seed._validate_seed_ref("../../attacker/repo/main")


def test_validate_seed_ref_rejects_slash() -> None:
    with pytest.raises(ValueError, match="invalid KUBERAG_SEED_REF"):
        seed._validate_seed_ref("main/foo")


def test_validate_seed_ref_rejects_empty() -> None:
    with pytest.raises(ValueError, match="invalid KUBERAG_SEED_REF"):
        seed._validate_seed_ref("")


def test_validate_seed_ref_rejects_too_long() -> None:
    with pytest.raises(ValueError, match="invalid KUBERAG_SEED_REF"):
        seed._validate_seed_ref("a" * 65)


# ─── SEED_REF default + env override ───────────────────────────────────────


def test_seed_ref_defaults_to_pinned_sha_when_no_env_override() -> None:
    """When KUBERAG_SEED_REF is unset, SEED_REF must be a 40-char hex SHA."""
    if os.environ.get("KUBERAG_SEED_REF"):
        pytest.skip("env override active; can't assert default")
    assert re.match(r"^[a-f0-9]{40}$", seed.SEED_REF) is not None, (
        f"SEED_REF should default to a 40-char hex SHA, got {seed.SEED_REF!r}"
    )


def test_seed_ref_honors_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting KUBERAG_SEED_REF should change SEED_REF after module reload."""
    monkeypatch.setenv("KUBERAG_SEED_REF", "v1.30.0")
    reloaded = importlib.reload(seed)
    try:
        assert reloaded.SEED_REF == "v1.30.0"
    finally:
        # Restore the unset state for any tests that follow in this run.
        monkeypatch.delenv("KUBERAG_SEED_REF", raising=False)
        importlib.reload(seed)


def test_seed_ref_validation_runs_at_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid KUBERAG_SEED_REF should fail loudly at import, not silently."""
    monkeypatch.setenv("KUBERAG_SEED_REF", "../../bad")
    try:
        with pytest.raises(ValueError, match="invalid KUBERAG_SEED_REF"):
            importlib.reload(seed)
    finally:
        monkeypatch.delenv("KUBERAG_SEED_REF", raising=False)
        importlib.reload(seed)


# ─── SEED_DOCS list invariants (pre-existing, unchanged) ───────────────────


def test_seed_docs_list_is_substantive() -> None:
    assert 20 <= len(seed.SEED_DOCS) <= 30


def test_seed_docs_all_point_to_english() -> None:
    assert all(p.startswith("content/en/docs/") for p in seed.SEED_DOCS)


def test_seed_docs_all_markdown() -> None:
    assert all(p.endswith(".md") for p in seed.SEED_DOCS)


def test_seed_docs_no_duplicates() -> None:
    assert len(seed.SEED_DOCS) == len(set(seed.SEED_DOCS))


# ─── download_doc (pre-existing, unchanged) ────────────────────────────────


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
