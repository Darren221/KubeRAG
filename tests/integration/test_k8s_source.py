from pathlib import Path

import pytest

from kuberag.ingest.k8s_source import K8sDocsSource

pytestmark = [pytest.mark.integration, pytest.mark.network]


def test_fetch_clones_and_returns_markdown_paths(tmp_path: Path) -> None:
    source = K8sDocsSource()
    files = source.fetch(tmp_path)
    assert len(files) > 100
    assert all(f.suffix == ".md" for f in files)
    for f in files:
        assert any(subtree in str(f) for subtree in K8sDocsSource.SUBTREES)


def test_fetch_is_idempotent_on_rerun(tmp_path: Path) -> None:
    source = K8sDocsSource()
    files_first = source.fetch(tmp_path)
    files_second = source.fetch(tmp_path)
    assert files_first == files_second
