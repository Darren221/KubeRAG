import hashlib
from pathlib import Path

from kuberag.eval.cache import EvalCache, build_config_hash, build_corpus_version


def test_get_miss_returns_none(tmp_path: Path) -> None:
    cache = EvalCache(tmp_path)
    assert cache.get("nonexistent-key") is None


def test_put_and_get_round_trip(tmp_path: Path) -> None:
    cache = EvalCache(tmp_path)
    cache.put("key-1", {"score": 0.9, "text": "hello"})
    assert cache.get("key-1") == {"score": 0.9, "text": "hello"}


def test_key_combines_question_corpus_and_config(tmp_path: Path) -> None:
    cache = EvalCache(tmp_path)
    k = cache.key(question="q", corpus_version="cv", config_hash="ch")
    expected = hashlib.sha256(b"q\x00cv\x00ch").hexdigest()
    assert k == expected


def test_key_changes_when_question_changes(tmp_path: Path) -> None:
    cache = EvalCache(tmp_path)
    k1 = cache.key(question="q1", corpus_version="cv", config_hash="ch")
    k2 = cache.key(question="q2", corpus_version="cv", config_hash="ch")
    assert k1 != k2


def test_key_changes_when_corpus_changes(tmp_path: Path) -> None:
    cache = EvalCache(tmp_path)
    k1 = cache.key(question="q", corpus_version="cv1", config_hash="ch")
    k2 = cache.key(question="q", corpus_version="cv2", config_hash="ch")
    assert k1 != k2


def test_key_changes_when_config_changes(tmp_path: Path) -> None:
    cache = EvalCache(tmp_path)
    k1 = cache.key(question="q", corpus_version="cv", config_hash="ch1")
    k2 = cache.key(question="q", corpus_version="cv", config_hash="ch2")
    assert k1 != k2


def test_persists_across_instances(tmp_path: Path) -> None:
    cache1 = EvalCache(tmp_path)
    cache1.put("k", {"v": 1})
    cache2 = EvalCache(tmp_path)
    assert cache2.get("k") == {"v": 1}


def test_creates_directory_if_missing(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "cache"
    EvalCache(path)
    assert path.is_dir()


def test_build_config_hash_is_stable() -> None:
    h1 = build_config_hash(
        {"chunker": "fixed", "k": 10, "model": "gpt-4o"}
    )
    h2 = build_config_hash(
        {"k": 10, "model": "gpt-4o", "chunker": "fixed"}  # different key order
    )
    assert h1 == h2  # sort_keys eliminates key-order influence


def test_build_config_hash_changes_on_value_change() -> None:
    h1 = build_config_hash({"k": 10})
    h2 = build_config_hash({"k": 11})
    assert h1 != h2


def test_build_corpus_version_from_counts() -> None:
    v = build_corpus_version(chroma_count=50, bm25_count=50)
    assert "50" in v


def test_build_corpus_version_changes_on_count_change() -> None:
    v1 = build_corpus_version(chroma_count=50, bm25_count=50)
    v2 = build_corpus_version(chroma_count=51, bm25_count=51)
    assert v1 != v2


def test_get_corrupt_file_returns_none(tmp_path: Path) -> None:
    cache = EvalCache(tmp_path)
    (tmp_path / "corrupt.json").write_text("not valid json {{{")
    assert cache.get("corrupt") is None
