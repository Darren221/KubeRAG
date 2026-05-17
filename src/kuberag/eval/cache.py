import hashlib
import json
from pathlib import Path
from typing import Any


class EvalCache:
    """Disk-backed JSON cache for eval results.

    Keys are derived from (question, corpus_version, config_hash) so that
    cache entries invalidate automatically when the question, corpus, or
    pipeline config changes.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def key(self, *, question: str, corpus_version: str, config_hash: str) -> str:
        h = hashlib.sha256()
        h.update(question.encode("utf-8"))
        h.update(b"\0")
        h.update(corpus_version.encode("utf-8"))
        h.update(b"\0")
        h.update(config_hash.encode("utf-8"))
        return h.hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        file = self.path / f"{key}.json"
        if not file.exists():
            return None
        try:
            return json.loads(file.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        file = self.path / f"{key}.json"
        tmp = file.with_suffix(file.suffix + ".tmp")
        tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
        tmp.replace(file)


def build_config_hash(relevant_config: dict[str, Any]) -> str:
    serialized = json.dumps(relevant_config, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def build_corpus_version(*, chroma_count: int, bm25_count: int) -> str:
    return f"chroma={chroma_count},bm25={bm25_count}"
