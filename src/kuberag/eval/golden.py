from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

GoldenType = Literal["lookup", "multi_hop", "unanswerable", "ambiguous"]


class GoldenEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    golden_answer: str
    expected_source_files: list[str] = Field(default_factory=list)
    type: GoldenType


class InvalidGoldenSetError(ValueError):
    """Raised when a golden-set file fails to parse or validate."""


def load_golden_set(path: Path) -> list[GoldenEntry]:
    if not path.exists():
        raise FileNotFoundError(path)

    entries: list[GoldenEntry] = []
    seen_ids: set[str] = set()

    with path.open(encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                entry = GoldenEntry.model_validate_json(line)
            except ValidationError as exc:
                raise InvalidGoldenSetError(
                    f"{path}:{line_num} — invalid entry: {exc}"
                ) from exc
            if entry.id in seen_ids:
                raise InvalidGoldenSetError(
                    f"{path}:{line_num} — duplicate id '{entry.id}'"
                )
            seen_ids.add(entry.id)
            entries.append(entry)

    return entries


def summarize_distribution(entries: list[GoldenEntry]) -> dict[str, int]:
    return dict(Counter(entry.type for entry in entries))
