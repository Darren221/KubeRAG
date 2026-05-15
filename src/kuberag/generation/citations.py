import re

from pydantic import BaseModel, ConfigDict, Field

from kuberag.generation.models import Answer
from kuberag.retrieval.fusion import FusedHit

_MARKER_RE = re.compile(r"\[(\d+)\]")


class CitationError(ValueError):
    """Raised when a citation marker does not reference a valid chunk."""


class ParsedCitation(BaseModel):
    model_config = ConfigDict(frozen=True)

    marker: int = Field(ge=1)
    claim_span: str
    chunk_id: str
    source: str
    section: str | None = None
    chunk_text: str


def find_text_markers(text: str) -> set[int]:
    return {int(m.group(1)) for m in _MARKER_RE.finditer(text)}


def parse_citations(answer: Answer, chunks: list[FusedHit]) -> list[ParsedCitation]:
    text_markers = find_text_markers(answer.text)
    structured_markers = {c.marker for c in answer.citations}
    all_markers = text_markers | structured_markers

    if all_markers and not chunks:
        raise CitationError(
            "answer contains citation markers but no chunks were supplied"
        )

    for marker in all_markers:
        if marker < 1 or marker > len(chunks):
            raise CitationError(
                f"citation marker [{marker}] does not map to any chunk "
                f"(supplied {len(chunks)} chunks)"
            )

    result: list[ParsedCitation] = []
    for citation in answer.citations:
        chunk = chunks[citation.marker - 1]
        result.append(
            ParsedCitation(
                marker=citation.marker,
                claim_span=citation.claim_span,
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                section=chunk.section,
                chunk_text=chunk.text,
            )
        )
    return result
