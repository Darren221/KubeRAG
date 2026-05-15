from pydantic import BaseModel, ConfigDict, Field

from kuberag.stores import Hit


class FusedHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    text: str
    source: str
    section: str | None = None
    chunking_strategy: str = ""
    rrf_score: float
    rank: int = Field(ge=0)
    dense_rank: int | None = None
    sparse_rank: int | None = None


def rrf(
    dense_hits: list[Hit],
    sparse_hits: list[Hit],
    *,
    dense_weight: float = 0.7,
    k_constant: int = 60,
) -> list[FusedHit]:
    if not 0.0 <= dense_weight <= 1.0:
        raise ValueError("dense_weight must be in [0, 1]")
    if k_constant <= 0:
        raise ValueError("k_constant must be positive")

    representative: dict[str, Hit] = {}
    dense_ranks: dict[str, int] = {}
    sparse_ranks: dict[str, int] = {}

    for hit in dense_hits:
        representative.setdefault(hit.chunk_id, hit)
        dense_ranks[hit.chunk_id] = hit.rank

    for hit in sparse_hits:
        representative.setdefault(hit.chunk_id, hit)
        sparse_ranks[hit.chunk_id] = hit.rank

    scored: list[tuple[float, str]] = []
    for chunk_id in representative:
        d = dense_ranks.get(chunk_id)
        s = sparse_ranks.get(chunk_id)
        dense_contrib = (
            dense_weight / (k_constant + d + 1) if d is not None else 0.0
        )
        sparse_contrib = (
            (1.0 - dense_weight) / (k_constant + s + 1) if s is not None else 0.0
        )
        scored.append((dense_contrib + sparse_contrib, chunk_id))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    return [
        FusedHit(
            chunk_id=chunk_id,
            text=representative[chunk_id].text,
            source=representative[chunk_id].source,
            section=representative[chunk_id].section,
            chunking_strategy=representative[chunk_id].chunking_strategy,
            rrf_score=score,
            rank=rank,
            dense_rank=dense_ranks.get(chunk_id),
            sparse_rank=sparse_ranks.get(chunk_id),
        )
        for rank, (score, chunk_id) in enumerate(scored)
    ]
