from kuberag.retrieval.dense import DenseRetriever
from kuberag.retrieval.fusion import FusedHit, rrf
from kuberag.retrieval.reranker import Reranker
from kuberag.retrieval.sparse import SparseRetriever
from kuberag.stores import Hit


def _hit_to_fused(hit: Hit) -> FusedHit:
    return FusedHit(
        chunk_id=hit.chunk_id,
        text=hit.text,
        source=hit.source,
        section=hit.section,
        chunking_strategy=hit.chunking_strategy,
        rrf_score=hit.score,
        rank=hit.rank,
        dense_rank=hit.rank,
        sparse_rank=None,
    )


class HybridSearch:
    def __init__(
        self,
        *,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        reranker: Reranker,
        dense_weight: float = 0.7,
        k_constant: int = 60,
    ) -> None:
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.dense_weight = dense_weight
        self.k_constant = k_constant

    async def search(
        self,
        query: str,
        *,
        k: int = 10,
        top_n: int = 5,
        dense_only: bool = False,
    ) -> list[FusedHit]:
        if not query.strip():
            return []

        dense_hits = await self.dense.retrieve(query, k=k)

        if dense_only:
            candidates = [_hit_to_fused(h) for h in dense_hits]
        else:
            sparse_hits = await self.sparse.retrieve(query, k=k)
            candidates = rrf(
                dense_hits,
                sparse_hits,
                dense_weight=self.dense_weight,
                k_constant=self.k_constant,
            )

        return await self.reranker.rerank(query, candidates, top_n=top_n)
