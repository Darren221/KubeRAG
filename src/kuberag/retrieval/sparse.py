from kuberag.stores import BM25Store, Hit


class SparseRetriever:
    def __init__(self, *, store: BM25Store) -> None:
        self.store = store

    async def retrieve(self, query: str, k: int = 10) -> list[Hit]:
        if not query.strip():
            return []
        return self.store.query(query, k=k)
