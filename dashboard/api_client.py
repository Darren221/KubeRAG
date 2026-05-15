import httpx
from pydantic import TypeAdapter

from kuberag.api import HealthResponse
from kuberag.generation.orchestrator import AnswerResult
from kuberag.stores import DocumentSummary

_ANSWER_ADAPTER: TypeAdapter[AnswerResult] = TypeAdapter(AnswerResult)
_DOCUMENTS_ADAPTER: TypeAdapter[list[DocumentSummary]] = TypeAdapter(
    list[DocumentSummary]
)


class KubeRAGClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 60.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url
        self._client = http_client or httpx.Client(
            base_url=base_url, timeout=timeout
        )

    def ask(
        self,
        question: str,
        *,
        dense_only: bool = False,
        k: int = 10,
        top_n: int = 5,
    ) -> AnswerResult:
        response = self._client.post(
            "/v1/ask",
            json={
                "question": question,
                "dense_only": dense_only,
                "k": k,
                "top_n": top_n,
            },
        )
        response.raise_for_status()
        return _ANSWER_ADAPTER.validate_python(response.json())

    def health(self) -> HealthResponse:
        response = self._client.get("/v1/health")
        response.raise_for_status()
        return HealthResponse.model_validate(response.json())

    def list_documents(self) -> list[DocumentSummary]:
        response = self._client.get("/v1/documents")
        response.raise_for_status()
        return _DOCUMENTS_ADAPTER.validate_python(response.json())

    def close(self) -> None:
        self._client.close()
