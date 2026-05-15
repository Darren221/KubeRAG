import os

import httpx
import streamlit as st

from dashboard.api_client import KubeRAGClient
from kuberag.generation.orchestrator import GroundedAnswer, InsufficientAnswer

st.set_page_config(page_title="KubeRAG", page_icon=":mag:", layout="wide")
st.title("KubeRAG")
st.caption(
    "Hybrid-search RAG over Kubernetes documentation. "
    "Dense + sparse retrieval, LLM-as-judge reranking, "
    "cited generation with per-citation verification."
)


def _get_client() -> KubeRAGClient:
    base_url = os.environ.get("KUBERAG_API_URL", "http://localhost:8000")
    if "api_client" not in st.session_state:
        st.session_state.api_client = KubeRAGClient(base_url=base_url)
    client: KubeRAGClient = st.session_state.api_client
    return client


client = _get_client()

with st.sidebar:
    st.subheader("Settings")
    dense_only = st.toggle(
        "Dense-only retrieval",
        value=False,
        help="Skip sparse (BM25) retrieval and RRF. Useful for A/B-comparing hybrid vs. dense.",
    )
    k = st.slider("k (per-retriever candidates)", min_value=3, max_value=30, value=10)
    top_n = st.slider("top_n (final results)", min_value=1, max_value=10, value=5)

question = st.text_input(
    "Ask a question",
    placeholder="How does a readiness probe affect Service endpoints?",
)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Searching and answering..."):
        try:
            answer = client.ask(
                question, dense_only=dense_only, k=k, top_n=top_n
            )
            st.session_state.last_answer = answer
            st.session_state.last_question = question
        except httpx.HTTPError as exc:
            st.error(f"Request failed: {exc}")

if "last_answer" in st.session_state:
    answer = st.session_state.last_answer
    st.divider()
    if isinstance(answer, GroundedAnswer):
        st.markdown("### Answer")
        st.markdown(answer.text)
        st.json(answer.model_dump(mode="json"))
    elif isinstance(answer, InsufficientAnswer):
        st.warning(f"Insufficient context: {answer.reason}")
        if answer.suggested_documents:
            st.markdown("**Suggested documents to check:**")
            for doc in answer.suggested_documents:
                st.markdown(f"- `{doc}`")
        if answer.generated_text:
            with st.expander("What the model wrote (rejected by confidence check)"):
                st.markdown(answer.generated_text)
