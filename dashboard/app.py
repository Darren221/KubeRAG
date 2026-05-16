import os

import httpx
import streamlit as st

from dashboard.api_client import KubeRAGClient
from dashboard.components import (
    render_chunks_panel,
    render_grounded_answer,
    render_insufficient_answer,
    render_styles,
)
from kuberag.generation.orchestrator import GroundedAnswer, InsufficientAnswer

st.set_page_config(page_title="KubeRAG", page_icon=":mag:", layout="wide")
render_styles()
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
    compare_mode = st.toggle(
        "Side-by-side comparison",
        value=False,
        help="Run both hybrid and dense-only and render them in two columns.",
    )
    dense_only = st.toggle(
        "Dense-only retrieval",
        value=False,
        disabled=compare_mode,
        help="Skip sparse (BM25) retrieval and RRF. Disabled in comparison mode.",
    )
    k = st.slider("k (per-retriever candidates)", min_value=3, max_value=30, value=10)
    top_n = st.slider("top_n (final results)", min_value=1, max_value=10, value=5)

question = st.text_input(
    "Ask a question",
    placeholder="How does a readiness probe affect Service endpoints?",
)


def _render_answer_block(
    answer: GroundedAnswer | InsufficientAnswer, *, anchor_prefix: str = "chunk-"
) -> None:
    if isinstance(answer, GroundedAnswer):
        render_grounded_answer(answer, anchor_prefix=anchor_prefix)
        st.divider()
        render_chunks_panel(
            answer.retrieved_chunks,
            citations=answer.citations,
            anchor_prefix=anchor_prefix,
        )
    else:
        render_insufficient_answer(answer)
        if answer.retrieved_chunks:
            st.divider()
            render_chunks_panel(
                answer.retrieved_chunks, anchor_prefix=anchor_prefix
            )


if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Searching and answering..."):
        try:
            if compare_mode:
                st.session_state.last_hybrid = client.ask(
                    question, dense_only=False, k=k, top_n=top_n
                )
                st.session_state.last_dense = client.ask(
                    question, dense_only=True, k=k, top_n=top_n
                )
                st.session_state.last_mode = "compare"
            else:
                st.session_state.last_answer = client.ask(
                    question, dense_only=dense_only, k=k, top_n=top_n
                )
                st.session_state.last_mode = "single"
            st.session_state.last_question = question
        except httpx.HTTPError as exc:
            st.error(f"Request failed: {exc}")


if st.session_state.get("last_mode") == "compare":
    st.divider()
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("## Hybrid (dense + sparse + RRF)")
        _render_answer_block(
            st.session_state.last_hybrid, anchor_prefix="hybrid-chunk-"
        )
    with col_right:
        st.markdown("## Dense-only")
        _render_answer_block(
            st.session_state.last_dense, anchor_prefix="dense-chunk-"
        )
elif st.session_state.get("last_mode") == "single":
    st.divider()
    _render_answer_block(st.session_state.last_answer)
