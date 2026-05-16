import re

import streamlit as st

from kuberag.generation.citations import VerifiedCitation
from kuberag.generation.orchestrator import GroundedAnswer, InsufficientAnswer
from kuberag.retrieval.fusion import FusedHit

_MARKER_RE = re.compile(r"\[(\d+)\]")

CITATION_CSS = """
<style>
.kr-citation {
    display: inline-block;
    padding: 0 6px;
    margin: 0 2px;
    border-radius: 4px;
    background: #e7f0ff;
    color: #1f4ea8;
    text-decoration: none;
    font-weight: 600;
    font-size: 0.9em;
}
.kr-citation:hover {
    background: #c9deff;
}
.kr-chunk-anchor {
    scroll-margin-top: 80px;
}
.kr-chunk-header {
    font-size: 0.85em;
    color: #555;
    margin-bottom: 4px;
}
.kr-citation-supported {
    color: #1a7a3a;
}
.kr-citation-unsupported {
    color: #b00020;
}
</style>
"""


def linkify_citations(text: str, *, anchor_prefix: str = "chunk-") -> str:
    return _MARKER_RE.sub(
        lambda m: (
            f'<a href="#{anchor_prefix}{m.group(1)}" '
            f'class="kr-citation">[{m.group(1)}]</a>'
        ),
        text,
    )


def format_chunk_provenance(chunk: FusedHit) -> str:
    parts = [f"rank {chunk.rank}", f"rrf {chunk.rrf_score:.4f}"]
    if chunk.dense_rank is not None:
        parts.append(f"dense #{chunk.dense_rank}")
    if chunk.sparse_rank is not None:
        parts.append(f"sparse #{chunk.sparse_rank}")
    return " · ".join(parts)


def render_styles() -> None:
    st.markdown(CITATION_CSS, unsafe_allow_html=True)


def render_grounded_answer(
    answer: GroundedAnswer, *, anchor_prefix: str = "chunk-"
) -> None:
    st.subheader("Answer")
    st.markdown(
        linkify_citations(answer.text, anchor_prefix=anchor_prefix),
        unsafe_allow_html=True,
    )


def render_insufficient_answer(answer: InsufficientAnswer) -> None:
    st.warning(f"Insufficient context: {answer.reason}")
    if answer.suggested_documents:
        st.markdown("**Suggested documents to check:**")
        for doc in answer.suggested_documents:
            st.markdown(f"- `{doc}`")
    if answer.generated_text:
        with st.expander("What the model wrote (rejected by confidence check)"):
            st.markdown(answer.generated_text)


def render_chunks_panel(
    chunks: list[FusedHit],
    citations: list[VerifiedCitation] | None = None,
    *,
    anchor_prefix: str = "chunk-",
) -> None:
    if not chunks:
        st.caption("No chunks retrieved.")
        return

    st.subheader("Retrieved chunks")
    citations_by_marker = (
        {c.marker: c for c in (citations or [])} if citations else {}
    )

    for index, chunk in enumerate(chunks):
        marker = index + 1
        cite = citations_by_marker.get(marker)
        st.markdown(
            f'<div id="{anchor_prefix}{marker}" class="kr-chunk-anchor"></div>',
            unsafe_allow_html=True,
        )
        header = f"**[{marker}]** `{chunk.source}`"
        if chunk.section:
            header += f" — *{chunk.section}*"
        if cite is not None:
            cls = "kr-citation-supported" if cite.supported else "kr-citation-unsupported"
            flag = "✓ supported" if cite.supported else "✗ unsupported"
            header += f' <span class="{cls}">{flag}</span>'
        st.markdown(header, unsafe_allow_html=True)
        st.caption(format_chunk_provenance(chunk))
        st.markdown(f"> {chunk.text}")
