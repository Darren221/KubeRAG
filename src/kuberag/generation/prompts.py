from kuberag.retrieval.fusion import FusedHit

SYSTEM_PROMPT = """You answer questions strictly from the numbered context passages provided.

Rules:
1. Use ONLY information from the numbered passages. Do not draw on outside knowledge.
2. Cite the passage(s) supporting each claim using bracketed markers like [1], [2]. \
If multiple passages support a claim, cite all of them: [1][3].
3. If the passages do not contain enough information to answer the question, say so \
explicitly. Do not guess.
4. Summarize the passages in your own words while still citing them. Quote sparingly.

Example: "Pods are the smallest deployable unit in Kubernetes [1]. They host one or \
more containers that share a network namespace [1][3]."
"""


def build_prompt(question: str, chunks: list[FusedHit]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _format_user_message(question, chunks)},
    ]


def _format_user_message(question: str, chunks: list[FusedHit]) -> str:
    context = _format_context(chunks)
    return f"Context:\n\n{context}\n\nQuestion: {question}"


def _format_context(chunks: list[FusedHit]) -> str:
    if not chunks:
        return "(no context passages were retrieved)"
    return "\n\n".join(_format_chunk(i + 1, c) for i, c in enumerate(chunks))


def _format_chunk(marker: int, chunk: FusedHit) -> str:
    header_parts = [f"[{marker}]", f"source={chunk.source}"]
    if chunk.section:
        header_parts.append(f"section={chunk.section}")
    header = " ".join(header_parts)
    return f"{header}\n{chunk.text}"
