from kuberag.generation.prompts import SYSTEM_PROMPT, build_prompt
from kuberag.retrieval.fusion import FusedHit


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    source: str = "/test/doc.md",
    section: str | None = None,
    rank: int = 0,
) -> FusedHit:
    return FusedHit(
        chunk_id=chunk_id,
        text=text,
        source=source,
        section=section,
        chunking_strategy="fixed",
        rrf_score=0.5,
        rank=rank,
        dense_rank=rank,
        sparse_rank=rank,
    )


def test_returns_system_and_user_messages() -> None:
    msgs = build_prompt("what?", [make_chunk("a", "hello")])
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_system_prompt_demands_context_only_answers() -> None:
    lower = SYSTEM_PROMPT.lower()
    assert "only" in lower
    assert "context" in lower or "passage" in lower


def test_system_prompt_instructs_citation_format() -> None:
    assert "[" in SYSTEM_PROMPT and "]" in SYSTEM_PROMPT
    assert "cite" in SYSTEM_PROMPT.lower()


def test_system_prompt_instructs_refusal_on_insufficient() -> None:
    lower = SYSTEM_PROMPT.lower()
    assert any(
        phrase in lower
        for phrase in ("do not guess", "say so", "not contain", "insufficient")
    )


def test_chunks_numbered_starting_at_one() -> None:
    chunks = [make_chunk("a", "first"), make_chunk("b", "second")]
    msgs = build_prompt("q", chunks)
    user = msgs[1]["content"]
    assert "[1]" in user
    assert "[2]" in user
    assert user.index("[1]") < user.index("[2]")


def test_chunks_preserved_in_order() -> None:
    chunks = [
        make_chunk("a", "ALPHA"),
        make_chunk("b", "BRAVO"),
        make_chunk("c", "CHARLIE"),
    ]
    user = build_prompt("q", chunks)[1]["content"]
    assert user.index("ALPHA") < user.index("BRAVO") < user.index("CHARLIE")


def test_source_appears_in_chunk_block() -> None:
    chunks = [make_chunk("a", "hello", source="/docs/intro.md")]
    user = build_prompt("q", chunks)[1]["content"]
    assert "/docs/intro.md" in user


def test_section_appears_when_present() -> None:
    chunks = [make_chunk("a", "hello", section="Pods")]
    user = build_prompt("q", chunks)[1]["content"]
    assert "Pods" in user


def test_section_omitted_when_none() -> None:
    chunks = [make_chunk("a", "hello", section=None)]
    user = build_prompt("q", chunks)[1]["content"]
    assert "section=None" not in user


def test_question_appears_in_user_message() -> None:
    chunks = [make_chunk("a", "hello")]
    user = build_prompt("how do pods work?", chunks)[1]["content"]
    assert "how do pods work?" in user


def test_chunk_text_appears_in_prompt() -> None:
    chunks = [make_chunk("a", "kubelet manages node lifecycle")]
    user = build_prompt("q", chunks)[1]["content"]
    assert "kubelet manages node lifecycle" in user


def test_empty_chunks_still_produces_valid_messages() -> None:
    msgs = build_prompt("q", [])
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "q" in msgs[1]["content"]


def test_question_separated_from_context() -> None:
    chunks = [make_chunk("a", "hello world")]
    user = build_prompt("question text", chunks)[1]["content"]
    # Question should come after the context block
    assert user.index("hello world") < user.index("question text")
