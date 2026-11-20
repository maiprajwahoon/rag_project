"""Tests for markdown chunker."""

import re

from rag_project.ingestion.chunker import chunk_documents, count_tokens
from rag_project.ingestion.models import Document


def test_every_chunk_is_within_max_tokens() -> None:
    """Every generated chunk must have token_count <= max_tokens."""
    long_paragraph = "FastAPI is a modern, fast, high-performance web framework for building APIs with Python. " * 30
    text = (
        "# Overview\n\n"
        f"{long_paragraph}\n\n"
        "## Deep Dive\n\n"
        f"{long_paragraph}\n\n"
        "### Details\n\n"
        f"{long_paragraph}\n"
    )

    doc = Document(source_path="docs/overview.md", title="Overview", text=text)
    max_tokens = 60
    chunks = chunk_documents([doc], max_tokens=max_tokens, overlap_tokens=15)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= max_tokens
        assert count_tokens(chunk.text) == chunk.token_count


def test_running_chunker_twice_gives_identical_ids() -> None:
    """Running chunk_documents twice on the same input produces identical IDs and chunks."""
    doc1 = Document(
        source_path="tutorial/first_steps.md",
        title="First Steps",
        text="# First Steps\n\nParagraph 1.\n\nParagraph 2.\n\n## Next\n\nParagraph 3.",
    )
    doc2 = Document(
        source_path="tutorial/query_params.md",
        title="Query Parameters",
        text="# Query Parameters\n\nInfo on params.",
    )

    docs = [doc1, doc2]
    run1 = chunk_documents(docs, max_tokens=50, overlap_tokens=10)
    run2 = chunk_documents(docs, max_tokens=50, overlap_tokens=10)

    assert len(run1) == len(run2)
    assert [c.chunk_id for c in run1] == [c.chunk_id for c in run2]
    assert [c.model_dump() for c in run1] == [c.model_dump() for c in run2]

    # Verify ID formatting: source_path::0000 with forward slashes
    assert run1[0].chunk_id == "tutorial/first_steps.md::0000"
    assert run1[0].chunk_index == 0


def test_heading_paths_are_preserved() -> None:
    """Heading hierarchy breadcrumbs must be properly preserved as 'H1 > H2 > H3'."""
    text = (
        "# Tutorial\n\n"
        "## Query Parameters\n\n"
        "Here are query parameters.\n\n"
        "### Optional Parameters\n\n"
        "Optional parameters use None or default values.\n\n"
        "## Path Parameters\n\n"
        "Path parameters are part of the path.\n"
    )

    doc = Document(source_path="tutorial.md", title="Tutorial", text=text)
    chunks = chunk_documents([doc], max_tokens=200, overlap_tokens=20)

    paths = [c.heading_path for c in chunks]
    assert "Tutorial > Query Parameters" in paths
    assert "Tutorial > Query Parameters > Optional Parameters" in paths
    assert "Tutorial > Path Parameters" in paths


def test_code_block_not_split_mid_block() -> None:
    """A fenced code block must not be split mid-block when the section is split by paragraph."""
    code = "```python\ndef get_items(skip: int = 0, limit: int = 10):\n    return items[skip : skip + limit]\n```"
    p1 = "Paragraph before code that contains some introductory text for the reader. " * 3
    p2 = "Paragraph after code that contains explanatory remarks and following details. " * 3
    text = f"# Section\n\n{p1}\n\n{code}\n\n{p2}"

    doc = Document(source_path="code.md", title="Section", text=text)

    # Pick max_tokens such that section must split, but the code block alone fits comfortably
    code_tokens = count_tokens(code)
    max_tokens = code_tokens + 25

    chunks = chunk_documents([doc], max_tokens=max_tokens, overlap_tokens=10)

    # Assert that the code block is never split across chunks:
    # Any chunk containing ```python must also contain the closing ``` and the entire body
    found_code_chunk = False
    for chunk in chunks:
        if "def get_items" in chunk.text:
            assert code in chunk.text
            found_code_chunk = True

    assert found_code_chunk, "The code block should be present intact in at least one chunk."


def test_oversized_code_block_is_split() -> None:
    """When a code block alone exceeds max_tokens, it must be split so every chunk <= max_tokens."""
    huge_code = "```python\n" + "\n".join(f"    line_{i} = {i} * 2" for i in range(100)) + "\n```"
    doc = Document(source_path="huge.md", title="Huge Code", text=f"# Huge\n\n{huge_code}")

    max_tokens = 50
    chunks = chunk_documents([doc], max_tokens=max_tokens, overlap_tokens=10)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= max_tokens


def test_overlap_only_between_consecutive_chunks_of_same_section() -> None:
    """Overlap must not carry over from the end of one section into the start of the next."""
    p_long = "This is a detailed paragraph explaining section content. " * 6

    text = (
        "# Section One\n\n"
        f"{p_long}\n\n"
        f"{p_long}\n\n"
        "## Section Two\n\n"
        "Short start of section two.\n\n"
        f"{p_long}\n"
    )

    doc = Document(source_path="sections.md", title="Sections", text=text)
    chunks = chunk_documents([doc], max_tokens=80, overlap_tokens=25)

    sec1_chunks = [c for c in chunks if c.heading_path == "Section One"]
    sec2_chunks = [c for c in chunks if c.heading_path == "Section One > Section Two"]

    assert len(sec1_chunks) >= 2
    assert len(sec2_chunks) >= 1

    # First chunk of Section Two must not contain any content from Section One
    sec2_first_chunk = sec2_chunks[0]
    assert "Section One" not in sec2_first_chunk.text
    assert sec2_first_chunk.text.startswith("Short start of section two.")


def test_no_chunk_below_min_tokens_unless_only_chunk_in_section() -> None:
    """No chunk should be below min_tokens unless it is the only chunk in its section."""
    p_med = "FastAPI builds on Starlette and Pydantic for high performance. " * 4
    p_tiny = "Short remark."

    text = (
        "# Section Alpha\n\n"
        f"{p_med}\n\n"
        f"{p_tiny}\n\n"
        f"{p_med}\n\n"
        "## Section Beta\n\n"
        f"{p_tiny}\n\n"
        "## Section Gamma\n"
    )

    doc = Document(source_path="min_tokens.md", title="Min Tokens Test", text=text)
    min_tokens = 30
    max_tokens = 100
    chunks = chunk_documents([doc], max_tokens=max_tokens, overlap_tokens=10, min_tokens=min_tokens)

    # Section Gamma is a lone heading with no body; it must be dropped
    assert not any("Section Gamma" in c.heading_path for c in chunks)

    # Group by heading path to evaluate multi-chunk vs single-chunk sections
    section_chunks: dict[str, list] = {}
    for c in chunks:
        section_chunks.setdefault(c.heading_path, []).append(c)

    for heading_path, chs in section_chunks.items():
        if len(chs) > 1:
            for c in chs:
                assert (
                    c.token_count >= min_tokens
                ), f"Chunk {c.chunk_id} in multi-chunk section {heading_path} has {c.token_count} < {min_tokens}"
        else:
            assert len(chs) == 1


def test_all_chunks_are_still_at_most_max_tokens() -> None:
    """After merging small chunks, all chunks must still be <= max_tokens."""
    p_long = "Python type hints enable automatic validation, serialization, and documentation. " * 4
    p_small = "Tiny remark."

    text = (
        "# Main Section\n\n"
        f"{p_long}\n\n"
        f"{p_small}\n\n"
        f"{p_long}\n\n"
        f"{p_small}\n"
    )

    doc = Document(source_path="merge_max.md", title="Merge Max", text=text)
    max_tokens = 100
    min_tokens = 30
    chunks = chunk_documents([doc], max_tokens=max_tokens, overlap_tokens=10, min_tokens=min_tokens)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= max_tokens
        assert count_tokens(chunk.text) <= max_tokens


def test_heading_anchor_stripped_and_stored_in_chunk() -> None:
    """Trailing { #anchor } must be stripped from heading_path and stored in Chunk.anchor."""
    text = (
        "# FastAPI { #fastapi }\n\n"
        "FastAPI framework overview.\n\n"
        "## Opinions { #opinions }\n\n"
        "Here are user opinions.\n\n"
        "## Unanchored Section\n\n"
        "No anchor on this heading.\n"
    )

    doc = Document(source_path="fastapi.md", title="FastAPI", text=text)
    chunks = chunk_documents([doc], max_tokens=100, overlap_tokens=10, min_tokens=10)

    chunk_map = {c.heading_path: c for c in chunks}

    # Verify { #anchor } is completely absent from all heading paths
    for path in chunk_map:
        assert "{ #" not in path
        assert "}" not in path

    assert "FastAPI" in chunk_map
    assert chunk_map["FastAPI"].anchor == "fastapi"

    assert "FastAPI > Opinions" in chunk_map
    assert chunk_map["FastAPI > Opinions"].anchor == "opinions"

    assert "FastAPI > Unanchored Section" in chunk_map
    assert chunk_map["FastAPI > Unanchored Section"].anchor is None


def test_drop_html_comment_and_directive_only_chunks() -> None:
    """Chunks that consist only of HTML comments or only ::: directives must be dropped."""
    text = (
        "# Reference Guide\n\n"
        "Here is real documentation content that should be preserved in a chunk.\n\n"
        "## Comment Section\n\n"
        "<!-- agent-banner -->\n\n"
        "## Another Comment\n\n"
        "<!-- sponsors -->\n"
        "<!-- /sponsors -->\n\n"
        "## API Router\n\n"
        "::: fastapi.APIRouter\n"
        "::: fastapi.routing.APIRoute\n\n"
        "## Working Section\n\n"
        "::: note\n"
        "This note has accompanying body text so it is not only directive lines.\n"
        ":::\n"
    )

    doc = Document(source_path="ref.md", title="Reference Guide", text=text)
    chunks = chunk_documents([doc], max_tokens=150, overlap_tokens=10, min_tokens=10)

    # Verify no chunk contains solely comments or solely ::: lines
    for c in chunks:
        assert re.sub(r"<!--[\s\S]*?-->", "", c.text).strip() != ""
        lines = [line.strip() for line in c.text.strip().splitlines() if line.strip()]
        assert not (bool(lines) and all(line.startswith(":::") for line in lines))

    # Only Reference Guide and Working Section remain; Comment Section, Another Comment, and API Router were dropped
    paths = [c.heading_path for c in chunks]
    assert paths == ["Reference Guide", "Reference Guide > Working Section"]
    assert len(chunks) == 2

    # Verify IDs are deterministic and chunk_index is sequentially renumbered (0, 1, ...)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
        assert chunk.chunk_id == f"ref.md::{i:04d}"
