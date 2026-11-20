"""Tests for document loader."""

from pathlib import Path

from rag_project.ingestion.loader import load_documents


def test_front_matter_is_removed(tmp_path: Path) -> None:
    """YAML front matter should be stripped from document text."""
    doc_path = tmp_path / "article.md"
    doc_path.write_text(
        "---\ntitle: Metadata Title\nauthor: Test Author\ntags: [a, b]\n---\n\n# Real Heading\n\nBody content.",
        encoding="utf-8",
    )

    docs = load_documents(tmp_path)
    assert len(docs) == 1
    doc = docs[0]
    assert "Metadata Title" not in doc.text
    assert "author: Test Author" not in doc.text
    assert "tags: [a, b]" not in doc.text
    assert "---" not in doc.text
    assert doc.title == "Real Heading"
    assert "Body content." in doc.text


def test_empty_files_are_skipped(tmp_path: Path) -> None:
    """Empty files, whitespace-only files, and files with only front matter must be skipped."""
    # Completely empty file
    (tmp_path / "empty.md").write_text("", encoding="utf-8")

    # Whitespace-only file
    (tmp_path / "whitespace.md").write_text("   \n\n\t  \n", encoding="utf-8")

    # Only front matter, no body
    (tmp_path / "only_front_matter.md").write_text(
        "---\ntitle: Ghost\n---\n\n   \n",
        encoding="utf-8",
    )

    # Valid file
    (tmp_path / "valid.md").write_text("# Valid Doc\n\nHello world.", encoding="utf-8")

    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0].source_path == "valid.md"
    assert docs[0].title == "Valid Doc"


def test_title_from_first_h1_or_filename(tmp_path: Path) -> None:
    """Document title must come from the first # heading, or filename if none exists."""
    sub_dir = tmp_path / "guides"
    sub_dir.mkdir()

    # Document with H1 heading
    (sub_dir / "with_h1.md").write_text(
        "# Advanced Guide\n\nSome paragraph.\n\n## Subheading\nMore text.",
        encoding="utf-8",
    )

    # Document without H1 heading
    (sub_dir / "no_h1.md").write_text(
        "## Only Subheading\n\nParagraph text.",
        encoding="utf-8",
    )

    docs = load_documents(tmp_path)
    doc_map = {d.source_path: d for d in docs}

    assert doc_map["guides/with_h1.md"].title == "Advanced Guide"
    assert doc_map["guides/no_h1.md"].title == "no_h1.md"


def test_deterministic_sort_by_path(tmp_path: Path) -> None:
    """Loaded documents must be sorted deterministically by their relative source path."""
    (tmp_path / "c.md").write_text("# C\nContent C", encoding="utf-8")
    (tmp_path / "a.md").write_text("# A\nContent A", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\nContent B", encoding="utf-8")

    docs = load_documents(tmp_path)
    paths = [d.source_path for d in docs]
    assert paths == ["a.md", "b.md", "c.md"]


def test_excluded_files_are_skipped(tmp_path: Path) -> None:
    """Files matching relative paths in exclude should be skipped."""
    (tmp_path / "included.md").write_text("# Included\n\nContent here.", encoding="utf-8")
    (tmp_path / "release-notes.md").write_text("# Release Notes\n\nNotes here.", encoding="utf-8")
    (tmp_path / "skip_me.md").write_text("# Skip Me\n\nDo not include.", encoding="utf-8")

    # Default exclude skips release-notes.md
    docs_default = load_documents(tmp_path)
    paths_default = [d.source_path for d in docs_default]
    assert "release-notes.md" not in paths_default
    assert "included.md" in paths_default
    assert "skip_me.md" in paths_default

    # Custom exclude
    docs_custom = load_documents(tmp_path, exclude={"skip_me.md"})
    paths_custom = [d.source_path for d in docs_custom]
    assert "skip_me.md" not in paths_custom
    assert "release-notes.md" in paths_custom
    assert "included.md" in paths_custom
