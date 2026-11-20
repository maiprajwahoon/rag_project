"""Document loader for recursively scanning and reading markdown files."""

import re
from pathlib import Path

from rag_project.ingestion.models import Document

_FRONT_MATTER_PATTERN = re.compile(
    r"^---\s*\r?\n.*?\r?\n(?:---|\.\.\.)\s*(?:\r?\n|$)",
    re.DOTALL,
)

_H1_PATTERN = re.compile(r"^#\s+(.+)$")


def _strip_front_matter(content: str) -> str:
    """Strip YAML front matter delimited by --- or ... at start of content."""
    if content.startswith("---"):
        return _FRONT_MATTER_PATTERN.sub("", content, count=1)
    return content


def _extract_title(text: str, filename: str) -> str:
    """Extract document title from the first # heading outside code blocks, or fallback to filename."""
    in_code = False
    current_fence = ""

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
            if not in_code:
                in_code = True
                current_fence = fence
            elif fence == current_fence:
                in_code = False
                current_fence = ""
            continue

        if not in_code:
            match = _H1_PATTERN.match(line)
            if match:
                return match.group(1).strip()

    return filename


def load_documents(
    root: Path,
    exclude: set[str] | None = None,
) -> list[Document]:
    """Recursively read all markdown files (*.md) under the specified root directory.

    YAML front matter is removed. The title is taken from the first level-1 heading (#),
    or the file's name if no level-1 heading exists. UTF-8 encoding is used.
    Empty files (or files with only whitespace / front matter) are skipped.
    Files matching relative paths in exclude are skipped.
    Results are sorted deterministically by relative source path.

    Args:
        root: Root directory to search recursively for markdown files.
        exclude: Relative paths of files to skip (defaults to {"release-notes.md"}).

    Returns:
        List of loaded Document models sorted by source_path.
    """
    if exclude is None:
        exclude = {"release-notes.md"}

    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        return []

    documents: list[Document] = []

    for file_path in root_path.rglob("*.md"):
        if not file_path.is_file():
            continue

        rel_path = file_path.relative_to(root_path).as_posix()
        if rel_path in exclude:
            continue

        raw_content = file_path.read_text(encoding="utf-8")
        if not raw_content.strip():
            continue

        stripped_content = _strip_front_matter(raw_content).strip()
        if not stripped_content:
            continue

        title = _extract_title(stripped_content, file_path.name)

        documents.append(
            Document(
                source_path=rel_path,
                title=title,
                text=stripped_content,
            )
        )

    documents.sort(key=lambda doc: doc.source_path)
    return documents
