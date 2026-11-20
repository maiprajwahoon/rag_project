"""Document loader for recursively scanning and reading markdown files."""

import re
from pathlib import Path

from rag_project.ingestion.models import Document

_FRONT_MATTER_PATTERN = re.compile(
    r"^---\s*\r?\n.*?\r?\n(?:---|\.\.\.)\s*(?:\r?\n|$)",
    re.DOTALL,
)

_H1_PATTERN = re.compile(r"^#\s+(.+)$")

_INCLUDE_LINE_PATTERN = re.compile(
    r"^\s*\{\*\s+.*?docs_src/(?P<relpath>[^\s*]+)(?:\s+[^*]+?)?\s*\*\}\s*$"
)

_EXT_LANG_MAP: dict[str, str] = {
    ".py": "python",
    ".sh": "bash",
    ".js": "javascript",
    ".ts": "typescript",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".html": "html",
    ".css": "css",
}


def _get_language(path: Path) -> str:
    """Determine the code fence language identifier from a file's extension."""
    ext = path.suffix.lower()
    return _EXT_LANG_MAP.get(ext, ext.lstrip("."))


def resolve_includes(text: str, docs_src_root: Path) -> str:
    """Resolve markdown snippet includes from docs_src_root into fenced code blocks.

    Matches lines of the form {* ../../docs_src/<path> <options> *}. If the referenced
    file exists under docs_src_root, the line is replaced by a fenced code block with
    the language derived from the file extension (.py -> python) and a leading comment
    line # File: <path under docs_src>. The hl[...] options are ignored.
    If the file does not exist, the original line is left unchanged.
    Lines that do not start with {* (after optional whitespace) are left untouched.

    Args:
        text: Input markdown text.
        docs_src_root: Root directory of docs_src.

    Returns:
        Markdown text with existing includes resolved.
    """
    resolved_lines: list[str] = []
    docs_root = Path(docs_src_root)

    for line in text.splitlines():
        stripped_left = line.lstrip()
        if not stripped_left.startswith("{*"):
            resolved_lines.append(line)
            continue

        match = _INCLUDE_LINE_PATTERN.match(line)
        if not match:
            resolved_lines.append(line)
            continue

        rel_path_str = match.group("relpath").replace("\\", "/")
        target_path = docs_root / rel_path_str
        if not target_path.is_file():
            resolved_lines.append(line)
            continue

        content = target_path.read_text(encoding="utf-8")
        lang = _get_language(target_path)
        body = content.rstrip("\r\n")

        if body:
            fenced = f"```{lang}\n# File: {rel_path_str}\n{body}\n```"
        else:
            fenced = f"```{lang}\n# File: {rel_path_str}\n```"

        resolved_lines.append(fenced)

    return "\n".join(resolved_lines)


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
    docs_src_root: Path | None = None,
) -> list[Document]:
    """Recursively read all markdown files (*.md) under the specified root directory.

    YAML front matter is removed. The title is taken from the first level-1 heading (#),
    or the file's name if no level-1 heading exists. UTF-8 encoding is used.
    Empty files (or files with only whitespace / front matter) are skipped.
    Files matching relative paths in exclude are skipped.
    If docs_src_root is provided, snippet includes ({* ... *}) are resolved.
    Results are sorted deterministically by relative source path.

    Args:
        root: Root directory to search recursively for markdown files.
        exclude: Relative paths of files to skip (defaults to {"release-notes.md", "_llm-test.md"}).
        docs_src_root: Optional root directory of docs_src to resolve snippet includes.

    Returns:
        List of loaded Document models sorted by source_path.
    """
    if exclude is None:
        exclude = {"release-notes.md", "_llm-test.md"}

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

        if docs_src_root is not None:
            stripped_content = resolve_includes(stripped_content, docs_src_root)

        documents.append(
            Document(
                source_path=rel_path,
                title=title,
                text=stripped_content,
            )
        )

    documents.sort(key=lambda doc: doc.source_path)
    return documents
