"""Markdown document chunker with heading hierarchy tracking and token limits."""

import re
from collections.abc import Sequence
from functools import lru_cache

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from rag_project.ingestion.models import Chunk, Document

_HEADING_REGEX = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_ANCHOR_REGEX = re.compile(r"\s*\{\s*#([^\s}]+)\s*\}\s*$")


def _strip_anchor(heading_raw: str) -> tuple[str, str | None]:
    """Strip trailing { #anchor } from a heading and return (clean_heading, anchor)."""
    match = _ANCHOR_REGEX.search(heading_raw)
    if match:
        anchor = match.group(1)
        clean = heading_raw[: match.start()].strip()
        return clean, anchor
    return heading_raw.strip(), None


def _is_html_comment_only(text: str) -> bool:
    """Check if text after stripping consists exclusively of HTML comment(s)."""
    stripped = text.strip()
    if not stripped:
        return False
    without_comments = re.sub(r"<!--[\s\S]*?-->", "", stripped).strip()
    return without_comments == ""


def _is_directive_lines_only(text: str) -> bool:
    """Check if text after stripping consists exclusively of ::: directive lines."""
    stripped = text.strip()
    if not stripped:
        return False
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    return bool(lines) and all(line.startswith(":::") for line in lines)


def _is_dropped_chunk(text: str) -> bool:
    """Check if chunk text after stripping is only an HTML comment or only ::: directive lines."""
    return _is_html_comment_only(text) or _is_directive_lines_only(text)


@lru_cache(maxsize=1)
def get_tokenizer() -> PreTrainedTokenizerBase:
    """Load and return the BAAI/bge-small-en-v1.5 tokenizer (cached)."""
    return AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")


def count_tokens(text: str) -> int:
    """Count tokens in text using the BAAI/bge-small-en-v1.5 tokenizer."""
    if not text:
        return 0
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def _split_document_by_headings(text: str) -> list[tuple[str, str | None, str]]:
    """Split markdown text into sections by heading hierarchy.

    Fenced code blocks are tracked so `#` lines within code are not mistaken for headings.

    Returns:
        A list of tuples: (heading_path, anchor, section_text).
    """
    lines = text.splitlines()
    sections: list[tuple[str, str | None, str]] = []
    heading_stack: list[tuple[int, str, str | None]] = []
    current_lines: list[str] = []
    current_heading_path = ""
    current_anchor: str | None = None
    in_code = False
    current_fence = ""

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
            if not in_code:
                in_code = True
                current_fence = fence
            elif fence == current_fence:
                in_code = False
                current_fence = ""
            current_lines.append(line)
            continue

        if not in_code:
            match = _HEADING_REGEX.match(line)
            if match:
                # Flush previous section content if non-empty
                section_content = "\n".join(current_lines).strip()
                if section_content:
                    sections.append(
                        (current_heading_path, current_anchor, section_content)
                    )
                current_lines = []

                # Update heading stack
                level = len(match.group(1))
                raw_heading_name = match.group(2).strip()
                clean_heading_name, anchor = _strip_anchor(raw_heading_name)
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, clean_heading_name, anchor))
                current_heading_path = " > ".join(h[1] for h in heading_stack)
                current_anchor = heading_stack[-1][2] if heading_stack else None
                continue

        current_lines.append(line)

    # Flush final section content
    section_content = "\n".join(current_lines).strip()
    if section_content:
        sections.append((current_heading_path, current_anchor, section_content))

    # Fallback if document has text but no non-heading content was captured
    if not sections and text.strip():
        fallback_path = " > ".join(h[1] for h in heading_stack) if heading_stack else ""
        fallback_anchor = heading_stack[-1][2] if heading_stack else None
        sections.append((fallback_path, fallback_anchor, text.strip()))

    return sections


def _split_section_into_blocks(text: str) -> list[str]:
    """Split a section's text into blocks (paragraphs and fenced code blocks).

    Fenced code blocks are kept intact as single blocks unless they alone exceed max_tokens.
    """
    lines = text.splitlines()
    blocks: list[str] = []
    current_lines: list[str] = []
    in_code = False
    current_fence = ""

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
            if not in_code:
                # Flush normal paragraph before entering code block
                if current_lines:
                    block_content = "\n".join(current_lines).strip()
                    if block_content:
                        blocks.append(block_content)
                    current_lines = []
                in_code = True
                current_fence = fence
                current_lines.append(line)
            else:
                current_lines.append(line)
                if fence == current_fence:
                    in_code = False
                    current_fence = ""
                    # Flush complete code block as a single unit
                    block_content = "\n".join(current_lines).strip()
                    if block_content:
                        blocks.append(block_content)
                    current_lines = []
            continue

        if in_code:
            current_lines.append(line)
        else:
            if stripped == "":
                # Blank line outside code fence separates paragraphs
                if current_lines:
                    block_content = "\n".join(current_lines).strip()
                    if block_content:
                        blocks.append(block_content)
                    current_lines = []
            else:
                current_lines.append(line)

    if current_lines:
        block_content = "\n".join(current_lines).strip()
        if block_content:
            blocks.append(block_content)

    return blocks


def _is_code_block(block: str) -> bool:
    """Return True if block is a complete fenced code block."""
    lines = block.strip().splitlines()
    if len(lines) < 2:
        return False
    first = lines[0].strip()
    last = lines[-1].strip()
    return (first.startswith(("```", "~~~"))) and (
        last == first[:3] or last == "```" or last == "~~~"
    )


def _parse_fence_header(block: str) -> tuple[str, str, str]:
    """Parse a fenced code block into (fence_open_line, inner_lines_text, fence_close_line).

    fence_open_line is e.g. '```python', inner_lines_text is everything between, fence_close_line is '```'.
    """
    lines = block.splitlines()
    fence_open = lines[0]
    fence_close = lines[-1]
    inner = "\n".join(lines[1:-1])
    return fence_open, inner, fence_close


def _is_safe_split_point(line: str) -> bool:
    """Return True if this line is a safe place to split a code block.

    Safe points are blank lines or lines that start at column 0 (def, class, @, etc.)
    without leading whitespace — i.e., top-level Python statements.
    """
    if line == "" or line.strip() == "":
        return True
    # starts at column 0 (no leading whitespace) — top-level statement
    return len(line) > 0 and not line[0].isspace()


def _split_oversized_code_block(
    block: str, max_tokens: int, min_tokens: int
) -> list[str]:
    """Split a fenced code block that exceeds max_tokens into balanced, self-contained pieces.

    Each piece is a complete fenced code block (open fence ... close fence).
    Splits happen only at blank lines or column-0 lines (def, class, @, etc.).
    Fence overhead tokens are counted toward max_tokens.
    Tail pieces below min_tokens are merged into the previous piece when they fit.
    """
    fence_open, inner, fence_close = _parse_fence_header(block)
    lang = fence_open.strip()[3:].strip()  # e.g. "python" from "```python"
    continuation_open = f"```{lang}" if lang else "```"

    # Token overhead per piece: open_fence\n...\nclose_fence — count the wrapper tokens
    wrapper_tokens = count_tokens(f"{fence_open}\n{fence_close}")
    content_budget = max_tokens - wrapper_tokens
    if content_budget <= 0:
        # pathological case: fence alone exceeds budget; return as-is
        return [block]

    inner_lines = inner.splitlines()
    pieces: list[str] = []
    current_inner_lines: list[str] = []

    def _build_piece(inner_lines_list: list[str]) -> str:
        inner_text = "\n".join(inner_lines_list)
        return f"{fence_open}\n{inner_text}\n{fence_close}"

    def _build_continuation_piece(inner_lines_list: list[str]) -> str:
        inner_text = "\n".join(inner_lines_list)
        return f"{continuation_open}\n{inner_text}\n{fence_close}"

    def _piece_fits(inner_lines_list: list[str], is_first: bool) -> bool:
        p = (
            _build_piece(inner_lines_list)
            if is_first
            else _build_continuation_piece(inner_lines_list)
        )
        return count_tokens(p) <= max_tokens

    is_first_piece = True

    for idx, line in enumerate(inner_lines):
        candidate_lines = current_inner_lines + [line]

        if _piece_fits(candidate_lines, is_first_piece):
            current_inner_lines.append(line)
        else:
            # Need to cut before this line. Find last safe split point.
            if current_inner_lines:
                # Find the latest safe split point within current_inner_lines
                split_at = len(current_inner_lines)  # default: cut before current line
                for back in range(len(current_inner_lines) - 1, 0, -1):
                    if _is_safe_split_point(current_inner_lines[back]):
                        split_at = back
                        break

                # If no safe split found, still cut at boundary (line boundary is guaranteed)
                piece_lines = current_inner_lines[:split_at]
                remaining_lines = current_inner_lines[split_at:]

                if piece_lines:
                    piece = (
                        _build_piece(piece_lines)
                        if is_first_piece
                        else _build_continuation_piece(piece_lines)
                    )
                    pieces.append(piece)
                    is_first_piece = False
                    current_inner_lines = remaining_lines + [line]
                else:
                    # Can't split further — just append line and let it overflow
                    current_inner_lines.append(line)
            else:
                current_inner_lines.append(line)

    if current_inner_lines:
        piece = (
            _build_piece(current_inner_lines)
            if is_first_piece
            else _build_continuation_piece(current_inner_lines)
        )
        pieces.append(piece)

    if not pieces:
        return [block]

    # Merge tail pieces below min_tokens into previous piece when they fit
    result: list[str] = []
    for piece in pieces:
        if (
            result
            and count_tokens(piece) < min_tokens
            and count_tokens(result[-1] + "\n\n" + piece) <= max_tokens
        ):
            result[-1] = result[-1] + "\n\n" + piece
        else:
            result.append(piece)

    return result


def _split_oversized_block(
    block: str, max_tokens: int, min_tokens: int = 0
) -> list[str]:
    """Split an individual block that alone exceeds max_tokens.

    For fenced code blocks, splits at safe points (blank lines, column-0 lines like def/class/@)
    so every sub-piece is a balanced, self-contained fenced code block.
    For plain text blocks, splits at line boundaries.
    """
    if _is_code_block(block):
        return _split_oversized_code_block(block, max_tokens, min_tokens)

    lines = block.splitlines()
    sub_blocks: list[str] = []
    current_lines: list[str] = []

    for line in lines:
        candidate = "\n".join(current_lines + [line]) if current_lines else line
        if count_tokens(candidate) <= max_tokens:
            current_lines.append(line)
        else:
            if current_lines:
                sub_blocks.append("\n".join(current_lines))
                current_lines = []

            if count_tokens(line) > max_tokens:
                words = line.split(" ")
                current_words: list[str] = []
                for word in words:
                    cand_word = (
                        " ".join(current_words + [word]) if current_words else word
                    )
                    if count_tokens(cand_word) <= max_tokens:
                        current_words.append(word)
                    else:
                        if current_words:
                            sub_blocks.append(" ".join(current_words))
                            current_words = []
                        if count_tokens(word) > max_tokens:
                            tokenizer = get_tokenizer()
                            ids = tokenizer.encode(word, add_special_tokens=False)
                            for start_idx in range(0, len(ids), max_tokens):
                                piece_ids = ids[start_idx : start_idx + max_tokens]
                                decoded = tokenizer.decode(piece_ids)
                                if isinstance(decoded, str):
                                    sub_blocks.append(decoded)
                                else:
                                    sub_blocks.extend(decoded)
                        else:
                            current_words.append(word)
                if current_words:
                    sub_blocks.append(" ".join(current_words))
            else:
                current_lines.append(line)

    if current_lines:
        sub_blocks.append("\n".join(current_lines))

    return [sb for sb in sub_blocks if sb.strip()]


def _text_starts_inside_fence(text: str) -> bool:
    """Return True if the given text starts inside a fenced code block.

    A text starts inside a fence if the first ``` or ~~~ line encountered when
    scanning from the top is a closing fence (i.e., the content began mid-block).
    We detect this by checking whether the text starts without an opening fence
    but has content that would only make sense inside a fence.

    Heuristic: scan lines; if we see a bare ``` or ~~~ (3 chars, no language tag)
    before we see one with a language or before any heading/paragraph, it's a closing fence.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            rest = stripped[3:]
            # A closing fence is exactly ``` or ~~~ with no extra content
            return rest == ""
        if stripped:  # non-empty, non-fence line
            return False
    return False


def _chunk_blocks(
    blocks: list[str], max_tokens: int, overlap_tokens: int, min_tokens: int = 0
) -> list[str]:
    """Group blocks into chunks respecting max_tokens and applying overlap.

    Overlap always starts at a line boundary.  If the overlap region would begin
    inside a fenced code block, the overlap is dropped for that chunk transition.
    """
    if not blocks:
        return []

    normalized_blocks: list[str] = []
    for block in blocks:
        if count_tokens(block) > max_tokens:
            normalized_blocks.extend(
                _split_oversized_block(block, max_tokens, min_tokens)
            )
        else:
            normalized_blocks.append(block)

    chunks: list[str] = []
    i = 0
    n = len(normalized_blocks)

    while i < n:
        current_blocks: list[str] = []
        j = i

        while j < n:
            cand_blocks = current_blocks + [normalized_blocks[j]]
            cand_text = "\n\n".join(cand_blocks)
            if count_tokens(cand_text) <= max_tokens:
                current_blocks.append(normalized_blocks[j])
                j += 1
            else:
                if not current_blocks:
                    current_blocks.append(normalized_blocks[j])
                    j += 1
                break

        chunks.append("\n\n".join(current_blocks))

        if j >= n:
            break

        next_i = j
        if overlap_tokens > 0:
            overlap_accum: list[str] = []
            for k in range(j - 1, i, -1):
                test_blocks = [normalized_blocks[k]] + overlap_accum
                test_text = "\n\n".join(test_blocks)
                if count_tokens(test_text) <= overlap_tokens:
                    overlap_accum = test_blocks
                    next_i = k
                else:
                    break

            # Validate overlap: must not start inside a fenced code block
            if overlap_accum:
                overlap_text = "\n\n".join(overlap_accum)
                if _text_starts_inside_fence(overlap_text):
                    overlap_accum = []
                    next_i = j

        if next_i <= i:
            next_i = i + 1
        i = next_i

    return chunks


def _is_lone_heading(text: str) -> bool:
    """Check if text consists exclusively of markdown heading line(s) with no body text."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return True
    return all(bool(_HEADING_REGEX.match(line)) for line in lines)


def _postprocess_section_chunks(
    chunks: list[str], max_tokens: int, min_tokens: int
) -> list[str]:
    """Merge chunks below min_tokens into adjacent chunks of the same section, or drop lone headings / comments / directives."""
    if not chunks:
        return []

    # Drop lone headings, pure HTML comments, and pure ::: directive lines
    filtered: list[str] = []
    for c in chunks:
        if _is_dropped_chunk(c):
            continue
        if count_tokens(c) < min_tokens and _is_lone_heading(c):
            continue
        filtered.append(c)

    if len(filtered) <= 1:
        return [c for c in filtered if not _is_dropped_chunk(c)]

    result = list(filtered)
    idx = 0
    while idx < len(result):
        if count_tokens(result[idx]) < min_tokens:
            if len(result) == 1:
                break

            # Try merging into previous chunk of the same section if it fits within max_tokens
            if (
                idx > 0
                and count_tokens(result[idx - 1] + "\n\n" + result[idx]) <= max_tokens
            ):
                result[idx - 1] = result[idx - 1] + "\n\n" + result[idx]
                result.pop(idx)
                idx = max(0, idx - 1)
                continue

            # Otherwise into next chunk if it fits within max_tokens
            if (
                idx < len(result) - 1
                and count_tokens(result[idx] + "\n\n" + result[idx + 1]) <= max_tokens
            ):
                result[idx + 1] = result[idx] + "\n\n" + result[idx + 1]
                result.pop(idx)
                continue

        idx += 1

    return [c for c in result if c.strip() and not _is_dropped_chunk(c)]


def chunk_documents(
    docs: Sequence[Document],
    max_tokens: int = 450,
    overlap_tokens: int = 60,
    min_tokens: int = 30,
) -> list[Chunk]:
    """Chunk markdown documents into retrieval-ready Chunk objects.

    Documents are first split by markdown heading hierarchy, tracking the heading path.
    Long sections are then split by paragraph. Fenced code blocks are kept intact unless
    the code block alone exceeds max_tokens. Every chunk is guaranteed to have at most
    max_tokens. Overlap applies only between consecutive chunks of the same section.
    Chunks below min_tokens are merged into adjacent chunks of the same section (or dropped
    if they are lone headings with no body). Trailing { #anchor } is stripped from headings
    and stored in Chunk.anchor. Pure HTML comments and pure ::: directive lines are dropped.
    Chunk IDs are deterministic and sequentially renumbered as f"{source_path}::{chunk_index:04d}".

    Args:
        docs: Sequence of Document models to chunk.
        max_tokens: Maximum allowed tokens per chunk (default: 450).
        overlap_tokens: Target overlap in tokens between chunks of the same section (default: 60).
        min_tokens: Minimum allowed tokens per chunk before merging (default: 30).

    Returns:
        List of Chunk models.
    """
    all_chunks: list[Chunk] = []

    for doc in docs:
        source_path = doc.source_path.replace("\\", "/")
        sections = _split_document_by_headings(doc.text)
        chunk_index = 0

        for heading_path, anchor, section_text in sections:
            sec_tokens = count_tokens(section_text)
            if sec_tokens <= max_tokens:
                raw_chunk_texts = [section_text]
            else:
                blocks = _split_section_into_blocks(section_text)
                raw_chunk_texts = _chunk_blocks(blocks, max_tokens, overlap_tokens)

            chunk_texts = _postprocess_section_chunks(
                raw_chunk_texts, max_tokens, min_tokens
            )

            for text in chunk_texts:
                token_count = count_tokens(text)
                chunk_id = f"{source_path}::{chunk_index:04d}"
                all_chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        source_path=source_path,
                        title=doc.title,
                        heading_path=heading_path,
                        chunk_index=chunk_index,
                        text=text,
                        token_count=token_count,
                        anchor=anchor,
                    )
                )
                chunk_index += 1

    return all_chunks
