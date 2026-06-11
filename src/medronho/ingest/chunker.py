"""Heading-aware Markdown chunking.

A note is split into one chunk per heading section; sections longer than
``max_chars`` are further split on paragraph boundaries with a character
overlap so context survives the cut. YAML frontmatter is stripped (it is
metadata, not prose worth embedding).
"""

import re

from pydantic import BaseModel

DEFAULT_MAX_CHARS = 1500
DEFAULT_OVERLAP = 200

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^(```|~~~)")


class Chunk(BaseModel):
    """One embeddable piece of a note.

    Attributes:
        heading_path: Heading hierarchy above the chunk, outermost first.
        text: Chunk content, without headings or embedding prefixes.
        position: 0-based chunk order within the note.
    """

    heading_path: list[str]
    text: str
    position: int


def chunk_markdown(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split a Markdown document into heading-aware chunks.

    Args:
        text: Full Markdown source of one note.
        max_chars: Maximum chunk length in characters.
        overlap: Characters of overlap between consecutive pieces of an
            oversized section. Must be smaller than ``max_chars``.

    Returns:
        Chunks in document order; empty list for an effectively empty note.
    """
    if overlap >= max_chars:
        raise ValueError(f"overlap ({overlap}) must be smaller than max_chars ({max_chars})")
    chunks: list[Chunk] = []
    for heading_path, section_text in _split_sections(_strip_frontmatter(text)):
        for piece in _split_long_text(section_text, max_chars=max_chars, overlap=overlap):
            chunks.append(Chunk(heading_path=heading_path, text=piece, position=len(chunks)))
    return chunks


def _strip_frontmatter(text: str) -> str:
    """Remove a leading YAML frontmatter block, if present."""
    if not text.startswith("---"):
        return text
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines[1:], start=1):
        if line.rstrip("\r\n") in ("---", "..."):
            return "".join(lines[i + 1 :])
    return text


def _split_sections(text: str) -> list[tuple[list[str], str]]:
    """Split on ATX headings, tracking the heading hierarchy.

    Headings inside fenced code blocks are ignored. Returns
    ``(heading_path, section_text)`` pairs for non-empty sections.
    """
    sections: list[tuple[list[str], str]] = []
    stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    in_fence = False

    def close_section() -> None:
        section = "\n".join(current_lines).strip()
        if section:
            sections.append(([title for _, title in stack], section))
        current_lines.clear()

    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING_RE.match(line)
        if heading is None:
            current_lines.append(line)
            continue
        close_section()
        level, title = len(heading.group(1)), heading.group(2)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
    close_section()
    return sections


def _split_long_text(text: str, *, max_chars: int, overlap: int) -> list[str]:
    """Split text into pieces of at most ``max_chars``, preferring paragraph breaks.

    Consecutive pieces share up to ``overlap`` characters: either the tail of
    the previous piece (when it still fits) or a true sliding window when a
    single paragraph exceeds ``max_chars``.
    """
    if len(text) <= max_chars:
        return [text]

    # Normalize to units of at most max_chars: paragraphs as-is, oversized
    # paragraphs hard-split with a sliding window (the window itself overlaps).
    step = max_chars - overlap
    units: list[str] = []
    for paragraph in (p.strip() for p in text.split("\n\n")):
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            units.append(paragraph)
        else:
            units.extend(paragraph[i : i + max_chars] for i in range(0, len(paragraph), step))

    # Greedily pack units; when a piece closes, seed the next one with its
    # tail (up to `overlap` chars) so context survives the cut.
    pieces: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
        elif len(current) + 2 + len(unit) <= max_chars:
            current = f"{current}\n\n{unit}"
        else:
            pieces.append(current)
            seed = current[-overlap:] if overlap else ""
            if seed and len(seed) + 2 + len(unit) <= max_chars:
                current = f"{seed}\n\n{unit}"
            else:
                current = unit
    if current:
        pieces.append(current)
    return pieces
