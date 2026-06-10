"""Tests for heading-aware Markdown chunking."""

from secondbrain.ingest.chunker import chunk_markdown


def test_empty_file_yields_no_chunks() -> None:
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n  ") == []


def test_no_headings_single_chunk() -> None:
    chunks = chunk_markdown("Just a short note.\n\nWith two paragraphs.")
    assert len(chunks) == 1
    assert chunks[0].heading_path == []
    assert "two paragraphs" in chunks[0].text
    assert chunks[0].position == 0


def test_heading_hierarchy_tracked() -> None:
    text = (
        "# Project X\n\nIntro paragraph.\n\n"
        "## Decisions\n\nUse SQLite.\n\n"
        "### Rationale\n\nOne file is a feature.\n\n"
        "## Open questions\n\nNone yet.\n"
    )
    chunks = chunk_markdown(text)
    assert [c.heading_path for c in chunks] == [
        ["Project X"],
        ["Project X", "Decisions"],
        ["Project X", "Decisions", "Rationale"],
        ["Project X", "Open questions"],
    ]
    assert [c.position for c in chunks] == [0, 1, 2, 3]


def test_frontmatter_stripped() -> None:
    text = "---\nstatus: active\ntags: [a, b]\n---\n\n# Title\n\nBody text.\n"
    chunks = chunk_markdown(text)
    assert len(chunks) == 1
    assert "status: active" not in chunks[0].text
    assert chunks[0].heading_path == ["Title"]


def test_heading_inside_code_fence_ignored() -> None:
    text = "# Real\n\nBefore.\n\n```sh\n# not a heading\necho hi\n```\n\nAfter.\n"
    chunks = chunk_markdown(text)
    assert len(chunks) == 1
    assert chunks[0].heading_path == ["Real"]
    assert "# not a heading" in chunks[0].text


def test_oversized_section_split_with_overlap() -> None:
    paragraphs = [f"Paragraph number {i:02d} with some filler text." for i in range(20)]
    text = "# Big\n\n" + "\n\n".join(paragraphs)
    max_chars, overlap = 120, 30
    chunks = chunk_markdown(text, max_chars=max_chars, overlap=overlap)
    assert len(chunks) > 1
    assert all(len(c.text) <= max_chars for c in chunks)
    assert all(c.heading_path == ["Big"] for c in chunks)
    # Consecutive pieces share the previous piece's tail as overlap.
    assert chunks[1].text.startswith(chunks[0].text[-overlap:])


def test_giant_paragraph_sliding_window() -> None:
    text = "x" * 1000  # one paragraph, no breaks anywhere
    max_chars, overlap = 100, 20
    chunks = chunk_markdown(text, max_chars=max_chars, overlap=overlap)
    assert all(len(c.text) <= max_chars for c in chunks)
    assert len(chunks) == 13  # ceil(1000 / (100 - 20)) windows
    assert chunks[1].text[:overlap] == chunks[0].text[-overlap:]


def test_overlap_must_be_smaller_than_max_chars() -> None:
    import pytest

    with pytest.raises(ValueError, match="overlap"):
        chunk_markdown("text", max_chars=100, overlap=100)
