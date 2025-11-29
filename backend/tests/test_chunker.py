"""Tests for the TextChunker service."""

import pytest
from app.services.chunker import TextChunker


@pytest.fixture
def chunker():
    """Chunker with small sizes for testing."""
    return TextChunker(chunk_size=100, chunk_overlap=20)


def test_count_tokens(chunker):
    """Should count tokens correctly."""
    assert chunker.count_tokens("") == 0
    assert chunker.count_tokens("Hello world") > 0


def test_split_text_small(chunker):
    """Small text returns single chunk."""
    chunks = chunker.split_text("Short text.")
    assert len(chunks) == 1


def test_split_text_large(chunker):
    """Large text returns multiple chunks."""
    large = " ".join(["word"] * 500)
    chunks = chunker.split_text(large)
    assert len(chunks) > 1


def test_split_by_headers(chunker):
    """Should split by ## headers."""
    text = """Intro.

## Section One

Content one.

## Section Two

Content two.
"""
    sections = chunker.split_by_headers(text)
    assert len(sections) == 3
    assert sections[0][0] is None  # intro has no header
    assert sections[1][0] == "Section One"
    assert sections[2][0] == "Section Two"


def test_chunk_document_structure(chunker):
    """Chunks should have required fields."""
    text = "## Header\n\nSome content here."
    chunks = chunker.chunk_document(text)

    assert len(chunks) > 0
    chunk = chunks[0]
    assert "content" in chunk
    assert "chunk_index" in chunk
    assert "token_count" in chunk
    assert "char_count" in chunk
    assert "header" in chunk