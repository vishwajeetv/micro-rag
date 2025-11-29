"""
Text chunking for RAG.

This module splits documents into smaller chunks for embedding and retrieval.
Uses tiktoken for accurate token counting (matches OpenAI's tokenization).
"""

import re

import tiktoken

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class TextChunker:
    """
    Splits text into chunks suitable for embedding.

    Why tiktoken?
    - OpenAI's official tokenizer
    - Exact token counts (not estimates)
    - Matches what the embedding model sees
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        model: str | None = None,
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Max tokens per chunk (default from config: 800)
            chunk_overlap: Overlap between chunks (default from config: 200)
            model: Tokenizer model name (default from config)
        """
        self.chunk_size = chunk_size or settings.chunk_size_tokens
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap_tokens
        self.model = model or settings.chunk_tokenizer_model

        # Load the tokenizer
        # GPT-5.1 and modern models use o200k_base (200k vocabulary)
        # GPT-4 used cl100k_base (100k vocabulary)
        # o200k_base is better for multilingual and code
        self._encoding = tiktoken.get_encoding("o200k_base")

        logger.info(
            "chunker_initialized",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            model=self.model,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self._encoding.encode(text))

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs."""
        return self._encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Convert token IDs back to text."""
        return self._encoding.decode(tokens)

    def split_text(self, text: str) -> list[str]:
        """
        Split text into chunks with overlap.

        Uses a sliding window approach:
        1. Encode entire text to tokens
        2. Take chunk_size tokens at a time
        3. Slide by (chunk_size - overlap) tokens for next chunk

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        tokens = self.encode(text)
        total_tokens = len(tokens)

        # If text fits in one chunk, return as-is
        if total_tokens <= self.chunk_size:
            return [text.strip()]

        chunks = []
        start = 0
        step = self.chunk_size - self.chunk_overlap

        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self.decode(chunk_tokens).strip()

            if chunk_text:
                chunks.append(chunk_text)

            # Move window forward
            start += step

            # Avoid tiny trailing chunks
            if total_tokens - start < self.chunk_overlap:
                break

        logger.debug(
            "text_split_complete",
            total_tokens=total_tokens,
            num_chunks=len(chunks),
        )

        return chunks

    def split_by_headers(self, text: str) -> list[tuple[str | None, str]]:
        """
        Split text by markdown headers (## Header).

        Returns:
            List of (header, content) tuples.
            First tuple may have header=None for content before any header.
        """
        if not text or not text.strip():
            return []

        # Pattern matches ## Header (from scraper output)
        header_pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)

        sections = []
        last_end = 0
        current_header = None

        for match in header_pattern.finditer(text):
            # Content before this header
            content = text[last_end : match.start()].strip()
            if content:
                sections.append((current_header, content))

            current_header = match.group(1).strip()
            last_end = match.end()

        # Don't forget content after last header
        remaining = text[last_end:].strip()
        if remaining:
            sections.append((current_header, remaining))

        return sections

    def chunk_document(self, text: str, title: str | None = None, url: str | None = None) -> list[dict]:
        """
        Main method: chunk a document with header awareness.

        Process:
        1. Split by headers first
        2. Chunk each section separately
        3. Preserve header context in metadata

        Args:
            text: Document text to chunk
            title: Source document title (for metadata)
            url: Source document URL (for metadata)

        Returns:
            List of chunk dicts matching ChunkBase schema:
            - content: chunk text
            - chunk_index: position in document
            - token_count: tokens in this chunk
            - char_count: characters in this chunk
            - header: section header (if any)
        """
        if not text or not text.strip():
            return []

        sections = self.split_by_headers(text)
        chunks = []
        chunk_index = 0

        for header, content in sections:
            # Split this section into token-sized chunks
            section_chunks = self.split_text(content)

            for chunk_text in section_chunks:
                chunks.append({
                    "content": chunk_text,
                    "chunk_index": chunk_index,
                    "token_count": self.count_tokens(chunk_text),
                    "char_count": len(chunk_text),
                    "header": header,
                })
                chunk_index += 1

        logger.info(
            "document_chunked",
            title=title,
            url=url,
            total_chunks=len(chunks),
            sections=len(sections),
        )

        return chunks
