"""
Text chunking for RAG.

This module splits documents into smaller chunks for embedding and retrieval.
Uses tiktoken for accurate token counting (matches OpenAI's tokenization).
"""

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
