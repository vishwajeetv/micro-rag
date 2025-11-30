"""
Embedding service for RAG.

This module generates embeddings using OpenAI's API.
Embeddings are numerical vectors that capture semantic meaning of text.
"""

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Generates embeddings using OpenAI's text-embedding-3-small model.

    Why async?
    - Non-blocking I/O for API calls
    - Works well with FastAPI's async nature
    - Can batch multiple requests efficiently
    """

    def __init__(self):
        """Initialize the OpenAI client."""
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model
        self.dimension = settings.openai_embedding_dimension

        logger.info(
            "embedding_service_initialized",
            model=self.model,
            dimension=self.dimension,
        )

    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats (embedding vector)
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimension,
        )

        embedding = response.data[0].embedding

        logger.debug(
            "text_embedded",
            text_length=len(text),
            embedding_dimension=len(embedding),
        )

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in a single API call.

        OpenAI supports batching up to 2048 texts per request.
        This is more efficient than individual calls.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (same order as input)
        """
        if not texts:
            return []

        # Filter empty texts but track indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)

        if not valid_texts:
            return [[] for _ in texts]

        response = await self.client.embeddings.create(
            model=self.model,
            input=valid_texts,
            dimensions=self.dimension,
        )

        # Map embeddings back to original indices
        embeddings = [[] for _ in texts]
        for i, data in enumerate(response.data):
            original_index = valid_indices[i]
            embeddings[original_index] = data.embedding

        logger.info(
            "batch_embedded",
            total_texts=len(texts),
            valid_texts=len(valid_texts),
            tokens_used=response.usage.total_tokens,
        )

        return embeddings
