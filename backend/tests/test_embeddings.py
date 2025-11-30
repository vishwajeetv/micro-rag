"""Tests for the EmbeddingService."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestEmbeddingService:
    """Tests for EmbeddingService class."""

    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI embeddings response."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_response.usage = MagicMock(total_tokens=10)
        return mock_response

    @pytest.fixture
    def mock_batch_response(self):
        """Mock OpenAI batch embeddings response."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
        ]
        mock_response.usage = MagicMock(total_tokens=20)
        return mock_response

    @pytest.mark.asyncio
    async def test_embed_text_returns_vector(self, mock_openai_response):
        """embed_text should return a list of floats."""
        with patch("app.services.embeddings.AsyncOpenAI") as mock_client:
            mock_client.return_value.embeddings.create = AsyncMock(
                return_value=mock_openai_response
            )

            from app.services.embeddings import EmbeddingService

            service = EmbeddingService()
            result = await service.embed_text("test text")

            assert isinstance(result, list)
            assert len(result) == 1536
            assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_embed_text_empty_raises_error(self):
        """embed_text should raise ValueError for empty text."""
        with patch("app.services.embeddings.AsyncOpenAI"):
            from app.services.embeddings import EmbeddingService

            service = EmbeddingService()

            with pytest.raises(ValueError, match="cannot be empty"):
                await service.embed_text("")

            with pytest.raises(ValueError, match="cannot be empty"):
                await service.embed_text("   ")

    @pytest.mark.asyncio
    async def test_embed_batch_returns_list(self, mock_batch_response):
        """embed_batch should return list of vectors."""
        with patch("app.services.embeddings.AsyncOpenAI") as mock_client:
            mock_client.return_value.embeddings.create = AsyncMock(
                return_value=mock_batch_response
            )

            from app.services.embeddings import EmbeddingService

            service = EmbeddingService()
            result = await service.embed_batch(["text one", "text two"])

            assert len(result) == 2
            assert len(result[0]) == 1536
            assert len(result[1]) == 1536

    @pytest.mark.asyncio
    async def test_embed_batch_empty_list(self):
        """embed_batch should return empty list for empty input."""
        with patch("app.services.embeddings.AsyncOpenAI"):
            from app.services.embeddings import EmbeddingService

            service = EmbeddingService()
            result = await service.embed_batch([])

            assert result == []