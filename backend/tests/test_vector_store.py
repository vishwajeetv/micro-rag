"""Tests for the VectorStore service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestVectorStore:
    """Tests for VectorStore class."""

    @pytest.fixture
    def mock_db(self):
        """Mock async database session."""
        db = AsyncMock()
        db.execute = AsyncMock()
        db.add = MagicMock()
        db.add_all = MagicMock()
        db.flush = AsyncMock()
        return db

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock EmbeddingService."""
        service = AsyncMock()
        service.embed_text = AsyncMock(return_value=[0.1] * 1536)
        service.embed_batch = AsyncMock(return_value=[[0.1] * 1536, [0.2] * 1536])
        return service

    @pytest.mark.asyncio
    async def test_get_stats_returns_counts(self, mock_db):
        """get_stats should return document and chunk counts."""
        # Mock count queries
        mock_db.execute = AsyncMock(
            side_effect=[
                MagicMock(scalar=MagicMock(return_value=10)),  # doc count
                MagicMock(scalar=MagicMock(return_value=50)),  # chunk count
                MagicMock(scalar=MagicMock(return_value=50)),  # embedded count
            ]
        )

        with patch("app.services.vector_store.EmbeddingService"):
            with patch("app.services.vector_store.TextChunker"):
                from app.services.vector_store import VectorStore

                store = VectorStore(mock_db)
                stats = await store.get_stats()

                assert "documents" in stats
                assert "chunks" in stats
                assert "chunks_with_embeddings" in stats

    @pytest.mark.asyncio
    async def test_ingest_document_creates_chunks(self, mock_db, mock_embedding_service):
        """ingest_document should create document and chunks."""
        # Mock no existing document
        mock_result = MagicMock()
        mock_result.scalar_one_or_none = MagicMock(return_value=None)
        mock_db.execute = AsyncMock(return_value=mock_result)

        with patch("app.services.vector_store.EmbeddingService", return_value=mock_embedding_service):
            with patch("app.services.vector_store.TextChunker") as mock_chunker:
                # Mock chunker output
                mock_chunker.return_value.chunk_document = MagicMock(
                    return_value=[
                        {"content": "chunk 1", "chunk_index": 0, "token_count": 10, "char_count": 50, "header": None},
                        {"content": "chunk 2", "chunk_index": 1, "token_count": 10, "char_count": 50, "header": "Section"},
                    ]
                )

                from app.services.vector_store import VectorStore

                store = VectorStore(mock_db)
                doc, chunks = await store.ingest_document(
                    collection_id=1,
                    url="https://example.com",
                    title="Test",
                    content="Test content",
                    content_hash="abc123",
                    word_count=100,
                )

                # Should have added document and chunks
                assert mock_db.add.called
                assert mock_db.add_all.called
                assert mock_db.flush.called