"""Tests for the RAGEngine."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestRAGEngine:
    """Tests for RAGEngine class."""

    @pytest.fixture
    def mock_chunks(self):
        """Mock retrieved chunks from vector store."""
        return [
            {
                "chunk_id": 1,
                "content": "Spain can be formed by Castile or Aragon.",
                "document_id": 1,
                "document_title": "Formation Decisions",
                "document_url": "https://eu5.wiki/Formation_Decisions",
                "chunk_index": 0,
                "score": 0.85,
                "header": "Forming Spain",
            },
            {
                "chunk_id": 2,
                "content": "You need admin tech 10 and own all Iberian provinces.",
                "document_id": 1,
                "document_title": "Formation Decisions",
                "document_url": "https://eu5.wiki/Formation_Decisions",
                "chunk_index": 1,
                "score": 0.78,
                "header": "Requirements",
            },
        ]

    @pytest.fixture
    def mock_openai_response(self):
        """Mock OpenAI chat completion response."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="To form Spain, play as Castile or Aragon and own all Iberian provinces with admin tech 10."))
        ]
        mock_response.usage = MagicMock(total_tokens=150)
        return mock_response

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_format_chunk_with_header(self, mock_db_session):
        """_format_chunk should include header when present."""
        with patch("app.services.rag_engine.AsyncOpenAI"):
            with patch("app.services.rag_engine.VectorStore"):
                from app.services.rag_engine import RAGEngine

                engine = RAGEngine(mock_db_session)
                chunk = {
                    "document_title": "Test Page",
                    "document_url": "https://example.com",
                    "header": "Section One",
                    "content": "Test content here.",
                }
                result = engine._format_chunk(chunk)

                assert "### Test Page" in result
                assert "Source: https://example.com" in result
                assert "Section: Section One" in result
                assert "Test content here." in result

    @pytest.mark.asyncio
    async def test_format_chunk_without_header(self, mock_db_session):
        """_format_chunk should work without header."""
        with patch("app.services.rag_engine.AsyncOpenAI"):
            with patch("app.services.rag_engine.VectorStore"):
                from app.services.rag_engine import RAGEngine

                engine = RAGEngine(mock_db_session)
                chunk = {
                    "document_title": "Test Page",
                    "document_url": "https://example.com",
                    "header": None,
                    "content": "Test content.",
                }
                result = engine._format_chunk(chunk)

                assert "### Test Page" in result
                assert "Section:" not in result

    @pytest.mark.asyncio
    async def test_build_messages_structure(self, mock_db_session, mock_chunks):
        """_build_messages should return proper message structure."""
        with patch("app.services.rag_engine.AsyncOpenAI"):
            with patch("app.services.rag_engine.VectorStore"):
                from app.services.rag_engine import RAGEngine

                engine = RAGEngine(mock_db_session)
                messages = engine._build_messages(
                    question="How do I form Spain?",
                    chunks=mock_chunks,
                    collection_name="EU5 Wiki",
                )

                assert len(messages) == 2
                assert messages[0]["role"] == "system"
                assert messages[1]["role"] == "user"
                assert "EU5 Wiki" in messages[0]["content"]
                assert "How do I form Spain?" in messages[1]["content"]

    @pytest.mark.asyncio
    async def test_build_sources(self, mock_db_session, mock_chunks):
        """_build_sources should format sources correctly."""
        with patch("app.services.rag_engine.AsyncOpenAI"):
            with patch("app.services.rag_engine.VectorStore"):
                from app.services.rag_engine import RAGEngine

                engine = RAGEngine(mock_db_session)
                sources = engine._build_sources(mock_chunks)

                assert len(sources) == 2
                assert sources[0]["document_title"] == "Formation Decisions"
                assert sources[0]["relevance_score"] == 0.85
                assert "Spain can be formed" in sources[0]["chunk_content"]

    @pytest.mark.asyncio
    async def test_query_returns_answer(self, mock_db_session, mock_chunks, mock_openai_response):
        """query should return structured response with answer."""
        with patch("app.services.rag_engine.AsyncOpenAI") as mock_client:
            mock_client.return_value.chat.completions.create = AsyncMock(
                return_value=mock_openai_response
            )
            with patch("app.services.rag_engine.VectorStore") as mock_store:
                mock_store.return_value.search = AsyncMock(return_value=mock_chunks)

                from app.services.rag_engine import RAGEngine

                engine = RAGEngine(mock_db_session)
                result = await engine.query("How do I form Spain?")

                assert "answer" in result
                assert "sources" in result
                assert "model" in result
                assert "tokens_used" in result
                assert "latency_ms" in result
                assert result["confidence"] == "high"  # score 0.85 >= 0.75

    @pytest.mark.asyncio
    async def test_query_no_chunks_returns_low_confidence(self, mock_db_session):
        """query should return low confidence when no chunks found."""
        with patch("app.services.rag_engine.AsyncOpenAI"):
            with patch("app.services.rag_engine.VectorStore") as mock_store:
                mock_store.return_value.search = AsyncMock(return_value=[])

                from app.services.rag_engine import RAGEngine

                engine = RAGEngine(mock_db_session)
                result = await engine.query("Unknown topic?")

                assert result["confidence"] == "low"
                assert result["sources"] == []
                assert "don't have enough" in result["answer"]