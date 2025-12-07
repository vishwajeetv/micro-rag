"""
RAG Query Engine for EU5 Wiki.

This module handles:
- Retrieving relevant chunks from vector store
- Building prompts (system + context + query)
- Calling GPT-5.1 Instant for generation
- Streaming responses with source citations
"""

import time
from typing import AsyncGenerator

from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.database import Collection
from app.core.logging import get_logger
from app.services.vector_store import VectorStore

logger = get_logger(__name__)

# System prompt template - placeholders filled at query time
SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant for {collection_name}.
{collection_description}

Rules:
- ONLY use information from the provided context below
- If the answer is not in the context, say "I don't have information about that in the knowledge base"
- Be concise: 1-3 sentences for simple questions, more for complex topics
- Use bullet points for lists
- Do not make up information

Context:
{context}
"""

# Fallback when no collection specified (searching all)
DEFAULT_COLLECTION_NAME = "this knowledge base"
DEFAULT_COLLECTION_DESCRIPTION = ""


class RAGEngine:
    """
    Orchestrates the RAG pipeline: retrieve → build prompt → generate.

    Usage:
        engine = RAGEngine(db_session)

        # Non-streaming
        response = await engine.query("How do I form Spain?")

        # Streaming
        async for chunk in engine.query_stream("How do I form Spain?"):
            print(chunk)
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize the RAG engine.

        Args:
            db: AsyncSession from FastAPI dependency injection
        """
        self.db = db
        self.vector_store = VectorStore(db)
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_chat_model
        self.temperature = settings.openai_temperature
        self.max_tokens = settings.openai_max_tokens

        logger.info(
            "rag_engine_initialized",
            model=self.model,
            temperature=self.temperature,
        )

    def _format_chunk(self, chunk: dict) -> str:
        """
        Format a single chunk with its metadata for the prompt.

        Args:
            chunk: Dict with content, document_title, document_url, header

        Returns:
            Formatted string with source info and content
        """
        header_line = f"Section: {chunk['header']}\n" if chunk.get("header") else ""
        return f"""### {chunk['document_title']}
Source: {chunk['document_url']}
{header_line}
{chunk['content']}
"""

    def _build_messages(
        self,
        question: str,
        chunks: list[dict],
        collection_name: str | None = None,
        collection_description: str | None = None,
    ) -> list[dict]:
        """
        Build the messages list for OpenAI chat completion.

        Args:
            question: User's question
            chunks: Retrieved chunks from vector store
            collection_name: Name of the collection (or default)
            collection_description: Description of the collection

        Returns:
            List of message dicts for OpenAI API
        """
        # Format context from chunks
        if chunks:
            context = "\n---\n".join(self._format_chunk(c) for c in chunks)
        else:
            context = "(No relevant context found)"

        # Build system prompt from template
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            collection_name=collection_name or DEFAULT_COLLECTION_NAME,
            collection_description=collection_description or DEFAULT_COLLECTION_DESCRIPTION,
            context=context,
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

    async def _get_collection(self, collection_id: int) -> Collection | None:
        """Fetch collection by ID."""
        result = await self.db.execute(
            select(Collection).where(Collection.id == collection_id)
        )
        return result.scalar_one_or_none()

    def _build_sources(self, chunks: list[dict]) -> list[dict]:
        """
        Build source citations from retrieved chunks.

        We return sources separately (not embedded in LLM response)
        to avoid hallucinated citations (Quiz Q6).
        """
        return [
            {
                "document_id": c["document_id"],
                "document_title": c["document_title"],
                "document_url": c["document_url"],
                "chunk_content": c["content"][:200] + "..." if len(c["content"]) > 200 else c["content"],
                "relevance_score": c["score"],
            }
            for c in chunks
        ]

    async def query(
        self,
        question: str,
        collection_id: int | None = None,
        top_k: int | None = None,
    ) -> dict:
        """
        Execute a RAG query (non-streaming).

        Args:
            question: User's question
            collection_id: Optional collection to search in
            top_k: Number of chunks to retrieve (default from config)

        Returns:
            Dict with answer, sources, model, tokens_used, latency_ms
        """
        start_time = time.time()

        # Get collection info if specified
        collection_name = None
        collection_description = None
        if collection_id:
            collection = await self._get_collection(collection_id)
            if collection:
                collection_name = collection.name
                collection_description = collection.description

        # Retrieve relevant chunks
        chunks = await self.vector_store.search(
            query=question,
            collection_id=collection_id,
            limit=top_k or settings.rag_top_k,
            score_threshold=settings.rag_min_score,
        )

        # Handle low-confidence case (Quiz Q9)
        if not chunks:
            logger.warning("no_relevant_chunks", question=question[:50])
            return {
                "answer": "I don't have enough relevant information to answer this question. Try rephrasing or check if this topic is covered in the wiki.",
                "sources": [],
                "model": self.model,
                "tokens_used": 0,
                "latency_ms": (time.time() - start_time) * 1000,
                "confidence": "low",
            }

        # Build messages and call LLM
        messages = self._build_messages(
            question=question,
            chunks=chunks,
            collection_name=collection_name,
            collection_description=collection_description,
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if response.usage else 0
        latency_ms = (time.time() - start_time) * 1000

        logger.info(
            "rag_query_completed",
            question=question[:50],
            chunks_used=len(chunks),
            tokens_used=tokens_used,
            latency_ms=round(latency_ms, 2),
        )

        return {
            "answer": answer,
            "sources": self._build_sources(chunks),
            "model": self.model,
            "tokens_used": tokens_used,
            "latency_ms": latency_ms,
            "confidence": "high" if chunks[0]["score"] >= 0.75 else "medium",
        }

    async def query_stream(
        self,
        question: str,
        collection_id: int | None = None,
        top_k: int | None = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Execute a RAG query with streaming response.

        Yields chunks in order (Quiz Q10 - sources first):
        1. {"type": "sources", "data": [...]}  - All sources upfront
        2. {"type": "content", "data": "token"} - Each token as generated
        3. {"type": "done", "data": {...}}     - Final metadata

        Args:
            question: User's question
            collection_id: Optional collection to search in
            top_k: Number of chunks to retrieve

        Yields:
            Dict with type and data
        """
        start_time = time.time()

        # Get collection info if specified
        collection_name = None
        collection_description = None
        if collection_id:
            collection = await self._get_collection(collection_id)
            if collection:
                collection_name = collection.name
                collection_description = collection.description

        # Retrieve relevant chunks
        chunks = await self.vector_store.search(
            query=question,
            collection_id=collection_id,
            limit=top_k or settings.rag_top_k,
            score_threshold=settings.rag_min_score,
        )

        # Handle low-confidence case
        if not chunks:
            logger.warning("no_relevant_chunks_stream", question=question[:50])
            yield {
                "type": "sources",
                "data": [],
            }
            yield {
                "type": "content",
                "data": "I don't have enough relevant information to answer this question. Try rephrasing or check if this topic is covered in the wiki.",
            }
            yield {
                "type": "done",
                "data": {
                    "model": self.model,
                    "tokens_used": 0,
                    "latency_ms": (time.time() - start_time) * 1000,
                    "confidence": "low",
                },
            }
            return

        # Send sources FIRST
        yield {
            "type": "sources",
            "data": self._build_sources(chunks),
        }

        # Build messages and stream LLM response
        messages = self._build_messages(
            question=question,
            chunks=chunks,
            collection_name=collection_name,
            collection_description=collection_description,
        )

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield {
                    "type": "content",
                    "data": chunk.choices[0].delta.content,
                }

        latency_ms = (time.time() - start_time) * 1000

        logger.info(
            "rag_stream_completed",
            question=question[:50],
            chunks_used=len(chunks),
            latency_ms=round(latency_ms, 2),
        )

        # Send completion with metadata
        yield {
            "type": "done",
            "data": {
                "model": self.model,
                "latency_ms": latency_ms,
                "confidence": "high" if chunks[0]["score"] >= 0.75 else "medium",
            },
        }