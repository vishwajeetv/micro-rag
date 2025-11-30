"""
Vector store service for RAG.

This module handles:
- Storing documents and chunks in PostgreSQL
- Generating and storing embeddings
- Similarity search using pgvector
"""

import json
from typing import Sequence

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.database import Document, Chunk, Collection
from app.services.embeddings import EmbeddingService
from app.services.chunker import TextChunker

logger = get_logger(__name__)


class VectorStore:
    """
    Manages document storage and vector similarity search.

    This is the main interface for:
    1. Ingesting scraped pages (document → chunks → embeddings)
    2. Searching for similar chunks given a query
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize the vector store.

        Args:
            db: AsyncSession from FastAPI dependency injection
        """
        self.db = db
        self.embeddings = EmbeddingService()
        self.chunker = TextChunker()

    async def ingest_document(
        self,
        collection_id: int,
        url: str,
        title: str,
        content: str,
        content_hash: str,
        word_count: int,
    ) -> tuple[Document, list[Chunk]]:
        """
        Ingest a scraped page: store document, chunk, embed, and save.

        Args:
            collection_id: Parent collection ID
            url: Source page URL
            title: Page title
            content: Cleaned text content
            content_hash: SHA-256 hash for change detection
            word_count: Word count of content

        Returns:
            Tuple of (Document, list of Chunks)
        """
        # Check if document already exists (by URL in this collection)
        existing = await self.db.execute(
            select(Document).where(
                Document.collection_id == collection_id,
                Document.url == url,
            )
        )
        existing_doc = existing.scalar_one_or_none()

        if existing_doc:
            # Check if content changed
            if existing_doc.content_hash == content_hash:
                logger.debug("document_unchanged", url=url)
                # Return existing document and chunks
                chunks_result = await self.db.execute(
                    select(Chunk).where(Chunk.document_id == existing_doc.id)
                )
                return existing_doc, list(chunks_result.scalars().all())

            # Content changed - delete old chunks (will re-create)
            logger.info("document_changed", url=url)
            await self.db.execute(
                text("DELETE FROM chunks WHERE document_id = :doc_id"),
                {"doc_id": existing_doc.id},
            )
            # Update document
            existing_doc.raw_content = content
            existing_doc.content_hash = content_hash
            existing_doc.word_count = word_count
            existing_doc.title = title
            document = existing_doc
        else:
            # Create new document
            document = Document(
                collection_id=collection_id,
                url=url,
                title=title,
                raw_content=content,
                content_hash=content_hash,
                word_count=word_count,
            )
            self.db.add(document)
            await self.db.flush()  # Get document ID

        # Chunk the content
        chunk_dicts = self.chunker.chunk_document(content, title=title, url=url)

        if not chunk_dicts:
            logger.warning("no_chunks_created", url=url)
            return document, []

        # Create chunk objects (without embeddings yet)
        chunks = []
        for chunk_dict in chunk_dicts:
            metadata = {"header": chunk_dict.get("header")}
            chunk = Chunk(
                document_id=document.id,
                content=chunk_dict["content"],
                chunk_index=chunk_dict["chunk_index"],
                token_count=chunk_dict["token_count"],
                char_count=chunk_dict["char_count"],
                metadata_json=json.dumps(metadata),
            )
            chunks.append(chunk)

        self.db.add_all(chunks)
        await self.db.flush()  # Get chunk IDs

        # Generate embeddings in batch
        texts = [c.content for c in chunks]
        embeddings = await self.embeddings.embed_batch(texts)

        # Update chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        await self.db.flush()

        logger.info(
            "document_ingested",
            url=url,
            title=title[:50],
            chunks_created=len(chunks),
        )

        return document, chunks

    async def search(
        self,
        query: str,
        collection_id: int | None = None,
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> list[dict]:
        """
        Search for chunks similar to the query.

        Uses cosine similarity via pgvector's <=> operator.

        Args:
            query: Search query text
            collection_id: Optional filter by collection
            limit: Max results to return
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of dicts with chunk info and similarity score
        """
        # Generate query embedding
        query_embedding = await self.embeddings.embed_text(query)

        # Build the search query
        # pgvector uses <=> for cosine distance (0 = identical, 2 = opposite)
        # We convert to similarity: 1 - (distance / 2)
        distance_expr = Chunk.embedding.cosine_distance(query_embedding)
        similarity_expr = (1 - distance_expr).label("similarity")

        stmt = (
            select(
                Chunk,
                Document.title.label("document_title"),
                Document.url.label("document_url"),
                similarity_expr,
            )
            .join(Document)
            .where(Chunk.embedding.isnot(None))
            .order_by(distance_expr)
            .limit(limit)
        )

        # Filter by collection if specified
        if collection_id is not None:
            stmt = stmt.where(Document.collection_id == collection_id)

        result = await self.db.execute(stmt)
        rows = result.all()

        # Format results
        results = []
        for chunk, doc_title, doc_url, similarity in rows:
            if similarity >= score_threshold:
                metadata = json.loads(chunk.metadata_json) if chunk.metadata_json else {}
                results.append({
                    "chunk_id": chunk.id,
                    "content": chunk.content,
                    "document_id": chunk.document_id,
                    "document_title": doc_title,
                    "document_url": doc_url,
                    "chunk_index": chunk.chunk_index,
                    "score": float(similarity),
                    "header": metadata.get("header"),
                })

        logger.info(
            "search_completed",
            query=query[:50],
            results_found=len(results),
            collection_id=collection_id,
        )

        return results

    async def get_stats(self, collection_id: int | None = None) -> dict:
        """
        Get statistics about the vector store.

        Args:
            collection_id: Optional filter by collection

        Returns:
            Dict with document/chunk counts
        """
        # Count documents
        doc_stmt = select(func.count(Document.id))
        if collection_id:
            doc_stmt = doc_stmt.where(Document.collection_id == collection_id)
        doc_count = (await self.db.execute(doc_stmt)).scalar() or 0

        # Count chunks
        chunk_stmt = select(func.count(Chunk.id))
        if collection_id:
            chunk_stmt = chunk_stmt.join(Document).where(
                Document.collection_id == collection_id
            )
        chunk_count = (await self.db.execute(chunk_stmt)).scalar() or 0

        # Count chunks with embeddings
        embedded_stmt = select(func.count(Chunk.id)).where(Chunk.embedding.isnot(None))
        if collection_id:
            embedded_stmt = embedded_stmt.join(Document).where(
                Document.collection_id == collection_id
            )
        embedded_count = (await self.db.execute(embedded_stmt)).scalar() or 0

        return {
            "documents": doc_count,
            "chunks": chunk_count,
            "chunks_with_embeddings": embedded_count,
            "collection_id": collection_id,
        }