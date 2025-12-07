"""
Vector store service for RAG.

This module handles:
- Storing documents and chunks in PostgreSQL
- Generating and storing embeddings
- Similarity search using pgvector
- Hybrid search (vector + keyword boost)
"""

import json
import re
from typing import Sequence

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.database import Document, Chunk, Collection
from app.services.embeddings import EmbeddingService
from app.services.chunker import TextChunker

logger = get_logger(__name__)

# Keyword boost settings for hybrid search
KEYWORD_BOOST = 0.35  # Boost score by this amount for keyword matches
MIN_KEYWORD_LENGTH = 3  # Skip short words like "in", "a", "the"
STOP_WORDS = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "can"}


def _simple_stem(word: str) -> str:
    """
    Simple stemmer - removes common suffixes for better keyword matching.
    """
    word = word.lower()
    # Order matters - check longer suffixes first
    suffixes = ['ies', 'es', 's', 'ing', 'ed', 'ly', 'tion', 'ness']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word


class VectorStore:
    """
    Manages document storage and vector similarity search.

    This is the main interface for:
    1. Ingesting scraped pages (document → chunks → embeddings)
    2. Searching for similar chunks given a query (hybrid: vector + keyword)
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

    def _extract_keywords(self, query: str) -> list[str]:
        """
        Extract meaningful keywords from query for hybrid search.

        Args:
            query: User query string

        Returns:
            List of lowercase keywords (no stop words, min length)
        """
        # Split on non-word characters
        words = re.findall(r'\w+', query.lower())
        # Filter: min length and not stop words
        keywords = [
            w for w in words
            if len(w) >= MIN_KEYWORD_LENGTH and w not in STOP_WORDS
        ]
        return keywords

    def _calculate_keyword_boost(self, content: str, keywords: list[str]) -> float:
        """
        Calculate keyword boost score for a chunk using stemmed matching.

        Gives extra bonus for chunks where content STARTS with keywords,
        indicating the chunk is specifically about that topic.

        Args:
            content: Chunk text content
            keywords: List of query keywords (already stemmed)

        Returns:
            Boost score (0 to KEYWORD_BOOST * 1.5)
        """
        if not keywords:
            return 0.0

        # Extract and stem words from content
        content_words = set(re.findall(r'\w+', content.lower()))
        content_stems = {_simple_stem(w) for w in content_words}

        # Stem keywords and check for matches
        matches = sum(1 for kw in keywords if _simple_stem(kw) in content_stems)
        # Score based on percentage of keywords matched
        match_ratio = matches / len(keywords)
        base_boost = KEYWORD_BOOST * match_ratio

        # Extra bonus if content STARTS with keyword (e.g., "List of cultures...")
        # This indicates the chunk is specifically ABOUT this topic
        content_start = content[:100].lower()
        start_bonus = 0.0

        # Count how many keywords appear in the start
        start_matches = sum(1 for kw in keywords if _simple_stem(kw) in content_start[:50])
        if start_matches >= 2:
            # Multiple keywords at start = very relevant
            start_bonus = KEYWORD_BOOST * 1.0  # Full extra boost
        elif start_matches == 1:
            start_bonus = KEYWORD_BOOST * 0.5  # Half extra boost

        return base_boost + start_bonus

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
        Hybrid search: vector similarity + keyword text search.

        Two-phase approach:
        1. Vector search for semantic matches
        2. Text search for keyword matches (catches tabular data)
        Merge and re-rank results.

        Args:
            query: Search query text
            collection_id: Optional filter by collection
            limit: Max results to return
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of dicts with chunk info and similarity score
        """
        # Extract keywords for text search
        keywords = self._extract_keywords(query)

        # Generate query embedding
        query_embedding = await self.embeddings.embed_text(query)

        # Build the search query
        # pgvector uses <=> for cosine distance (0 = identical, 2 = opposite)
        # We convert to similarity: 1 - (distance / 2)
        distance_expr = Chunk.embedding.cosine_distance(query_embedding)
        similarity_expr = (1 - distance_expr).label("similarity")

        # Fetch vector search candidates
        fetch_limit = max(limit * 5, 50)

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
            .limit(fetch_limit)
        )

        if collection_id is not None:
            stmt = stmt.where(Document.collection_id == collection_id)

        result = await self.db.execute(stmt)
        vector_rows = result.all()

        # Also do a keyword text search using ILIKE
        # This catches chunks that have low vector scores but contain query terms
        keyword_rows = []
        if keywords and len(keywords) >= 2:
            # Search for chunks containing at least 2 keywords
            # Use multiple OR combinations of keyword pairs
            from sqlalchemy import or_, and_

            # Generate all 2-keyword combinations
            keyword_pairs = []
            for i in range(len(keywords)):
                for j in range(i + 1, len(keywords)):
                    keyword_pairs.append(
                        and_(
                            Chunk.content.ilike(f"%{keywords[i]}%"),
                            Chunk.content.ilike(f"%{keywords[j]}%"),
                        )
                    )

            keyword_stmt = (
                select(
                    Chunk,
                    Document.title.label("document_title"),
                    Document.url.label("document_url"),
                    similarity_expr,
                )
                .join(Document)
                .where(Chunk.embedding.isnot(None))
            )
            if collection_id is not None:
                keyword_stmt = keyword_stmt.where(Document.collection_id == collection_id)

            # Match chunks containing at least 2 keywords
            keyword_stmt = keyword_stmt.where(or_(*keyword_pairs)).limit(fetch_limit)

            kw_result = await self.db.execute(keyword_stmt)
            keyword_rows = kw_result.all()

        # Merge results (dedupe by chunk_id)
        seen_ids = set()
        all_rows = []
        for row in vector_rows:
            if row[0].id not in seen_ids:
                seen_ids.add(row[0].id)
                all_rows.append(row)
        for row in keyword_rows:
            if row[0].id not in seen_ids:
                seen_ids.add(row[0].id)
                all_rows.append(row)

        # Format results with keyword boost
        candidates = []
        for chunk, doc_title, doc_url, similarity in all_rows:
            # Calculate keyword boost
            keyword_boost = self._calculate_keyword_boost(chunk.content, keywords)
            # Also boost if keywords appear in document title
            title_boost = self._calculate_keyword_boost(doc_title, keywords) * 0.5

            boosted_score = float(similarity) + keyword_boost + title_boost

            if boosted_score >= score_threshold:
                metadata = json.loads(chunk.metadata_json) if chunk.metadata_json else {}
                candidates.append({
                    "chunk_id": chunk.id,
                    "content": chunk.content,
                    "document_id": chunk.document_id,
                    "document_title": doc_title,
                    "document_url": doc_url,
                    "chunk_index": chunk.chunk_index,
                    "score": boosted_score,
                    "vector_score": float(similarity),
                    "keyword_boost": keyword_boost + title_boost,
                    "header": metadata.get("header"),
                })

        # Re-sort by boosted score and take top N
        candidates.sort(key=lambda x: x["score"], reverse=True)
        results = candidates[:limit]

        logger.info(
            "hybrid_search_completed",
            query=query[:50],
            keywords=keywords,
            results_found=len(results),
            vector_candidates=len(vector_rows),
            keyword_candidates=len(keyword_rows),
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