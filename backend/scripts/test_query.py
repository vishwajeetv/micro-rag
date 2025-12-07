"""
Test script for querying the RAG pipeline.

Usage:
    cd backend
    source .venv/bin/activate
    PYTHONPATH=. python scripts/test_query.py
"""

import asyncio
from sqlalchemy import select

import app.models.database as database
from app.models.database import init_db, close_db, get_db, Collection
from app.services.rag_engine import RAGEngine
from app.services.vector_store import VectorStore
from app.core.config import settings


async def main():
    await init_db()

    async for db in get_db():
        # Get collection
        result = await db.execute(
            select(Collection).where(Collection.slug == "eu5-wiki")
        )
        collection = result.scalar_one()
        print(f"Collection: {collection.name}")
        print(f"RAG min score: {settings.rag_min_score}")
        print("=" * 60)

        # Test question
        question = "How does combat work in EU5?"

        # First, show what chunks we retrieve
        vs = VectorStore(db)
        chunks = await vs.search(
            query=question,
            collection_id=collection.id,
            limit=settings.rag_top_k,
            score_threshold=settings.rag_min_score,
        )

        print(f"\nQ: {question}")
        print(f"\nRetrieved {len(chunks)} chunks:")
        for i, c in enumerate(chunks):
            print(f"\n  [{i+1}] {c['document_title']} (score: {c['score']:.3f})")
            print(f"      {c['content'][:150]}...")

        # Build and show the prompt
        rag = RAGEngine(db)
        messages = rag._build_messages(
            question=question,
            chunks=chunks,
            collection_name=collection.name,
            collection_description=collection.description,
        )

        print("\n" + "=" * 60)
        print("PROMPT SENT TO LLM:")
        print("=" * 60)
        for msg in messages:
            print(f"\n[{msg['role'].upper()}]")
            print("-" * 40)
            # Truncate system message if too long
            content = msg['content']
            if msg['role'] == 'system' and len(content) > 2000:
                print(content[:2000] + "\n... (truncated)")
            else:
                print(content)

        # Now get the actual answer
        print("\n" + "=" * 60)
        print("LLM RESPONSE:")
        print("=" * 60)

        result = await rag.query(
            question=question,
            collection_id=collection.id,
            top_k=settings.rag_top_k,
        )

        print(f"\nA: {result['answer']}")
        print(f"\nConfidence: {result['confidence']}")
        print(f"Sources: {len(result['sources'])}")
        print(f"Tokens: {result['tokens_used']}, Latency: {result['latency_ms']:.0f}ms")

        break

    await close_db()


if __name__ == "__main__":
    asyncio.run(main())