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


async def main():
    await init_db()

    async for db in get_db():
        # Get collection
        result = await db.execute(
            select(Collection).where(Collection.slug == "eu5-wiki")
        )
        collection = result.scalar_one()
        print(f"Collection: {collection.name}\n")

        # Create RAG engine
        rag = RAGEngine(db)

        # Test questions
        questions = [
            "What is Europa Universalis 5?",
            "What are the main features of EU5?",
            "How does combat work?",
        ]

        for q in questions:
            print(f"Q: {q}")
            print("-" * 50)

            result = await rag.query(
                question=q,
                collection_id=collection.id,
                top_k=3,
            )

            print(f"A: {result['answer']}\n")
            print(f"Confidence: {result['confidence']}")
            print(f"Sources: {len(result['sources'])}")
            for s in result['sources'][:2]:
                print(f"  - {s['document_title']} (score: {s['relevance_score']:.2f})")
            print(f"Tokens: {result['tokens_used']}, Latency: {result['latency_ms']:.0f}ms")
            print("=" * 50 + "\n")

        break

    await close_db()


if __name__ == "__main__":
    asyncio.run(main())