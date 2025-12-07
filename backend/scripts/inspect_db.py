"""
Inspect the database contents.

Usage:
    cd backend
    source .venv/bin/activate
    PYTHONPATH=. python scripts/inspect_db.py
"""

import asyncio
from sqlalchemy import text

import app.models.database as database
from app.models.database import init_db, close_db


async def main():
    await init_db()

    async with database.engine.connect() as conn:
        # Check documents
        print("=" * 60)
        print("DOCUMENTS")
        print("=" * 60)
        result = await conn.execute(text(
            "SELECT id, title, word_count, url FROM documents ORDER BY id"
        ))
        docs = result.fetchall()
        for d in docs:
            print(f"\n{d[0]}. {d[1]}")
            print(f"   Words: {d[2]}")
            print(f"   URL: {d[3]}")

        print(f"\nTotal: {len(docs)} documents")

        # Check chunks per document
        print("\n" + "=" * 60)
        print("CHUNKS PER DOCUMENT")
        print("=" * 60)
        result = await conn.execute(text("""
            SELECT d.title, COUNT(c.id) as chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id, d.title
            ORDER BY d.id
        """))
        for row in result.fetchall():
            print(f"  {row[0]}: {row[1]} chunks")

        # Check total chunks and embeddings
        print("\n" + "=" * 60)
        print("STATS")
        print("=" * 60)
        result = await conn.execute(text("SELECT COUNT(*) FROM chunks"))
        total_chunks = result.scalar()

        result = await conn.execute(text("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL"))
        embedded_chunks = result.scalar()

        print(f"  Total chunks: {total_chunks}")
        print(f"  With embeddings: {embedded_chunks}")

        # Sample chunk content
        print("\n" + "=" * 60)
        print("SAMPLE CHUNK CONTENT (first 3)")
        print("=" * 60)
        result = await conn.execute(text("""
            SELECT c.id, d.title, LEFT(c.content, 200) as preview
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            LIMIT 3
        """))
        for row in result.fetchall():
            print(f"\nChunk {row[0]} from '{row[1]}':")
            print(f"  {row[2]}...")

    await close_db()


if __name__ == "__main__":
    asyncio.run(main())