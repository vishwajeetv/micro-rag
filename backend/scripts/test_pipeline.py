"""
Test script for the full RAG pipeline.

Usage:
    cd backend
    source .venv/bin/activate
    PYTHONPATH=. python scripts/test_pipeline.py
"""

import asyncio
from sqlalchemy import select

import app.models.database as database
from app.models.database import init_db, close_db, get_db, Base, Collection
from app.services.scraper import WikiScraper
from app.services.vector_store import VectorStore


async def main():
    # 1. Initialize database connection
    await init_db()
    print("✓ Database connected")

    # 2. Create tables (engine is set after init_db)
    async with database.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✓ Tables created")

    # Get a database session
    async for db in get_db():
        # 3. Check if collection exists, create if not
        result = await db.execute(
            select(Collection).where(Collection.slug == "eu5-wiki")
        )
        collection = result.scalar_one_or_none()

        if collection:
            print(f"✓ Collection already exists: {collection.slug} (id={collection.id})")
        else:
            collection = Collection(
                name="EU5 Wiki",
                slug="eu5-wiki",
                description="Europa Universalis 5 Wiki",
                base_url="https://eu5.paradoxwikis.com",
                start_url="https://eu5.paradoxwikis.com/Europa_Universalis_5_Wiki",
            )
            db.add(collection)
            await db.commit()
            await db.refresh(collection)
            print(f"✓ Collection created: {collection.slug} (id={collection.id})")

        # 4. Scrape and ingest pages
        vector_store = VectorStore(db)
        print("\nScraping EU5 Wiki (ALL pages)...")
        print("This may take a while...\n")

        # Get already scraped URLs to skip them
        from app.models.database import Document
        existing_result = await db.execute(
            select(Document.url).where(Document.collection_id == collection.id)
        )
        existing_urls = set(row[0] for row in existing_result.fetchall())
        print(f"Already scraped: {len(existing_urls)} pages (will skip)\n")

        async with WikiScraper(base_url="https://eu5.paradoxwikis.com") as scraper:
            count = 0
            skipped = 0
            errors = 0
            async for result in scraper.crawl(
                start_url="https://eu5.paradoxwikis.com/Europa_Universalis_5_Wiki",
                max_pages=10000,  # Scrape all pages (wiki likely has <1000)
            ):
                if hasattr(result, "title"):  # ScrapedPage
                    # Skip already scraped
                    if result.url in existing_urls:
                        skipped += 1
                        continue

                    print(f"\n  [{count+1}] Ingesting: {result.title}")
                    print(f"      Words: {result.word_count}, Links found: {len(result.links)}")

                    doc, chunks = await vector_store.ingest_document(
                        collection_id=collection.id,
                        url=result.url,
                        title=result.title,
                        content=result.content,
                        content_hash=result.content_hash,
                        word_count=result.word_count,
                    )
                    await db.commit()
                    existing_urls.add(result.url)  # Track newly added
                    print(f"      → Created {len(chunks)} chunks with embeddings")
                    count += 1
                else:  # ScrapeError
                    errors += 1
                    print(f"\n  ✗ Error scraping: {result.url}")

        # 5. Show stats
        stats = await vector_store.get_stats(collection.id)
        print(f"\n{'='*50}")
        print(f"Done!")
        print(f"  New pages ingested: {count}")
        print(f"  Skipped (already in DB): {skipped}")
        print(f"  Errors: {errors}")
        print(f"\nTotal in database:")
        print(f"  Documents: {stats['documents']}")
        print(f"  Chunks: {stats['chunks']}")
        print(f"  Embeddings: {stats['chunks_with_embeddings']}")
        print(f"{'='*50}")

        break  # Exit the generator after one iteration

    # Cleanup
    await close_db()


if __name__ == "__main__":
    asyncio.run(main())