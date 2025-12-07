"""
Services module.

Contains business logic for:
- Web scraping (scraper.py)
- Embeddings (embeddings.py) - Phase 5
- Vector store operations (vector_store.py) - Phase 5
- RAG pipeline (rag_engine.py) - Phase 6
"""

from app.services.scraper import WikiScraper, ScrapedPage, ScrapeError, scrape_single_page
from app.services.rag_engine import RAGEngine

__all__ = [
    "WikiScraper",
    "ScrapedPage",
    "ScrapeError",
    "scrape_single_page",
    "RAGEngine",
]
