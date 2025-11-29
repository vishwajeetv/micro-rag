"""
Database models and Pydantic schemas.

This package exports:
- SQLAlchemy models (Collection, Document, Chunk, ScrapeJob)
- Pydantic schemas for API validation
- Database utilities (get_db, init_db, close_db)
"""

from app.models.database import (
    Base,
    Collection,
    Document,
    Chunk,
    ScrapeJob,
    get_db,
    init_db,
    close_db,
)

from app.models.schemas import (
    HealthResponse,
    CollectionCreate,
    CollectionResponse,
    CollectionDetail,
    DocumentResponse,
    ChunkResponse,
    ChunkWithScore,
    ScrapeRequest,
    ScrapeJobResponse,
    ScrapeStatusResponse,
    ChatRequest,
    ChatResponse,
    IndexStatsResponse,
    ErrorResponse,
)

__all__ = [
    # Database
    "Base",
    "Collection",
    "Document",
    "Chunk",
    "ScrapeJob",
    "get_db",
    "init_db",
    "close_db",
    # Schemas
    "HealthResponse",
    "CollectionCreate",
    "CollectionResponse",
    "CollectionDetail",
    "DocumentResponse",
    "ChunkResponse",
    "ChunkWithScore",
    "ScrapeRequest",
    "ScrapeJobResponse",
    "ScrapeStatusResponse",
    "ChatRequest",
    "ChatResponse",
    "IndexStatsResponse",
    "ErrorResponse",
]
