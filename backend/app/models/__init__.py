"""
Database models and Pydantic schemas.

This package exports:
- SQLAlchemy models (Document, Chunk, ScrapeJob)
- Pydantic schemas for API validation
- Database utilities (get_db, init_db, close_db)
"""

from app.models.database import (
    Base,
    Document,
    Chunk,
    ScrapeJob,
    get_db,
    init_db,
    close_db,
)

from app.models.schemas import (
    HealthResponse,
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
    "Document",
    "Chunk",
    "ScrapeJob",
    "get_db",
    "init_db",
    "close_db",
    # Schemas
    "HealthResponse",
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