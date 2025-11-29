"""
Database configuration and SQLAlchemy models.

This module provides:
- Async database engine and session management
- Base model class for all database models
- Document and chunk models with pgvector support
- HNSW index configuration for vector similarity search
"""

from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    DateTime,
    ForeignKey,
    Index,
    func,
    text,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
    AsyncEngine,
)
from sqlalchemy.orm import DeclarativeBase, relationship

# pgvector support
from pgvector.sqlalchemy import Vector

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ============================================================================
# DATABASE ENGINE & SESSION
# ============================================================================

# Global engine instance (initialized in init_db)
engine: AsyncEngine | None = None

# Session factory (initialized in init_db)
async_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_db() -> None:
    """
    Initialize the database connection pool.

    Call this once at application startup (in main.py lifespan).

    What this does:
    - Creates the async engine with connection pooling
    - Creates session factory for generating sessions
    - Enables pgvector extension if not already enabled
    - Creates tables if they don't exist

    If database is unavailable, logs warning but doesn't crash.
    """
    global engine, async_session_factory

    logger.info(
        "initializing_database",
        host=settings.postgres_host,
        database=settings.postgres_db,
        pool_size=settings.db_pool_size,
    )

    # Create async engine with connection pooling
    engine = create_async_engine(
        settings.database_url,
        echo=settings.db_echo,  # Log SQL queries if enabled
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_pre_ping=True,  # Check connection health before using
    )

    # Create session factory
    async_session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,  # Don't expire objects after commit
        autocommit=False,
        autoflush=False,
    )

    # Try to connect and set up database
    try:
        async with engine.begin() as conn:
            # Enable pgvector extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            logger.info("pgvector_extension_enabled")

            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("database_tables_created")
    except Exception as e:
        logger.warning(
            "database_connection_failed",
            error=str(e),
            hint="Start PostgreSQL with: docker-compose up -d postgres",
        )


async def close_db() -> None:
    """
    Close database connections gracefully.

    Call this at application shutdown (in main.py lifespan).
    """
    global engine

    if engine is not None:
        await engine.dispose()
        logger.info("database_connections_closed")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency injection for database sessions.

    Usage in FastAPI:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()

    Why use this pattern?
    - Ensures sessions are properly closed after each request
    - Handles transaction rollback on errors
    - Works with FastAPI's dependency injection system
    """
    if async_session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ============================================================================
# BASE MODEL
# ============================================================================


class Base(DeclarativeBase):
    """
    Base class for all database models.

    Provides:
    - Common columns (id, created_at, updated_at)
    - Type hints for SQLAlchemy
    """

    pass


# ============================================================================
# COLLECTION MODEL (for multi-site support)
# ============================================================================


class Collection(Base):
    """
    Represents a collection of documents from a single source/site.

    Examples:
    - "eu5-wiki" for Europa Universalis 5 Wiki
    - "stellaris-wiki" for Stellaris Wiki
    - "company-docs" for internal documentation

    Why use collections?
    - Supports multiple RAG sources in one system
    - Each collection can have its own scraping config
    - Chat queries can be scoped to specific collections
    - Makes future multi-tenancy easier (collections belong to tenants)
    """

    __tablename__ = "collections"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Collection identity
    name = Column(String(100), unique=True, nullable=False, index=True)
    slug = Column(String(100), unique=True, nullable=False, index=True)  # URL-safe identifier
    description = Column(Text, nullable=True)

    # Source configuration
    base_url = Column(String(2048), nullable=False)  # e.g., "https://eu5.paradoxwikis.com"
    start_url = Column(String(2048), nullable=False)  # e.g., ".../Europa_Universalis_5_Wiki"

    # Scraping settings (can override global defaults)
    scraper_max_pages = Column(Integer, nullable=True)  # NULL = use default
    scraper_delay_seconds = Column(Integer, nullable=True)

    # Status
    is_active = Column(Integer, nullable=False, default=1)  # 1 = active, 0 = disabled
    last_scraped_at = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    documents = relationship("Document", back_populates="collection", cascade="all, delete-orphan")
    scrape_jobs = relationship("ScrapeJob", back_populates="collection", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Collection(id={self.id}, slug='{self.slug}')>"


# ============================================================================
# DOCUMENT MODEL
# ============================================================================


class Document(Base):
    """
    Represents a scraped wiki page.

    Each wiki page becomes one Document. Documents are then split into
    multiple Chunks for embedding and retrieval.

    Why separate Document and Chunk?
    - Documents store metadata about the source page
    - Chunks store the actual text pieces for RAG
    - This allows updating/re-scraping without losing chunk references
    """

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Parent collection reference
    collection_id = Column(
        Integer,
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Source information
    url = Column(String(2048), nullable=False, index=True)
    title = Column(String(500), nullable=False)

    # Content
    raw_content = Column(Text, nullable=False)  # Original HTML or cleaned text
    content_hash = Column(String(64), nullable=False)  # SHA-256 for change detection

    # Metadata
    scraped_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_modified = Column(DateTime, nullable=True)  # From wiki's Last-Modified header
    word_count = Column(Integer, nullable=False, default=0)

    # Timestamps
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    collection = relationship("Collection", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    # Unique constraint: same URL can exist in different collections
    __table_args__ = (
        Index("ix_documents_collection_url", "collection_id", "url", unique=True),
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, title='{self.title[:50]}...')>"


# ============================================================================
# CHUNK MODEL (with Vector Embedding)
# ============================================================================


class Chunk(Base):
    """
    Represents a text chunk from a document with its embedding.

    This is the core model for RAG retrieval:
    - Each chunk is a piece of text (800 tokens, 200 overlap)
    - Each chunk has a vector embedding for similarity search
    - Chunks reference their source document for citations

    Why use HNSW index instead of IVFFlat?
    - HNSW is faster for queries (no need to search multiple lists)
    - HNSW has better recall (more accurate results)
    - HNSW builds incrementally (IVFFlat needs rebuild after updates)
    - Trade-off: HNSW uses more memory and slower insert
    """

    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Parent document reference
    document_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Chunk content
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)  # Position in document (0, 1, 2...)

    # Token information
    token_count = Column(Integer, nullable=False)
    char_count = Column(Integer, nullable=False)

    # Vector embedding
    # Dimension must match OpenAI text-embedding-3-small (1536)
    embedding = Column(
        Vector(settings.openai_embedding_dimension),
        nullable=True,  # Nullable so we can insert before embedding
    )

    # Metadata for retrieval context
    # Store section headers, page title, etc. for better RAG prompts
    metadata_json = Column(Text, nullable=True)  # JSON string for flexibility

    # Timestamps
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    document = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<Chunk(id={self.id}, doc_id={self.document_id}, index={self.chunk_index})>"


# ============================================================================
# INDEXES
# ============================================================================

# HNSW index for vector similarity search
# This index dramatically speeds up nearest neighbor queries
#
# Parameters explained:
# - m (default 16): Number of bi-directional links per node
#   - Higher = better recall, more memory, slower build
#   - Typical: 12-48, 16 is a good default
#
# - ef_construction (default 64): Size of dynamic candidate list during construction
#   - Higher = better recall, slower build
#   - Typical: 64-200, 64 is a good default for development
#
# Distance operators:
# - vector_cosine_ops: Cosine similarity (what we use)
# - vector_l2_ops: L2/Euclidean distance
# - vector_ip_ops: Inner product

# Create HNSW index using raw SQL (pgvector-specific)
# Note: Index is created via init_pgvector.sql or Alembic migration
# The SQLAlchemy Index here is for documentation and potential future use

hnsw_index = Index(
    "ix_chunks_embedding_hnsw",
    Chunk.embedding,
    postgresql_using="hnsw",
    postgresql_with={
        "m": settings.hnsw_m,
        "ef_construction": settings.hnsw_ef_construction,
    },
    postgresql_ops={"embedding": "vector_cosine_ops"},
)


# ============================================================================
# SCRAPE JOB MODEL (for tracking background scraping tasks)
# ============================================================================


class ScrapeJob(Base):
    """
    Tracks background scraping jobs.

    When a user initiates a scrape, we create a ScrapeJob to track:
    - Progress (pages scraped, pages remaining)
    - Status (pending, running, completed, failed)
    - Errors encountered

    This allows the frontend to poll for progress updates.
    """

    __tablename__ = "scrape_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Parent collection reference
    collection_id = Column(
        Integer,
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Status tracking
    status = Column(
        String(20),
        nullable=False,
        default="pending",
        index=True,
    )  # pending, running, completed, failed

    # Progress
    total_pages = Column(Integer, nullable=False, default=0)
    pages_scraped = Column(Integer, nullable=False, default=0)
    pages_failed = Column(Integer, nullable=False, default=0)
    chunks_created = Column(Integer, nullable=False, default=0)

    # Error information
    error_message = Column(Text, nullable=True)

    # Timestamps
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    collection = relationship("Collection", back_populates="scrape_jobs")

    def __repr__(self) -> str:
        return f"<ScrapeJob(id={self.id}, collection_id={self.collection_id}, status='{self.status}', progress={self.pages_scraped}/{self.total_pages})>"