"""
Pydantic schemas for API request/response validation.

Why separate schemas from SQLAlchemy models?
- SQLAlchemy models represent database structure
- Pydantic schemas represent API contracts
- They can evolve independently
- Pydantic provides automatic validation and serialization

Naming conventions:
- *Request: Input from client (POST/PUT body)
- *Response: Output to client
- *Base: Shared fields between Request/Response
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# HEALTH CHECK
# ============================================================================


class HealthResponse(BaseModel):
    """Response for health check endpoint."""

    status: str = Field(description="Overall health status", examples=["healthy"])
    app_name: str = Field(description="Application name")
    environment: str = Field(description="Current environment")
    version: str = Field(description="Application version")
    database: str = Field(
        description="Database connection status", examples=["connected", "disconnected"]
    )
    timestamp: datetime = Field(description="Response timestamp")


class DatabaseStats(BaseModel):
    """Database statistics for admin endpoints."""

    total_documents: int = Field(description="Total number of scraped documents")
    total_chunks: int = Field(description="Total number of text chunks")
    index_status: str = Field(description="Vector index status")


# ============================================================================
# COLLECTION SCHEMAS
# ============================================================================


class CollectionBase(BaseModel):
    """Base fields for Collection."""

    name: str = Field(
        min_length=2,
        max_length=100,
        description="Display name for the collection",
        examples=["Europa Universalis 5 Wiki", "Stellaris Wiki"],
    )
    description: str | None = Field(
        default=None,
        max_length=500,
        description="Description of what this collection contains",
    )
    base_url: str = Field(
        description="Base URL of the website",
        examples=["https://eu5.paradoxwikis.com"],
    )
    start_url: str = Field(
        description="Starting URL for scraping",
        examples=["https://eu5.paradoxwikis.com/Europa_Universalis_5_Wiki"],
    )


class CollectionCreate(CollectionBase):
    """Request to create a new collection."""

    slug: str | None = Field(
        default=None,
        min_length=2,
        max_length=100,
        pattern=r"^[a-z0-9-]+$",
        description="URL-safe identifier (auto-generated from name if not provided)",
        examples=["eu5-wiki", "stellaris-wiki"],
    )
    scraper_max_pages: int | None = Field(
        default=None,
        ge=1,
        le=10000,
        description="Max pages to scrape (default: from config)",
    )


class CollectionResponse(BaseModel):
    """Collection data returned from API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    slug: str
    description: str | None
    base_url: str
    start_url: str
    is_active: int
    document_count: int = Field(default=0, description="Number of documents in collection")
    last_scraped_at: datetime | None
    created_at: datetime


class CollectionDetail(CollectionResponse):
    """Detailed collection information."""

    scraper_max_pages: int | None
    scraper_delay_seconds: int | None
    chunk_count: int = Field(default=0, description="Total chunks in collection")


# ============================================================================
# DOCUMENT SCHEMAS
# ============================================================================


class DocumentBase(BaseModel):
    """Base fields for Document."""

    url: str = Field(description="Source URL of the wiki page")
    title: str = Field(description="Page title")


class DocumentResponse(DocumentBase):
    """Document data returned from API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    collection_id: int
    word_count: int
    chunk_count: int = Field(default=0, description="Number of chunks from this document")
    scraped_at: datetime
    created_at: datetime


class DocumentDetail(DocumentResponse):
    """Detailed document information including content."""

    raw_content: str = Field(description="Full content of the document")


# ============================================================================
# CHUNK SCHEMAS
# ============================================================================


class ChunkBase(BaseModel):
    """Base fields for Chunk."""

    content: str = Field(description="Chunk text content")
    chunk_index: int = Field(description="Position in source document")


class ChunkResponse(ChunkBase):
    """Chunk data returned from API (without embedding)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    document_id: int
    token_count: int
    char_count: int
    created_at: datetime


class ChunkWithScore(ChunkResponse):
    """Chunk with similarity score for search results."""

    score: float = Field(description="Similarity score (0-1, higher is more similar)")
    document_title: str = Field(description="Title of source document")
    document_url: str = Field(description="URL of source document")


# ============================================================================
# SCRAPE SCHEMAS
# ============================================================================


class ScrapeRequest(BaseModel):
    """Request to start a scraping job."""

    max_pages: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Maximum pages to scrape (default: from config)",
    )
    start_url: str | None = Field(
        default=None, description="Starting URL (default: EU5 Wiki main page)"
    )
    force_rescrape: bool = Field(
        default=False, description="Re-scrape pages even if already in database"
    )


class ScrapeJobResponse(BaseModel):
    """Response when starting a scrape job."""

    model_config = ConfigDict(from_attributes=True)

    job_id: int = Field(description="Unique job identifier for tracking")
    status: str = Field(description="Job status")
    message: str = Field(description="Human-readable status message")


class ScrapeStatusResponse(BaseModel):
    """Detailed status of a scraping job."""

    model_config = ConfigDict(from_attributes=True)

    job_id: int
    collection_id: int
    collection_slug: str | None = Field(default=None)
    status: str = Field(description="pending, running, completed, or failed")
    total_pages: int
    pages_scraped: int
    pages_failed: int
    chunks_created: int
    progress_percent: float = Field(description="Progress as percentage")
    error_message: str | None = Field(default=None)
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime


# ============================================================================
# CHAT / RAG SCHEMAS
# ============================================================================


class ChatRequest(BaseModel):
    """Request to chat with the RAG system."""

    question: str = Field(
        min_length=3,
        max_length=2000,
        description="Question to ask",
        examples=["How do I form Spain in EU5?", "What are the best trade nodes?"],
    )
    collection_slug: str | None = Field(
        default=None,
        description="Collection to search in (default: search all collections)",
        examples=["eu5-wiki"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of chunks to retrieve (default: from config)",
    )
    include_sources: bool = Field(
        default=True, description="Include source citations in response"
    )


class SourceCitation(BaseModel):
    """A source document cited in the response."""

    document_id: int
    document_title: str
    document_url: str
    chunk_content: str = Field(description="Relevant excerpt from the source")
    relevance_score: float = Field(description="How relevant this source is (0-1)")


class ChatResponse(BaseModel):
    """Response from the RAG chat endpoint."""

    answer: str = Field(description="Generated answer to the question")
    sources: list[SourceCitation] = Field(
        default_factory=list, description="Source documents used"
    )
    model: str = Field(description="LLM model used for generation")
    tokens_used: int = Field(description="Total tokens consumed")
    latency_ms: float = Field(description="Response time in milliseconds")


class ChatStreamChunk(BaseModel):
    """A chunk of streaming chat response."""

    type: str = Field(description="content, source, or done")
    content: str | None = Field(default=None)
    source: SourceCitation | None = Field(default=None)
    metadata: dict[str, Any] | None = Field(default=None)


# ============================================================================
# INDEX STATS
# ============================================================================


class IndexStatsResponse(BaseModel):
    """Statistics about the vector index."""

    total_documents: int
    total_chunks: int
    total_embeddings: int = Field(description="Chunks with embeddings")
    pending_embeddings: int = Field(description="Chunks awaiting embedding")
    index_type: str = Field(description="Type of vector index (HNSW)")
    index_params: dict[str, Any] = Field(description="Index configuration")
    storage_mb: float = Field(description="Estimated storage size")


# ============================================================================
# ERROR SCHEMAS
# ============================================================================


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error: str = Field(description="Error type")
    detail: str = Field(description="Detailed error message")
    request_id: str | None = Field(default=None, description="Request ID for debugging")


class ValidationErrorResponse(BaseModel):
    """Response for validation errors (422)."""

    error: str = "Validation Error"
    detail: list[dict[str, Any]] = Field(description="List of validation errors")


# ============================================================================
# AGENT SCHEMAS
# ============================================================================


class AgentRequest(BaseModel):
    """Request to run an agent query."""

    message: str = Field(
        min_length=1,
        max_length=5000,
        description="User message/question for the agent",
        examples=["What are the key features of Europa Universalis 5?"],
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversation continuity (auto-generated if not provided)",
    )
    collection_slug: str | None = Field(
        default=None,
        description="Collection to search in (default: search all collections)",
    )


class AgentEventData(BaseModel):
    """Data payload for agent events."""

    # For thinking events
    iteration: int | None = Field(default=None, description="Current iteration number")

    # For tool_call events
    tool: str | None = Field(default=None, description="Tool name being called")
    arguments: str | None = Field(default=None, description="Tool arguments as JSON")

    # For tool_result events
    success: bool | None = Field(default=None, description="Whether tool succeeded")
    output: str | None = Field(default=None, description="Tool output (truncated)")

    # For answer events
    content: str | None = Field(default=None, description="Final answer content")

    # For error events
    message: str | None = Field(default=None, description="Error message")


class AgentEventResponse(BaseModel):
    """A single event from the agent stream."""

    type: str = Field(
        description="Event type: thinking, tool_call, tool_result, answer, error"
    )
    data: AgentEventData = Field(description="Event-specific data")