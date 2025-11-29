"""
API routes for the Micro-RAG application.

This module defines all API endpoints:
- Health check
- Collection management
- Scraping endpoints
- Chat/RAG endpoints
- Index statistics

Routes are organized by feature and will be expanded in later phases.
"""

import re
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text, select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.logging import get_logger
from app.models.database import get_db, Collection, Document, Chunk, ScrapeJob
from app.models.schemas import (
    HealthResponse,
    CollectionCreate,
    CollectionResponse,
    CollectionDetail,
    IndexStatsResponse,
    ScrapeRequest,
    ScrapeJobResponse,
    ScrapeStatusResponse,
    ChatRequest,
    ChatResponse,
)

logger = get_logger(__name__)

# Create main router
router = APIRouter()


def slugify(text: str) -> str:
    """Convert text to URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text


# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Check if the API and database are healthy",
)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    """
    Health check endpoint.

    Returns application status and database connectivity.
    Used by Docker health checks and load balancers.
    """
    try:
        await db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        logger.error("health_check_db_failed", error=str(e))
        db_status = "disconnected"

    return HealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        app_name=settings.app_name,
        environment=settings.environment,
        version="0.1.0",
        database=db_status,
        timestamp=datetime.utcnow(),
    )


@router.get("/health/ready", tags=["Health"], summary="Readiness check")
async def readiness_check(db: AsyncSession = Depends(get_db)) -> dict:
    """Readiness probe for Kubernetes/Docker."""
    try:
        await db.execute(text("SELECT 1"))
        return {"ready": True}
    except Exception as e:
        logger.error("readiness_check_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )


@router.get("/health/live", tags=["Health"], summary="Liveness check")
async def liveness_check() -> dict:
    """Liveness probe for Kubernetes/Docker."""
    return {"alive": True}


# ============================================================================
# COLLECTION ENDPOINTS
# ============================================================================


@router.post(
    "/collections",
    response_model=CollectionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Collections"],
    summary="Create a collection",
    description="Create a new document collection for a website",
)
async def create_collection(
    request: CollectionCreate,
    db: AsyncSession = Depends(get_db),
) -> CollectionResponse:
    """
    Create a new collection.

    Collections group documents from a single source (e.g., a wiki site).
    Each collection has its own scraping configuration.
    """
    # Generate slug if not provided
    slug = request.slug or slugify(request.name)

    # Check if slug already exists
    existing = await db.execute(
        select(Collection).where(Collection.slug == slug)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Collection with slug '{slug}' already exists",
        )

    # Create collection
    collection = Collection(
        name=request.name,
        slug=slug,
        description=request.description,
        base_url=request.base_url,
        start_url=request.start_url,
        scraper_max_pages=request.scraper_max_pages,
    )
    db.add(collection)
    await db.commit()
    await db.refresh(collection)

    logger.info("collection_created", collection_id=collection.id, slug=slug)

    return CollectionResponse(
        id=collection.id,
        name=collection.name,
        slug=collection.slug,
        description=collection.description,
        base_url=collection.base_url,
        start_url=collection.start_url,
        is_active=collection.is_active,
        document_count=0,
        last_scraped_at=collection.last_scraped_at,
        created_at=collection.created_at,
    )


@router.get(
    "/collections",
    response_model=list[CollectionResponse],
    tags=["Collections"],
    summary="List all collections",
)
async def list_collections(
    db: AsyncSession = Depends(get_db),
) -> list[CollectionResponse]:
    """List all document collections."""
    result = await db.execute(
        select(Collection).order_by(Collection.created_at.desc())
    )
    collections = result.scalars().all()

    responses = []
    for c in collections:
        # Count documents
        doc_count = await db.execute(
            select(func.count()).select_from(Document).where(Document.collection_id == c.id)
        )
        responses.append(
            CollectionResponse(
                id=c.id,
                name=c.name,
                slug=c.slug,
                description=c.description,
                base_url=c.base_url,
                start_url=c.start_url,
                is_active=c.is_active,
                document_count=doc_count.scalar() or 0,
                last_scraped_at=c.last_scraped_at,
                created_at=c.created_at,
            )
        )

    return responses


@router.get(
    "/collections/{slug}",
    response_model=CollectionDetail,
    tags=["Collections"],
    summary="Get collection details",
)
async def get_collection(
    slug: str,
    db: AsyncSession = Depends(get_db),
) -> CollectionDetail:
    """Get detailed information about a collection."""
    result = await db.execute(
        select(Collection).where(Collection.slug == slug)
    )
    collection = result.scalar_one_or_none()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{slug}' not found",
        )

    # Count documents and chunks
    doc_count = await db.execute(
        select(func.count()).select_from(Document).where(Document.collection_id == collection.id)
    )
    chunk_count = await db.execute(
        select(func.count())
        .select_from(Chunk)
        .join(Document)
        .where(Document.collection_id == collection.id)
    )

    return CollectionDetail(
        id=collection.id,
        name=collection.name,
        slug=collection.slug,
        description=collection.description,
        base_url=collection.base_url,
        start_url=collection.start_url,
        is_active=collection.is_active,
        scraper_max_pages=collection.scraper_max_pages,
        scraper_delay_seconds=collection.scraper_delay_seconds,
        document_count=doc_count.scalar() or 0,
        chunk_count=chunk_count.scalar() or 0,
        last_scraped_at=collection.last_scraped_at,
        created_at=collection.created_at,
    )


@router.delete(
    "/collections/{slug}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Collections"],
    summary="Delete a collection",
)
async def delete_collection(
    slug: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a collection and all its documents/chunks."""
    result = await db.execute(
        select(Collection).where(Collection.slug == slug)
    )
    collection = result.scalar_one_or_none()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{slug}' not found",
        )

    await db.delete(collection)
    await db.commit()

    logger.info("collection_deleted", slug=slug)


# ============================================================================
# INDEX STATISTICS
# ============================================================================


@router.get(
    "/index/stats",
    response_model=IndexStatsResponse,
    tags=["Index"],
    summary="Get index statistics",
)
async def get_index_stats(db: AsyncSession = Depends(get_db)) -> IndexStatsResponse:
    """Get statistics about the document index."""
    doc_result = await db.execute(select(func.count()).select_from(Document))
    total_documents = doc_result.scalar() or 0

    chunk_result = await db.execute(select(func.count()).select_from(Chunk))
    total_chunks = chunk_result.scalar() or 0

    embedded_result = await db.execute(
        select(func.count()).select_from(Chunk).where(Chunk.embedding.isnot(None))
    )
    total_embeddings = embedded_result.scalar() or 0

    pending_embeddings = total_chunks - total_embeddings
    storage_bytes = total_embeddings * settings.openai_embedding_dimension * 4
    storage_mb = storage_bytes / (1024 * 1024)

    return IndexStatsResponse(
        total_documents=total_documents,
        total_chunks=total_chunks,
        total_embeddings=total_embeddings,
        pending_embeddings=pending_embeddings,
        index_type="HNSW",
        index_params={
            "m": settings.hnsw_m,
            "ef_construction": settings.hnsw_ef_construction,
            "ef_search": settings.hnsw_ef_search,
            "distance_metric": settings.vector_distance_metric,
        },
        storage_mb=round(storage_mb, 2),
    )


# ============================================================================
# SCRAPING ENDPOINTS
# ============================================================================


@router.post(
    "/collections/{slug}/scrape",
    response_model=ScrapeJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Scraping"],
    summary="Start scraping job for a collection",
)
async def start_scrape(
    slug: str,
    request: ScrapeRequest,
    db: AsyncSession = Depends(get_db),
) -> ScrapeJobResponse:
    """
    Start a new scraping job for a collection.

    This endpoint:
    1. Creates a new ScrapeJob in the database
    2. Starts a background task to scrape pages
    3. Returns immediately with a job ID for tracking
    """
    # Get collection
    result = await db.execute(
        select(Collection).where(Collection.slug == slug)
    )
    collection = result.scalar_one_or_none()

    if not collection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{slug}' not found",
        )

    # Check for running jobs for this collection
    running_result = await db.execute(
        select(ScrapeJob).where(
            ScrapeJob.collection_id == collection.id,
            ScrapeJob.status.in_(["pending", "running"]),
        )
    )
    running_job = running_result.scalar_one_or_none()

    if running_job:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Scraping job {running_job.id} is already {running_job.status} for this collection",
        )

    # Determine max pages
    max_pages = (
        request.max_pages
        or collection.scraper_max_pages
        or settings.scraper_max_pages
    )

    # Create new job
    job = ScrapeJob(
        collection_id=collection.id,
        status="pending",
        total_pages=max_pages,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    logger.info(
        "scrape_job_created",
        job_id=job.id,
        collection_slug=slug,
        max_pages=max_pages,
    )

    # TODO: Phase 3 - Start background scraping task

    return ScrapeJobResponse(
        job_id=job.id,
        status=job.status,
        message=f"Scraping job {job.id} created for collection '{slug}'.",
    )


@router.get(
    "/scrape/{job_id}",
    response_model=ScrapeStatusResponse,
    tags=["Scraping"],
    summary="Get scraping job status",
)
async def get_scrape_status(
    job_id: int,
    db: AsyncSession = Depends(get_db),
) -> ScrapeStatusResponse:
    """Get the status of a scraping job."""
    result = await db.execute(
        select(ScrapeJob)
        .options(selectinload(ScrapeJob.collection))
        .where(ScrapeJob.id == job_id)
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scrape job {job_id} not found",
        )

    progress = 0.0
    if job.total_pages > 0:
        progress = (job.pages_scraped / job.total_pages) * 100

    return ScrapeStatusResponse(
        job_id=job.id,
        collection_id=job.collection_id,
        collection_slug=job.collection.slug if job.collection else None,
        status=job.status,
        total_pages=job.total_pages,
        pages_scraped=job.pages_scraped,
        pages_failed=job.pages_failed,
        chunks_created=job.chunks_created,
        progress_percent=round(progress, 1),
        error_message=job.error_message,
        started_at=job.started_at,
        completed_at=job.completed_at,
        created_at=job.created_at,
    )


# ============================================================================
# CHAT/RAG ENDPOINTS
# ============================================================================


@router.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Ask a question",
    description="Ask a question and get an AI-generated answer with sources",
)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """
    RAG chat endpoint.

    This endpoint:
    1. Embeds the user's question
    2. Searches for relevant chunks (optionally filtered by collection)
    3. Generates an answer using GPT-4
    4. Returns answer with source citations

    To be implemented in Phase 6.
    """
    # If collection specified, verify it exists
    if request.collection_slug:
        result = await db.execute(
            select(Collection).where(Collection.slug == request.collection_slug)
        )
        if not result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{request.collection_slug}' not found",
            )

    # Check if we have any chunks
    chunk_count = await db.execute(select(func.count()).select_from(Chunk))
    if (chunk_count.scalar() or 0) == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No documents indexed yet. Please create a collection and run a scrape job first.",
        )

    logger.info(
        "chat_request",
        question=request.question[:100],
        collection=request.collection_slug,
    )

    # TODO: Phase 6 - Implement RAG pipeline

    return ChatResponse(
        answer="RAG pipeline not yet implemented. Coming in Phase 6!",
        sources=[],
        model=settings.openai_chat_model,
        tokens_used=0,
        latency_ms=0,
    )


# ============================================================================
# DOCUMENTATION ENDPOINT
# ============================================================================


@router.get("/", tags=["Info"], summary="API information")
async def api_info() -> dict:
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": "0.1.0",
        "description": "Multi-collection RAG system",
        "docs_url": f"{settings.api_v1_prefix}/docs",
        "health_url": f"{settings.api_v1_prefix}/health",
    }
