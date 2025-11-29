"""
API routes for the Micro-RAG application.

This module defines all API endpoints:
- Health check
- Scraping endpoints
- Chat/RAG endpoints
- Index statistics

Routes are organized by feature and will be expanded in later phases.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.models.database import get_db, Document, Chunk, ScrapeJob
from app.models.schemas import (
    HealthResponse,
    IndexStatsResponse,
    ScrapeRequest,
    ScrapeJobResponse,
    ScrapeStatusResponse,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
)

logger = get_logger(__name__)

# Create main router
router = APIRouter()


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
    # Check database connection
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


@router.get(
    "/health/ready",
    tags=["Health"],
    summary="Readiness check",
    description="Check if the service is ready to receive traffic",
)
async def readiness_check(db: AsyncSession = Depends(get_db)) -> dict:
    """
    Readiness probe for Kubernetes/Docker.

    Different from health check:
    - Health: Is the app running?
    - Ready: Is the app ready to serve requests?

    We check database connectivity here.
    """
    try:
        await db.execute(text("SELECT 1"))
        return {"ready": True}
    except Exception as e:
        logger.error("readiness_check_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )


@router.get(
    "/health/live",
    tags=["Health"],
    summary="Liveness check",
    description="Check if the service is alive",
)
async def liveness_check() -> dict:
    """
    Liveness probe for Kubernetes/Docker.

    Simple check - if we can respond, we're alive.
    No external dependency checks here.
    """
    return {"alive": True}


# ============================================================================
# INDEX STATISTICS
# ============================================================================


@router.get(
    "/index/stats",
    response_model=IndexStatsResponse,
    tags=["Index"],
    summary="Get index statistics",
    description="Get statistics about the vector index and stored documents",
)
async def get_index_stats(db: AsyncSession = Depends(get_db)) -> IndexStatsResponse:
    """
    Get statistics about the document index.

    Returns counts of documents, chunks, and embeddings.
    Useful for monitoring and debugging.
    """
    # Count documents
    doc_result = await db.execute(select(func.count()).select_from(Document))
    total_documents = doc_result.scalar() or 0

    # Count chunks
    chunk_result = await db.execute(select(func.count()).select_from(Chunk))
    total_chunks = chunk_result.scalar() or 0

    # Count chunks with embeddings (not null)
    embedded_result = await db.execute(
        select(func.count()).select_from(Chunk).where(Chunk.embedding.isnot(None))
    )
    total_embeddings = embedded_result.scalar() or 0

    # Calculate pending
    pending_embeddings = total_chunks - total_embeddings

    # Estimate storage (rough estimate: 1536 floats * 4 bytes * num_embeddings)
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
# SCRAPING ENDPOINTS (Stubs - to be implemented in Phase 3)
# ============================================================================


@router.post(
    "/scrape",
    response_model=ScrapeJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Scraping"],
    summary="Start scraping job",
    description="Start a background job to scrape EU5 Wiki pages",
    responses={
        202: {"description": "Scraping job started"},
        409: {"description": "Scraping job already running"},
    },
)
async def start_scrape(
    request: ScrapeRequest,
    db: AsyncSession = Depends(get_db),
) -> ScrapeJobResponse:
    """
    Start a new scraping job.

    This endpoint:
    1. Creates a new ScrapeJob in the database
    2. Starts a background task to scrape pages
    3. Returns immediately with a job ID for tracking

    To be implemented in Phase 3.
    """
    # Check for running jobs
    running_result = await db.execute(
        select(ScrapeJob).where(ScrapeJob.status.in_(["pending", "running"]))
    )
    running_job = running_result.scalar_one_or_none()

    if running_job:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Scraping job {running_job.id} is already {running_job.status}",
        )

    # Create new job
    job = ScrapeJob(
        status="pending",
        total_pages=request.max_pages or settings.scraper_max_pages,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    logger.info("scrape_job_created", job_id=job.id, max_pages=job.total_pages)

    # TODO: Phase 3 - Start background scraping task

    return ScrapeJobResponse(
        job_id=job.id,
        status=job.status,
        message=f"Scraping job {job.id} created. Implementation coming in Phase 3.",
    )


@router.get(
    "/scrape/{job_id}",
    response_model=ScrapeStatusResponse,
    tags=["Scraping"],
    summary="Get scraping job status",
    description="Get the current status of a scraping job",
    responses={
        200: {"description": "Job status"},
        404: {"description": "Job not found"},
    },
)
async def get_scrape_status(
    job_id: int,
    db: AsyncSession = Depends(get_db),
) -> ScrapeStatusResponse:
    """
    Get the status of a scraping job.

    Use this to poll for progress updates.
    """
    result = await db.execute(select(ScrapeJob).where(ScrapeJob.id == job_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scrape job {job_id} not found",
        )

    # Calculate progress
    progress = 0.0
    if job.total_pages > 0:
        progress = (job.pages_scraped / job.total_pages) * 100

    return ScrapeStatusResponse(
        job_id=job.id,
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
# CHAT/RAG ENDPOINTS (Stubs - to be implemented in Phase 6)
# ============================================================================


@router.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Ask a question",
    description="Ask a question about EU5 and get an AI-generated answer with sources",
    responses={
        200: {"description": "Generated answer"},
        503: {"description": "No documents indexed yet"},
    },
)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """
    RAG chat endpoint.

    This endpoint:
    1. Embeds the user's question
    2. Searches for relevant chunks
    3. Generates an answer using GPT-4
    4. Returns answer with source citations

    To be implemented in Phase 6.
    """
    # Check if we have any chunks
    chunk_count = await db.execute(select(func.count()).select_from(Chunk))
    if (chunk_count.scalar() or 0) == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No documents indexed yet. Please run a scrape job first.",
        )

    logger.info("chat_request", question=request.question[:100])

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


@router.get(
    "/",
    tags=["Info"],
    summary="API information",
    description="Get basic API information and available endpoints",
)
async def api_info() -> dict:
    """
    Root endpoint with API information.

    Useful for verifying the API is running.
    """
    return {
        "name": settings.app_name,
        "version": "0.1.0",
        "description": "RAG system for Europa Universalis 5 Wiki",
        "docs_url": f"{settings.api_v1_prefix}/docs",
        "health_url": f"{settings.api_v1_prefix}/health",
    }