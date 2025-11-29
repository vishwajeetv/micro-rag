"""
FastAPI application entry point.

This module creates and configures the FastAPI application with:
- Lifespan events (startup/shutdown)
- CORS middleware for frontend integration
- Request logging middleware
- API routes
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars
import uuid
import time

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.models.database import init_db, close_db
from app.api.routes import router

# Initialize logger
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    This handles startup and shutdown events:
    - Startup: Configure logging, initialize database connection
    - Shutdown: Close database connections gracefully

    Why use lifespan instead of on_event decorators?
    - on_event is deprecated in FastAPI
    - lifespan provides cleaner resource management
    - Works better with async context managers
    """
    # ========================================================================
    # STARTUP
    # ========================================================================

    # Configure structured logging first
    configure_logging()
    logger.info(
        "application_starting",
        app_name=settings.app_name,
        environment=settings.environment,
        debug=settings.debug,
    )

    # Initialize database connection pool
    await init_db()
    logger.info("database_initialized", host=settings.postgres_host)

    logger.info(
        "application_started",
        host=settings.host,
        port=settings.port,
        api_prefix=settings.api_v1_prefix,
    )

    yield  # Application runs here

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    logger.info("application_shutting_down")

    # Close database connections
    await close_db()
    logger.info("database_connections_closed")

    logger.info("application_stopped")


def create_application() -> FastAPI:
    """
    Application factory function.

    Why use a factory function?
    - Makes testing easier (create fresh app instances)
    - Allows different configurations for different environments
    - Follows the factory pattern best practice
    """

    app = FastAPI(
        title=settings.app_name,
        description="Micro RAG System",
        version="0.1.0",
        debug=settings.debug,
        lifespan=lifespan,
        # OpenAPI documentation configuration
        docs_url=f"{settings.api_v1_prefix}/docs" if settings.debug else None,
        redoc_url=f"{settings.api_v1_prefix}/redoc" if settings.debug else None,
        openapi_url=f"{settings.api_v1_prefix}/openapi.json" if settings.debug else None,
    )

    # ========================================================================
    # MIDDLEWARE (order matters - last added runs first)
    # ========================================================================

    # CORS middleware - allows frontend to call backend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_methods_list,
        allow_headers=["*"],
    )

    # ========================================================================
    # REQUEST LOGGING MIDDLEWARE
    # ========================================================================

    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        """
        Log every request with timing and context.

        This middleware:
        - Generates a unique request ID for tracing
        - Logs request start and completion
        - Measures request duration
        - Binds context for all logs in the request
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())[:8]

        # Bind context for this request (all logs will include these)
        bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        # Log request start
        start_time = time.perf_counter()
        logger.info(
            "request_started",
            client_host=request.client.host if request.client else "unknown",
        )

        try:
            # Process the request
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log request completion
            logger.info(
                "request_completed",
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )

            # Add request ID to response headers for debugging
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as exc:
            # Calculate duration even for errors
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log error
            logger.error(
                "request_failed",
                error=str(exc),
                duration_ms=round(duration_ms, 2),
                exc_info=True,
            )

            # Re-raise to let FastAPI handle it
            raise

        finally:
            # Always clear context after request
            clear_contextvars()

    # ========================================================================
    # EXCEPTION HANDLERS
    # ========================================================================

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """
        Catch-all exception handler.

        In production, we don't want to expose internal error details.
        In development, we show the full error message.
        """
        logger.error(
            "unhandled_exception",
            error=str(exc),
            path=request.url.path,
            exc_info=True,
        )

        if settings.debug:
            # Development: show error details
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "detail": str(exc),
                    "type": type(exc).__name__,
                },
            )
        else:
            # Production: hide details
            return JSONResponse(
                status_code=500,
                content={"error": "Internal Server Error"},
            )

    # ========================================================================
    # ROUTES
    # ========================================================================

    # Include API routes with prefix
    app.include_router(router, prefix=settings.api_v1_prefix)

    return app


# Create the application instance
app = create_application()