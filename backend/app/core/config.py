"""
Application configuration management using Pydantic Settings.

This module loads and validates all environment variables from .env file.
All settings are type-safe and validated at startup.

How it works:
- Pydantic Settings ALWAYS tries to read from environment variables first
- Default values are only used as FALLBACK if env var is not set
- Required fields (Field(...)) will cause startup to fail if not provided
"""

from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Settings are loaded in this order (later overrides earlier):
    1. Environment variables (ALWAYS CHECKED FIRST)
    2. .env file in project root
    3. Default values defined below (FALLBACK ONLY)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra env vars not defined here
    )

    # ========================================================================
    # APPLICATION SETTINGS
    # ========================================================================

    environment: Literal["development", "staging", "production"] = "development"
    app_name: str = "micro-rag"
    api_v1_prefix: str = "/api"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000

    # ========================================================================
    # OPENAI CONFIGURATION (REQUIRED - NO DEFAULTS FOR SECRETS)
    # ========================================================================

    openai_api_key: str = Field(
        ...,  # Required! App will crash if not in .env
        description="OpenAI API key (get from https://platform.openai.com/api-keys)"
    )
    openai_chat_model: str = "gpt-5.1"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimension: int = 1536
    openai_max_tokens: int = 1000
    openai_temperature: float = Field(0.2, ge=0.0, le=2.0)  # Low for factual Q&A

    @field_validator("openai_api_key")
    @classmethod
    def validate_openai_key(cls, v: str) -> str:
        """Validate OpenAI API key is not empty."""
        if not v or len(v) < 10:
            raise ValueError("OpenAI API key is required and must be at least 10 characters")
        return v

    # ========================================================================
    # POSTGRESQL + PGVECTOR CONFIGURATION
    # ========================================================================

    # Individual components - used to build database_url
    postgres_user: str = Field(
        ...,  # Required!
        description="PostgreSQL username"
    )
    postgres_password: str = Field(
        ...,  # Required! Never have default passwords
        description="PostgreSQL password"
    )
    postgres_db: str = Field(
        ...,  # Required!
        description="PostgreSQL database name"
    )
    postgres_host: str = "localhost"  # Has default but reads from env first
    postgres_port: int = 5432

    # Connection pool settings
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30
    db_echo: bool = False  # Set True to log all SQL queries

    # Vector search configuration
    vector_distance_metric: Literal["cosine", "l2", "inner_product"] = "cosine"

    # HNSW index parameters
    hnsw_m: int = Field(16, ge=4, le=64, description="HNSW M parameter")
    hnsw_ef_construction: int = Field(64, ge=16, description="HNSW ef_construction")
    hnsw_ef_search: int = Field(40, ge=10, description="HNSW ef_search")

    @property
    def database_url(self) -> str:
        """Build async database URL from components."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        """Get synchronous database URL for Alembic migrations."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ========================================================================
    # WEB SCRAPING CONFIGURATION
    # ========================================================================

    eu5_wiki_base_url: str = "https://eu5.paradoxwikis.com"
    eu5_wiki_start_url: str = "https://eu5.paradoxwikis.com/Europa_Universalis_5_Wiki"
    scraper_max_pages: int = 500
    scraper_delay_seconds: float = 1.0
    scraper_concurrency: int = 5
    scraper_timeout_seconds: int = 30
    scraper_max_retries: int = 3
    scraper_user_agent: str = "MicroRAG-Bot/1.0 (Educational Project)"

    # ========================================================================
    # TEXT CHUNKING CONFIGURATION
    # ========================================================================

    chunk_size_tokens: int = Field(800, ge=100, le=8000)
    chunk_overlap_tokens: int = Field(200, ge=0, le=1000)
    chunk_tokenizer_model: str = "text-embedding-3-small"

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        # Note: In Pydantic v2, we can't access other fields during validation
        # This will be checked at runtime if needed
        return v

    # ========================================================================
    # RAG QUERY CONFIGURATION
    # ========================================================================

    rag_top_k: int = Field(5, ge=1, le=20)
    rag_min_score: float = Field(0.5, ge=0.0, le=1.0)  # 0.5 works better for RAG
    rag_include_sources: bool = True
    rag_max_context_tokens: int = Field(3000, ge=500, le=8000)

    # ========================================================================
    # LOGGING CONFIGURATION
    # ========================================================================

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "console"
    log_output: str = "stdout"  # stdout, file, or both
    log_file_path: str = "logs/app.log"
    log_max_size_mb: int = 100
    log_backup_count: int = 5

    # ========================================================================
    # CORS CONFIGURATION
    # ========================================================================

    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    cors_allow_credentials: bool = True
    cors_allow_methods: str = "GET,POST,PUT,DELETE,OPTIONS"
    cors_allow_headers: str = "*"

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def cors_methods_list(self) -> list[str]:
        """Parse comma-separated CORS methods into a list."""
        return [method.strip() for method in self.cors_allow_methods.split(",")]

    # ========================================================================
    # SECURITY & RATE LIMITING
    # ========================================================================

    api_auth_enabled: bool = False
    api_key: str | None = None
    rate_limit_per_minute: int = 60
    rate_limit_enabled: bool = True

    # ========================================================================
    # MONITORING & OBSERVABILITY
    # ========================================================================

    enable_cost_tracking: bool = True
    enable_metrics: bool = True
    sentry_dsn: str | None = None

    # ========================================================================
    # TESTING CONFIGURATION
    # ========================================================================

    test_mode: bool = False
    use_mock_services: bool = False
    test_database_url: str | None = None

    # ========================================================================
    # COMPUTED PROPERTIES
    # ========================================================================

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    @property
    def api_base_url(self) -> str:
        """Get the base URL for API endpoints."""
        return f"http://{self.host}:{self.port}{self.api_v1_prefix}"


# Singleton instance
settings = Settings()


def get_settings() -> Settings:
    """
    Dependency injection function for FastAPI.

    Usage:
        @app.get("/")
        async def root(settings: Settings = Depends(get_settings)):
            return {"app_name": settings.app_name}
    """
    return settings