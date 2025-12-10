# ============================================================================
# Development Dockerfile for FastAPI Backend
# ============================================================================
# Simple, single-stage Dockerfile for local development
# Uses uv for blazing fast package installation
# Just run: docker-compose up
# ============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies
# - build-essential: for compiling Python packages with C extensions
# - libpq-dev: PostgreSQL client library (needed for psycopg2)
# - curl: for health checks and uv installation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv - ultra-fast Python package installer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy requirements first (layer caching - only reinstall if requirements change)
COPY requirements.txt .

# Install Python dependencies with uv (10-100x faster than pip)
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Run uvicorn with hot reload (code changes auto-reload)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]