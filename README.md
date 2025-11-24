# Micro-RAG: Europa Universalis 5 Wiki RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for answering questions about Europa Universalis 5 game mechanics using the official wiki.

## Tech Stack

- **Backend**: FastAPI + Python 3.11
- **Vector Database**: PostgreSQL + pgvector (HNSW index)
- **LLM**: OpenAI GPT-4 + text-embedding-3-small
- **Orchestration**: LangChain
- **Frontend**: React + Vite + TypeScript (coming soon)
- **Infrastructure**: Docker + Docker Compose

## Quick Start

### 1. Prerequisites

- Docker & Docker Compose installed
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### 2. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
# Required: OPENAI_API_KEY=sk-...
# Optional: Customize other settings (defaults are provided)
```

### 3. Start Services

```bash
# Start PostgreSQL + pgvector and FastAPI backend
docker-compose up

# First run will:
# 1. Download Docker images (~2 min)
# 2. Install Python dependencies (~3 min)
# 3. Initialize PostgreSQL with pgvector extension
# 4. Start FastAPI server on http://localhost:8000
```

### 4. Verify Installation

```bash
# Check if services are running
docker-compose ps

# Check API health
curl http://localhost:8000/api/health

# View logs
docker-compose logs -f backend
```

## Project Structure

```
micro-rag/
├── backend/
│   ├── app/
│   │   ├── api/          # API routes
│   │   ├── core/         # Config, logging
│   │   ├── models/       # Pydantic schemas
│   │   └── services/     # Business logic (scraper, embeddings, RAG)
│   ├── tests/            # Tests
│   ├── scripts/          # Database init scripts
│   ├── requirements.txt  # Python dependencies
│   └── Dockerfile        # Backend container
├── frontend/             # React app (coming soon)
├── docker-compose.yml    # Service orchestration
├── .env.example          # Environment template
└── README.md            # This file
```

## Development

### Running Locally (Without Docker)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
cd backend
pip install -r requirements.txt

# Start PostgreSQL separately (or use Docker for just Postgres)
docker-compose up postgres

# Run FastAPI with hot reload
uvicorn app.main:app --reload
```

### Database Management

```bash
# Access PostgreSQL CLI
docker-compose exec postgres psql -U microrag -d microrag

# Run SQL commands
SELECT * FROM pg_extension WHERE extname = 'vector';

# Stop services
docker-compose down

# Stop and remove volumes (WARNING: deletes data!)
docker-compose down -v
```

## Configuration

All settings are in `.env` file:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ | - | OpenAI API key |
| `POSTGRES_USER` | ✅ | - | Database username |
| `POSTGRES_PASSWORD` | ✅ | - | Database password |
| `POSTGRES_DB` | ✅ | - | Database name |
| `CHUNK_SIZE_TOKENS` | ❌ | 800 | Chunk size for text splitting |
| `RAG_TOP_K` | ❌ | 5 | Number of chunks to retrieve |
| `LOG_LEVEL` | ❌ | INFO | Logging level |

See `.env.example` for full list of configuration options.

## What's Next? (Phase 2)

- [ ] Build FastAPI application structure (main.py, routes)
- [ ] Add health check and basic endpoints
- [ ] Set up database models with SQLAlchemy
- [ ] Configure Alembic for migrations

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## License

Educational project - MIT License

## Troubleshooting

**Port already in use:**
```bash
# Change ports in .env
POSTGRES_PORT=5433
PORT=8001
```

**Database connection failed:**
```bash
# Check if Postgres is healthy
docker-compose ps postgres

# View Postgres logs
docker-compose logs postgres
```

**Hot reload not working:**
```bash
# Check volume mount in docker-compose.yml
# Make sure source code is mounted correctly
```