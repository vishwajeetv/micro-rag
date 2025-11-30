# Micro-RAG Learning Project - Session Notes

**Date:** 2025-11-24
**Last Updated:** 2025-12-01 (Session 4 - Phase 5 complete)
**Project:** Europa Universalis 5 Wiki RAG System
**Tech Stack:** Python + FastAPI + PostgreSQL/pgvector + OpenAI + React (Vite) + TypeScript

---

## 📊 Your Quiz Assessment Results

**Score: 12/12 (100%)** - Excellent conceptual understanding!

### Quiz Results:
✅ **Embeddings:** Numerical vectors that capture semantic meaning
✅ **Chunking:** All reasons (context limits, embedding constraints, retrieval precision)
✅ **RAG Basics:** Retrieves relevant documents from knowledge base
✅ **Vector Search:** Similarity search using vector distance/similarity
✅ **Model Separation:** Embedding model converts text to vectors; LLM generates answers
✅ **Chunk Overlap:** Prevents context loss at chunk boundaries
✅ **Prompt Structure:** System prompt + context chunks + user question + instructions
✅ **Production Challenges:** All of these and more
✅ **Embedding Models:** Small is faster/cheaper; Large is more accurate but costs more
✅ **Retrieval Issues:** Possible issues: poor chunking, not enough context, or info doesn't exist
✅ **Evaluation:** Retrieval metrics (precision/recall) + generation metrics (answer quality)
✅ **Security:** Prompt injection attack - could expose data or bypass intended behavior

**Assessment:** You have strong conceptual foundations. Ready to build production-quality code.

---

## 🎯 Project Goals

**End Goal:** Build a production-ready RAG system that:
1. Scrapes https://eu5.paradoxwikis.com/Europa_Universalis_5_Wiki
2. Stores content as embeddings in PostgreSQL + pgvector (local, HNSW index)
3. Enables RAG-based chat about EU5 game mechanics
4. Has a React chat UI for user interaction
5. Follows production best practices (testing, monitoring, security, Docker)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    REACT UI (Vite + TypeScript)                 │
│  Components: ChatInterface, MessageList, SourceCitations        │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP/SSE
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND                              │
│  Endpoints: /api/scrape, /api/chat, /api/status, /api/health   │
│  Modules: Scraper, Chunking, Embeddings, RAG Pipeline          │
└─────────┬───────────────────────┬───────────────────────────────┘
          │                       │
          │ Scrape & Process      │ Query & Generate
          ▼                       ▼
┌──────────────────┐    ┌─────────────────────────────────────┐
│  EU5 Wiki Pages  │    │  POSTGRESQL + PGVECTOR (Local)      │
│  (HTML Content)  │    │  - Vector data type (embeddings)    │
└──────────────────┘    │  - HNSW index (similarity search)   │
                        │  - Document metadata                │
                        └──────────┬──────────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────────────┐
                        │  OPENAI API              │
                        │  - GPT-4 (generation)    │
                        │  - text-embedding-3-small│
                        └──────────────────────────┘
```

---

## 📋 Implementation Plan (20 Days)

### **Phase 1: Project Foundation (Day 1)** ✅ COMPLETED
1. ✅ Create project structure (backend + frontend directories)
2. ✅ Set up requirements.txt with detailed dependency explanations (PostgreSQL + pgvector)
3. ✅ Configure environment management (.env.example, config.py with Pydantic Settings)
4. ✅ Initialize git repository with .gitignore
5. ✅ Create Docker setup (Dockerfile, docker-compose.yml with PostgreSQL + pgvector)
6. ✅ Set up structured logging configuration (structlog with JSON/console modes)

### **Phase 2: Backend Core (Day 2)** ✅ COMPLETED
- ✅ Build FastAPI app with layered architecture
- ✅ Add middleware (CORS, error handling, logging)
- ✅ Create Pydantic models
- ⏳ Set up pytest and pre-commit hooks (deferred)

### **Phase 3: Web Scraping (Days 3-4)**
- Implement EU5 Wiki scraper (BeautifulSoup + aiohttp)
- Add rate limiting and retry logic
- Background task management

### **Phase 4: Chunking & Text Processing (Day 5)**
- RecursiveCharacterTextSplitter from LangChain
- Chunk size: 800 tokens, overlap: 200 tokens
- Metadata extraction

### **Phase 5: Embeddings & Vector Store (Days 6-7)**
- PostgreSQL + pgvector setup with SQLAlchemy models
- OpenAI embeddings (text-embedding-3-small)
- HNSW index creation and optimization
- Batch upsert implementation

### **Phase 6: RAG Query Engine (Days 8-10)**
- Semantic search with pgvector (cosine similarity)
- Prompt engineering (system + context + query)
- GPT-4 integration with streaming
- Source citations

### **Phase 7: API Endpoints (Day 11)**
- POST /api/scrape, /api/chat
- GET /api/status/{job_id}, /api/index/stats
- OpenAPI documentation

### **Phase 8: React Frontend (Days 12-14)**
- Vite + React + TypeScript setup
- Chat interface with streaming
- Source citation display
- Admin panel for scraping

### **Phase 9: Observability (Day 15)**
- Structured logging (JSON)
- Metrics collection
- Cost tracking

### **Phase 10: Security (Day 16)**
- Input sanitization
- Prompt injection detection
- Rate limiting
- API authentication

### **Phase 11: Testing (Days 17-18)**
- Unit, integration, e2e tests
- >80% coverage
- CI/CD pipeline (GitHub Actions)

### **Phase 12: Deployment & Docs (Days 19-20)**
- Docker multi-stage builds
- Comprehensive README
- Deployment guide

---

## 🛠️ Tech Stack Details

### Backend
- **FastAPI 0.104+** - Modern async web framework
- **Python 3.11+** - Latest stable Python
- **LangChain** - RAG orchestration framework
- **PostgreSQL + pgvector** - Local vector database with HNSW index
- **SQLAlchemy 2.0+** - Async ORM for database operations
- **OpenAI API** - GPT-4 + embeddings
- **BeautifulSoup4 + aiohttp** - Web scraping
- **Structlog** - Structured logging

### Frontend
- **React 18** - UI library
- **Vite** - Fast build tool
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling

### DevOps
- **Docker** - Containerization
- **GitHub Actions** - CI/CD
- **pytest** - Testing framework

---

## 📂 Project Structure (Updated - Session 4)

```
micro-rag/
├── .git/                            ✅ Git repository
├── .gitignore                       ✅ Created
├── README.md                        ✅ Quick start guide
├── SESSION_NOTES.md                 ✅ This file
├── REQUIREMENTS_DRAFT.md            ✅ Initial requirements
├── backend/                         # All backend code lives here
│   ├── .venv/                       ✅ Python 3.11 virtual environment
│   ├── .env                         ✅ Environment config (API keys, DB)
│   ├── .env.example                 ✅ Template for .env
│   ├── docker-compose.yml           ✅ PostgreSQL + pgvector
│   ├── Dockerfile                   ✅ Backend container
│   ├── requirements.txt             ✅ Python dependencies
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 ✅ FastAPI app + middleware
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   └── routes.py           ✅ Collection + API endpoints
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py           ✅ Pydantic Settings
│   │   │   └── logging.py          ✅ Structlog (fixed)
│   │   ├── models/
│   │   │   ├── __init__.py         ✅ Exports
│   │   │   ├── database.py         ✅ SQLAlchemy + pgvector + Collections
│   │   │   └── schemas.py          ✅ Pydantic schemas
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── scraper.py          ✅ Async wiki scraper (Phase 3)
│   │       ├── chunker.py          ✅ Token-based chunker (Phase 4)
│   │       ├── embeddings.py       ✅ OpenAI embeddings (Phase 5)
│   │       ├── vector_store.py     ✅ DB + search (Phase 5)
│   │       └── rag_engine.py       ⏳ Phase 6
│   ├── scripts/
│   │   └── init_pgvector.sql       ✅ Auto-enable pgvector
│   └── tests/
│       ├── __init__.py
│       ├── test_chunker.py         ✅ 5 tests
│       ├── test_embeddings.py      ✅ 4 tests
│       └── test_vector_store.py    ✅ 2 tests
└── frontend/                        ⏳ Phase 8
    └── src/
```

---

## 📝 Next Steps (Phase 3 - Web Scraping)

**Before you start:**
- Make sure your `OPENAI_API_KEY` in `.env` is a real key (not placeholder)
- Start Docker: `docker-compose up -d postgres` (or start Docker Desktop first)

**Phase 3 Tasks:**
1. Create `backend/app/services/scraper.py` - EU5 Wiki scraper
2. Implement async HTTP client with aiohttp
3. Parse wiki pages with BeautifulSoup
4. Add rate limiting and retry logic with tenacity
5. Extract clean text from wiki HTML
6. Background task management for long-running scrapes
7. Test scraping a few pages

---

## 📚 Key Learning Questions to Research (Optional Deep Dives)

While the implementation will teach you most concepts, here are topics to research if you want deeper understanding:

### Embeddings & Vector Search
1. What are embeddings and how do they capture semantic meaning?
2. Cosine similarity vs dot product vs euclidean distance?
3. Different embedding models and their tradeoffs?

### Chunking Strategy
4. Why chunk documents? What happens if chunks are too small/large?
5. What's chunk overlap and why is it important?
6. Different chunking strategies (fixed-size, sentence-based, semantic)?

### RAG Fundamentals
7. What is RAG and how is it different from fine-tuning?
8. Retrieval parameters (top_k, score threshold)?
9. LLM context window limitations?

### Vector Databases
10. PostgreSQL + pgvector vs dedicated vector DBs (Pinecone, Weaviate, Qdrant)?
11. HNSW vs IVFFlat indexes - tradeoffs and when to use each?
12. What metadata to store with vectors? Index optimization strategies?

### System Design
13. On-demand vs batch scraping?
14. Handling wiki updates?
15. Detecting and preventing LLM hallucinations?

### Prompt Engineering
16. How to structure RAG prompts?
17. Including source citations?
18. Protecting against prompt injection?

---

## 💡 Your Preferences (From Quiz)

- **Learning Style:** Learn concepts first, then build
- **LLM:** Cloud-based (OpenAI/Anthropic)
- **UI:** React.js with Vite (not Next.js)
- **Scope:** Full production setup

---

## 🚀 When You're Ready to Continue

Just say: **"Let's continue from where we left off"** and I'll:
1. Start Phase 2 (Backend Core - FastAPI application)
2. Reference this session document
3. Keep building step-by-step with explanations
4. **Use Sonnet model** (as requested)

---

## ✅ Session 2 Complete! (2025-11-24)

**What We Built:**
- ✅ PostgreSQL + pgvector setup (replaced Pinecone)
- ✅ HNSW index configuration (better than IVFFlat)
- ✅ Complete requirements.txt with all dependencies
- ✅ Pydantic Settings with env-first pattern
- ✅ Docker Compose with simple dev setup
- ✅ Structured logging (structlog)
- ✅ Git repository initialized
- ✅ Comprehensive .gitignore
- ✅ README with quick start guide

---

## ✅ Session 3 Complete! (2025-11-29 to 2025-11-30)

**What We Built:**
- ✅ `main.py` - FastAPI application with lifespan events
- ✅ CORS middleware for frontend integration
- ✅ Request logging middleware with request IDs
- ✅ Global exception handler
- ✅ `database.py` - SQLAlchemy async engine + pgvector models
  - Document model (stores wiki pages)
  - Chunk model (stores text chunks with embeddings)
  - ScrapeJob model (tracks scraping progress)
  - HNSW index configuration
  - Graceful DB connection handling (app starts without DB)
- ✅ `schemas.py` - Pydantic schemas for all endpoints
- ✅ `routes.py` - 8 API endpoints:
  - `GET /api/` - API info
  - `GET /api/health` - Health check
  - `GET /api/health/ready` - Readiness probe
  - `GET /api/health/live` - Liveness probe
  - `GET /api/index/stats` - Index statistics
  - `POST /api/scrape` - Start scraping (stub)
  - `GET /api/scrape/{job_id}` - Get scrape status
  - `POST /api/chat` - RAG chat (stub)

**Fixes Applied (2025-11-30):**
- ✅ Fixed structlog configuration (removed stdlib-specific processors)
- ✅ Made database connection graceful (app starts even without PostgreSQL)
- ✅ Removed strict `sk-` API key validation (supports Azure OpenAI)
- ✅ Restructured project: moved `.env`, `.venv`, `docker-compose.yml` to `backend/`
- ✅ Recreated venv with Python 3.11 (3.13 had wheel issues)

**To Run:**
```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload
# Visit http://localhost:8000/api/docs
```

**Next:** Phase 3 - Web Scraping

---

## ✅ Session 4 Progress (2025-11-30)

**What We Built:**

### Collection Support (Multi-Site RAG)
- ✅ Added `Collection` model to `database.py`
- ✅ Updated `Document` and `ScrapeJob` to reference collections
- ✅ Added Collection CRUD endpoints to `routes.py`:
  - `POST /api/collections` - Create collection
  - `GET /api/collections` - List collections
  - `GET /api/collections/{slug}` - Get collection detail
  - `DELETE /api/collections/{slug}` - Delete collection
  - `POST /api/collections/{slug}/scrape` - Start scraping for collection
- ✅ `ChatRequest` now supports `collection_slug` for filtering

### Phase 3 - Web Scraping (COMPLETED)
- ✅ `backend/app/services/scraper.py` - Full async wiki scraper
  - `WikiScraper` class with async context manager
  - `aiohttp` for async HTTP requests
  - `BeautifulSoup` for HTML parsing
  - `tenacity` for retry with exponential backoff (3 attempts)
  - Rate limiting with configurable delay
  - Concurrent request limiting (semaphore)
  - Link extraction for breadth-first crawling
  - Content cleaning (removes navboxes, edit links, cookie banners)
  - Tested on EU5 Wiki - successfully scraped pages

**Key Classes:**
```python
# Data classes
ScrapedPage(url, title, content, content_hash, word_count, links)
ScrapeError(url, error, status_code)

# Main scraper
async with WikiScraper(base_url="https://eu5.paradoxwikis.com") as scraper:
    async for result in scraper.crawl(start_url="...", max_pages=100):
        # Process each page
```

**Test Results:**
- Main wiki page: 498 words, 50 links found
- Europa Universalis V page: 3,424 words
- Patch 1.0.X page: 16,814 words
- Content extraction working well (headers marked with ##)

**Database Schema with Collections:**
```
collections (parent)
    └── documents (scraped pages, FK to collection)
            └── chunks (text chunks with embeddings, FK to document)
    └── scrape_jobs (background job tracking, FK to collection)
```

### Phase 4 - Chunking & Text Processing (COMPLETED)
- ✅ `backend/app/services/chunker.py` - Token-based text chunker
  - `TextChunker` class with tiktoken (o200k_base encoding)
  - `count_tokens()` - Accurate token counting
  - `encode()`/`decode()` - Token conversion
  - `split_text()` - Sliding window with overlap
  - `split_by_headers()` - Splits on `## Header` markers (from scraper output)
  - `chunk_document()` - Main method combining both, returns metadata
- ✅ `backend/tests/test_chunker.py` - 5 minimal tests (all passing)
- ✅ Uses tiktoken `o200k_base` encoding (same as GPT-5.1 and text-embedding-3)

**Key Features:**
```python
chunker = TextChunker(chunk_size=800, chunk_overlap=200)
chunks = chunker.chunk_document(text, title="Page Title", url="https://...")

# Returns list of dicts:
# {
#     "content": "chunk text",
#     "chunk_index": 0,
#     "token_count": 150,
#     "char_count": 800,
#     "header": "Section Name"  # or None for intro
# }
```

**Research Notes (This Session):**
- **tiktoken** is OpenAI's tokenizer - `o200k_base` for modern models
- **Alternatives explored:**
  - LangChain `RecursiveCharacterTextSplitter` / `MarkdownHeaderTextSplitter`
  - LlamaIndex splitters
  - `semantic-text-splitter` (Rust-based)
  - Google Vertex AI RAG Engine (managed service)
- **Google vs OpenAI tooling:** OpenAI open-sourced tiktoken; Google has `countTokens` API but no offline tokenizer. Alternatives: Kitoken, ai-tokenizer
- **Embedding model:** `text-embedding-3-small` is still current (Nov 2025). GPT-5.1 Instant is chat model, not embedding.

### Phase 5 - Embeddings & Vector Store (COMPLETED)
- ✅ `backend/app/services/embeddings.py` - OpenAI embeddings service
  - `EmbeddingService` class with async OpenAI client
  - `embed_text()` - Single text → 1536-dim vector
  - `embed_batch()` - Batch texts → vectors (efficient)
  - Uses `text-embedding-3-small` ($0.02/M tokens)
- ✅ `backend/app/services/vector_store.py` - Document storage + search
  - `VectorStore` class (needs AsyncSession from FastAPI DI)
  - `ingest_document()` - Scrape → chunk → embed → store in DB
  - `search()` - Query → cosine similarity via pgvector
  - `get_stats()` - Document/chunk counts
- ✅ `backend/tests/test_embeddings.py` - 4 tests (mocked OpenAI)
- ✅ `backend/tests/test_vector_store.py` - 2 tests (mocked DB)

**Total Tests: 11 passing**

**Embedding Model Decision:**
- Chose `text-embedding-3-small` (1536 dims, $0.02/M tokens)
- `text-embedding-3-large` is 6.5x more expensive, marginal accuracy gain
- Can upgrade later if needed

**Next Steps:**
- Phase 6: RAG Query Engine (prompt engineering + GPT-4 integration)
