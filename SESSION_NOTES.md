# Micro-RAG Learning Project - Session Notes

**Date:** 2025-11-24
**Last Updated:** 2025-11-24 (Session 2)
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

### **Phase 2: Backend Core (Day 2)**
- Build FastAPI app with layered architecture
- Add middleware (CORS, error handling, logging)
- Create Pydantic models
- Set up pytest and pre-commit hooks

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

## 📂 Project Structure (Session 2)

```
micro-rag/
├── .venv/                           # Python virtual environment (existing)
├── .git/                            ✅ Git repository initialized
├── .gitignore                       ✅ Created
├── .env.example                     ✅ Created (HNSW config, PostgreSQL)
├── .env                             ✅ Created (copy of .env.example)
├── docker-compose.yml               ✅ Created (PostgreSQL + pgvector + backend)
├── README.md                        ✅ Created (quick start guide)
├── SESSION_NOTES.md                 ✅ Updated (this file)
├── REQUIREMENTS_DRAFT.md            ✅ Created (Session 1)
├── backend/
│   ├── app/
│   │   ├── __init__.py             ✅ Created
│   │   ├── main.py                 ⏳ Next - Phase 2
│   │   ├── api/
│   │   │   ├── __init__.py         ✅ Created
│   │   │   └── routes.py           ⏳ To create
│   │   ├── core/
│   │   │   ├── __init__.py         ✅ Created
│   │   │   ├── config.py           ✅ Created (Pydantic Settings)
│   │   │   └── logging.py          ✅ Created (Structlog)
│   │   ├── models/
│   │   │   ├── __init__.py         ✅ Created
│   │   │   ├── database.py         ⏳ To create (SQLAlchemy models)
│   │   │   └── schemas.py          ⏳ To create (Pydantic schemas)
│   │   └── services/
│   │       ├── __init__.py         ✅ Created
│   │       ├── scraper.py          ⏳ To create
│   │       ├── embeddings.py       ⏳ To create
│   │       ├── vector_store.py     ⏳ To create
│   │       └── rag_engine.py       ⏳ To create
│   ├── tests/
│   │   └── __init__.py             ✅ Created
│   ├── scripts/
│   │   └── init_pgvector.sql       ✅ Created (auto-enable pgvector)
│   ├── requirements.txt            ✅ Created (PostgreSQL + pgvector)
│   └── Dockerfile                  ✅ Created (dev-only, simple)
└── frontend/
    ├── src/
    │   ├── components/             ✅ Created (empty)
    │   ├── pages/                  ✅ Created (empty)
    │   └── utils/                  ✅ Created (empty)
    ├── package.json                ⏳ To create (Vite + React + TS)
    └── Dockerfile                  ⏳ To create
```

---

## 📝 Next Steps (Phase 2 - Backend Core)

**Before you start:**
- Add your `OPENAI_API_KEY` to `.env` file

**Phase 2 Tasks:**
1. Create `backend/app/main.py` - FastAPI application with lifespan events
2. Add CORS middleware for frontend integration
3. Add request logging middleware
4. Create database connection with SQLAlchemy async engine
5. Create `backend/app/models/database.py` - SQLAlchemy base and database models
6. Create health check endpoint (`GET /api/health`)
7. Test Docker setup: `docker-compose up`

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

**What We Built Today:**
- ✅ PostgreSQL + pgvector setup (replaced Pinecone)
- ✅ HNSW index configuration (better than IVFFlat)
- ✅ Complete requirements.txt with all dependencies
- ✅ Pydantic Settings with env-first pattern
- ✅ Docker Compose with simple dev setup
- ✅ Structured logging (structlog)
- ✅ Git repository initialized
- ✅ Comprehensive .gitignore
- ✅ README with quick start guide

**Next Session:** Phase 2 - Backend Core (FastAPI app, database models, health check)

**Session saved! You can restart your laptop safely. See you tomorrow! 🎮**
