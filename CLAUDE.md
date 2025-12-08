# Claude Code Session Notes

## User Preferences

### Performance Settings
- **Maximize parallel tool calls** - Always use maximum parallelism when calling tools
- **Large token usage is OK** - Don't be conservative with tokens; prioritize thoroughness
- **Hardware**: MacBook Pro M4 Pro 16", 24GB RAM, 20-core GPU
- **Use hardware efficiently** - Leverage M4 Pro capabilities for fast parallel operations

### Communication Style
- Comprehensive and detailed responses preferred
- Include code examples when relevant
- Don't abbreviate or truncate explanations
- **MVP-first approach** - Build bare minimum first, add complexity later
- **Simplify for first-time builders** - User is building agentic AI for the first time

### Project Context
- This is a learning project for Vishwajeet (Engineering Manager)
- Goal: Build baseline framework for AI automation agents
- Build for learning to guide team better
- Uses: OpenAI for LLMs (not local), but local compute for framework operations
- **First time building agentic AI** - Keep explanations simple and beginner-friendly

### Tech Stack Preferences
- Backend: Python/FastAPI (async)
- Frontend: React/TypeScript
- Database: PostgreSQL + pgvector
- LLM: OpenAI API (GPT-4o/GPT-5.1)
- No local LLM inference needed (Ollama installed but not primary)

## Development Notes

### Plans
- **`AGENTIC_MVP_PLAN.md`** - Simple MVP plan (START HERE)
- **`AGENTIC_FRAMEWORK_PLAN.md`** - Full comprehensive plan (reference later)

### Key Files
- `backend/app/services/rag_engine.py` - Core RAG pipeline
- `backend/app/services/vector_store.py` - Hybrid search
- `backend/app/core/config.py` - Settings management
- `backend/app/api/routes.py` - API endpoints with SSE streaming

### Next Steps (from plan)
1. Phase 1: LLM & Tool abstractions
2. Phase 2: Agent core with ReAct pattern
3. Phase 3: Memory management
4. Phase 4: Guardrails & observability
5. Phase 5: Multi-agent & workflows
6. Phase 6: Customer deployment

## Commands Reference
```bash
# Start PostgreSQL
cd backend && docker-compose up -d postgres

# Run backend
cd backend && source .venv/bin/activate && PYTHONPATH=. uvicorn app.main:app --reload --port 8000

# Run frontend
cd frontend && npm run dev

# Run evaluation
cd backend && PYTHONPATH=. python evaluation/run_eval.py
```
