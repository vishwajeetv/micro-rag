# Micro-RAG

RAG system built for learning. With a sample implementation for Europa Universalis 5 wiki website.

## Tech Stack

- **Backend**: FastAPI, PostgreSQL + pgvector, OpenAI (GPT-5.1, embeddings)
- **Frontend**: React + Vite + TypeScript + MUI
- **Evaluation**: RAGAS

## Setup

### Prerequisites
- Docker
- Python 3.11
- Node.js 18+
- OpenAI API key

### 1. Database
```bash
cd backend
docker-compose up -d postgres
```

### 2. Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add OPENAI_API_KEY
```

### 3. Scrape & Index
```bash
cd backend
PYTHONPATH=. python scripts/test_pipeline.py
```

### 4. Run Backend
```bash
cd backend
source .venv/bin/activate
PYTHONPATH=. uvicorn app.main:app --reload --port 8000
```

### 5. Run Frontend
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

## Evaluation

Run RAGAS evaluation:
```bash
cd backend
PYTHONPATH=. python evaluation/run_eval.py
```

View reports:
```bash
cd backend/evaluation
python serve_report.py
```
Open http://localhost:8080

## API Docs

- Swagger: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
