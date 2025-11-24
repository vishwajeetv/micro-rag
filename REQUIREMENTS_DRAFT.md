# Backend Requirements with Detailed Explanations

Copy this content to `backend/requirements.txt` when ready.

```txt
# ============================================================================
# WEB FRAMEWORK
# ============================================================================

# FastAPI - Modern, high-performance web framework for building APIs
# Why: Provides async support, automatic API documentation (Swagger/OpenAPI),
# data validation via Pydantic, and excellent developer experience
fastapi==0.104.1

# Uvicorn - Lightning-fast ASGI server implementation
# Why: Needed to run FastAPI applications. The [standard] extra includes
# performance optimizations like uvloop (faster event loop) and httptools
uvicorn[standard]==0.24.0

# Python Multipart - Parser for multipart/form-data
# Why: Required for handling file uploads if we want to add file-based
# document ingestion in the future (e.g., upload PDFs to add to RAG)
python-multipart==0.0.6


# ============================================================================
# DATA VALIDATION & CONFIGURATION
# ============================================================================

# Pydantic - Data validation using Python type annotations
# Why: FastAPI uses this for request/response validation. Also helps catch
# bugs early by validating data types at runtime
pydantic==2.5.0

# Pydantic Settings - Settings management for Pydantic
# Why: Manages environment variables and configuration in a type-safe way.
# Reads from .env files and validates all config values (API keys, URLs, etc.)
pydantic-settings==2.1.0


# ============================================================================
# LANGCHAIN ECOSYSTEM - RAG Orchestration
# ============================================================================

# LangChain - Framework for building LLM applications
# Why: Provides abstractions for RAG pipelines, text splitting (chunking),
# prompt templates, and chains that connect multiple steps together.
# Saves you from writing boilerplate code
langchain==0.1.0

# LangChain OpenAI - OpenAI integrations for LangChain
# Why: Provides ready-to-use wrappers for OpenAI's embeddings and chat models.
# Handles API calls, retries, and error handling for you
langchain-openai==0.0.2

# LangChain Community - Community-contributed integrations
# Why: Contains additional integrations and utilities, including document
# loaders and text splitters we'll use for chunking
langchain-community==0.0.10


# ============================================================================
# VECTOR DATABASE
# ============================================================================

# Pinecone Client - Official Pinecone SDK
# Why: Connects to Pinecone cloud vector database. Handles vector upserts,
# queries, index management, and similarity search. We use Pinecone because
# it's managed (no infrastructure to maintain) and scales automatically
pinecone-client==3.0.0


# ============================================================================
# LLM PROVIDER
# ============================================================================

# OpenAI - Official OpenAI Python SDK
# Why: Provides access to GPT-4 (for text generation) and text-embedding-3-small
# (for creating embeddings). Handles authentication, rate limiting, and streaming
openai==1.6.1


# ============================================================================
# WEB SCRAPING
# ============================================================================

# BeautifulSoup4 - HTML/XML parser
# Why: Parses the HTML from Europa Universalis 5 Wiki pages. Extracts text
# content while ignoring navigation, ads, and other irrelevant elements.
# Easy to use for simple scraping tasks
beautifulsoup4==4.12.2

# aiohttp - Async HTTP client/server
# Why: Makes HTTP requests to download wiki pages. Async version means we can
# scrape multiple pages concurrently (faster than requests library).
# Works well with FastAPI's async nature
aiohttp==3.9.1

# lxml - Fast XML/HTML parser
# Why: BeautifulSoup can use different parsers. lxml is the fastest and most
# feature-complete. Also handles malformed HTML better than Python's built-in parser
lxml==4.9.3


# ============================================================================
# ENVIRONMENT MANAGEMENT
# ============================================================================

# Python Dotenv - Load environment variables from .env files
# Why: Keeps secrets (API keys, DB credentials) out of code. Loads variables
# from .env file into os.environ. Works with pydantic-settings
python-dotenv==1.0.0


# ============================================================================
# LOGGING & MONITORING
# ============================================================================

# Structlog - Structured logging library
# Why: Produces JSON logs with structured data (timestamps, request IDs, context).
# Much better than print() or basic logging. Makes debugging production issues
# easier and enables log aggregation tools (ELK, Datadog) to parse logs
structlog==23.2.0


# ============================================================================
# TESTING
# ============================================================================

# Pytest - Testing framework
# Why: Most popular Python testing framework. Simple syntax, powerful fixtures,
# great plugin ecosystem. Used for unit, integration, and e2e tests
pytest==7.4.3

# Pytest Asyncio - Async test support for pytest
# Why: Allows testing async functions (needed since FastAPI uses async/await).
# Without this, you can't test async endpoints or services
pytest-asyncio==0.21.1

# Pytest Cov - Coverage plugin for pytest
# Why: Measures test coverage (% of code tested). Helps identify untested code.
# Generates coverage reports to ensure we hit >80% coverage
pytest-cov==4.1.0

# HTTPX - Async HTTP client
# Why: FastAPI's TestClient uses this under the hood. Also useful for integration
# tests that call external APIs (OpenAI, Pinecone) where you mock responses
httpx==0.25.2


# ============================================================================
# CODE QUALITY & FORMATTING
# ============================================================================

# Black - Opinionated code formatter
# Why: Automatically formats Python code to a consistent style. No arguments
# about formatting in code reviews. Just run black and you're done
black==23.12.0

# Flake8 - Linter for style guide enforcement
# Why: Checks for PEP 8 compliance (Python style guide), unused imports,
# undefined variables, and other code quality issues. Catches bugs early
flake8==6.1.0

# Mypy - Static type checker
# Why: Checks type hints at development time (before running code). Catches
# type errors like passing a string where an int is expected. Makes refactoring safer
mypy==1.7.1

# Pre-commit - Git hook framework
# Why: Runs checks (black, flake8, mypy) automatically before each commit.
# Prevents committing code that doesn't pass quality checks. Saves CI time
pre-commit==3.6.0


# ============================================================================
# UTILITIES
# ============================================================================

# Tiktoken - OpenAI's tokenizer
# Why: Counts tokens in text accurately (using OpenAI's tokenization method).
# Critical for staying within chunk size limits and LLM context windows.
# Example: "Hello world" is 2 tokens, but "Hello" might be 1 token
tiktoken==0.5.2

# Tenacity - Retry library
# Why: Implements retry logic with exponential backoff for API calls.
# If OpenAI or Pinecone API fails temporarily (network issue, rate limit),
# tenacity automatically retries instead of crashing. Production-essential
tenacity==8.2.3
```

---

## Summary of Dependencies by Category

**Total: 23 dependencies**

- **Web Framework:** 3 packages (FastAPI, Uvicorn, python-multipart)
- **Validation:** 2 packages (Pydantic, pydantic-settings)
- **LangChain:** 3 packages (langchain, langchain-openai, langchain-community)
- **Vector DB:** 1 package (pinecone-client)
- **LLM:** 1 package (openai)
- **Scraping:** 3 packages (beautifulsoup4, aiohttp, lxml)
- **Config:** 1 package (python-dotenv)
- **Logging:** 1 package (structlog)
- **Testing:** 4 packages (pytest, pytest-asyncio, pytest-cov, httpx)
- **Code Quality:** 4 packages (black, flake8, mypy, pre-commit)
- **Utilities:** 2 packages (tiktoken, tenacity)
