# Agentic AI Framework Evolution Plan

> **From micro-rag to Enterprise Agentic Framework**
>
> A comprehensive plan for transforming the micro-rag codebase into a baseline framework for building AI automation agents and agentic workflows for customers.

---

## Executive Summary

### Current State
Your `micro-rag` is a well-architected RAG system with:
- Clean layered architecture (API → Services → Models → Data)
- Hybrid search (pgvector + keyword boosting)
- Streaming SSE support
- Full async/await throughout
- Strong patterns: dependency injection, factory pattern, service abstraction
- RAGAS evaluation infrastructure
- React frontend with real-time streaming

### Target State
An **enterprise-ready agentic AI framework** that enables:
- Rapid deployment of AI agents for customers
- Multi-step reasoning and task execution
- Tool orchestration (including RAG as a tool)
- Memory management (short, working, long-term)
- Multi-agent collaboration
- Human-in-the-loop workflows
- Production observability and guardrails

### Why This Approach?
Rather than adopting a framework like Agno wholesale, building on your existing codebase provides:
1. **Deep understanding** - You'll know every line of code
2. **Customization** - Tailor exactly to your customer needs
3. **No vendor lock-in** - Full control over evolution
4. **Learning** - Best way to guide your team

---

## Part 1: Architecture Analysis

### 1.1 What You Already Have (Reusable Assets)

| Component | Location | Reuse Strategy |
|-----------|----------|----------------|
| **Async Database Layer** | `models/database.py` | Extend for agent memory/state |
| **Config Management** | `core/config.py` | Add agent-specific settings |
| **SSE Streaming** | `api/routes.py:604` | Foundation for agent event streaming |
| **Vector Search** | `services/vector_store.py` | Becomes `RAGSearchTool` |
| **Embedding Service** | `services/embeddings.py` | Shared across tools |
| **Chunking** | `services/chunker.py` | Document processing for knowledge |
| **Job Tracking** | `models/database.py:394` | Pattern for `AgentJob` |
| **RAGAS Evaluation** | `evaluation/run_eval.py` | Extend for agent evaluation |
| **React Streaming UI** | `frontend/src/Chat.tsx` | Adapt for agent UI |

### 1.2 What Needs to Be Built

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AGENTIC FRAMEWORK                                │
├─────────────────────────────────────────────────────────────────────────┤
│  NEW LAYERS (to build)                                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │   Agents    │ │   Tools     │ │   Memory    │ │  Guardrails │       │
│  │   System    │ │   Registry  │ │   Manager   │ │   Layer     │       │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                       │
│  │  Workflow   │ │   LLM       │ │  Telemetry  │                       │
│  │   Engine    │ │  Providers  │ │   & Tracing │                       │
│  └─────────────┘ └─────────────┘ └─────────────┘                       │
├─────────────────────────────────────────────────────────────────────────┤
│  EXISTING LAYERS (extend)                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  RAG Services (VectorStore, Embeddings, Chunker, RAGEngine)     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Core Infrastructure (Config, Logging, Database, API)            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 2: Framework Options Analysis

### 2.1 Build vs Integrate Decision Matrix

| Approach | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **Build from scratch** | Full control, deep learning | Time-consuming | Core patterns only |
| **Agno** | Fast, 529x faster than LangGraph, memory built-in | Less learning, dependency | Consider for production |
| **LangGraph** | Industry standard, visualizable | Steeper learning curve, verbose | Learn patterns, don't depend |
| **Hybrid** | Best of both worlds | Complexity | **Recommended** |

### 2.2 Recommended Hybrid Approach

1. **Build core abstractions yourself** (Tools, Agents, Memory) - for learning
2. **Study Agno patterns** - for best practices
3. **Use Agno in production** (optional) - for speed when needed
4. **Maintain compatibility** - your tools should work with Agno if needed

---

## Part 3: Detailed Technical Design

### 3.1 Directory Structure Evolution

```
micro-rag/  →  agentic-framework/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── config.py           # EXTEND: Add agent settings
│   │   │   ├── logging.py          # EXTEND: Add trace IDs
│   │   │   └── exceptions.py       # NEW: Custom exceptions
│   │   │
│   │   ├── models/
│   │   │   ├── database.py         # EXTEND: Agent tables
│   │   │   ├── schemas.py          # EXTEND: Agent schemas
│   │   │   ├── memory.py           # NEW: Memory models
│   │   │   └── tenant.py           # NEW: Multi-tenant models
│   │   │
│   │   ├── services/
│   │   │   ├── rag/                # RESTRUCTURE existing
│   │   │   │   ├── __init__.py
│   │   │   │   ├── engine.py       # Current rag_engine.py
│   │   │   │   ├── vector_store.py # Current vector_store.py
│   │   │   │   ├── embeddings.py   # Current embeddings.py
│   │   │   │   └── chunker.py      # Current chunker.py
│   │   │   │
│   │   │   ├── llm/                # NEW: LLM abstraction
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py         # Abstract LLMProvider
│   │   │   │   ├── openai.py       # OpenAI implementation
│   │   │   │   ├── anthropic.py    # Claude implementation
│   │   │   │   └── registry.py     # Provider registry
│   │   │   │
│   │   │   ├── tools/              # NEW: Tool framework
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py         # BaseTool abstract class
│   │   │   │   ├── registry.py     # ToolRegistry
│   │   │   │   ├── decorators.py   # @tool decorator
│   │   │   │   └── builtin/
│   │   │   │       ├── __init__.py
│   │   │   │       ├── rag_search.py    # RAG as a tool
│   │   │   │       ├── web_search.py    # Web search tool
│   │   │   │       ├── calculator.py    # Math operations
│   │   │   │       ├── code_executor.py # Sandboxed code
│   │   │   │       └── http_client.py   # API calls
│   │   │   │
│   │   │   ├── agents/             # NEW: Agent framework
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py         # BaseAgent class
│   │   │   │   ├── executor.py     # AgentExecutor (ReAct loop)
│   │   │   │   ├── planner.py      # Task decomposition
│   │   │   │   ├── types.py        # AgentState, Action, Observation
│   │   │   │   └── prompts.py      # Agent prompt templates
│   │   │   │
│   │   │   ├── memory/             # NEW: Memory management
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py         # BaseMemory abstract
│   │   │   │   ├── conversation.py # Short-term chat history
│   │   │   │   ├── working.py      # Task context memory
│   │   │   │   └── long_term.py    # Persistent vector memory
│   │   │   │
│   │   │   ├── workflows/          # NEW: Workflow engine
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py         # Workflow, Step classes
│   │   │   │   ├── engine.py       # WorkflowEngine
│   │   │   │   └── templates/      # Pre-built workflows
│   │   │   │
│   │   │   ├── guardrails/         # NEW: Safety layer
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py         # BaseGuardrail
│   │   │   │   ├── input_filter.py # PII, injection defense
│   │   │   │   ├── output_filter.py# Content policy
│   │   │   │   └── cost_limiter.py # Budget management
│   │   │   │
│   │   │   └── telemetry/          # NEW: Observability
│   │   │       ├── __init__.py
│   │   │       ├── tracer.py       # Distributed tracing
│   │   │       ├── metrics.py      # Prometheus metrics
│   │   │       └── cost_tracker.py # Token/cost tracking
│   │   │
│   │   └── api/
│   │       ├── routes/             # RESTRUCTURE
│   │       │   ├── __init__.py
│   │       │   ├── health.py       # Health endpoints
│   │       │   ├── chat.py         # RAG chat endpoints
│   │       │   ├── collections.py  # Collection management
│   │       │   ├── agents.py       # NEW: Agent endpoints
│   │       │   └── workflows.py    # NEW: Workflow endpoints
│   │       └── middleware/         # NEW
│   │           ├── __init__.py
│   │           ├── auth.py         # Authentication
│   │           ├── rate_limit.py   # Rate limiting
│   │           └── tracing.py      # Request tracing
│   │
│   ├── evaluation/
│   │   ├── run_eval.py             # EXTEND: Agent evaluation
│   │   ├── agent_eval.py           # NEW: Agent metrics
│   │   └── workflow_eval.py        # NEW: Workflow metrics
│   │
│   └── tests/
│       ├── services/
│       │   ├── test_tools.py       # NEW
│       │   ├── test_agents.py      # NEW
│       │   └── test_memory.py      # NEW
│       └── integration/
│           └── test_agent_flow.py  # NEW
│
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── Chat.tsx            # EXTEND for agents
│       │   ├── AgentView.tsx       # NEW: Agent UI
│       │   └── WorkflowView.tsx    # NEW: Workflow UI
│       └── services/
│           └── api.ts              # EXTEND: Agent APIs
│
├── configs/                        # NEW: Customer configs
│   └── customers/
│       ├── acme_corp.yaml
│       └── example.yaml
│
└── docs/                           # NEW: Documentation
    ├── architecture.md
    ├── tools.md
    ├── agents.md
    └── deployment.md
```

### 3.2 Core Abstractions - Detailed Design

#### 3.2.1 LLM Provider Abstraction

```python
# services/llm/base.py
from abc import ABC, abstractmethod
from typing import AsyncGenerator
from pydantic import BaseModel

class LLMMessage(BaseModel):
    role: str  # system, user, assistant, tool
    content: str
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None

class LLMResponse(BaseModel):
    content: str | None
    tool_calls: list[dict] | None
    usage: dict  # prompt_tokens, completion_tokens, total_tokens
    model: str
    finish_reason: str  # stop, tool_calls, length

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Generate a completion."""
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[LLMMessage],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> AsyncGenerator[LLMResponse, None]:
        """Stream a completion."""
        pass

    @abstractmethod
    def to_tool_schema(self, tools: list["BaseTool"]) -> list[dict]:
        """Convert tools to provider-specific schema."""
        pass


# services/llm/openai.py
class OpenAIProvider(BaseLLMProvider):
    """OpenAI implementation."""

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        self.client = AsyncOpenAI(api_key=api_key or settings.openai_api_key)
        self.model = model

    async def complete(self, messages, tools=None, temperature=0.7, max_tokens=1000):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[m.model_dump() for m in messages],
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            tool_calls=response.choices[0].message.tool_calls,
            usage=response.usage.model_dump(),
            model=response.model,
            finish_reason=response.choices[0].finish_reason,
        )
```

#### 3.2.2 Tool Framework

```python
# services/tools/base.py
from abc import ABC, abstractmethod
from typing import Any, Type
from pydantic import BaseModel, Field

class ToolResult(BaseModel):
    """Result from tool execution."""
    success: bool
    output: Any
    error: str | None = None
    metadata: dict = Field(default_factory=dict)

class BaseTool(ABC):
    """Abstract base class for all tools."""

    name: str
    description: str
    parameters_schema: Type[BaseModel]  # Pydantic model for params

    @abstractmethod
    async def execute(self, **params) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def to_openai_schema(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema.model_json_schema(),
            }
        }

    def to_anthropic_schema(self) -> dict:
        """Convert to Anthropic tool use format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters_schema.model_json_schema(),
        }


# services/tools/decorators.py
def tool(name: str, description: str):
    """Decorator to create a tool from a function."""
    def decorator(func):
        # Introspect function signature to create parameter schema
        sig = inspect.signature(func)
        fields = {}
        for param_name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                fields[param_name] = (param.annotation, ...)

        ParamsModel = create_model(f"{name}Params", **fields)

        class DecoratedTool(BaseTool):
            name = name
            description = description
            parameters_schema = ParamsModel

            async def execute(self, **params):
                try:
                    result = await func(**params) if asyncio.iscoroutinefunction(func) else func(**params)
                    return ToolResult(success=True, output=result)
                except Exception as e:
                    return ToolResult(success=False, output=None, error=str(e))

        return DecoratedTool()
    return decorator


# services/tools/builtin/rag_search.py
class RAGSearchParams(BaseModel):
    query: str = Field(description="Search query")
    collection: str | None = Field(default=None, description="Collection to search in")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")

class RAGSearchTool(BaseTool):
    """Search the knowledge base using RAG."""

    name = "rag_search"
    description = "Search the knowledge base for information. Use this when you need to find specific information from documents."
    parameters_schema = RAGSearchParams

    def __init__(self, vector_store: VectorStore, db: AsyncSession):
        self.vector_store = vector_store
        self.db = db

    async def execute(self, query: str, collection: str | None = None, top_k: int = 5) -> ToolResult:
        # Resolve collection
        collection_id = None
        if collection:
            result = await self.db.execute(
                select(Collection).where(Collection.slug == collection)
            )
            col = result.scalar_one_or_none()
            if col:
                collection_id = col.id

        # Search
        chunks = await self.vector_store.search(
            query=query,
            collection_id=collection_id,
            limit=top_k,
        )

        return ToolResult(
            success=True,
            output=[{
                "content": c["content"],
                "source": c["document_title"],
                "url": c["document_url"],
                "score": c["score"],
            } for c in chunks],
            metadata={"num_results": len(chunks)},
        )
```

#### 3.2.3 Agent Framework (ReAct Pattern)

```python
# services/agents/types.py
from enum import Enum
from pydantic import BaseModel

class AgentActionType(str, Enum):
    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer"
    ASK_USER = "ask_user"

class AgentAction(BaseModel):
    """An action the agent wants to take."""
    type: AgentActionType
    tool_name: str | None = None
    tool_input: dict | None = None
    content: str | None = None  # For final_answer or ask_user
    reasoning: str | None = None  # Agent's thought process

class AgentObservation(BaseModel):
    """Result of an action."""
    action: AgentAction
    result: ToolResult | str
    timestamp: datetime

class AgentState(BaseModel):
    """Current state of agent execution."""
    messages: list[LLMMessage]
    actions: list[AgentAction] = Field(default_factory=list)
    observations: list[AgentObservation] = Field(default_factory=list)
    iteration: int = 0
    status: str = "running"  # running, completed, failed, needs_input


# services/agents/base.py
class BaseAgent:
    """Base class for all agents."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm: BaseLLMProvider,
        tools: list[BaseTool] = None,
        max_iterations: int = 10,
        memory: "BaseMemory" = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools = tools or []
        self.tool_map = {t.name: t for t in self.tools}
        self.max_iterations = max_iterations
        self.memory = memory

    def get_tool_schemas(self) -> list[dict]:
        """Get tool schemas for LLM."""
        return [t.to_openai_schema() for t in self.tools]


# services/agents/executor.py
class AgentExecutor:
    """Executes agent reasoning loop (ReAct pattern)."""

    def __init__(self, agent: BaseAgent):
        self.agent = agent

    async def run(self, user_input: str) -> AsyncGenerator[dict, None]:
        """
        Run the agent loop, yielding events.

        Event types:
        - {"type": "thinking", "content": "..."}
        - {"type": "tool_call", "tool": "...", "input": {...}}
        - {"type": "tool_result", "tool": "...", "output": {...}}
        - {"type": "answer", "content": "..."}
        - {"type": "error", "message": "..."}
        - {"type": "done", "metadata": {...}}
        """
        state = AgentState(
            messages=[
                LLMMessage(role="system", content=self.agent.system_prompt),
                LLMMessage(role="user", content=user_input),
            ]
        )

        # Add memory context if available
        if self.agent.memory:
            memory_context = await self.agent.memory.get_context(user_input)
            if memory_context:
                state.messages.insert(1, LLMMessage(
                    role="system",
                    content=f"Relevant context from memory:\n{memory_context}"
                ))

        while state.iteration < self.agent.max_iterations:
            state.iteration += 1

            # Get LLM response
            response = await self.agent.llm.complete(
                messages=state.messages,
                tools=self.agent.get_tool_schemas() if self.agent.tools else None,
                temperature=0.2,
            )

            # Check if agent wants to use a tool
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call.function.name
                    tool_input = json.loads(tool_call.function.arguments)

                    yield {
                        "type": "tool_call",
                        "tool": tool_name,
                        "input": tool_input,
                        "iteration": state.iteration,
                    }

                    # Execute tool
                    if tool_name in self.agent.tool_map:
                        result = await self.agent.tool_map[tool_name].execute(**tool_input)

                        yield {
                            "type": "tool_result",
                            "tool": tool_name,
                            "output": result.output if result.success else result.error,
                            "success": result.success,
                        }

                        # Add tool result to messages
                        state.messages.append(LLMMessage(
                            role="assistant",
                            content=None,
                            tool_calls=[tool_call],
                        ))
                        state.messages.append(LLMMessage(
                            role="tool",
                            content=json.dumps(result.output) if result.success else f"Error: {result.error}",
                            tool_call_id=tool_call.id,
                        ))
                    else:
                        yield {
                            "type": "error",
                            "message": f"Unknown tool: {tool_name}",
                        }

            # Check if agent has final answer
            elif response.content:
                yield {
                    "type": "thinking",
                    "content": response.content,
                }

                # Check for final answer marker or finish_reason
                if response.finish_reason == "stop":
                    yield {
                        "type": "answer",
                        "content": response.content,
                    }
                    state.status = "completed"
                    break

                # Add response to messages for next iteration
                state.messages.append(LLMMessage(
                    role="assistant",
                    content=response.content,
                ))

        # Save to memory if available
        if self.agent.memory and state.status == "completed":
            await self.agent.memory.save(user_input, state)

        yield {
            "type": "done",
            "metadata": {
                "iterations": state.iteration,
                "status": state.status,
                "tools_used": [a.tool_name for a in state.actions if a.tool_name],
            }
        }
```

#### 3.2.4 Memory Management

```python
# services/memory/base.py
from abc import ABC, abstractmethod

class BaseMemory(ABC):
    """Abstract base for memory implementations."""

    @abstractmethod
    async def get_context(self, query: str) -> str | None:
        """Get relevant context for a query."""
        pass

    @abstractmethod
    async def save(self, input: str, state: AgentState) -> None:
        """Save interaction to memory."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all memory."""
        pass


# services/memory/conversation.py
class ConversationMemory(BaseMemory):
    """Short-term conversation history."""

    def __init__(self, max_messages: int = 20, max_tokens: int = 4000):
        self.messages: list[LLMMessage] = []
        self.max_messages = max_messages
        self.max_tokens = max_tokens

    async def get_context(self, query: str) -> str | None:
        if not self.messages:
            return None

        # Return recent messages formatted as context
        context_parts = []
        for msg in self.messages[-self.max_messages:]:
            context_parts.append(f"{msg.role}: {msg.content}")

        return "Previous conversation:\n" + "\n".join(context_parts)

    async def save(self, input: str, state: AgentState) -> None:
        # Add user input and final answer to history
        self.messages.append(LLMMessage(role="user", content=input))

        # Find final answer
        for msg in reversed(state.messages):
            if msg.role == "assistant" and msg.content:
                self.messages.append(msg)
                break

        # Trim if too many messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    async def clear(self) -> None:
        self.messages = []


# services/memory/long_term.py
class LongTermMemory(BaseMemory):
    """Persistent vector-based memory using your existing vector store."""

    def __init__(self, vector_store: VectorStore, embeddings: EmbeddingService, user_id: str):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.user_id = user_id
        self.memory_collection = f"memory_{user_id}"

    async def get_context(self, query: str) -> str | None:
        # Search for relevant memories
        results = await self.vector_store.search(
            query=query,
            collection_id=None,  # Would filter by memory collection
            limit=5,
            score_threshold=0.7,
        )

        if not results:
            return None

        context_parts = []
        for r in results:
            context_parts.append(f"- {r['content']}")

        return "Relevant memories:\n" + "\n".join(context_parts)

    async def save(self, input: str, state: AgentState) -> None:
        # Create a memory summary
        summary = f"User asked: {input}\nAgent response: {state.messages[-1].content if state.messages else 'N/A'}"

        # Would save to vector store with memory collection
        # await self.vector_store.ingest_memory(...)
        pass
```

#### 3.2.5 Streaming Agent Events (Extend Current SSE)

```python
# api/routes/agents.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/agents", tags=["Agents"])

@router.post("/{agent_id}/run")
async def run_agent(
    agent_id: str,
    request: AgentRunRequest,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """
    Run an agent with streaming events.

    Event types:
    - thinking: Agent's reasoning
    - tool_call: Tool being called
    - tool_result: Result from tool
    - answer: Final answer
    - done: Execution complete
    """
    # Load agent configuration
    agent_config = await get_agent_config(agent_id)

    # Build agent with tools
    llm = OpenAIProvider(model=agent_config.model)
    tools = [get_tool(t) for t in agent_config.tools]

    agent = BaseAgent(
        name=agent_config.name,
        system_prompt=agent_config.system_prompt,
        llm=llm,
        tools=tools,
    )

    executor = AgentExecutor(agent)

    async def event_generator():
        async for event in executor.run(request.input):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
```

### 3.3 Database Schema Extensions

```python
# models/database.py - New tables

class AgentConfig(Base):
    """Agent configuration stored in database."""
    __tablename__ = "agent_configs"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    slug = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    system_prompt = Column(Text, nullable=False)
    model = Column(String(100), default="gpt-4o")
    tools = Column(Text)  # JSON array of tool names
    max_iterations = Column(Integer, default=10)
    temperature = Column(Float, default=0.2)
    is_active = Column(Integer, default=1)

    # Multi-tenant support
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=True)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class AgentRun(Base):
    """Tracks agent execution history."""
    __tablename__ = "agent_runs"

    id = Column(Integer, primary_key=True)
    agent_id = Column(Integer, ForeignKey("agent_configs.id"), nullable=False)
    user_id = Column(String(100))  # External user identifier
    session_id = Column(String(100))  # For conversation grouping

    input = Column(Text, nullable=False)
    output = Column(Text)
    status = Column(String(20), default="running")  # running, completed, failed

    iterations = Column(Integer, default=0)
    tools_used = Column(Text)  # JSON array
    total_tokens = Column(Integer, default=0)
    latency_ms = Column(Float)

    error_message = Column(Text)

    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())


class Tenant(Base):
    """Multi-tenant support for customer isolation."""
    __tablename__ = "tenants"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    slug = Column(String(100), unique=True, nullable=False)

    # Configuration
    config = Column(Text)  # JSON config

    # Limits
    max_agents = Column(Integer, default=10)
    max_requests_per_day = Column(Integer, default=1000)
    max_tokens_per_day = Column(Integer, default=100000)

    # Usage tracking
    requests_today = Column(Integer, default=0)
    tokens_today = Column(Integer, default=0)
    usage_reset_at = Column(DateTime)

    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    agents = relationship("AgentConfig", backref="tenant")
    collections = relationship("Collection", backref="tenant")  # Add tenant_id to Collection
```

---

## Part 4: Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
**Goal: Core abstractions without breaking existing functionality**

#### Week 1: LLM & Tool Abstractions
- [ ] Create `services/llm/` directory structure
- [ ] Implement `BaseLLMProvider` abstract class
- [ ] Implement `OpenAIProvider` (extract from current `rag_engine.py`)
- [ ] Create `services/tools/` directory structure
- [ ] Implement `BaseTool` abstract class
- [ ] Implement `ToolRegistry`
- [ ] Write unit tests for LLM and Tool abstractions

**Learning Goals:**
- Understand abstraction patterns for swappable components
- Learn OpenAI function calling format deeply

#### Week 2: RAG as a Tool
- [ ] Refactor RAG into `services/rag/` module
- [ ] Create `RAGSearchTool` that wraps `VectorStore.search()`
- [ ] Test RAG tool independently
- [ ] Ensure existing `/chat` endpoint still works

**Learning Goals:**
- See how RAG becomes just another tool in the agentic world
- Understand tool composition patterns

---

### Phase 2: Agent Core (Weeks 3-4)
**Goal: Working ReAct agent with tool use**

#### Week 3: Agent Types & Executor
- [ ] Create `services/agents/` directory structure
- [ ] Implement `AgentState`, `AgentAction`, `AgentObservation` types
- [ ] Implement `BaseAgent` class
- [ ] Implement `AgentExecutor` with ReAct loop
- [ ] Create agent prompt templates

**Learning Goals:**
- Deep understanding of ReAct pattern
- How LLMs reason with tools

#### Week 4: Streaming & API
- [ ] Extend SSE protocol with agent event types
- [ ] Create `/agents/{id}/run` endpoint
- [ ] Implement agent configuration storage
- [ ] Test end-to-end agent execution
- [ ] Create basic agent via config (RAG + search tools)

**Learning Goals:**
- Event-driven architecture for AI
- Real-time streaming patterns

---

### Phase 3: Memory & Context (Weeks 5-6)
**Goal: Agents that remember**

#### Week 5: Memory System
- [ ] Create `services/memory/` directory structure
- [ ] Implement `ConversationMemory` (short-term)
- [ ] Implement `WorkingMemory` (task context)
- [ ] Integrate memory with `AgentExecutor`

**Learning Goals:**
- Context window management
- Memory hierarchy for AI agents

#### Week 6: Long-term Memory
- [ ] Implement `LongTermMemory` using vector store
- [ ] Add memory tables to database
- [ ] Create memory search and retrieval
- [ ] Test memory persistence across sessions

**Learning Goals:**
- Vector-based memory systems
- When to use different memory types

---

### Phase 4: Production Hardening (Weeks 7-8)
**Goal: Enterprise-ready features**

#### Week 7: Guardrails & Safety
- [ ] Create `services/guardrails/` directory structure
- [ ] Implement input validation (PII detection, injection defense)
- [ ] Implement output filtering (content policy)
- [ ] Implement cost/token budgets
- [ ] Add guardrails to agent execution pipeline

**Learning Goals:**
- AI safety in production
- Cost management strategies

#### Week 8: Observability
- [ ] Enhance logging with trace IDs
- [ ] Implement token/cost tracking per request
- [ ] Create agent execution metrics
- [ ] Set up basic monitoring dashboard
- [ ] Extend RAGAS evaluation for agents

**Learning Goals:**
- Distributed tracing for AI
- What metrics matter for agents

---

### Phase 5: Multi-Agent & Workflows (Weeks 9-10)
**Goal: Complex orchestration patterns**

#### Week 9: Workflow Engine
- [ ] Create `services/workflows/` directory structure
- [ ] Implement `Workflow` and `Step` classes
- [ ] Implement `WorkflowEngine` for deterministic flows
- [ ] Create sample workflows (approval chain, data processing)

**Learning Goals:**
- When to use workflows vs agents
- State machine patterns

#### Week 10: Multi-Agent
- [ ] Implement agent-to-agent communication
- [ ] Create supervisor/coordinator pattern
- [ ] Build sample multi-agent system
- [ ] Test handoff between agents

**Learning Goals:**
- Multi-agent orchestration
- When single vs multi-agent

---

### Phase 6: Customer Deployment (Weeks 11-12)
**Goal: Deployable for customers**

#### Week 11: Multi-Tenancy
- [ ] Add tenant isolation to database
- [ ] Implement per-tenant configuration
- [ ] Create tenant management APIs
- [ ] Add usage tracking and limits

**Learning Goals:**
- Multi-tenant architecture
- Customer isolation patterns

#### Week 12: Configuration-Driven Agents
- [ ] Create YAML-based agent configuration
- [ ] Implement config validation
- [ ] Create customer onboarding workflow
- [ ] Document deployment process

**Learning Goals:**
- Config-as-code for AI
- Rapid customer deployment

---

## Part 5: Recommended Additional Tools

### 5.1 For Your M4 Pro Development Environment

Install these to enhance your development workflow:

```bash
# Python environment management
brew install pyenv
pyenv install 3.12.0
pyenv global 3.12.0

# Package management
pip install uv  # 10-100x faster than pip

# Code quality
pip install ruff  # Replaces flake8, black, isort - much faster

# Async development
pip install ipython[async]  # For testing async code

# API testing
brew install httpie  # Better than curl for API testing

# Local LLM testing (optional, for when OpenAI is down)
brew install ollama
ollama pull llama3.2  # 3B model, runs well on M4 Pro

# Observability
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp

# Database tools
brew install pgcli  # Better PostgreSQL CLI
```

### 5.2 Development Patterns to Adopt

```python
# 1. Use Protocol classes for type hints (better than ABC for interfaces)
from typing import Protocol, runtime_checkable

@runtime_checkable
class ToolProtocol(Protocol):
    name: str
    description: str
    async def execute(self, **params) -> ToolResult: ...

# 2. Use Pydantic for all data classes
from pydantic import BaseModel, Field, ConfigDict

class AgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)  # Immutable
    name: str = Field(min_length=1)
    tools: list[str] = Field(default_factory=list)

# 3. Dependency injection pattern (you already use this!)
async def get_agent(
    agent_id: str,
    db: AsyncSession = Depends(get_db),
    llm: BaseLLMProvider = Depends(get_llm),
) -> BaseAgent:
    ...

# 4. Context managers for resource cleanup
from contextlib import asynccontextmanager

@asynccontextmanager
async def agent_session(agent: BaseAgent):
    try:
        yield agent
    finally:
        await agent.cleanup()
```

---

## Part 6: Learning Path for Engineering Manager

### 6.1 Hands-On Implementation Order

| Week | Build | Learn | Outcome |
|------|-------|-------|---------|
| 1 | LLM abstraction | Provider patterns | Can swap GPT/Claude |
| 2 | Tool framework | Function calling | Tools work with any LLM |
| 3 | ReAct agent | Reasoning loops | Agent thinks and acts |
| 4 | Streaming events | Real-time AI | UI shows agent thinking |
| 5 | Memory system | Context management | Agents remember |
| 6 | Long-term memory | Vector memory | Cross-session learning |
| 7 | Guardrails | AI safety | Production protection |
| 8 | Observability | AI ops | Know what agents do |
| 9 | Workflows | Deterministic AI | Predictable processes |
| 10 | Multi-agent | Orchestration | Complex systems |
| 11 | Multi-tenant | Enterprise patterns | Customer isolation |
| 12 | Config-driven | Deployment | Rapid customer setup |

### 6.2 Key Concepts to Master

1. **ReAct Pattern** - The foundation of modern agents
   - Think → Act → Observe → Repeat
   - Know when to stop

2. **Tool Abstraction** - Everything is a tool
   - RAG = tool, API calls = tool, code execution = tool
   - Tools have schemas for LLM understanding

3. **Memory Hierarchy** - What agents need to remember
   - Short-term: Current conversation
   - Working: Current task state
   - Long-term: Learned knowledge

4. **Streaming Architecture** - Users need to see progress
   - Event types for different stages
   - Graceful handling of interrupts

5. **Guardrails** - Production AI needs safety
   - Input validation (prevent jailbreaks)
   - Output filtering (prevent harmful content)
   - Cost limits (prevent runaway costs)

6. **Multi-agent Patterns**
   - Supervisor: One agent delegates
   - Collaborative: Agents discuss
   - Pipeline: Agent chains

### 6.3 Questions to Guide Your Team

After building this, you'll be able to ask your team:

1. "How does the ReAct loop handle tool failures?"
2. "What's the difference between conversation and working memory?"
3. "How do we prevent prompt injection in our agents?"
4. "When should we use a workflow vs an agent?"
5. "How do we trace a request through a multi-agent system?"
6. "What metrics should we monitor for agent health?"
7. "How do we handle rate limits across multiple customers?"

---

## Part 7: Integration with Agno (Optional)

If you decide to use Agno for production speed:

### 7.1 Agno-Compatible Tool Pattern

```python
# Your tool that works with both custom and Agno agents
from agno.tools import tool

@tool(name="rag_search", description="Search knowledge base")
def rag_search_agno(query: str, collection: str = None, top_k: int = 5) -> str:
    """Agno-compatible wrapper around your RAG tool."""
    # Reuse your existing implementation
    result = asyncio.run(your_rag_tool.execute(
        query=query, collection=collection, top_k=top_k
    ))
    return json.dumps(result.output)
```

### 7.2 Migration Path

1. Build your abstractions first (learning)
2. Ensure tool interface matches Agno's `@tool` decorator
3. Test agents with your framework
4. Optionally swap executor to Agno's `Agent` class
5. Keep your custom tools, memory, guardrails

---

## Part 8: Success Metrics

### 8.1 Technical Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Agent latency (simple) | <3s | End-to-end time |
| Agent latency (complex) | <30s | With multiple tool calls |
| Tool execution success | >95% | Track failures |
| Memory recall accuracy | >80% | Test with known facts |
| Guardrail catch rate | >99% | Adversarial testing |

### 8.2 Business Metrics

| Metric | Target | Meaning |
|--------|--------|---------|
| Time to deploy customer agent | <1 day | Config-driven setup |
| Customer onboarding | <1 week | Full integration |
| Agent task completion | >85% | Users achieve goals |

---

## Appendix A: Sample Agent Configuration

```yaml
# configs/customers/acme_corp.yaml
tenant:
  name: ACME Corporation
  slug: acme-corp
  limits:
    max_requests_per_day: 10000
    max_tokens_per_day: 500000

agents:
  - name: Support Agent
    slug: support-agent
    description: Customer support assistant for ACME products
    model: gpt-4o
    temperature: 0.3
    max_iterations: 8

    system_prompt: |
      You are ACME Corp's support assistant. You help customers with:
      - Product questions
      - Order status
      - Technical issues

      Always be helpful and professional. If you can't help, offer to
      escalate to a human agent.

    tools:
      - rag_search:
          collection: acme-docs
      - order_lookup:
          api_url: ${ACME_API_URL}
      - ticket_create:
          system: zendesk
      - human_handoff:
          queue: support-tier-2

    guardrails:
      - pii_filter
      - content_policy
      - cost_limit:
          max_tokens_per_request: 4000

  - name: Sales Agent
    slug: sales-agent
    # ... similar configuration
```

---

## Appendix B: Quick Reference

### Event Types for Agent Streaming

```typescript
// Frontend TypeScript types
type AgentEventType =
  | 'thinking'      // Agent's reasoning
  | 'tool_call'     // About to call a tool
  | 'tool_result'   // Tool returned
  | 'answer'        // Final answer
  | 'ask_user'      // Needs user input
  | 'error'         // Something went wrong
  | 'done';         // Execution complete

interface AgentEvent {
  type: AgentEventType;
  content?: string;
  tool?: string;
  input?: Record<string, unknown>;
  output?: unknown;
  metadata?: {
    iteration?: number;
    tokens_used?: number;
    latency_ms?: number;
  };
}
```

### Key Imports Structure

```python
# After restructuring, imports will look like:
from app.services.llm import OpenAIProvider, AnthropicProvider
from app.services.tools import BaseTool, ToolRegistry, tool
from app.services.tools.builtin import RAGSearchTool, WebSearchTool
from app.services.agents import BaseAgent, AgentExecutor, AgentState
from app.services.memory import ConversationMemory, LongTermMemory
from app.services.guardrails import PIIFilter, CostLimiter
from app.services.workflows import Workflow, WorkflowEngine
```

---

## Sources & References

- [Agno Framework Documentation](https://docs.agno.com)
- [Agno GitHub Repository](https://github.com/agno-agi/agno)
- [Best AI Agent Frameworks 2025](https://langwatch.ai/blog/best-ai-agent-frameworks-in-2025-comparing-langgraph-dspy-crewai-agno-and-more)
- [ReAct Pattern Implementation](https://www.dailydoseofds.com/ai-agents-crash-course-part-10-with-implementation/)
- [ReAct Best Practices](https://metadesignsolutions.com/using-the-react-pattern-in-ai-agents-best-practices-pitfalls-implementation-tips/)
- [Multi-Agent Orchestration Patterns](https://www.kore.ai/blog/choosing-the-right-orchestration-pattern-for-multi-agent-systems)
- [Google Cloud Agentic AI Design Patterns](https://cloud.google.com/architecture/choose-design-pattern-agentic-ai-system)
- [MLX LM for Local LLMs](https://github.com/ml-explore/mlx-lm)
- [MLX Omni Server](https://github.com/madroidmaq/mlx-omni-server)
