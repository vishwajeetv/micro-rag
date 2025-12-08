# Agentic AI Framework - MVP Plan

> **Simple first. Add complexity later.**
>
> This is a beginner-friendly guide to build your first agentic AI system.

---

## What Are We Building?

### Current State: RAG System
```
User asks question → Search documents → Generate answer
```
That's it. One step. No decision making.

### Target State: Agentic System
```
User asks question → Agent THINKS → Agent DECIDES what to do → Agent ACTS → Repeat until done
```
The agent can use multiple tools, make decisions, and solve complex problems.

---

## The Simplest Possible Agent

An agent is just a **loop**:

```
┌─────────────────────────────────────────┐
│                                         │
│   1. THINK  →  "What should I do?"      │
│       │                                 │
│       ▼                                 │
│   2. ACT    →  Use a tool OR answer     │
│       │                                 │
│       ▼                                 │
│   3. OBSERVE → See the result           │
│       │                                 │
│       ▼                                 │
│   4. REPEAT → Until task is done        │
│                                         │
└─────────────────────────────────────────┘
```

That's the entire concept. Everything else is details.

---

## MVP Scope: What We're Building

### YES (MVP)
- ✅ One working agent that can use tools
- ✅ Tools: RAG search + one simple tool (calculator or web search)
- ✅ Streaming output so user sees agent thinking
- ✅ Basic conversation memory (remember last few messages)

### NO (Later)
- ❌ Multiple agents talking to each other
- ❌ Complex workflows with branches
- ❌ Guardrails and safety filters
- ❌ Multi-tenant/customer support
- ❌ Long-term memory
- ❌ Multiple LLM providers (just use OpenAI)

---

## Part 1: Core Concepts (Simple Explanations)

### What is a Tool?

A tool is a **function the agent can call**.

```python
# This is a tool
def search_knowledge_base(query: str) -> str:
    """Search our documents for information."""
    results = vector_store.search(query)
    return results

# This is also a tool
def calculate(expression: str) -> str:
    """Do math calculations."""
    return eval(expression)  # simplified
```

The agent doesn't run these directly. It tells the LLM "here are tools you can use" and the LLM decides when to call them.

### What is an Agent?

An agent is:
1. An LLM (like GPT-4)
2. A list of tools it can use
3. A system prompt telling it how to behave
4. A loop that keeps going until the task is done

```python
# Simplified agent
class Agent:
    def __init__(self, tools, system_prompt):
        self.llm = OpenAI()
        self.tools = tools
        self.system_prompt = system_prompt

    def run(self, user_question):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_question}
        ]

        while True:
            # Ask LLM what to do
            response = self.llm.chat(messages, tools=self.tools)

            # If LLM wants to use a tool
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    result = self.execute_tool(tool_call)
                    messages.append({"role": "tool", "content": result})

            # If LLM has final answer
            else:
                return response.content
```

That's ~20 lines. That's an agent.

### What is Memory?

Memory is just **keeping track of the conversation**.

```python
# Simplest memory: a list
memory = []

# User says something
memory.append({"role": "user", "content": "What is Spain?"})

# Agent responds
memory.append({"role": "assistant", "content": "Spain is a country..."})

# Next turn, include memory in the prompt
response = llm.chat(messages=memory + [new_message])
```

For MVP, that's all we need.

---

## Part 2: MVP Architecture

### What We're Adding to Your Codebase

```
backend/app/
├── services/
│   ├── rag_engine.py      # KEEP (existing)
│   ├── vector_store.py    # KEEP (existing)
│   ├── embeddings.py      # KEEP (existing)
│   │
│   ├── tools/             # NEW (simple)
│   │   ├── __init__.py
│   │   ├── base.py        # BaseTool class (~50 lines)
│   │   └── rag_tool.py    # RAG as a tool (~30 lines)
│   │
│   └── agents/            # NEW (simple)
│       ├── __init__.py
│       ├── base.py        # Agent class (~100 lines)
│       └── executor.py    # Run loop (~80 lines)
│
├── api/
│   └── routes.py          # ADD: /agent/run endpoint
```

**Total new code: ~300 lines**

### How It Connects

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  POST /api/agent/run                                        │
│  { "message": "How do I form Spain in EU5?" }              │
│                                                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      AGENT                                  │
│                                                             │
│  System Prompt: "You are a helpful assistant. You have     │
│  access to a knowledge base about EU5. Use the search      │
│  tool to find information before answering."               │
│                                                             │
│  Tools: [rag_search]                                       │
│                                                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            │                           │
            ▼                           ▼
┌───────────────────┐       ┌───────────────────┐
│   LLM (GPT-4)     │       │   RAG Tool        │
│                   │       │                   │
│   Decides what    │       │   Your existing   │
│   to do next      │       │   vector_store    │
│                   │       │   .search()       │
└───────────────────┘       └───────────────────┘
```

---

## Part 3: MVP Implementation Plan

### Week 1: Tool Abstraction

**Goal:** Create a simple way to define tools

**File: `services/tools/base.py`**
```python
from abc import ABC, abstractmethod
from pydantic import BaseModel

class ToolResult(BaseModel):
    """What a tool returns."""
    success: bool
    output: str
    error: str | None = None

class BaseTool(ABC):
    """Base class for all tools."""

    name: str           # e.g., "rag_search"
    description: str    # LLM reads this to know when to use the tool

    @abstractmethod
    async def run(self, **kwargs) -> ToolResult:
        """Execute the tool."""
        pass

    def to_openai_format(self) -> dict:
        """Convert to OpenAI function calling format."""
        # This tells OpenAI what parameters the tool accepts
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters_schema()
            }
        }

    @abstractmethod
    def get_parameters_schema(self) -> dict:
        """Define what parameters this tool accepts."""
        pass
```

**File: `services/tools/rag_tool.py`**
```python
from app.services.tools.base import BaseTool, ToolResult
from app.services.vector_store import VectorStore

class RAGSearchTool(BaseTool):
    """Search the knowledge base."""

    name = "rag_search"
    description = "Search the knowledge base for information. Use this when you need to find facts or answer questions about the documented topics."

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }

    async def run(self, query: str) -> ToolResult:
        try:
            results = await self.vector_store.search(query=query, limit=5)

            # Format results as readable text
            output_parts = []
            for r in results:
                output_parts.append(f"Source: {r['document_title']}\n{r['content']}\n")

            return ToolResult(
                success=True,
                output="\n---\n".join(output_parts) if output_parts else "No results found."
            )
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
```

**Test it:**
```python
# Test the tool directly
tool = RAGSearchTool(vector_store)
result = await tool.run(query="How to form Spain")
print(result.output)
```

---

### Week 2: Simple Agent

**Goal:** Create an agent that uses tools

**File: `services/agents/base.py`**
```python
from openai import AsyncOpenAI
from app.core.config import settings

class Agent:
    """A simple agent that can use tools."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.tool_map = {t.name: t for t in self.tools}
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = "gpt-4o"  # or gpt-4o-mini for cheaper testing

    def get_tools_for_llm(self) -> list[dict]:
        """Get tools in OpenAI format."""
        return [t.to_openai_format() for t in self.tools]
```

**File: `services/agents/executor.py`**
```python
import json
from typing import AsyncGenerator
from app.services.agents.base import Agent

class AgentExecutor:
    """Runs the agent loop."""

    def __init__(self, agent: Agent, max_iterations: int = 5):
        self.agent = agent
        self.max_iterations = max_iterations

    async def run(self, user_message: str) -> AsyncGenerator[dict, None]:
        """
        Run the agent and stream events.

        Yields events like:
        - {"type": "thinking", "content": "..."}
        - {"type": "tool_call", "tool": "rag_search", "input": {...}}
        - {"type": "tool_result", "output": "..."}
        - {"type": "answer", "content": "..."}
        """

        # Start with system prompt and user message
        messages = [
            {"role": "system", "content": self.agent.system_prompt},
            {"role": "user", "content": user_message}
        ]

        for iteration in range(self.max_iterations):
            # Call LLM
            response = await self.agent.client.chat.completions.create(
                model=self.agent.model,
                messages=messages,
                tools=self.agent.get_tools_for_llm() if self.agent.tools else None,
            )

            choice = response.choices[0]

            # Check if LLM wants to use tools
            if choice.message.tool_calls:
                # Add assistant message with tool calls
                messages.append(choice.message)

                # Execute each tool
                for tool_call in choice.message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_input = json.loads(tool_call.function.arguments)

                    yield {
                        "type": "tool_call",
                        "tool": tool_name,
                        "input": tool_input
                    }

                    # Run the tool
                    if tool_name in self.agent.tool_map:
                        result = await self.agent.tool_map[tool_name].run(**tool_input)
                        tool_output = result.output if result.success else f"Error: {result.error}"
                    else:
                        tool_output = f"Unknown tool: {tool_name}"

                    yield {
                        "type": "tool_result",
                        "tool": tool_name,
                        "output": tool_output[:500]  # Truncate for display
                    }

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_output
                    })

            # Check if LLM has final answer (no tool calls)
            elif choice.message.content:
                yield {
                    "type": "answer",
                    "content": choice.message.content
                }
                return  # Done!

            # Safety check
            if choice.finish_reason == "stop" and not choice.message.tool_calls:
                yield {
                    "type": "answer",
                    "content": choice.message.content or "I couldn't find an answer."
                }
                return

        # Max iterations reached
        yield {
            "type": "answer",
            "content": "I've done multiple searches but couldn't find a complete answer. Here's what I found so far based on my research."
        }
```

---

### Week 3: API Endpoint

**Goal:** Expose the agent via API with streaming

**Add to `api/routes.py`:**
```python
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from app.services.agents.base import Agent
from app.services.agents.executor import AgentExecutor
from app.services.tools.rag_tool import RAGSearchTool
from app.services.vector_store import VectorStore

# Request schema
class AgentRunRequest(BaseModel):
    message: str
    collection_slug: str | None = None

# Create agent endpoint
@router.post("/agent/run", tags=["Agent"])
async def run_agent(
    request: AgentRunRequest,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """
    Run the agent with streaming output.

    Returns Server-Sent Events:
    - tool_call: Agent is calling a tool
    - tool_result: Tool returned results
    - answer: Final answer
    """

    # Create tools
    vector_store = VectorStore(db)
    rag_tool = RAGSearchTool(vector_store)

    # Create agent
    agent = Agent(
        name="Assistant",
        system_prompt="""You are a helpful assistant with access to a knowledge base.

When asked a question:
1. Use the rag_search tool to find relevant information
2. Based on the search results, provide a clear answer
3. If you can't find the information, say so

Always search before answering factual questions.""",
        tools=[rag_tool]
    )

    # Create executor
    executor = AgentExecutor(agent, max_iterations=5)

    # Stream events
    async def event_stream():
        async for event in executor.run(request.message):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream"
    )
```

---

### Week 4: Simple Frontend Update

**Goal:** Show agent's thinking process in the UI

**Update `Chat.tsx` to handle new event types:**
```typescript
// Add new state for showing agent activity
const [agentActivity, setAgentActivity] = useState<string[]>([]);

// In the streaming handler, add cases for agent events:
for await (const event of streamAgentMessage({ message: userMessage })) {
  if (event.type === 'tool_call') {
    setAgentActivity(prev => [...prev, `🔍 Searching: "${event.input.query}"`]);
  } else if (event.type === 'tool_result') {
    setAgentActivity(prev => [...prev, `✓ Found results`]);
  } else if (event.type === 'answer') {
    // Clear activity, show answer
    setAgentActivity([]);
    setMessages(prev => {
      const newMessages = [...prev];
      const lastMsg = newMessages[newMessages.length - 1];
      if (lastMsg.role === 'assistant') {
        lastMsg.content = event.content;
        lastMsg.loading = false;
      }
      return newMessages;
    });
  }
}
```

---

## Part 4: Testing Your Agent

### Manual Test
```bash
# Start the server
cd backend
PYTHONPATH=. uvicorn app.main:app --reload

# Test with curl
curl -X POST http://localhost:8000/api/agent/run \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I form Spain in EU5?"}' \
  --no-buffer
```

### Expected Output
```
data: {"type": "tool_call", "tool": "rag_search", "input": {"query": "form Spain EU5"}}
data: {"type": "tool_result", "tool": "rag_search", "output": "Source: Formation of Spain..."}
data: {"type": "answer", "content": "To form Spain in EU5, you need to..."}
```

### Python Test
```python
# tests/test_agent.py
import pytest
from app.services.agents.base import Agent
from app.services.agents.executor import AgentExecutor
from app.services.tools.rag_tool import RAGSearchTool

@pytest.mark.asyncio
async def test_agent_uses_tool(db_session):
    """Test that agent calls RAG tool for questions."""
    vector_store = VectorStore(db_session)
    rag_tool = RAGSearchTool(vector_store)

    agent = Agent(
        name="Test",
        system_prompt="Use rag_search to answer questions.",
        tools=[rag_tool]
    )

    executor = AgentExecutor(agent)
    events = []

    async for event in executor.run("What is EU5?"):
        events.append(event)

    # Should have called the tool
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_calls) > 0
    assert tool_calls[0]["tool"] == "rag_search"

    # Should have final answer
    answers = [e for e in events if e["type"] == "answer"]
    assert len(answers) == 1
```

---

## Part 5: What You've Built (MVP)

After 4 weeks, you have:

```
┌─────────────────────────────────────────────────────────────┐
│                     YOUR MVP AGENT                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User: "How do I form Spain?"                              │
│                                                             │
│         ┌──────────────────────────────────────┐           │
│         │            AGENT                      │           │
│         │                                       │           │
│         │  1. Thinks: "I should search first"  │           │
│         │  2. Calls: rag_search("form Spain")  │           │
│         │  3. Gets: [search results]           │           │
│         │  4. Answers: "To form Spain..."      │           │
│         │                                       │           │
│         └──────────────────────────────────────┘           │
│                                                             │
│  Tools: [RAG Search]                                       │
│  Memory: Last 20 messages                                  │
│  Streaming: Yes                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Lines of new code: ~300**
**Time: 4 weeks**
**Result: Working agent!**

---

## Part 6: After MVP - What to Add Next

Once MVP works, add these **one at a time**:

### Phase 2: More Tools (Week 5-6)
```python
# Add a web search tool
class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the internet for current information"

    async def run(self, query: str) -> ToolResult:
        # Use Tavily, Serper, or Brave Search API
        results = await search_api.search(query)
        return ToolResult(success=True, output=results)

# Add a calculator tool
class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Do math calculations"

    async def run(self, expression: str) -> ToolResult:
        try:
            result = eval(expression)  # Use safer eval in production
            return ToolResult(success=True, output=str(result))
        except:
            return ToolResult(success=False, output="", error="Invalid expression")
```

### Phase 3: Better Memory (Week 7-8)
```python
# Add conversation memory
class ConversationMemory:
    def __init__(self, max_messages: int = 20):
        self.messages = []
        self.max_messages = max_messages

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_context(self) -> list[dict]:
        return self.messages.copy()
```

### Phase 4: Multiple LLM Providers (Week 9-10)
```python
# Abstract the LLM
class LLMProvider:
    async def chat(self, messages, tools=None): ...

class OpenAIProvider(LLMProvider):
    async def chat(self, messages, tools=None):
        return await openai_client.chat.completions.create(...)

class AnthropicProvider(LLMProvider):
    async def chat(self, messages, tools=None):
        return await anthropic_client.messages.create(...)
```

### Phase 5: Guardrails (Week 11-12)
```python
# Add safety checks
class InputGuard:
    def check(self, user_input: str) -> bool:
        # Check for prompt injection
        if "ignore previous instructions" in user_input.lower():
            return False
        return True

class CostGuard:
    def __init__(self, max_tokens_per_request: int = 4000):
        self.max_tokens = max_tokens_per_request

    def check(self, token_count: int) -> bool:
        return token_count < self.max_tokens
```

---

## Summary: MVP in 4 Weeks

| Week | Build | Lines | Outcome |
|------|-------|-------|---------|
| 1 | Tool base class + RAG tool | ~80 | Tools work |
| 2 | Agent + Executor | ~180 | Agent thinks |
| 3 | API endpoint | ~40 | Agent accessible |
| 4 | Frontend updates | ~30 | Users see thinking |

**Total: ~330 lines of new code**

---

## Key Takeaways

1. **An agent is just a loop** - Think → Act → Observe → Repeat

2. **Tools are just functions** - Wrap your existing code as tools

3. **Start simple** - One agent, one tool, get it working

4. **Your RAG becomes a tool** - The biggest mental shift

5. **Add complexity later** - Memory, guardrails, multi-agent = later

---

## Quick Reference

### File Structure (MVP)
```
backend/app/services/
├── tools/
│   ├── __init__.py
│   ├── base.py          # BaseTool, ToolResult
│   └── rag_tool.py      # RAGSearchTool
└── agents/
    ├── __init__.py
    ├── base.py          # Agent class
    └── executor.py      # AgentExecutor (the loop)
```

### New API Endpoint
```
POST /api/agent/run
Body: { "message": "your question" }
Response: Server-Sent Events stream
```

### Event Types
```json
{"type": "tool_call", "tool": "rag_search", "input": {"query": "..."}}
{"type": "tool_result", "tool": "rag_search", "output": "..."}
{"type": "answer", "content": "Final answer here"}
```

---

Now go build it! Start with Week 1 and get the RAG tool working first.
