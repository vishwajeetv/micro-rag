"""
AgentExecutor - The brain's control loop.

This is where the magic happens:
1. Send user query + tools to LLM
2. LLM decides: answer directly OR use tools
3. If tools needed: execute them, send results back to LLM
4. Repeat until LLM gives final answer

The loop continues until:
- LLM responds without tool calls (final answer)
- Max iterations reached (safety limit)
- Error occurs
"""

import json
from typing import AsyncGenerator, Any

from app.core.logging import get_logger
from app.services.agents.base import Agent
from app.services.agents.memory import SessionMemory
from app.services.tools.base import ToolResult

logger = get_logger(__name__)

# Event types for streaming
EVENT_THINKING = "thinking"       # LLM is processing
EVENT_TOOL_CALL = "tool_call"     # LLM wants to use a tool
EVENT_TOOL_RESULT = "tool_result" # Tool execution completed
EVENT_ANSWER = "answer"           # Final answer from LLM
EVENT_ERROR = "error"             # Something went wrong


class AgentEvent:
    """
    An event emitted during agent execution.

    Used for streaming real-time updates to the client.
    """
    def __init__(self, event_type: str, data: dict):
        self.type = event_type
        self.data = data

    def to_dict(self) -> dict:
        return {"type": self.type, "data": self.data}

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        return f"data: {json.dumps(self.to_dict())}\n\n"


class AgentExecutor:
    """
    Runs the agent loop.

    This is the orchestrator that:
    1. Manages conversation with the LLM
    2. Executes tools when LLM requests them
    3. Streams events back to the caller
    4. Handles memory (conversation history)

    Example:
        agent = Agent(name="Assistant", system_prompt="...", tools=[...])
        executor = AgentExecutor(agent)

        async for event in executor.run("What is Python?"):
            print(event.type, event.data)
    """

    def __init__(
        self,
        agent: Agent,
        memory: SessionMemory | None = None,
        max_iterations: int = 10,
    ):
        """
        Initialize the executor.

        Args:
            agent: The agent to run
            memory: Optional session memory for conversation history
            max_iterations: Safety limit to prevent infinite loops
        """
        self.agent = agent
        self.memory = memory or SessionMemory()
        self.max_iterations = max_iterations

    async def run(self, user_input: str) -> AsyncGenerator[AgentEvent, None]:
        """
        Run the agent loop.

        This is an async generator that yields events as they happen:
        - thinking: LLM is processing
        - tool_call: LLM wants to use a tool
        - tool_result: Tool finished executing
        - answer: Final response from LLM
        - error: Something went wrong

        Args:
            user_input: The user's question/request

        Yields:
            AgentEvent objects for each step
        """
        logger.info(
            "agent_run_start",
            agent=self.agent.name,
            input=user_input[:100]  # Log first 100 chars
        )

        # Add user message to memory
        self.memory.add_message("user", user_input)

        iteration = 0
        total_tokens = 0  # Track cumulative token usage

        while iteration < self.max_iterations:
            iteration += 1

            logger.debug(
                "agent_iteration",
                iteration=iteration,
                max=self.max_iterations
            )

            # Emit thinking event with token count so far
            yield AgentEvent(EVENT_THINKING, {
                "iteration": iteration,
                "tokens_so_far": total_tokens
            })

            try:
                # Call LLM with conversation history and tools
                response = await self._call_llm()

                # Track token usage from this call
                if response.usage:
                    total_tokens += response.usage.total_tokens

                # Get the assistant's message
                message = response.choices[0].message

                # Check if LLM wants to use tools
                if message.tool_calls:
                    # LLM wants to use tools - execute them

                    # Add assistant message (with tool calls) to memory
                    self.memory.add_message(
                        "assistant",
                        message.content or "",
                        tool_calls=[
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in message.tool_calls
                        ]
                    )

                    # Execute each tool call
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args_str = tool_call.function.arguments

                        # Emit tool_call event
                        yield AgentEvent(EVENT_TOOL_CALL, {
                            "tool": tool_name,
                            "arguments": tool_args_str
                        })

                        # Execute the tool
                        result = await self._execute_tool(
                            tool_name,
                            tool_args_str
                        )

                        # Add tool result to memory
                        self.memory.add_tool_result(tool_call.id, result.output)

                        # Emit tool_result event
                        yield AgentEvent(EVENT_TOOL_RESULT, {
                            "tool": tool_name,
                            "success": result.success,
                            "output": str(result.output)[:500]  # Limit for streaming
                        })

                    # Continue loop - LLM needs to process tool results
                    continue

                else:
                    # No tool calls - this is the final answer
                    final_answer = message.content or ""

                    # Add to memory
                    self.memory.add_message("assistant", final_answer)

                    # Emit answer event with total tokens
                    yield AgentEvent(EVENT_ANSWER, {
                        "content": final_answer,
                        "tokens_used": total_tokens,
                        "iterations": iteration
                    })

                    logger.info(
                        "agent_run_complete",
                        agent=self.agent.name,
                        iterations=iteration,
                        tokens_used=total_tokens
                    )

                    # Done!
                    return

            except Exception as e:
                logger.error(
                    "agent_run_error",
                    agent=self.agent.name,
                    error=str(e),
                    iteration=iteration
                )
                yield AgentEvent(EVENT_ERROR, {"message": str(e)})
                return

        # Max iterations reached
        logger.warning(
            "agent_max_iterations",
            agent=self.agent.name,
            max=self.max_iterations
        )
        yield AgentEvent(EVENT_ERROR, {
            "message": f"Max iterations ({self.max_iterations}) reached"
        })

    async def _call_llm(self) -> Any:
        """
        Call the LLM with current conversation and tools.

        Returns:
            OpenAI chat completion response
        """
        messages = self.memory.get_messages_for_api()

        # Add system prompt at the beginning
        full_messages = [
            {"role": "system", "content": self.agent.system_prompt},
            *messages
        ]

        # Get tools schema (None if no tools)
        tools = self.agent.get_tools_schema()

        # Call OpenAI
        kwargs = {
            "model": self.agent.model,
            "messages": full_messages,
        }

        # Only add tools if we have them
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"  # Let LLM decide

        response = await self.agent.client.chat.completions.create(**kwargs)

        return response

    async def _execute_tool(self, tool_name: str, args_str: str) -> ToolResult:
        """
        Execute a tool by name with given arguments.

        Args:
            tool_name: Name of the tool to execute
            args_str: JSON string of arguments

        Returns:
            ToolResult from the tool execution
        """
        # Get the tool
        tool = self.agent.get_tool(tool_name)

        if not tool:
            logger.error("tool_not_found", tool=tool_name)
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{tool_name}' not found"
            )

        try:
            # Parse arguments
            args = json.loads(args_str) if args_str else {}

            logger.info(
                "tool_execute",
                tool=tool_name,
                args=args
            )

            # Execute the tool
            result = await tool.run(**args)

            logger.info(
                "tool_complete",
                tool=tool_name,
                success=result.success
            )

            return result

        except json.JSONDecodeError as e:
            logger.error(
                "tool_args_invalid",
                tool=tool_name,
                args=args_str,
                error=str(e)
            )
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid arguments: {str(e)}"
            )
        except Exception as e:
            logger.error(
                "tool_execute_error",
                tool=tool_name,
                error=str(e)
            )
            return ToolResult(
                success=False,
                output="",
                error=f"Tool execution failed: {str(e)}"
            )
