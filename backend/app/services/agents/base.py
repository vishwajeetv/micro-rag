"""
Agent class - The brain that decides what to do.

An Agent has:
- A name
- A system prompt (personality/instructions)
- A list of tools it can use
- An LLM to think with
"""

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logging import get_logger
from app.services.tools.base import BaseTool

logger = get_logger(__name__)


class Agent:
    """
    An Agent that can use tools to accomplish tasks.

    The agent itself doesn't run the loop - that's AgentExecutor's job.
    The agent just holds the configuration: name, prompt, tools, LLM.

    Example:
        # Create an agent
        agent = Agent(
            name="Assistant",
            system_prompt="You are a helpful assistant...",
            tools=[rag_search_tool, calculator_tool]
        )

        # Run it with the executor
        executor = AgentExecutor(agent)
        async for event in executor.run("What is 2+2?"):
            print(event)
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list[BaseTool] | None = None,
        model: str = "gpt-5.1",
    ):
        """
        Create a new agent.

        Args:
            name: Agent's name (for logging/display)
            system_prompt: Instructions for the LLM (personality, rules, etc.)
            tools: List of tools the agent can use
            model: OpenAI model to use (default: gpt-4o)
        """
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.model = model

        # Create tool lookup map: name -> tool
        self.tool_map: dict[str, BaseTool] = {
            tool.name: tool for tool in self.tools
        }

        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

        logger.info(
            "agent_created",
            name=name,
            model=model,
            tools=[t.name for t in self.tools]
        )

    def get_tools_schema(self) -> list[dict] | None:
        """
        Get tools in OpenAI format for API calls.

        Returns None if no tools (OpenAI doesn't want empty list).
        """
        if not self.tools:
            return None
        return [tool.to_openai_schema() for tool in self.tools]

    def get_tool(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        return self.tool_map.get(name)

    def __repr__(self) -> str:
        return f"<Agent: {self.name}, tools={[t.name for t in self.tools]}>"
