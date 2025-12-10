"""
Base class for all tools.

A Tool is something an agent can use to interact with the world:
- Search a knowledge base
- Call an API
- Do calculations
- etc.

The LLM decides WHEN to use tools based on their descriptions.
Your code EXECUTES the tools and returns results to the LLM.
"""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """
    Result returned by a tool after execution.

    Attributes:
        success: Whether the tool executed successfully
        output: The result data (string, dict, list, etc.)
        error: Error message if success=False
    """
    success: bool
    output: Any
    error: str | None = None


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    To create a new tool:
    1. Inherit from BaseTool
    2. Set name and description (LLM reads these!)
    3. Implement get_parameters_schema()
    4. Implement run()

    Example:
        class MyTool(BaseTool):
            name = "my_tool"
            description = "Does something useful"

            def get_parameters_schema(self) -> dict:
                return {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "..."}
                    },
                    "required": ["param1"]
                }

            async def run(self, param1: str) -> ToolResult:
                result = do_something(param1)
                return ToolResult(success=True, output=result)
    """

    # Subclasses must set these
    name: str = ""
    description: str = ""

    @abstractmethod
    def get_parameters_schema(self) -> dict:
        """
        Define what parameters this tool accepts.

        Returns JSON Schema format that OpenAI understands.
        The LLM uses this to know what arguments to pass.

        Example:
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return"
                    }
                },
                "required": ["query"]
            }
        """
        pass

    @abstractmethod
    async def run(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.

        This is where your actual logic goes:
        - Call APIs
        - Query databases
        - Do calculations
        - etc.

        Args:
            **kwargs: Parameters matching get_parameters_schema()

        Returns:
            ToolResult with success status and output/error
        """
        pass

    def to_openai_schema(self) -> dict:
        """
        Convert this tool to OpenAI's function calling format.

        This is what you send to OpenAI so the LLM knows about the tool.
        The LLM reads 'description' to decide when to use this tool.

        Returns:
            Dict in OpenAI's tool format
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters_schema()
            }
        }

    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"
