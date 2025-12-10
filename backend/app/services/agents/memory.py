"""
Session Memory - Keeps track of conversation history.

This stores the full conversation thread so the agent can:
- Remember what was said earlier
- Build context across multiple exchanges
- Include tool calls and results in history

Memory Format:
    [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is...", "tool_calls": [...]},
        {"role": "tool", "tool_call_id": "...", "content": "..."},
        {"role": "assistant", "content": "Based on that..."}
    ]
"""

from typing import Any
from app.core.logging import get_logger

logger = get_logger(__name__)


class SessionMemory:
    """
    In-memory conversation history for a single session.

    Stores messages in OpenAI's format so they can be
    directly used in API calls.

    Example:
        memory = SessionMemory()
        memory.add_message("user", "Hello!")
        memory.add_message("assistant", "Hi there!")

        messages = memory.get_messages_for_api()
        # [{"role": "user", "content": "Hello!"}, ...]
    """

    def __init__(self, max_messages: int = 50):
        """
        Initialize memory.

        Args:
            max_messages: Maximum messages to keep (prevents context overflow)
        """
        self.messages: list[dict[str, Any]] = []
        self.max_messages = max_messages

    def add_message(
        self,
        role: str,
        content: str,
        tool_calls: list[dict] | None = None
    ) -> None:
        """
        Add a message to the conversation history.

        Args:
            role: "user", "assistant", or "system"
            content: The message content
            tool_calls: Optional tool calls (for assistant messages)
        """
        message: dict[str, Any] = {
            "role": role,
            "content": content
        }

        # Add tool_calls if present (assistant requesting tools)
        if tool_calls:
            message["tool_calls"] = tool_calls

        self.messages.append(message)

        # Trim if too long (keep recent messages)
        if len(self.messages) > self.max_messages:
            # Keep first message (might be important context) and recent ones
            self.messages = self.messages[:1] + self.messages[-(self.max_messages - 1):]

        logger.debug(
            "memory_add_message",
            role=role,
            content_length=len(content),
            total_messages=len(self.messages)
        )

    def add_tool_result(self, tool_call_id: str, result: Any) -> None:
        """
        Add a tool result to the conversation.

        Tool results must reference the tool_call_id they're responding to.
        This is how OpenAI knows which tool call the result belongs to.

        Args:
            tool_call_id: The ID of the tool call this is responding to
            result: The result from the tool execution
        """
        # Convert result to string if needed
        content = str(result) if not isinstance(result, str) else result

        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content
        })

        logger.debug(
            "memory_add_tool_result",
            tool_call_id=tool_call_id,
            result_length=len(content)
        )

    def get_messages_for_api(self) -> list[dict[str, Any]]:
        """
        Get messages formatted for OpenAI API.

        Returns:
            List of message dicts ready for chat.completions.create()
        """
        return self.messages.copy()

    def get_last_user_message(self) -> str | None:
        """Get the most recent user message."""
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                return msg["content"]
        return None

    def get_last_assistant_message(self) -> str | None:
        """Get the most recent assistant message."""
        for msg in reversed(self.messages):
            if msg["role"] == "assistant":
                return msg["content"]
        return None

    def clear(self) -> None:
        """Clear all messages."""
        self.messages = []
        logger.debug("memory_cleared")

    def to_dict(self) -> dict:
        """
        Export memory state for persistence.

        Returns:
            Dict containing all messages
        """
        return {
            "messages": self.messages.copy(),
            "max_messages": self.max_messages
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionMemory":
        """
        Restore memory from exported state.

        Args:
            data: Dict from to_dict()

        Returns:
            Restored SessionMemory instance
        """
        memory = cls(max_messages=data.get("max_messages", 50))
        memory.messages = data.get("messages", [])
        return memory

    def __len__(self) -> int:
        return len(self.messages)

    def __repr__(self) -> str:
        return f"<SessionMemory: {len(self.messages)} messages>"
