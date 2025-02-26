"""
Short memory module for the BMW Agents framework.
This module implements the short-term memory used by agents during task execution.
"""

from typing import Dict, Iterator, List, Optional

from bmw_agents.core.prompt_strategies.base import Message
from bmw_agents.utils.llm_providers import LLMProvider
from bmw_agents.utils.logger import get_logger

logger = get_logger("memory.short_memory")


class ShortMemory:
    """
    Short-term memory used by agents during task execution.

    Short Memory (SM) operates within the confines of a single task,
    remains isolated to each agent, and is purged upon task completion.
    It stores a history of messages in the order they were created.
    """

    def __init__(self, max_size: Optional[int] = None) -> None:
        """
        Initialize a short memory.

        Args:
            max_size: Maximum number of messages to store (None for unlimited)
        """
        self.messages: List[Message] = []
        self.max_size = max_size

    def add(self, message: Message) -> None:
        """
        Add a message to the memory.

        Args:
            message: The message to add
        """
        self.messages.append(message)

        # If max_size is set, trim the oldest messages
        if self.max_size is not None and len(self.messages) > self.max_size:
            excess = len(self.messages) - self.max_size
            self.messages = self.messages[excess:]
            logger.debug(f"Trimmed {excess} old messages from short memory")

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the memory.

        Args:
            content: The content of the message
        """
        self.add(Message("user", content))

    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to the memory.

        Args:
            content: The content of the message
        """
        self.add(Message("assistant", content))

    def add_system_message(self, content: str) -> None:
        """
        Add a system message to the memory.

        Args:
            content: The content of the message
        """
        self.add(Message("system", content))

    def get_all(self) -> List[Message]:
        """
        Get all messages in the memory.

        Returns:
            List of all messages
        """
        return self.messages

    def get_last(self, n: int = 1) -> List[Message]:
        """
        Get the last n messages from the memory.

        Args:
            n: Number of messages to get

        Returns:
            List of the last n messages
        """
        return self.messages[-n:] if self.messages else []

    def get_by_role(self, role: str) -> List[Message]:
        """
        Get all messages with the specified role.

        Args:
            role: The role to filter by (user, assistant, system)

        Returns:
            List of messages with the specified role
        """
        return [msg for msg in self.messages if msg.role == role]

    def clear(self) -> None:
        """Clear all messages from the memory."""
        self.messages = []

    def to_dict_list(self) -> List[Dict[str, str]]:
        """
        Convert all messages to a list of dictionaries.

        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return [msg.to_dict() for msg in self.messages]

    def get_token_count(self, llm_provider: LLMProvider) -> int:
        """
        Get the total token count of all messages in the memory.

        Args:
            llm_provider: LLM provider with a count_tokens method

        Returns:
            Total token count
        """
        total_tokens = 0
        for message in self.messages:
            total_tokens += llm_provider.count_tokens(message.content)
        return total_tokens

    def __len__(self) -> int:
        """Get the number of messages in the memory."""
        return len(self.messages)

    def __getitem__(self, index: int) -> Message:
        """Get a message by index."""
        return self.messages[index]

    def __iter__(self) -> Iterator[Message]:
        """Iterate over messages."""
        return iter(self.messages)
