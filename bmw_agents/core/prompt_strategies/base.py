"""
Base class for all prompt strategies used in the BMW Agents framework.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from bmw_agents.utils.llm_providers import LLMProvider
from bmw_agents.utils.logger import get_logger

logger = get_logger("prompt_strategies.base")


class Message:
    """
    Represents a message in a conversation.
    """

    def __init__(self, role: str, content: str) -> None:
        """
        Initialize a message.

        Args:
            role: The role of the sender (system, user, assistant)
            content: The content of the message
        """
        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        """Convert message to a dictionary representation."""
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "Message":
        """Create a message from a dictionary representation."""
        return cls(role=data["role"], content=data["content"])

    def __str__(self) -> str:
        return f"{self.role.upper()}: {self.content}"


class PromptStrategy(ABC):
    """
    Base class for all prompt strategies.

    A prompt strategy defines how messages are processed and sent to an LLM,
    and how results are processed.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        template_path: Optional[str] = None,
        template_content: Optional[str] = None,
    ) -> None:
        """
        Initialize a prompt strategy.

        Args:
            llm_provider: The LLM provider to use
            template_path: Path to the template file (optional)
            template_content: Template content as string (optional)

        Note:
            Either template_path or template_content must be provided.
        """
        self.llm_provider = llm_provider

        # Get template content
        if template_path:
            # Get absolute path to the template
            if not os.path.isabs(template_path):
                # Try to find the template in the configs/prompt_templates directory
                base_dir = os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
                template_dir = os.path.join(base_dir, "configs", "prompt_templates")
                abs_template_path = os.path.join(template_dir, template_path)

                # If not found, use the provided path
                if not os.path.exists(abs_template_path):
                    abs_template_path = template_path
            else:
                abs_template_path = template_path

            with open(abs_template_path, "r") as f:
                self.template = f.read()
        elif template_content:
            self.template = template_content
        else:
            raise ValueError("Either template_path or template_content must be provided")

        # Storage for conversation history
        self.messages: List[Message] = []

    def add_system_message(self, content: str) -> None:
        """
        Add a system message to the conversation.

        Args:
            content: The content of the system message
        """
        self.messages.append(Message("system", content))

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the conversation.

        Args:
            content: The content of the user message
        """
        self.messages.append(Message("user", content))

    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to the conversation.

        Args:
            content: The content of the assistant message
        """
        self.messages.append(Message("assistant", content))

    def render_template(self, **kwargs: Any) -> str:
        """
        Render the template with the provided variables.

        Args:
            **kwargs: Variables to inject into the template

        Returns:
            Rendered template
        """
        return self.template.format(**kwargs)

    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """
        Get messages in the format expected by the LLM provider.

        Returns:
            List of message dictionaries
        """
        return [msg.to_dict() for msg in self.messages]

    @abstractmethod
    async def execute(self, instruction: str, **kwargs: Any) -> str:
        """
        Execute the prompt strategy with the given instruction.

        Args:
            instruction: The instruction to process
            **kwargs: Additional arguments for processing

        Returns:
            The result of the strategy execution
        """
        pass

    @abstractmethod
    def post_process(self, response: Dict[str, Any]) -> str:
        """
        Process the response from the LLM.

        Args:
            response: The raw response from the LLM

        Returns:
            Processed content
        """
        pass

    async def get_llm_response(
        self, temperature: float = 0.7, max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get a response from the LLM.

        Args:
            temperature: Controls randomness in the output
            max_tokens: Maximum number of tokens to generate

        Returns:
            Raw LLM response
        """
        messages = self.get_messages_for_llm()
        logger.debug(f"Sending {len(messages)} messages to LLM")

        response = await self.llm_provider.generate(
            messages=messages, temperature=temperature, max_tokens=max_tokens
        )

        return response

    def clear_messages(self) -> None:
        """Clear all messages in the conversation."""
        self.messages = []
