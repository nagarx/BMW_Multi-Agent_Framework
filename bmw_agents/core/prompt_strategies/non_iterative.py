"""
Non-iterative prompt strategy for the BMW Agents framework.
This module implements a simple one-off strategy for LLM calls.
"""

import json
from typing import Any, Callable, Dict, Optional, Coroutine

from bmw_agents.core.prompt_strategies.base import PromptStrategy
from bmw_agents.utils.llm_providers import LLMProvider
from bmw_agents.utils.logger import get_logger

logger = get_logger(__name__)


class NonIterativePromptStrategy(PromptStrategy):
    """
    A simple non-iterative prompt strategy that makes a single LLM call.

    This strategy is suitable for tasks that don't require iterative execution
    or tool usage, such as planning and verification.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        template_path: Optional[str] = None,
        template_content: Optional[str] = None,
        post_processor: Optional[Callable[[str], str]] = None,
    ) -> None:
        """
        Initialize a non-iterative prompt strategy.

        Args:
            llm_provider: The LLM provider to use
            template_path: Path to the template file (optional)
            template_content: Template content as string (optional)
            post_processor: Function to process the LLM response (optional)
        """
        super().__init__(llm_provider, template_path, template_content)
        self.post_processor = post_processor

    async def execute(self, instruction: str, **kwargs: Any) -> str:
        """
        Execute the prompt strategy with the given instruction.

        Args:
            instruction: The instruction/query to process
            **kwargs: Additional variables for the template

        Returns:
            The result of the LLM call
        """
        # Start with a clean slate
        self.clear_messages()

        # Render the system message template
        variables = {"instruction": instruction, **kwargs}
        system_content = self.render_template(**variables)

        # Add system message
        self.add_system_message(system_content)

        # Add the user instruction
        self.add_user_message(instruction)

        # Get response from LLM
        response = await self.get_llm_response(
            temperature=kwargs.get("temperature", 0.7), max_tokens=kwargs.get("max_tokens", None)
        )

        # Process the response
        result = self.post_process(response)

        # Add the assistant's response to the message history
        self.add_assistant_message(result)

        return result

    def post_process(self, response: Dict[str, Any]) -> str:
        """
        Process the response from the LLM.

        Args:
            response: The raw response from the LLM

        Returns:
            Processed content
        """
        content = response["content"]

        # Apply custom post-processor if provided
        if self.post_processor:
            content = self.post_processor(content)

        return content


class JSONPromptStrategy(NonIterativePromptStrategy):
    """
    A non-iterative prompt strategy that expects and validates JSON output.

    This strategy is suitable for structured tasks where the LLM should return
    data in a specific JSON format, such as the output of a planner.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        template_path: Optional[str] = None,
        template_content: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        retries: int = 2,
    ) -> None:
        """
        Initialize a JSON prompt strategy.

        Args:
            llm_provider: The LLM provider to use
            template_path: Path to the template file (optional)
            template_content: Template content as string (optional)
            schema: JSON schema to validate against (optional)
            retries: Number of retries if JSON parsing fails (default: 2)
        """
        super().__init__(llm_provider, template_path, template_content)
        self.schema = schema
        self.retries = retries

    async def execute(self, instruction: str, **kwargs: Any) -> str:
        """
        Execute the prompt strategy with the given instruction.

        This overrides the parent method to process JSON responses.

        Args:
            instruction: The instruction to execute
            **kwargs: Additional arguments to pass to the template renderer

        Returns:
            A JSON string containing the parsed response
        """
        # Add the instruction as a user message
        self.add_user_message(instruction)

        # Render the template and get the LLM response
        response = await self.get_llm_response(**kwargs)
        
        # Process the response
        result = self._parse_json_response(response)
        
        # Convert result to JSON string and return
        return json.dumps(result)

    def _parse_json_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse and validate the JSON response.
        
        Args:
            response: The response from the LLM
            
        Returns:
            The parsed JSON response
        """
        try:
            result_str = super().post_process(response)
            result_json = json.loads(result_str)
            
            # Validate against schema if provided
            if self.schema:
                # TODO: Implement schema validation
                pass
                
            return result_json
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise

    def post_process(self, response: Dict[str, Any]) -> str:
        """
        Process the LLM response for verification.

        Args:
            response: The raw response from the LLM

        Returns:
            The verification result as a string ("true" or "false")
        """
        # Extract the content from the response
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Look for indications of verification success
        lowercase_content = content.lower()
        verified = (
            "verified" in lowercase_content
            or "verification passed" in lowercase_content
            or "passes verification" in lowercase_content
            or "yes" in lowercase_content
        )

        # Return as string to match parent class return type
        return "true" if verified else "false"


class PlannerPromptStrategy(JSONPromptStrategy):
    """
    A specialized JSON prompt strategy for the Planner agent.

    This strategy expects a JSON output containing tasks and their dependencies.
    """

    def __init__(self, llm_provider: LLMProvider, template_path: str = "planner.txt") -> None:
        """
        Initialize a planner prompt strategy.

        Args:
            llm_provider: The LLM provider to use
            template_path: Path to the template file (default: "planner.txt")
        """
        super().__init__(llm_provider, template_path=template_path)

        # Define the expected JSON schema for planner output
        self.schema = {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "description": {"type": "string"},
                            "dependencies": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["id", "description", "dependencies"],
                    },
                }
            },
            "required": ["tasks"],
        }


class VerifierPromptStrategy(NonIterativePromptStrategy):
    """
    A specialized non-iterative prompt strategy for the Verifier agent.

    This strategy expects a boolean output indicating whether the
    execution result satisfies the original instruction.
    """

    def __init__(self, llm_provider: LLMProvider, template_path: str = "verifier.txt") -> None:
        """
        Initialize a verifier prompt strategy.

        Args:
            llm_provider: The LLM provider to use
            template_path: Path to the template file (default: "verifier.txt")
        """
        super().__init__(llm_provider, template_path=template_path)

    def post_process(self, response: Dict[str, Any]) -> str:
        """
        Process the LLM response and determine if verification passes.

        Args:
            response: The raw response from the LLM

        Returns:
            String indicating verification result ("true" or "false")
        """
        # Extract the content from the response
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Look for indications of verification success
        lowercase_content = content.lower()
        verified = (
            "verified" in lowercase_content
            or "verification passed" in lowercase_content
            or "passes verification" in lowercase_content
            or "yes" in lowercase_content
        )

        # Return as string to match parent class return type
        return "true" if verified else "false"
