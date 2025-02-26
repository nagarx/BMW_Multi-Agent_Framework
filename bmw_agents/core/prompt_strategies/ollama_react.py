"""
Ollama-specific ReAct prompt strategy for the BMW Agents framework.
This module implements ReAct prompt strategy optimized for Ollama models.
"""

from typing import Any, Dict, List

from bmw_agents.core.prompt_strategies.react import ReActPromptStrategy, Tool
from bmw_agents.utils.llm_providers import OllamaProvider
from bmw_agents.utils.logger import get_logger

logger = get_logger("prompt_strategies.ollama_react")


class OllamaReActPromptStrategy(ReActPromptStrategy):
    """
    Implementation of the ReAct prompt strategy optimized for Ollama models.

    This strategy adapts the ReAct strategy to handle Ollama's specific output format
    and templating requirements.
    """

    def __init__(
        self,
        llm_provider: OllamaProvider,
        tools: List[Tool] = None,
        template_path: str = "ollama_react.txt",
        max_iterations: int = 10,
        termination_sequence: str = "FINAL ANSWER:",
    ) -> None:
        """
        Initialize an Ollama-specific ReAct prompt strategy.

        Args:
            llm_provider: The Ollama LLM provider to use
            tools: List of tools available to the agent
            template_path: Path to the template file optimized for Ollama
            max_iterations: Maximum number of iterations
            termination_sequence: Sequence that marks the end of execution
        """
        # Ensure provider is an OllamaProvider
        if not isinstance(llm_provider, OllamaProvider):
            logger.warning(
                f"OllamaReActPromptStrategy is optimized for OllamaProvider, "
                f"but received {type(llm_provider).__name__}. This may cause issues."
            )

        super().__init__(llm_provider, tools, template_path, max_iterations, termination_sequence)

    async def execute(self, instruction: str, **kwargs: Any) -> str:
        """
        Execute the Ollama-specific ReAct prompt strategy with the given instruction.

        This implementation provides additional context in system messages to optimize
        for Ollama model behavior.

        Args:
            instruction: The instruction to execute
            **kwargs: Additional variables for the template

        Returns:
            The final result after the ReAct loop
        """
        # Add special hint for Ollama models about the expected output format
        enhanced_instruction = (
            f"{instruction}\n\n"
            f"Remember to follow the exact format specified. For each step, first provide your "
            f"reasoning as 'Thought:', then specify your action as 'Action:' with valid JSON."
        )

        # Execute the base class implementation
        logger.debug(
            f"Before execute - Thoughts: {len(self.thoughts)}, "
            f"Actions: {len(self.actions)}, Observations: {len(self.observations)}"
        )
        result = await super().execute(enhanced_instruction, **kwargs)
        logger.debug(
            f"After execute - Thoughts: {len(self.thoughts)}, "
            f"Actions: {len(self.actions)}, Observations: {len(self.observations)}"
        )

        return result

    async def run(self, instruction: str) -> Dict[str, Any]:
        """
        Run the Ollama-specific ReAct prompt strategy with the given instruction.

        This method exists for backwards compatibility. It calls execute() and formats the result.

        Args:
            instruction: The instruction to execute

        Returns:
            Dictionary with result and execution trace
        """
        # Reset the arrays
        self.thoughts = []
        self.actions = []
        self.observations = []

        # Execute the strategy
        result = await self.execute(instruction)

        # Debug trace arrays
        logger.debug(f"Thoughts: {len(self.thoughts)}")
        logger.debug(f"Actions: {len(self.actions)}")
        logger.debug(f"Observations: {len(self.observations)}")

        return {
            "result": result,
            "trace": {
                "thoughts": self.thoughts,
                "actions": self.actions,
                "observations": self.observations,
            },
        }
