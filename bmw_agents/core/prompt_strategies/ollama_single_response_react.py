"""
Ollama-specific implementation of the SingleResponseTracedReAct prompt strategy.
"""

import os
from typing import Any, List, Optional

from bmw_agents.core.prompt_strategies.react import Tool
from bmw_agents.core.prompt_strategies.single_response_traced_react import (
    SingleResponseTracedReAct,
)
from bmw_agents.utils.llm_providers import OllamaProvider
from bmw_agents.utils.logger import get_logger

logger = get_logger("prompt_strategies.ollama_single_response_react")


class OllamaSingleResponseTracedReActPromptStrategy(SingleResponseTracedReAct):
    """
    Implementation of the SingleResponseTracedReAct strategy optimized for Ollama models.

    This strategy adapts the SingleResponseTracedReAct strategy to handle Ollama's specific
    output format and templating requirements.
    """

    def __init__(
        self,
        llm_provider: OllamaProvider,
        tools: Optional[List[Tool]] = None,
        template_path: Optional[str] = None,
        termination_sequence: str = "FINAL ANSWER:",
    ) -> None:
        """
        Initialize the Ollama-specific SingleResponseTracedReAct strategy.

        Args:
            llm_provider: Ollama provider instance
            tools: List of tools available to the agent
            template_path: Path to the prompt template
            termination_sequence: Sequence indicating the final answer
        """
        super().__init__(
            llm_provider=llm_provider,
            tools=tools,
            template_path=template_path or "ollama_react.txt",
            termination_sequence=termination_sequence,
        )

    async def execute(self, instruction: str, **kwargs: Any) -> str:
        """
        Execute the Ollama-specific SingleResponseTracedReAct strategy with the given instruction.

        This implementation enhances the instruction with specific formatting guidance for Ollama
        models, which tend to generate the entire execution trace in a single response. The enhanced
        instruction includes explicit reminders about the expected format (Thought/Action/Observation
        sequence), helping the model generate responses that can be properly parsed.

        Args:
            instruction: The instruction to execute
            **kwargs: Additional variables for the template including:
                - temperature: Controls randomness of the output (default: 0.7)
                - max_tokens: Maximum tokens to generate (optional)
                - Any other variables needed by the template

        Returns:
            The final result after extracting from the response
        """
        # Add special hint for Ollama models about the expected output format
        enhanced_instruction = (
            f"{instruction}\n\n"
            f"Remember to follow the exact format specified. For each step, first provide your "
            f"reasoning as 'Thought:', then specify your action as 'Action:' with valid JSON, "
            f"followed by 'Observation:' with the result. Repeat this format for each step until "
            f"you have enough information to provide the final answer."
        )

        logger.debug("Executing SingleResponseTracedReAct with Ollama-enhanced instruction")
        return await super().execute(enhanced_instruction, **kwargs)


# Alias for backward compatibility
OllamaSingleResponseReAct = OllamaSingleResponseTracedReActPromptStrategy
