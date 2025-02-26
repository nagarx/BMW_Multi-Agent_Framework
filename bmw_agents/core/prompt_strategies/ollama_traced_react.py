"""
Ollama-specific TracedReAct prompt strategy for the BMW Agents framework.
This module implements a version of TracedReAct optimized for Ollama models.
"""

from typing import Dict, Any, Optional, List, Tuple

from bmw_agents.core.prompt_strategies.traced_react import TracedReAct
from bmw_agents.utils.llm_providers import OllamaProvider
from bmw_agents.utils.logger import get_logger

logger = get_logger("prompt_strategies.ollama_traced_react")

class OllamaTracedReAct(TracedReAct):
    """
    Implementation of the TracedReAct prompt strategy optimized for Ollama models.
    
    This strategy adapts the TracedReAct strategy to handle Ollama's specific output format
    and templating requirements.
    """
    
    def __init__(self, 
                 llm_provider: OllamaProvider, 
                 tools: List[Any] = None,
                 template_path: str = "ollama_react.txt",
                 max_iterations: int = 10,
                 termination_sequence: str = "FINAL ANSWER:"):
        """
        Initialize an Ollama-specific TracedReAct prompt strategy.
        
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
                f"OllamaTracedReAct is optimized for OllamaProvider, "
                f"but received {type(llm_provider).__name__}. This may cause issues."
            )
            
        super().__init__(
            llm_provider, 
            tools, 
            template_path, 
            max_iterations, 
            termination_sequence
        )
        
    async def execute(self, instruction: str, **kwargs) -> str:
        """
        Execute the Ollama-specific TracedReAct prompt strategy with the given instruction.
        
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
        
        logger.debug(f"Executing TracedReAct with Ollama-enhanced instruction")
        return await super().execute(enhanced_instruction, **kwargs) 