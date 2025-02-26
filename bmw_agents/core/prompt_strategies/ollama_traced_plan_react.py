"""
Ollama-specific TracedPlanReAct prompt strategy for the BMW Agents framework.
This module implements a version of TracedPlanReAct optimized for Ollama models.
"""

from typing import Dict, Any, Optional, List, Tuple

from bmw_agents.core.prompt_strategies.traced_plan_react import TracedPlanReAct
from bmw_agents.utils.llm_providers import OllamaProvider
from bmw_agents.utils.logger import get_logger

logger = get_logger("prompt_strategies.ollama_traced_plan_react")

class OllamaTracedPlanReAct(TracedPlanReAct):
    """
    Implementation of the TracedPlanReAct prompt strategy optimized for Ollama models.
    
    This strategy adapts the TracedPlanReAct strategy to handle Ollama's specific output format
    and templating requirements.
    """
    
    def __init__(self, 
                 llm_provider: OllamaProvider, 
                 tools: List[Any] = None,
                 template_path: str = "ollama_plan_react.txt",
                 max_iterations: int = 10,
                 termination_sequence: str = "FINAL ANSWER:"):
        """
        Initialize an Ollama-specific TracedPlanReAct prompt strategy.
        
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
                f"OllamaTracedPlanReAct is optimized for OllamaProvider, "
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
        Execute the Ollama-specific TracedPlanReAct prompt strategy with the given instruction.
        
        This implementation provides additional context in system messages to optimize
        for Ollama model behavior.
        
        Args:
            instruction: The instruction to execute
            **kwargs: Additional variables for the template
            
        Returns:
            The final result after the PlanReAct loop
        """
        # Add special hint for Ollama models about the expected output format
        enhanced_instruction = (
            f"{instruction}\n\n"
            f"Remember to follow the exact format specified. First create a plan starting with 'Plan:', "
            f"then for each step provide your reasoning as 'Thought:', followed by your action as 'Action:' with valid JSON."
        )
        
        logger.debug(f"Executing TracedPlanReAct with Ollama-enhanced instruction")
        return await super().execute(enhanced_instruction, **kwargs) 