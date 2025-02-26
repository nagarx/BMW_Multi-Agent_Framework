"""
Ollama-specific SingleResponseTracedPlanReAct prompt strategy for the BMW Agents framework.
This module implements a version of TracedPlanReAct optimized for Ollama models that generate
a complete execution trace in a single response, including plan extraction.
"""

from typing import Dict, Any, Optional, List, Tuple

from bmw_agents.core.prompt_strategies.single_response_traced_plan_react import SingleResponseTracedPlanReAct
from bmw_agents.utils.llm_providers import OllamaProvider
from bmw_agents.utils.logger import get_logger

logger = get_logger("prompt_strategies.ollama_single_response_plan_react")

class OllamaSingleResponsePlanReAct(SingleResponseTracedPlanReAct):
    """
    Implementation of the SingleResponseTracedPlanReAct strategy optimized for Ollama models.
    
    This strategy adapts the SingleResponseTracedPlanReAct strategy to handle Ollama's specific output format
    and templating requirements, including plan extraction.
    """
    
    def __init__(self, 
                 llm_provider: OllamaProvider, 
                 tools: List[Any] = None,
                 template_path: str = "ollama_plan_react.txt",
                 termination_sequence: str = "FINAL ANSWER:"):
        """
        Initialize an Ollama-specific SingleResponseTracedPlanReAct strategy.
        
        Args:
            llm_provider: The Ollama LLM provider to use
            tools: List of tools available to the agent
            template_path: Path to the template file optimized for Ollama
            termination_sequence: Sequence that marks the end of execution
        """
        # Ensure provider is an OllamaProvider
        if not isinstance(llm_provider, OllamaProvider):
            logger.warning(
                f"OllamaSingleResponsePlanReAct is optimized for OllamaProvider, "
                f"but received {type(llm_provider).__name__}. This may cause issues."
            )
            
        super().__init__(
            llm_provider, 
            tools, 
            template_path, 
            termination_sequence
        )
        
    async def execute(self, instruction: str, **kwargs) -> str:
        """
        Execute the Ollama-specific SingleResponseTracedPlanReAct strategy with the given instruction.
        
        This implementation enhances the instruction with specific formatting guidance for Ollama models,
        which tend to generate the entire execution trace in a single response. The enhanced instruction
        includes explicit reminders about the expected format (Plan/Thought/Action/Observation sequence),
        helping the model generate responses that can be properly parsed. The key difference from the
        regular SingleResponseTracedReAct is the inclusion of an explicit planning step at the beginning.
        
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
            f"Remember to follow the exact format specified. First create a plan starting with 'Plan:', "
            f"then for each step provide your reasoning as 'Thought:', specify your action as 'Action:' with valid JSON, "
            f"followed by 'Observation:' with the result. Repeat this format for each step until "
            f"you have enough information to provide the final answer."
        )
        
        logger.debug(f"Executing SingleResponseTracedPlanReAct with Ollama-enhanced instruction")
        return await super().execute(enhanced_instruction, **kwargs) 