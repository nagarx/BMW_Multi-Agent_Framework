"""
Single-response TracedPlanReAct prompt strategy for the BMW Agents framework.
This module implements a version of TracedPlanReAct that parses a complete execution trace
from a single LLM response, including plan extraction.
"""

import json
import re
from typing import Dict, Any, Optional, List, Tuple, Callable

from bmw_agents.core.prompt_strategies.single_response_traced_react import SingleResponseTracedReAct
from bmw_agents.utils.logger import get_logger, LoggingContext
from bmw_agents.utils.llm_providers import LLMProvider

logger = get_logger("prompt_strategies.single_response_traced_plan_react")

class SingleResponseTracedPlanReAct(SingleResponseTracedReAct):
    """
    Implementation of the PlanReAct prompt strategy that extracts a complete execution trace
    from a single LLM response.
    
    This strategy extends SingleResponseTracedReAct by also extracting plan information
    from the LLM response.
    """
    
    def __init__(self, 
                 llm_provider: LLMProvider, 
                 tools: List[Any] = None,
                 template_path: str = "plan_react.txt",
                 termination_sequence: str = "FINAL ANSWER:"):
        """
        Initialize a SingleResponseTracedPlanReAct prompt strategy.
        
        Args:
            llm_provider: The LLM provider to use
            tools: List of tools available to the agent
            template_path: Path to the template file
            termination_sequence: Sequence that marks the end of execution
        """
        super().__init__(
            llm_provider, 
            tools, 
            template_path, 
            termination_sequence
        )
        
        # Track plans
        self.plans = []
    
    async def run(self, instruction: str) -> Dict[str, Any]:
        """
        Run the SingleResponseTracedPlanReAct prompt strategy.
        
        Args:
            instruction: The instruction/query to execute
            
        Returns:
            Dictionary with result and execution trace
        """
        # Reset trace arrays
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.plans = []
        self.result = ""
        
        # Execute the strategy
        await self.execute(instruction)
        
        # Return both the result and the trace
        return {
            "result": self.result,
            "trace": {
                "plan": self.plans[0] if self.plans else "",
                "thoughts": self.thoughts,
                "actions": self.actions,
                "observations": self.observations
            }
        }
    
    async def process_response(self, content: str) -> str:
        """
        Process the full LLM response to extract the execution trace, plan, and final answer.
        
        Extends the base implementation to also extract plans.
        
        Args:
            content: The full LLM response content
            
        Returns:
            The final answer extracted from the response
        """
        # Extract plan if present
        plan_pattern = r"\*{0,2}Plan:?\*{0,2}(.*?)(?:\*{0,2}Thought:?\*{0,2}|\*{0,2}Action:?\*{0,2}|\*{0,2}FINAL ANSWER:?\*{0,2}|$)"
        plan_match = re.search(plan_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if plan_match and plan_match.group(1).strip():
            plan = plan_match.group(1).strip()
            self.plans.append(plan)
            logger.debug(f"Extracted plan: {plan[:100]}...")
        
        # Call the parent implementation for the rest of the processing
        return await super().process_response(content) 