"""
Traced PlanReAct prompt strategy for the BMW Agents framework.
This module implements a version of PlanReAct strategy that explicitly tracks execution trace.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from bmw_agents.core.prompt_strategies.traced_react import TracedReAct
from bmw_agents.utils.llm_providers import LLMProvider
from bmw_agents.utils.logger import get_logger

logger = get_logger("prompt_strategies.traced_plan_react")


class TracedPlanReAct(TracedReAct):
    """
    Implementation of the PlanReAct prompt strategy with explicit trace tracking.

    This strategy extends TracedReAct by adding an explicit planning step
    before the thought-action-observation cycle.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        tools: List[Any] = None,
        template_path: str = "plan_react.txt",
        max_iterations: int = 10,
        termination_sequence: str = "FINAL ANSWER:",
    ) -> None:
        """
        Initialize a TracedPlanReAct prompt strategy.

        Args:
            llm_provider: The LLM provider to use
            tools: List of tools available to the agent
            template_path: Path to the template file
            max_iterations: Maximum number of iterations
            termination_sequence: Sequence that marks the end of execution
        """
        super().__init__(llm_provider, tools, template_path, max_iterations, termination_sequence)

        # Track plans
        self.plans = []

    async def run(self, instruction: str) -> Dict[str, Any]:
        """
        Run the TracedPlanReAct prompt strategy.

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

        # Execute the strategy
        final_answer = await self.execute(instruction)

        # Return both the result and the trace
        return {
            "result": final_answer,
            "trace": {
                "plan": self.plans[0] if self.plans else "",
                "thoughts": self.thoughts,
                "actions": self.actions,
                "observations": self.observations,
            },
        }

    async def execute(self, instruction: str, **kwargs: Any) -> str:
        """
        Execute the TracedPlanReAct prompt strategy.

        Args:
            instruction: The instruction/query to process
            **kwargs: Additional variables for the template

        Returns:
            The final result after the PlanReAct loop
        """
        logger.info("Starting TracedPlanReAct execution")
        
        self.clear_messages()

        # Call the base TracedReAct execution
        result = await super().execute(instruction, **kwargs)

        # Log trace
        logger.info(
            f"Completed TracedPlanReAct Execution with {len(self.plans)} plans, {len(self.thoughts)} thoughts, {len(self.actions)} actions, and {len(self.observations)} observations"
        )
        return result

    def extract_thought_and_action(self, content: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Extract thought, action, and plan from the LLM response.

        Extends the base implementation to also extract plans.

        Args:
            content: The LLM response content

        Returns:
            Tuple of (thought, action) where action is a dictionary or None
        """
        # Extract plan if present
        plan_pattern = r"\*{0,2}Plan:?\*{0,2}(.*?)(?:\*{0,2}Thought:?\*{0,2}|\*{0,2}Action:?\*{0,2}|\*{0,2}FINAL ANSWER:?\*{0,2}|$)"
        plan_match = re.search(plan_pattern, content, re.DOTALL | re.IGNORECASE)

        if plan_match and plan_match.group(1).strip():
            plan = plan_match.group(1).strip()
            if plan not in self.plans:
                self.plans.append(plan)

        # Use the parent implementation for thought and action
        return super().extract_thought_and_action(content)
