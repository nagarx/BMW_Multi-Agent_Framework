"""
PlanReAct prompt strategy for the BMW Agents framework.
This module implements the PlanReAct prompt strategy, which extends ReAct with explicit planning.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from bmw_agents.core.prompt_strategies.react import ReActPromptStrategy, Tool
from bmw_agents.utils.llm_providers import LLMProvider
from bmw_agents.utils.logger import get_logger

logger = get_logger("prompt_strategies.plan_react")


class PlanReActPromptStrategy(ReActPromptStrategy):
    """
    PlanReAct prompt strategy for the BMW Agents framework.

    This strategy extends ReAct by adding an explicit planning step at the beginning
    of the execution. The LLM first creates a plan, then follows the ReAct approach for
    each step in the plan.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        tools: Optional[List[Tool]] = None,
        template_path: Optional[str] = None,
        max_iterations: int = 10,
        termination_sequence: str = "FINAL ANSWER:",
    ) -> None:
        """
        Initialize a PlanReAct prompt strategy.

        Args:
            llm_provider: The LLM provider to use for generating responses.
            tools: A list of tools that can be used by the LLM.
            template_path: Path to the template file. If not provided, a default template is used.
            max_iterations: Maximum number of iterations to run.
            termination_sequence: Sequence indicating the LLM has reached a final answer.
        """
        super().__init__(llm_provider, tools, template_path, max_iterations, termination_sequence)
        self.plans: List[str] = []

    def extract_plan_thought_and_action(
        self, content: str
    ) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        """
        Extract plan, thought, and action from the LLM response.

        Args:
            content: The LLM response content

        Returns:
            Tuple of (plan, thought, action) where action is a dictionary or None
        """
        # Extract plan
        plan_match = re.search(r"Plan:(.*?)(?:Thought:|$)", content, re.DOTALL)
        plan = plan_match.group(1).strip() if plan_match else ""

        # Use parent class to extract thought and action
        thought, action = self.extract_thought_and_action(content)

        return plan, thought, action

    async def execute(self, instruction: str, **kwargs: Any) -> str:
        """
        Execute the PlanReAct prompt strategy with the given instruction.

        Args:
            instruction: The instruction to execute.
            **kwargs: Additional arguments.

        Returns:
            The final result of the execution.
        """
        # Start with a clean slate
        self.clear_messages()
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.plans = []

        # Render the system message template
        variables = {
            "instruction": instruction,
            "tools": self.get_tool_descriptions(),
            "termination_sequence": self.termination_sequence,
            **kwargs,
        }
        system_content = self.render_template(**variables)

        # Add system message
        self.add_system_message(system_content)

        # Add the user instruction
        self.add_user_message(instruction)

        # Main PlanReAct loop
        iteration = 0
        final_answer = None

        logger.info("Starting PlanReAct execution")
        while iteration < self.max_iterations:
            logger.info(f"Starting iteration {iteration+1}/{self.max_iterations}")

            # Get LLM response
            response = await self.get_llm_response(
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", None),
            )

            # Extract content
            content = response["content"]
            self.add_assistant_message(content)

            # Check for termination
            if self.termination_sequence in content:
                # Extract final answer
                final_answer = content.split(self.termination_sequence, 1)[1].strip()
                logger.info(
                    f"Termination sequence found, final answer: {final_answer[:100]}..."
                )
                break

            # Process the response to extract plan, thought and action
            plan, thought, action = self.extract_plan_thought_and_action(content)
            self.plans.append(plan)
            self.thoughts.append(thought)

            # If no action is found or action is invalid, ask for clarification
            if not action:
                logger.warning("No valid action found in response")
                self.add_user_message(
                    "I couldn't understand your action. Please try again with a valid action."
                )
                iteration += 1
                continue

            # Execute the action
            observation = await self.execute_action(action)
            self.actions.append(action)
            self.observations.append(observation)

            # Add observation as user message
            self.add_user_message(f"Observation: {observation}")

            iteration += 1

        else:
            # If we reach max iterations without termination
            logger.warning(
                f"Reached maximum iterations ({self.max_iterations}) without termination"
            )
            final_answer = (
                "I wasn't able to complete the task within the allowed number of steps."
            )

        return final_answer or "No final answer was produced."

    def get_execution_trace(self) -> List[Dict[str, str]]:
        """
        Get the execution trace of the PlanReAct loop.

        Returns:
            List of dictionaries with 'plan', 'thought', 'action', and 'observation' keys
        """
        return [
            {
                "plan": plan,
                "thought": thought,
                "action": json.dumps(action) if action else None,
                "observation": observation,
            }
            for plan, thought, action, observation in zip(
                self.plans, self.thoughts, self.actions, self.observations
            )
        ]
