"""
Single-response TracedReAct prompt strategy for the BMW Agents framework.
This module implements a version of TracedReAct that parses a complete execution trace from a single LLM response.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from bmw_agents.core.prompt_strategies.react import ReActPromptStrategy, Tool
from bmw_agents.core.prompt_strategies.base import PromptStrategy
from bmw_agents.core.toolbox.tool import BaseTool
from bmw_agents.utils.llm_providers import LLMProvider
from bmw_agents.utils.logger import get_logger

logger = get_logger("prompt_strategies.single_response_traced_react")


class SingleResponseTracedReAct(PromptStrategy):
    """
    Implementation of the ReAct prompt strategy that extracts a complete execution trace
    from a single LLM response.

    This strategy is optimized for models that generate the entire thought/action/observation
    sequence in a single response, rather than through iterative calls.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        tools: Optional[List[BaseTool]] = None,
        template_path: str = "react.txt",
        termination_sequence: str = "FINAL ANSWER:",
    ) -> None:
        """
        Initialize a SingleResponseTracedReAct prompt strategy.

        Args:
            llm_provider: The LLM provider to use
            tools: List of tools available to the agent
            template_path: Path to the template file
            termination_sequence: Sequence that marks the end of execution
        """
        super().__init__(llm_provider, template_path=template_path)
        self.tools = tools or []
        self.termination_sequence = termination_sequence

        # Initialize trace arrays
        self.thoughts: List[str] = []
        self.actions: List[Dict[str, Any]] = []
        self.observations: List[str] = []
        self.result = ""

    def get_tool_descriptions(self) -> str:
        """
        Get a formatted string describing all available tools.

        Returns:
            Formatted tool descriptions
        """
        if not self.tools:
            return "No tools available."

        descriptions = []
        for i, tool in enumerate(self.tools):
            descriptions.append(f"{i+1}. {tool.name}: {tool.description}")

        return "\n".join(descriptions)

    def find_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """
        Find a tool by its name.

        Args:
            name: The name of the tool to find

        Returns:
            The tool object if found, None otherwise
        """
        name = name.lower().strip()
        for tool in self.tools:
            if tool.name.lower() == name:
                return tool
        return None

    async def run(self, instruction: str) -> Dict[str, Any]:
        """
        Run the SingleResponseTracedReAct prompt strategy.

        Args:
            instruction: The instruction/query to execute

        Returns:
            Dictionary with result and execution trace
        """
        # Reset trace arrays
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.result = ""

        # Execute the strategy
        await self.execute(instruction)

        # Return both the result and the trace
        return {
            "result": self.result,
            "trace": {
                "thoughts": self.thoughts,
                "actions": self.actions,
                "observations": self.observations,
            },
        }

    async def execute(self, instruction: str, **kwargs: Any) -> str:
        """
        Execute the SingleResponseTracedReAct prompt strategy.

        Args:
            instruction: The instruction/query to process
            **kwargs: Additional variables for the template

        Returns:
            The final result after extracting from the response
        """
        logger.info("Starting SingleResponseTracedReAct execution")
        
        # Start with a clean slate
        self.clear_messages()

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

        # Get LLM response (single call)
        logger.info("Sending request to LLM")
        response = await self.get_llm_response(
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", None),
        )

        # Extract content
        content = response["content"]
        self.add_assistant_message(content)

        # Process the content to extract the execution trace
        logger.info("Processing LLM response")
        final_answer = await self.process_response(content)

        logger.info(
            f"Completed SingleResponseTracedReAct Execution with {len(self.thoughts)} thoughts, {len(self.actions)} actions, and {len(self.observations)} observations"
        )
        return final_answer

    async def process_response(self, content: str) -> str:
        """
        Process the full LLM response to extract the execution trace and final answer.

        Args:
            content: The full LLM response content

        Returns:
            The final answer extracted from the response
        """
        # First, extract the final answer if present
        final_answer = ""
        if self.termination_sequence in content:
            final_parts = content.split(self.termination_sequence, 1)
            if len(final_parts) > 1:
                final_answer = final_parts[1].strip()
                content = final_parts[
                    0
                ]  # Remove the final answer from the content for further processing

        # Store the final answer
        self.result = final_answer

        # Extract all thought-action-observation sequences
        pattern = r"(Thought:[\s\S]*?)(Action:[\s\S]*?)(Observation:[\s\S]*?)(?=Thought:|$)"
        sequences = re.finditer(pattern, content, re.IGNORECASE)

        for match in sequences:
            thought_text = match.group(1).strip()
            action_text = match.group(2).strip()
            observation_text = match.group(3).strip()

            # Clean up the texts and extract the content after the labels
            thought = re.sub(r"^Thought:\s*", "", thought_text, flags=re.IGNORECASE).strip()
            action_str = re.sub(r"^Action:\s*", "", action_text, flags=re.IGNORECASE).strip()
            observation = re.sub(
                r"^Observation:\s*", "", observation_text, flags=re.IGNORECASE
            ).strip()

            # Parse the action JSON
            action = None
            try:
                # Try to extract JSON object
                json_pattern = r"{.*}"
                json_match = re.search(json_pattern, action_str, re.DOTALL)
                if json_match:
                    action_json = json.loads(json_match.group(0))

                    # If we have action_json but no 'tool' key, see if we can infer it
                    if "tool" not in action_json and len(self.tools) == 1:
                        action_json["tool"] = self.tools[0].name

                    # If the action has a tool key, use it
                    if "tool" in action_json:
                        action = action_json
            except Exception as e:
                logger.warning(f"Failed to parse action as JSON: {e}")

            # Add to trace arrays
            if thought:
                self.thoughts.append(thought)

            if action:
                self.actions.append(action)

                # If observation is blank or doesn't look like a real observation,
                # actually execute the tool to get the real observation
                should_execute = not observation or observation.lower() in [
                    "i need to execute this action",
                    "executing the action",
                    "awaiting result",
                ]

                if should_execute:
                    try:
                        real_observation = await self.execute_action(action)
                        observation = real_observation
                    except Exception as e:
                        logger.error(f"Error executing action: {e}")

            if observation:
                self.observations.append(observation)

        # Print debug info
        logger.debug(
            f"Extracted {len(self.thoughts)} thoughts, {len(self.actions)} actions, and {len(self.observations)} observations"
        )

        return final_answer

    async def execute_action(self, action: Dict[str, Any]) -> str:
        """
        Execute the specified action using the appropriate tool.

        Args:
            action: The action to execute, with 'tool' and 'args' keys

        Returns:
            The result of the action execution
        """
        tool_name = action.get("tool")
        args = action.get("args", {})

        if not tool_name:
            return "Error: No tool specified in the action."

        tool = self.find_tool_by_name(tool_name)
        if not tool:
            return f"Error: Tool '{tool_name}' not found."

        try:
            logger.info(f"Executing tool: {tool_name}")
            result = await tool.execute(**args)
            return str(result)
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            logger.error(error_msg)
            return error_msg

    def post_process(self, response: Dict[str, Any]) -> str:
        """
        Process the response from the LLM.

        Args:
            response: The raw response from the LLM

        Returns:
            Processed content
        """
        return response["content"]
