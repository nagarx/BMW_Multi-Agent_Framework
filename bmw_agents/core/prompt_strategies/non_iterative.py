"""
Non-iterative prompt strategy for the BMW Agents framework.
This module implements a simple one-off strategy for LLM calls.
"""

import json
from typing import Dict, Any, Optional, Union, Callable, List

from bmw_agents.core.prompt_strategies.base import PromptStrategy, Message
from bmw_agents.utils.logger import get_logger
from bmw_agents.utils.llm_providers import LLMProvider

logger = get_logger("prompt_strategies.non_iterative")

class NonIterativePromptStrategy(PromptStrategy):
    """
    A simple non-iterative prompt strategy that makes a single LLM call.
    
    This strategy is suitable for tasks that don't require iterative execution
    or tool usage, such as planning and verification.
    """
    
    def __init__(self, 
                 llm_provider: LLMProvider, 
                 template_path: Optional[str] = None,
                 template_content: Optional[str] = None,
                 post_processor: Optional[Callable[[str], str]] = None):
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
    
    async def execute(self, instruction: str, **kwargs) -> str:
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
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", None)
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
    
    def __init__(self, 
                 llm_provider: LLMProvider, 
                 template_path: Optional[str] = None,
                 template_content: Optional[str] = None,
                 schema: Optional[Dict[str, Any]] = None,
                 retries: int = 2):
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
    
    async def execute(self, instruction: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the prompt strategy with the given instruction.
        
        This overrides the parent method to return a dictionary instead of a string.
        
        Args:
            instruction: The instruction/query to process
            **kwargs: Additional variables for the template
            
        Returns:
            Parsed JSON response as a dictionary
        """
        # Attempt to get a valid JSON response with retries
        for attempt in range(self.retries + 1):
            try:
                # Call the parent's execute method to get the raw response
                result_str = await super().execute(instruction, **kwargs)
                
                # Try to parse as JSON
                result_json = json.loads(result_str)
                
                # Validate against schema if provided
                if self.schema:
                    # TODO: Implement schema validation
                    pass
                
                return result_json
            
            except json.JSONDecodeError as e:
                if attempt < self.retries:
                    logger.warning(f"Failed to parse JSON (attempt {attempt+1}/{self.retries+1}): {e}")
                    # Provide feedback and retry
                    self.add_user_message(
                        "Your response was not valid JSON. Please provide a response in proper JSON format."
                    )
                else:
                    logger.error(f"Failed to parse JSON after {self.retries+1} attempts: {e}")
                    raise
    
    def post_process(self, response: Dict[str, Any]) -> str:
        """
        Process the response from the LLM.
        
        For the JSON strategy, we just return the raw content and handle JSON
        parsing in the execute method.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            Raw content string
        """
        return response["content"]


class PlannerPromptStrategy(JSONPromptStrategy):
    """
    A specialized JSON prompt strategy for the Planner agent.
    
    This strategy expects a JSON output containing tasks and their dependencies.
    """
    
    def __init__(self, 
                 llm_provider: LLMProvider, 
                 template_path: str = "planner.txt"):
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
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["id", "description", "dependencies"]
                    }
                }
            },
            "required": ["tasks"]
        }


class VerifierPromptStrategy(NonIterativePromptStrategy):
    """
    A specialized non-iterative prompt strategy for the Verifier agent.
    
    This strategy expects a boolean output indicating whether the 
    execution result satisfies the original instruction.
    """
    
    def __init__(self, 
                 llm_provider: LLMProvider, 
                 template_path: str = "verifier.txt"):
        """
        Initialize a verifier prompt strategy.
        
        Args:
            llm_provider: The LLM provider to use
            template_path: Path to the template file (default: "verifier.txt")
        """
        super().__init__(llm_provider, template_path=template_path)
    
    def post_process(self, response: Dict[str, Any]) -> bool:
        """
        Process the response from the LLM and convert it to a boolean.
        
        Args:
            response: The raw response from the LLM
            
        Returns:
            Boolean indicating verification success
        """
        content = response["content"].lower().strip()
        
        # Look for positive indicators
        if "yes" in content or "true" in content or "verified" in content or "success" in content:
            return True
        
        # Look for negative indicators
        elif "no" in content or "false" in content or "not verified" in content or "failure" in content:
            return False
        
        # If we can't determine, log a warning and default to False
        logger.warning(f"Unable to determine verification result from: {content}")
        return False 