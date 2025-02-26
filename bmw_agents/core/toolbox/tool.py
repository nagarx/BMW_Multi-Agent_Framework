"""
Tool module for the BMW Agents framework.
This module defines the interface for tools that agents can use.
"""

import inspect
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional, List, get_type_hints, Union

from bmw_agents.utils.logger import get_logger

logger = get_logger("toolbox.tool")

class Tool(ABC):
    """
    Base class for all tools that can be used by agents.
    
    A tool is a function that can be called by an agent to perform some action.
    It has a name, description, and schema for its input and output.
    """
    
    def __init__(self, name: str, description: str, function: Callable):
        """
        Initialize a tool.
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
            function: The function to call when the tool is executed
        """
        self.name = name
        self.description = description
        self.function = function
        
        # Extract parameter info from the function
        self.parameters = self._get_parameter_info()
    
    def _get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract parameter information from the function.
        
        Returns:
            Dictionary mapping parameter names to their type and description
        """
        params = {}
        sig = inspect.signature(self.function)
        type_hints = get_type_hints(self.function)
        
        # Get docstring to extract parameter descriptions
        docstring = inspect.getdoc(self.function) or ""
        param_descriptions = {}
        
        # Simple parsing of docstring to extract parameter descriptions
        lines = docstring.split("\n")
        in_params_section = False
        current_param = None
        
        for line in lines:
            line = line.strip()
            
            # Look for Parameters section
            if line.lower().startswith("parameters:") or line.lower().startswith("args:"):
                in_params_section = True
                continue
            
            # Exit parameters section when we hit another section
            if in_params_section and line and line.endswith(":") and not line.startswith(" "):
                in_params_section = False
                continue
            
            # Parse parameter descriptions
            if in_params_section and line:
                param_match = line.split(":", 1)
                if len(param_match) == 2:
                    current_param = param_match[0].strip()
                    param_descriptions[current_param] = param_match[1].strip()
                elif current_param and line.startswith(" "):
                    # Continuation of previous parameter description
                    param_descriptions[current_param] += " " + line.strip()
        
        # Build parameter info dictionary
        for name, param in sig.parameters.items():
            # Skip 'self' parameter for methods
            if name == 'self':
                continue
                
            param_type = type_hints.get(name, Any).__name__
            description = param_descriptions.get(name, f"Parameter '{name}'")
            
            params[name] = {
                "type": param_type,
                "description": description,
                "required": param.default == inspect.Parameter.empty
            }
        
        return params
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for the tool.
        
        Returns:
            Dictionary with tool name, description, and parameters
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
    
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool with the given arguments.
        
        Args:
            **kwargs: Arguments for the tool
            
        Returns:
            The result of the tool execution
        """
        try:
            # Check if the function is async
            if inspect.iscoroutinefunction(self.function):
                result = await self.function(**kwargs)
            else:
                result = self.function(**kwargs)
                
            return result
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {str(e)}")
            raise
    
    def __str__(self) -> str:
        """Get string representation of the tool."""
        params_str = ", ".join([f"{name}: {info['type']}" for name, info in self.parameters.items()])
        return f"{self.name}({params_str}): {self.description}"


class SimpleTool(Tool):
    """
    A simple implementation of the Tool interface.
    
    This class can be used to create tools from existing functions
    with minimal configuration.
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 function: Callable, 
                 parameter_descriptions: Optional[Dict[str, str]] = None):
        """
        Initialize a simple tool.
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
            function: The function to call when the tool is executed
            parameter_descriptions: Optional dictionary mapping parameter names to descriptions
        """
        # Override docstring with provided parameter descriptions
        if parameter_descriptions:
            original_doc = function.__doc__ or ""
            params_doc = "Parameters:\n"
            for param, desc in parameter_descriptions.items():
                params_doc += f"    {param}: {desc}\n"
            
            # Preserve original docstring if it exists, but replace or add parameters section
            if "Parameters:" in original_doc or "Args:" in original_doc:
                # Replace existing parameters section
                new_doc = original_doc.split("Parameters:")[0]
                new_doc += params_doc
            else:
                # Add parameters section
                new_doc = original_doc.strip() + "\n\n" + params_doc
            
            function.__doc__ = new_doc
        
        super().__init__(name, description, function)


class FunctionTool(Tool):
    """
    Tool that wraps a function with predefined schema.
    
    This class allows for more control over the schema than SimpleTool,
    and is useful when the function doesn't have appropriate type hints or docstrings.
    """
    
    def __init__(self, 
                 name: str, 
                 description: str, 
                 function: Callable, 
                 parameters: Dict[str, Dict[str, Any]]):
        """
        Initialize a function tool.
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
            function: The function to call when the tool is executed
            parameters: Dictionary mapping parameter names to their schemas
        """
        self.name = name
        self.description = description
        self.function = function
        self.parameters = parameters
    
    def _get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get parameter information from the provided schema.
        
        Returns:
            The parameters dictionary as provided in __init__
        """
        return self.parameters 