"""
Tool module for the BMW Agents framework.
This module defines the interface for tools that agents can use.
"""

import inspect
from typing import Any, Callable, Dict, Optional, get_type_hints

from bmw_agents.utils.logger import get_logger

logger = get_logger("toolbox.tool")


class BaseTool:
    """
    Base class for tools that can be used in the BMW Agents framework.
    """

    def __init__(self, name: str, description: str, function: Callable) -> None:
        """
        Initialize a tool.

        Args:
            name: The name of the tool
            description: A description of what the tool does
            function: The function that implements the tool's functionality
        """
        self.name = name
        self.description = description
        self._function = function

        # Extract parameter info from the function
        self.parameters = self._get_parameter_info()

    def _get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract parameter information from the function.

        Returns:
            Dictionary mapping parameter names to their type and description
        """
        params = {}
        sig = inspect.signature(self._function)
        type_hints = get_type_hints(self._function)

        # Get docstring to extract parameter descriptions
        docstring = inspect.getdoc(self._function) or ""
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
            if name == "self":
                continue

            param_type = type_hints.get(name, Any).__name__
            description = param_descriptions.get(name, f"Parameter '{name}'")

            params[name] = {
                "type": param_type,
                "description": description,
                "required": param.default == inspect.Parameter.empty,
            }

        return params

    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for the tool.

        Returns:
            Dictionary with tool name, description, and parameters
        """
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    async def execute(self, **kwargs: Any) -> Any:
        """
        Execute the tool with the given arguments.

        Args:
            **kwargs: Arguments to pass to the tool function

        Returns:
            The result of executing the tool
        """
        try:
            if inspect.iscoroutinefunction(self._function):
                result = await self._function(**kwargs)
            else:
                result = self._function(**kwargs)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {self.name}: {str(e)}")
            raise

    def __str__(self) -> str:
        """Get string representation of the tool."""
        params_str = ", ".join(
            [f"{name}: {info['type']}" for name, info in self.parameters.items()]
        )
        return f"{self.name}({params_str}): {self.description}"


class Tool(BaseTool):
    """
    A tool that can be used in the BMW Agents framework.
    Includes parameter information for use in LLM prompts.
    """

    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Initialize a tool with parameter information.

        Args:
            name: The name of the tool
            description: A description of what the tool does
            function: The function that implements the tool's functionality
            parameters: Parameter information for the tool
        """
        super().__init__(name, description, function)
        self.parameters = self._get_parameters(parameters)


class SimpleTool(Tool):
    """
    A simple implementation of the Tool interface.

    This class can be used to create tools from existing functions
    with minimal configuration.
    """

    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        parameter_descriptions: Optional[Dict[str, str]] = None,
    ) -> None:
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
    A tool created from a function with type annotations.
    This automatically extracts parameter information from the function signature.
    """

    def __init__(
        self, name: str, description: str, function: Callable, parameters: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Initialize a function tool.

        Args:
            name: The name of the tool
            description: A description of what the tool does
            function: The function that implements the tool's functionality
            parameters: Parameter information for the tool
        """
        super().__init__(name, description, function)
        self.parameters = parameters
