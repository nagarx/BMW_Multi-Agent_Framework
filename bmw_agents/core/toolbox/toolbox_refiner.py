"""
Toolbox Refiner module for the BMW Agents framework.
This module defines the ToolboxRefiner class for customizing toolboxes for specific agents.
"""

import re
from typing import Any, Callable, Dict, List, Optional, TypeVar

from bmw_agents.core.toolbox.tool import SimpleTool, Tool
from bmw_agents.core.toolbox.toolbox import Toolbox
from bmw_agents.utils.logger import get_logger

logger = get_logger("toolbox.toolbox_refiner")

T = TypeVar("T")  # Type variable for return values


class ToolboxRefiner:
    """
    A class for refining and customizing toolboxes based on specific agent needs.

    The ToolboxRefiner provides methods for:
    - Filtering tools by name patterns
    - Adding/removing specific tools
    - Creating specialized versions of tools with custom parameters
    - Transforming tool parameters and descriptions
    """

    def __init__(self, base_toolbox: Toolbox) -> None:
        """
        Initialize a toolbox refiner.

        Args:
            base_toolbox: The base toolbox to refine
        """
        self.base_toolbox = base_toolbox
        self.refined_toolbox = Toolbox(base_toolbox.get_all_tools())

    def include_tools(self, tool_names: List[str]) -> "ToolboxRefiner":
        """
        Include only tools with the specified names.

        Args:
            tool_names: List of tool names to include

        Returns:
            Self for method chaining
        """
        self.refined_toolbox = self.base_toolbox.filter_by_names(tool_names)
        return self

    def exclude_tools(self, tool_names: List[str]) -> "ToolboxRefiner":
        """
        Exclude tools with the specified names.

        Args:
            tool_names: List of tool names to exclude

        Returns:
            Self for method chaining
        """
        for name in tool_names:
            if name in self.refined_toolbox:
                self.refined_toolbox.remove_tool(name)
        return self

    def include_pattern(self, pattern: str) -> "ToolboxRefiner":
        """
        Include only tools whose names match the pattern.

        Args:
            pattern: Regex pattern to match against tool names

        Returns:
            Self for method chaining
        """
        self.refined_toolbox = self.base_toolbox.filter_by_pattern(pattern)
        return self

    def exclude_pattern(self, pattern: str) -> "ToolboxRefiner":
        """
        Exclude tools whose names match the pattern.

        Args:
            pattern: Regex pattern to match against tool names

        Returns:
            Self for method chaining
        """
        regex = re.compile(pattern)
        tools_to_remove = [
            tool.name for tool in self.refined_toolbox.get_all_tools() if regex.search(tool.name)
        ]

        for name in tools_to_remove:
            self.refined_toolbox.remove_tool(name)

        return self

    def add_custom_tool(self, tool: Tool) -> "ToolboxRefiner":
        """
        Add a custom tool to the refined toolbox.

        Args:
            tool: The tool to add

        Returns:
            Self for method chaining
        """
        self.refined_toolbox.add_tool(tool)
        return self

    def add_simple_tool(
        self, name: str, func: Callable, description: Optional[str] = None
    ) -> "ToolboxRefiner":
        """
        Add a simple tool based on a function to the refined toolbox.

        Args:
            name: Name for the tool
            func: Function to use for the tool
            description: Optional custom description

        Returns:
            Self for method chaining
        """
        tool = SimpleTool(name=name, function=func, description=description)
        self.refined_toolbox.add_tool(tool)
        return self

    def specialize_tool(
        self,
        original_name: str,
        new_name: str,
        description: Optional[str] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
        required_params: Optional[List[str]] = None,
        optional_params: Optional[List[str]] = None,
    ) -> "ToolboxRefiner":
        """
        Create a specialized version of an existing tool.

        This method creates a new tool based on an existing one, but with some
        parameters pre-filled and potentially a customized description.

        Args:
            original_name: Name of the original tool
            new_name: Name for the specialized tool
            description: Custom description for the specialized tool
            fixed_params: Parameters to fix with specific values
            required_params: Parameters to mark as required
            optional_params: Parameters to mark as optional

        Returns:
            Self for method chaining
        """
        original_tool = self.base_toolbox.get_tool(original_name)
        if not original_tool:
            logger.warning(f"Cannot specialize non-existent tool '{original_name}'")
            return self

        # Create a wrapper function that applies fixed parameters
        def specialized_function(*args: Any, **kwargs: Any) -> Any:
            if fixed_params:
                # Apply fixed parameters, but allow explicit parameters to override
                for param_name, param_value in fixed_params.items():
                    if param_name not in kwargs:
                        kwargs[param_name] = param_value

            return original_tool.execute(*args, **kwargs)

        # Create a clone of the original parameters
        parameters = {}
        for param_name, param_info in original_tool.parameters.items():
            # Skip parameters that are fixed and not required/optional
            if fixed_params and param_name in fixed_params:
                if not (required_params and param_name in required_params) and not (
                    optional_params and param_name in optional_params
                ):
                    continue

            # Clone the parameter info
            parameters[param_name] = param_info.copy()

            # Update required status based on required_params and optional_params
            if required_params and param_name in required_params:
                parameters[param_name]["required"] = True
            elif optional_params and param_name in optional_params:
                parameters[param_name]["required"] = False

        # Create the specialized tool
        specialized_tool = Tool(
            name=new_name,
            description=description or original_tool.description,
            parameters=parameters,
            function=specialized_function,
        )

        self.refined_toolbox.add_tool(specialized_tool)
        return self

    def modify_tool_description(self, tool_name: str, new_description: str) -> "ToolboxRefiner":
        """
        Modify the description of a tool.

        Args:
            tool_name: Name of the tool to modify
            new_description: New description for the tool

        Returns:
            Self for method chaining
        """
        tool = self.refined_toolbox.get_tool(tool_name)
        if tool:
            # Create a new tool with the updated description
            updated_tool = Tool(
                name=tool.name,
                description=new_description,
                parameters=tool.parameters,
                function=tool.function,
            )
            self.refined_toolbox.remove_tool(tool_name)
            self.refined_toolbox.add_tool(updated_tool)
        return self

    def modify_parameter_description(
        self, tool_name: str, param_name: str, new_description: str
    ) -> "ToolboxRefiner":
        """
        Modify the description of a parameter in a tool.

        Args:
            tool_name: Name of the tool to modify
            param_name: Name of the parameter to modify
            new_description: New description for the parameter

        Returns:
            Self for method chaining
        """
        tool = self.refined_toolbox.get_tool(tool_name)
        if tool and param_name in tool.parameters:
            # Create new parameters with the updated description
            parameters = {k: v.copy() for k, v in tool.parameters.items()}
            parameters[param_name]["description"] = new_description

            # Create a new tool with the updated parameters
            updated_tool = Tool(
                name=tool.name,
                description=tool.description,
                parameters=parameters,
                function=tool.function,
            )
            self.refined_toolbox.remove_tool(tool_name)
            self.refined_toolbox.add_tool(updated_tool)
        return self

    def build(self) -> Toolbox:
        """
        Build and return the refined toolbox.

        Returns:
            The refined toolbox
        """
        return self.refined_toolbox
