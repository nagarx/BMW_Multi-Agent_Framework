"""
Toolbox module for the BMW Agents framework.
This module defines the Toolbox class for managing collections of tools.
"""

from typing import Dict, List, Optional, Any, Set, Union
import re

from bmw_agents.core.toolbox.tool import Tool
from bmw_agents.utils.logger import get_logger

logger = get_logger("toolbox.toolbox")

class Toolbox:
    """
    Container for a collection of tools that can be used by agents.
    
    The toolbox provides methods for adding, removing, and accessing tools,
    as well as generating schema information for all tools.
    """
    
    def __init__(self, tools: Optional[List[Tool]] = None):
        """
        Initialize a toolbox.
        
        Args:
            tools: Initial list of tools (optional)
        """
        self.tools: Dict[str, Tool] = {}
        
        # Add initial tools if provided
        if tools:
            for tool in tools:
                self.add_tool(tool)
    
    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool to the toolbox.
        
        Args:
            tool: The tool to add
        """
        self.tools[tool.name] = tool
        logger.debug(f"Added tool '{tool.name}' to toolbox")
    
    def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool from the toolbox.
        
        Args:
            tool_name: The name of the tool to remove
            
        Returns:
            True if the tool was removed, False if it wasn't in the toolbox
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.debug(f"Removed tool '{tool_name}' from toolbox")
            return True
        else:
            logger.warning(f"Attempted to remove non-existent tool '{tool_name}'")
            return False
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        Get a tool by name.
        
        Args:
            tool_name: The name of the tool to get
            
        Returns:
            The tool if found, None otherwise
        """
        return self.tools.get(tool_name)
    
    def has_tool(self, tool_name: str) -> bool:
        """
        Check if the toolbox contains a tool with the given name.
        
        Args:
            tool_name: The name of the tool to check for
            
        Returns:
            True if the tool exists, False otherwise
        """
        return tool_name in self.tools
    
    def get_all_tools(self) -> List[Tool]:
        """
        Get all tools in the toolbox.
        
        Returns:
            List of all tools
        """
        return list(self.tools.values())
    
    def get_tool_names(self) -> List[str]:
        """
        Get the names of all tools in the toolbox.
        
        Returns:
            List of tool names
        """
        return list(self.tools.keys())
    
    def get_schema(self) -> List[Dict[str, Any]]:
        """
        Get the schema for all tools in the toolbox.
        
        Returns:
            List of tool schemas
        """
        return [tool.get_schema() for tool in self.tools.values()]
    
    def get_formatted_descriptions(self) -> str:
        """
        Get a formatted string with descriptions of all tools.
        
        Returns:
            Formatted string with tool descriptions
        """
        if not self.tools:
            return "No tools available."
        
        descriptions = []
        for i, tool in enumerate(self.tools.values()):
            param_desc = []
            for param_name, param_info in tool.parameters.items():
                required = "required" if param_info.get("required", False) else "optional"
                param_desc.append(f"  - {param_name} ({required}): {param_info.get('description', '')}")
            
            param_str = "\n".join(param_desc)
            descriptions.append(
                f"{i+1}. {tool.name}: {tool.description}\n"
                f"Parameters:\n{param_str}"
            )
        
        return "\n\n".join(descriptions)
    
    def filter_by_names(self, names: List[str]) -> 'Toolbox':
        """
        Create a new toolbox containing only the tools with the given names.
        
        Args:
            names: List of tool names to include
            
        Returns:
            New toolbox with the filtered tools
        """
        filtered_tools = []
        for name in names:
            tool = self.get_tool(name)
            if tool:
                filtered_tools.append(tool)
            else:
                logger.warning(f"Tool '{name}' not found in toolbox")
        
        return Toolbox(filtered_tools)
    
    def filter_by_pattern(self, pattern: str) -> 'Toolbox':
        """
        Create a new toolbox containing only the tools whose names match the pattern.
        
        Args:
            pattern: Regex pattern to match against tool names
            
        Returns:
            New toolbox with the filtered tools
        """
        regex = re.compile(pattern)
        filtered_tools = [
            tool for tool in self.tools.values()
            if regex.search(tool.name)
        ]
        
        return Toolbox(filtered_tools)
    
    def merge(self, other: 'Toolbox') -> 'Toolbox':
        """
        Merge this toolbox with another toolbox.
        
        Args:
            other: Another toolbox to merge with
            
        Returns:
            New toolbox containing tools from both toolboxes
        """
        merged = Toolbox(self.get_all_tools())
        for tool in other.get_all_tools():
            merged.add_tool(tool)
        
        return merged
    
    def __len__(self) -> int:
        """Get the number of tools in the toolbox."""
        return len(self.tools)
    
    def __contains__(self, tool_name: str) -> bool:
        """Check if the toolbox contains a tool with the given name."""
        return tool_name in self.tools
    
    def __iter__(self):
        """Iterate over the tools in the toolbox."""
        return iter(self.tools.values()) 