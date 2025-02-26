"""
Tool registry module for the BMW Agents framework.
This module provides functions to get predefined toolboxes.
"""

from bmw_agents.core.toolbox.toolbox import Toolbox
from bmw_agents.core.toolbox.tools.basic_tools import create_basic_toolbox
from bmw_agents.core.toolbox.tools.file_tools import create_file_toolbox
from bmw_agents.utils.logger import get_logger

logger = get_logger("toolbox.tools.registry")

def get_all_tools() -> Toolbox:
    """
    Get a toolbox with all built-in tools.
    
    Returns:
        Toolbox with all built-in tools
    """
    toolbox = Toolbox()
    
    # Merge in all toolboxes
    basic_toolbox = create_basic_toolbox()
    file_toolbox = create_file_toolbox()
    
    # Combine all toolboxes
    toolbox = basic_toolbox.merge(file_toolbox)
    
    logger.info(f"Created toolbox with {len(toolbox)} tools")
    return toolbox

def get_basic_tools() -> Toolbox:
    """
    Get a toolbox with basic tools only.
    
    Returns:
        Toolbox with basic tools
    """
    return create_basic_toolbox()

def get_file_tools() -> Toolbox:
    """
    Get a toolbox with file operation tools only.
    
    Returns:
        Toolbox with file tools
    """
    return create_file_toolbox()

def get_safe_tools() -> Toolbox:
    """
    Get a toolbox with safe tools only (no file system operations).
    
    Returns:
        Toolbox with safe tools
    """
    # Start with all basic tools
    toolbox = create_basic_toolbox()
    
    # Remove any potentially unsafe tools
    unsafe_tools = [
        "web.get", 
        "web.post"
    ]
    
    for tool_name in unsafe_tools:
        if tool_name in toolbox:
            toolbox.remove_tool(tool_name)
    
    return toolbox 