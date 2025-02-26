"""
File tools module for the BMW Agents framework.
This module provides tools for file operations.
"""

import csv
import json
import os
import pathlib
import shutil
from typing import Any, Dict, List, Optional, Union

from bmw_agents.core.toolbox.tool import SimpleTool
from bmw_agents.core.toolbox.toolbox import Toolbox
from bmw_agents.utils.logger import get_logger

logger = get_logger("toolbox.tools.file_tools")


def file_read(path: str, encoding: str = "utf-8") -> str:
    """
    Read a text file.

    Args:
        path: Path to the file
        encoding: File encoding

    Returns:
        File contents as a string
    """
    try:
        with open(path, "r", encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {path}: {str(e)}")
        raise


def file_write(path: str, content: str, encoding: str = "utf-8", append: bool = False) -> bool:
    """
    Write or append to a text file.

    Args:
        path: Path to the file
        content: Content to write
        encoding: File encoding
        append: Whether to append to the file (default: False)

    Returns:
        True if successful
    """
    try:
        mode = "a" if append else "w"
        with open(path, mode, encoding=encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Error writing to file {path}: {str(e)}")
        raise


def file_exists(path: str) -> bool:
    """
    Check if a file exists.

    Args:
        path: Path to the file

    Returns:
        True if the file exists, False otherwise
    """
    return os.path.isfile(path)


def file_delete(path: str) -> bool:
    """
    Delete a file.

    Args:
        path: Path to the file

    Returns:
        True if successful
    """
    try:
        if not os.path.isfile(path):
            logger.warning(f"File {path} does not exist")
            return False
        os.remove(path)
        return True
    except Exception as e:
        logger.error(f"Error deleting file {path}: {str(e)}")
        raise


def file_size(path: str) -> int:
    """
    Get the size of a file in bytes.

    Args:
        path: Path to the file

    Returns:
        File size in bytes
    """
    try:
        return os.path.getsize(path)
    except Exception as e:
        logger.error(f"Error getting size of file {path}: {str(e)}")
        raise


def file_copy(source: str, destination: str) -> bool:
    """
    Copy a file.

    Args:
        source: Path to the source file
        destination: Path to the destination file

    Returns:
        True if successful
    """
    try:
        shutil.copy2(source, destination)
        return True
    except Exception as e:
        logger.error(f"Error copying file {source} to {destination}: {str(e)}")
        raise


def file_move(source: str, destination: str) -> bool:
    """
    Move or rename a file.

    Args:
        source: Path to the source file
        destination: Path to the destination file

    Returns:
        True if successful
    """
    try:
        shutil.move(source, destination)
        return True
    except Exception as e:
        logger.error(f"Error moving file {source} to {destination}: {str(e)}")
        raise


def dir_create(path: str, exist_ok: bool = True) -> bool:
    """
    Create a directory.

    Args:
        path: Path to the directory
        exist_ok: Whether it's OK if the directory already exists

    Returns:
        True if successful
    """
    try:
        os.makedirs(path, exist_ok=exist_ok)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {path}: {str(e)}")
        raise


def dir_exists(path: str) -> bool:
    """
    Check if a directory exists.

    Args:
        path: Path to the directory

    Returns:
        True if the directory exists, False otherwise
    """
    return os.path.isdir(path)


def dir_list(path: str, pattern: Optional[str] = None) -> List[str]:
    """
    List files and directories in a directory.

    Args:
        path: Path to the directory
        pattern: Optional glob pattern to filter results

    Returns:
        List of file and directory names
    """
    try:
        if pattern:
            p = pathlib.Path(path)
            return [str(item.relative_to(path)) for item in p.glob(pattern)]
        else:
            return os.listdir(path)
    except Exception as e:
        logger.error(f"Error listing directory {path}: {str(e)}")
        raise


def dir_delete(path: str, recursive: bool = False) -> bool:
    """
    Delete a directory.

    Args:
        path: Path to the directory
        recursive: Whether to delete the directory recursively

    Returns:
        True if successful
    """
    try:
        if recursive:
            shutil.rmtree(path)
        else:
            os.rmdir(path)
        return True
    except Exception as e:
        logger.error(f"Error deleting directory {path}: {str(e)}")
        raise


def json_read(path: str, encoding: str = "utf-8") -> Any:
    """
    Read and parse a JSON file.

    Args:
        path: Path to the JSON file
        encoding: File encoding

    Returns:
        Parsed JSON content
    """
    try:
        with open(path, "r", encoding=encoding) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading JSON file {path}: {str(e)}")
        raise


def json_write(path: str, data: Any, encoding: str = "utf-8", indent: int = 2) -> bool:
    """
    Write data to a JSON file.

    Args:
        path: Path to the JSON file
        data: Data to write
        encoding: File encoding
        indent: Indentation level for formatting

    Returns:
        True if successful
    """
    try:
        with open(path, "w", encoding=encoding) as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        logger.error(f"Error writing to JSON file {path}: {str(e)}")
        raise


def csv_read(
    path: str, has_header: bool = True, delimiter: str = ",", encoding: str = "utf-8"
) -> List[Union[Dict[str, str], List[str]]]:
    """
    Read a CSV file.

    Args:
        path: Path to the CSV file
        has_header: Whether the CSV file has a header row
        delimiter: CSV delimiter
        encoding: File encoding

    Returns:
        List of dictionaries (if has_header=True) or list of lists (if has_header=False)
    """
    try:
        with open(path, "r", encoding=encoding, newline="") as f:
            if has_header:
                # We need to use Any type here to avoid mypy errors with csv.DictReader
                reader: Any = csv.DictReader(f, delimiter=delimiter)
                return [dict(row) for row in reader]  # Convert OrderedDict to dict
            else:
                reader = csv.reader(f, delimiter=delimiter)
                return list(reader)
    except Exception as e:
        logger.error(f"Error reading CSV file {path}: {str(e)}")
        raise


def csv_write(
    path: str,
    data: List[Union[Dict[str, str], List[str]]],
    fieldnames: Optional[List[str]] = None,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> bool:
    """
    Write data to a CSV file.

    Args:
        path: Path to the CSV file
        data: List of dictionaries or list of lists to write
        fieldnames: Column names (required if data is list of dicts and first row isn't complete)
        delimiter: CSV delimiter
        encoding: File encoding

    Returns:
        True if successful
    """
    try:
        with open(path, "w", encoding=encoding, newline="") as f:
            # Check if data is a list of dictionaries or a list of lists
            is_dict_list = bool(data) and all(isinstance(row, dict) for row in data)

            if is_dict_list:
                if not fieldnames and data:
                    # Try to get fieldnames from the first dict
                    dict_data = [d for d in data if isinstance(d, dict)]
                    if dict_data:
                        fieldnames = list(dict_data[0].keys())
                
                # We need to use Any type here to avoid mypy errors with csv.DictWriter
                writer: Any = csv.DictWriter(f, fieldnames=fieldnames or [], delimiter=delimiter)
                writer.writeheader()
                
                # Only write items that are dictionaries
                dict_data = [d for d in data if isinstance(d, dict)]
                writer.writerows(dict_data)
            else:
                writer = csv.writer(f, delimiter=delimiter)
                
                # Convert any dictionaries to lists of values
                list_data = []
                for row in data:
                    if isinstance(row, dict):
                        list_data.append(list(row.values()))
                    else:
                        list_data.append(row)
                writer.writerows(list_data)
        return True
    except Exception as e:
        logger.error(f"Error writing to CSV file {path}: {str(e)}")
        raise


def path_join(*paths: str) -> str:
    """
    Join path components.

    Args:
        *paths: Path components to join

    Returns:
        Joined path
    """
    return os.path.join(*paths)


def path_absolute(path: str) -> str:
    """
    Get the absolute path.

    Args:
        path: Relative or absolute path

    Returns:
        Absolute path
    """
    return os.path.abspath(path)


def path_basename(path: str) -> str:
    """
    Get the basename of a path.

    Args:
        path: Path

    Returns:
        Basename of the path
    """
    return os.path.basename(path)


def path_dirname(path: str) -> str:
    """
    Get the directory name of a path.

    Args:
        path: Path

    Returns:
        Directory name of the path
    """
    return os.path.dirname(path)


def create_file_toolbox() -> Toolbox:
    """
    Create a toolbox with file operation tools.

    Returns:
        Toolbox with file tools
    """
    toolbox = Toolbox()

    # File operations
    toolbox.add_tool(SimpleTool(name="file.read", function=file_read, description="Read content from a file"))
    toolbox.add_tool(SimpleTool(name="file.write", function=file_write, description="Write content to a file"))
    toolbox.add_tool(SimpleTool(name="file.exists", function=file_exists, description="Check if a file exists"))
    toolbox.add_tool(SimpleTool(name="file.delete", function=file_delete, description="Delete a file"))
    toolbox.add_tool(SimpleTool(name="file.size", function=file_size, description="Get the size of a file"))
    toolbox.add_tool(SimpleTool(name="file.copy", function=file_copy, description="Copy a file"))
    toolbox.add_tool(SimpleTool(name="file.move", function=file_move, description="Move or rename a file"))

    # Directory operations
    toolbox.add_tool(SimpleTool(name="dir.create", function=dir_create, description="Create a directory"))
    toolbox.add_tool(SimpleTool(name="dir.exists", function=dir_exists, description="Check if a directory exists"))
    toolbox.add_tool(SimpleTool(name="dir.list", function=dir_list, description="List contents of a directory"))
    toolbox.add_tool(SimpleTool(name="dir.delete", function=dir_delete, description="Delete a directory"))

    # JSON operations
    toolbox.add_tool(SimpleTool(name="json.read", function=json_read, description="Read and parse a JSON file"))
    toolbox.add_tool(SimpleTool(name="json.write", function=json_write, description="Write data to a JSON file"))

    # CSV operations
    toolbox.add_tool(SimpleTool(name="csv.read", function=csv_read, description="Read a CSV file"))
    toolbox.add_tool(SimpleTool(name="csv.write", function=csv_write, description="Write data to a CSV file"))

    # Path operations
    toolbox.add_tool(SimpleTool(name="path.join", function=path_join, description="Join path components"))
    toolbox.add_tool(SimpleTool(name="path.absolute", function=path_absolute, description="Get absolute path"))
    toolbox.add_tool(SimpleTool(name="path.basename", function=path_basename, description="Get the base name of a path"))
    toolbox.add_tool(SimpleTool(name="path.dirname", function=path_dirname, description="Get the directory name of a path"))

    return toolbox
