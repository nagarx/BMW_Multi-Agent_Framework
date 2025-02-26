"""
Basic tools module for the BMW Agents framework.
This module provides basic built-in tools for common operations.
"""

import datetime
import json
import math
import os
import random
import re
from typing import Any, Dict, List, Optional

import requests

from bmw_agents.core.toolbox.tool import SimpleTool
from bmw_agents.core.toolbox.toolbox import Toolbox
from bmw_agents.utils.logger import get_logger

logger = get_logger("toolbox.tools.basic_tools")

# Text processing tools


def text_split(text: str, delimiter: str = " ") -> List[str]:
    """
    Split text into a list of strings based on the delimiter.

    Args:
        text: The text to split
        delimiter: The delimiter to split on (default: space)

    Returns:
        List of strings after splitting
    """
    return text.split(delimiter)


def text_join(texts: List[str], delimiter: str = " ") -> str:
    """
    Join a list of strings into a single string with the given delimiter.

    Args:
        texts: List of strings to join
        delimiter: The delimiter to join with (default: space)

    Returns:
        Joined string
    """
    return delimiter.join(texts)


def text_replace(text: str, old: str, new: str) -> str:
    """
    Replace occurrences of a substring with another in the text.

    Args:
        text: The text to process
        old: The substring to replace
        new: The replacement substring

    Returns:
        Text with replacements
    """
    return text.replace(old, new)


def text_regex_replace(text: str, pattern: str, replacement: str) -> str:
    """
    Replace substrings matching a regex pattern in the text.

    Args:
        text: The text to process
        pattern: Regular expression pattern
        replacement: The replacement string

    Returns:
        Text with replacements
    """
    return re.sub(pattern, replacement, text)


def text_extract(text: str, pattern: str) -> List[str]:
    """
    Extract substrings matching a regex pattern from the text.

    Args:
        text: The text to process
        pattern: Regular expression pattern

    Returns:
        List of matching substrings
    """
    return re.findall(pattern, text)


def text_contains(text: str, substring: str, case_sensitive: bool = True) -> bool:
    """
    Check if text contains a substring.

    Args:
        text: The text to check
        substring: The substring to look for
        case_sensitive: Whether the search is case-sensitive

    Returns:
        True if the substring is found, False otherwise
    """
    if case_sensitive:
        return substring in text
    else:
        return substring.lower() in text.lower()


# Math tools


def math_add(a: float, b: float) -> float:
    """
    Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of the numbers
    """
    return a + b


def math_subtract(a: float, b: float) -> float:
    """
    Subtract one number from another.

    Args:
        a: Number to subtract from
        b: Number to subtract

    Returns:
        Difference of the numbers
    """
    return a - b


def math_multiply(a: float, b: float) -> float:
    """
    Multiply two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Product of the numbers
    """
    return a * b


def math_divide(a: float, b: float) -> float:
    """
    Divide one number by another.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Quotient of the numbers
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def math_power(base: float, exponent: float) -> float:
    """
    Raise a number to a power.

    Args:
        base: The base number
        exponent: The exponent

    Returns:
        Result of the exponentiation
    """
    return math.pow(base, exponent)


def math_sqrt(number: float) -> float:
    """
    Calculate the square root of a number.

    Args:
        number: The number to find the square root of

    Returns:
        Square root of the number
    """
    if number < 0:
        raise ValueError("Cannot calculate square root of a negative number")
    return math.sqrt(number)


# Date and time tools


def datetime_now() -> str:
    """
    Get the current date and time.

    Returns:
        Current date and time in ISO format
    """
    return datetime.datetime.now().isoformat()


def datetime_format(timestamp: str, format_str: str) -> str:
    """
    Format a timestamp according to a format string.

    Args:
        timestamp: ISO format timestamp
        format_str: Format string (e.g., "%Y-%m-%d")

    Returns:
        Formatted date/time string
    """
    dt = datetime.datetime.fromisoformat(timestamp)
    return dt.strftime(format_str)


def datetime_add(
    timestamp: str, days: int = 0, hours: int = 0, minutes: int = 0, seconds: int = 0
) -> str:
    """
    Add a time duration to a timestamp.

    Args:
        timestamp: ISO format timestamp
        days: Number of days to add
        hours: Number of hours to add
        minutes: Number of minutes to add
        seconds: Number of seconds to add

    Returns:
        New timestamp in ISO format
    """
    dt = datetime.datetime.fromisoformat(timestamp)
    delta = datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
    new_dt = dt + delta
    return new_dt.isoformat()


# Web tools


def web_get(url: str, headers: Optional[Dict[str, str]] = None) -> str:
    """
    Make an HTTP GET request to a URL.

    Args:
        url: The URL to request
        headers: Optional headers for the request

    Returns:
        Response text
    """
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"Error making GET request to {url}: {str(e)}")
        raise


def web_post(url: str, data: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> str:
    """
    Make an HTTP POST request to a URL.

    Args:
        url: The URL to request
        data: The data to send (will be converted to JSON)
        headers: Optional headers for the request

    Returns:
        Response text
    """
    try:
        if headers is None:
            headers = {}
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"Error making POST request to {url}: {str(e)}")
        raise


# JSON tools


def json_parse(text: str) -> Dict[str, Any]:
    """
    Parse a JSON string into a Python object.

    Args:
        text: JSON string to parse

    Returns:
        Parsed Python object
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON: {str(e)}")
        raise


def json_stringify(obj: Any, pretty: bool = False) -> str:
    """
    Convert a Python object to a JSON string.

    Args:
        obj: Python object to convert
        pretty: Whether to format the JSON with indentation

    Returns:
        JSON string
    """
    try:
        indent = 2 if pretty else None
        return json.dumps(obj, indent=indent)
    except TypeError as e:
        logger.error(f"Error stringifying object to JSON: {str(e)}")
        raise


# Utility tools


def random_number(min_value: float = 0, max_value: float = 1) -> float:
    """
    Generate a random number in the specified range.

    Args:
        min_value: Minimum value (inclusive)
        max_value: Maximum value (inclusive)

    Returns:
        Random number
    """
    return random.uniform(min_value, max_value)


def random_choice(options: List[Any]) -> Any:
    """
    Choose a random item from a list.

    Args:
        options: List of options to choose from

    Returns:
        Randomly selected item
    """
    if not options:
        raise ValueError("Cannot choose from empty list")
    return random.choice(options)


def env_var(name: str, default: Optional[str] = None) -> str:
    """
    Get the value of an environment variable.

    Args:
        name: Name of the environment variable
        default: Default value if the variable is not set

    Returns:
        Value of the environment variable or default
    """
    return os.environ.get(name, default)


def create_basic_toolbox() -> Toolbox:
    """
    Create a toolbox with all basic tools.

    Returns:
        Toolbox with basic tools
    """
    toolbox = Toolbox()

    # Text processing tools
    toolbox.add_tool(SimpleTool(name="text.split", function=text_split, description="Split text by delimiter"))
    toolbox.add_tool(SimpleTool(name="text.join", function=text_join, description="Join text parts with delimiter"))
    toolbox.add_tool(SimpleTool(name="text.replace", function=text_replace, description="Replace text in a string"))
    toolbox.add_tool(SimpleTool(name="text.regex_replace", function=text_regex_replace, description="Replace text using regex pattern"))
    toolbox.add_tool(SimpleTool(name="text.extract", function=text_extract, description="Extract text using a pattern"))
    toolbox.add_tool(SimpleTool(name="text.contains", function=text_contains, description="Check if text contains substring"))

    # Math tools
    toolbox.add_tool(SimpleTool(name="math.add", function=math_add, description="Add two numbers"))
    toolbox.add_tool(SimpleTool(name="math.subtract", function=math_subtract, description="Subtract second number from first"))
    toolbox.add_tool(SimpleTool(name="math.multiply", function=math_multiply, description="Multiply two numbers"))
    toolbox.add_tool(SimpleTool(name="math.divide", function=math_divide, description="Divide first number by second"))
    toolbox.add_tool(SimpleTool(name="math.power", function=math_power, description="Raise base to an exponent"))
    toolbox.add_tool(SimpleTool(name="math.sqrt", function=math_sqrt, description="Calculate square root"))

    # Date and time tools
    toolbox.add_tool(SimpleTool(name="datetime.now", function=datetime_now, description="Get current date and time"))
    toolbox.add_tool(SimpleTool(name="datetime.format", function=datetime_format, description="Format date and time"))
    toolbox.add_tool(SimpleTool(name="datetime.add", function=datetime_add, description="Add time to datetime"))

    # Web tools
    toolbox.add_tool(SimpleTool(name="web.get", function=web_get, description="Make HTTP GET request"))
    toolbox.add_tool(SimpleTool(name="web.post", function=web_post, description="Make HTTP POST request"))

    # JSON tools
    toolbox.add_tool(SimpleTool(name="json.parse", function=json_parse, description="Parse JSON string"))
    toolbox.add_tool(SimpleTool(name="json.stringify", function=json_stringify, description="Convert object to JSON string"))

    # Utility tools
    toolbox.add_tool(SimpleTool(name="random.number", function=random_number, description="Generate random number"))
    toolbox.add_tool(SimpleTool(name="random.choice", function=random_choice, description="Pick random item from list"))
    toolbox.add_tool(SimpleTool(name="env.get", function=env_var, description="Get environment variable"))

    return toolbox
