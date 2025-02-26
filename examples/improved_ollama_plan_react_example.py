#!/usr/bin/env python3
"""
Improved example demonstrating the Ollama-specific PlanReAct strategy.
"""

import asyncio
import datetime
import logging
import os
import sys

# Add parent directory to path so we can import from bmw_agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bmw_agents.core.prompt_strategies.ollama_plan_react import OllamaPlanReActPromptStrategy
from bmw_agents.core.prompt_strategies.react import Tool
from bmw_agents.utils.llm_providers import OllamaProvider
from bmw_agents.utils.logger import setup_logger

# Set up logging
logger = setup_logger(level=logging.INFO)


# Define some simple tools
async def get_current_time() -> str:
    """Get the current date and time."""
    now = datetime.datetime.now()
    return now.strftime("%B %d, %Y, at %I:%M %p")


async def calculate_sum(a: float, b: float) -> float:
    """Calculate the sum of two numbers."""
    return a + b


async def calculate_product(a: float, b: float) -> float:
    """Calculate the product of two numbers."""
    return a * b


async def get_random_number(min_value: float = 0, max_value: float = 100) -> float:
    """Generate a random number between min_value and max_value."""
    import random

    return random.uniform(min_value, max_value)


async def main() -> None:
    """Run an improved PlanReAct example with Ollama-specific strategy."""
    logger.info("Initializing Ollama provider with deepseek-r1:14b model...")

    # Initialize the LLM provider
    llm_provider = OllamaProvider(model_name="deepseek-r1:14b")

    # Create tools
    tools = [
        Tool(
            name="get_current_time",
            description="Get the current date and time",
            function=get_current_time,
        ),
        Tool(
            name="calculate_sum",
            description="Calculate the sum of two numbers",
            function=calculate_sum,
        ),
        Tool(
            name="calculate_product",
            description="Calculate the product of two numbers",
            function=calculate_product,
        ),
        Tool(
            name="get_random_number",
            description="Generate a random number between min_value and max_value",
            function=get_random_number,
        ),
    ]

    # Create the Ollama-specific PlanReAct prompt strategy
    prompt_strategy = OllamaPlanReActPromptStrategy(
        llm_provider=llm_provider, tools=tools, max_iterations=5
    )

    # Run the strategy with a user query
    user_query = "What is the current time? Then calculate the sum of 123.45 and 678.9, and finally multiply the result by a random number between 1 and 10."

    logger.info(f"Running query: {user_query}")
    result = await prompt_strategy.run(user_query)

    # Print the plan
    print("\n--- Plan ---")
    print(result["trace"]["plan"])
    print()

    # Print the execution trace
    print("--- Execution Trace ---")
    for i, (thought, action, observation) in enumerate(
        zip(
            result["trace"]["thoughts"], result["trace"]["actions"], result["trace"]["observations"]
        )
    ):
        print(f"Step {i+1}:")
        print(f"  Thought: {thought}")
        if action:
            print(f"  Action: {action}")
        else:
            print("  Action: None")
        print(f"  Observation: {observation}")
        print()

    # Print the final result
    print("--- Final Result ---")
    print(result["result"])
    print("-------------------")

    logger.info("Example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
