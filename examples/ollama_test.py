#!/usr/bin/env python3
"""
Example script for using the BMW Agents framework with Ollama.
This demonstrates how to set up and use the OllamaProvider with deepseek-r1:14b.
"""

import asyncio
import logging
import os
import sys

# Add parent directory to path so we can import from bmw_agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bmw_agents.core.prompt_strategies.non_iterative import NonIterativePromptStrategy
from bmw_agents.utils.llm_providers import OllamaProvider
from bmw_agents.utils.logger import setup_logger

# Set up logging
logger = setup_logger(level=logging.INFO)


async def main() -> None:
    """Run a simple test with Ollama provider."""
    logger.info("Initializing Ollama provider with deepseek-r1:14b model...")

    # Initialize the LLM provider
    llm_provider = OllamaProvider(model_name="deepseek-r1:14b")

    # Create a simple prompt strategy
    prompt_strategy = NonIterativePromptStrategy(
        llm_provider=llm_provider,
        template_content="You are a helpful AI assistant. Answer the user's question clearly and concisely.",
    )

    # Execute the prompt strategy
    logger.info("Sending test query to Ollama...")
    response = await prompt_strategy.execute("What is the BMW Agents framework?")

    # Print the response
    print("\n--- Response from Ollama deepseek-r1:14b ---")
    print(response)
    print("-------------------------------------------\n")

    logger.info("Test completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
