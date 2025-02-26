#!/usr/bin/env python3
"""
Advanced example for using the BMW Agents framework with Ollama.
This demonstrates the ReAct prompt strategy with tools using the Ollama provider.
"""

import os
import sys
import asyncio
import logging
import datetime
import tempfile
import json
import re

# Add parent directory to path so we can import from bmw_agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bmw_agents.utils.llm_providers import OllamaProvider
from bmw_agents.core.prompt_strategies.react import ReActPromptStrategy, Tool
from bmw_agents.utils.logger import setup_logger
from bmw_agents.core.toolbox.tools.basic_tools import (
    datetime_now,
    math_add,
    math_multiply,
    random_number
)

# Set up logging
logger = setup_logger(level=logging.DEBUG)  # Set to DEBUG for more detailed logs

# Define some simple tools
async def get_current_time():
    """Get the current date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

async def calculate_sum(a: float, b: float):
    """Calculate the sum of two numbers."""
    return a + b

async def calculate_product(a: float, b: float):
    """Calculate the product of two numbers."""
    return a * b

async def get_random_number(min_value: float = 0, max_value: float = 100):
    """Generate a random number between min_value and max_value."""
    import random
    return random.uniform(min_value, max_value)

# Custom extraction function to debug responses
def extract_thought_and_action(content):
    """Extract thought and action from the response for debugging."""
    # Extract thought
    thought_match = re.search(r"Thought:(.*?)(?:Action:|$)", content, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else "No thought found"
    
    # Extract action
    action_match = re.search(r"Action:(.*?)(?:(?:\r\n|\r|\n){2}|$)", content, re.DOTALL)
    action_str = action_match.group(1).strip() if action_match else "No action found"
    
    print(f"Extracted thought: {thought[:100]}...")
    print(f"Extracted action: {action_str[:100]}...")
    
    # Try to parse action as JSON
    try:
        if action_str != "No action found":
            # Check if there's a code block with JSON
            json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", action_str, re.DOTALL)
            if json_match:
                action_json = json.loads(json_match.group(1))
                print(f"Parsed JSON action: {json.dumps(action_json, indent=2)}")
            else:
                # Try to parse the whole action string
                action_json = json.loads(action_str)
                print(f"Parsed JSON action: {json.dumps(action_json, indent=2)}")
        else:
            print("No action to parse")
    except Exception as e:
        print(f"Error parsing action as JSON: {e}")
    
    return thought, action_str

async def main():
    """Run a ReAct example with Ollama provider."""
    logger.info("Initializing Ollama provider with deepseek-r1:14b model...")
    
    # Initialize the LLM provider
    llm_provider = OllamaProvider(model_name="deepseek-r1:14b")
    
    # Create tools
    tools = [
        Tool(
            name="get_current_time",
            description="Get the current date and time",
            function=get_current_time
        ),
        Tool(
            name="calculate_sum",
            description="Calculate the sum of two numbers",
            function=calculate_sum
        ),
        Tool(
            name="calculate_product",
            description="Calculate the product of two numbers",
            function=calculate_product
        ),
        Tool(
            name="get_random_number",
            description="Generate a random number between min_value and max_value",
            function=get_random_number
        )
    ]
    
    # Create a temporary file with our custom template
    template_content = """
You are a helpful AI assistant with access to the following tools:

{tools}

To use a tool, you MUST provide your response in the following format:

Thought: <your reasoning about what to do next>
Action: {{
  "tool": "<tool_name>",
  "args": {{
    "<arg_name>": "<arg_value>"
  }}
}}

The system will respond with the tool's output:
Observation: <result of the tool call>

You MUST repeat this Thought/Action/Observation format for EACH step. ALWAYS start with "Thought:" followed by your reasoning, then "Action:" followed by a JSON object.

After you have all the information needed to answer the user's question, use the following to provide your final answer:

{termination_sequence} <your final answer>

IMPORTANT: 
1. DO NOT skip steps or try to predict tool outputs.
2. Use valid JSON format for the Action section.
3. DO NOT include explanations outside the Thought/Action/Observation format.
4. ONLY use the tools provided.
"""
    
    # Write the template to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp:
        temp.write(template_content)
        template_path = temp.name
    
    try:
        # Create a ReAct prompt strategy with our monkey-patched extraction function
        react_strategy = ReActPromptStrategy(
            llm_provider=llm_provider,
            tools=tools,
            template_path=template_path,
            max_iterations=5,
            termination_sequence="FINAL ANSWER:"
        )
        
        # Monkey patch the extract_thought_and_action method for debugging
        original_extract = react_strategy.extract_thought_and_action
        react_strategy.extract_thought_and_action = lambda content: (extract_thought_and_action(content), original_extract(content))[1]
        
        # Execute the ReAct strategy with a query
        logger.info("Sending query to Ollama with ReAct strategy...")
        query = "What is the current time? Then calculate the sum of 123.45 and 678.9, and finally multiply the result by a random number between 1 and 10."
        
        # Print all messages after each iteration
        original_get_llm_response = react_strategy.get_llm_response
        async def wrapped_get_llm_response(*args, **kwargs):
            print("\n--- Current Messages ---")
            for i, msg in enumerate(react_strategy.messages):
                print(f"Message {i+1} - {msg.role}: {msg.content[:100]}...")
            result = await original_get_llm_response(*args, **kwargs)
            print(f"\n--- Full Response from LLM ---")
            print(result['content'])
            print("\n--- End Response ---")
            return result
        react_strategy.get_llm_response = wrapped_get_llm_response
        
        response = await react_strategy.execute(query)
        
        # Debug: Print raw thoughts, actions, observations
        print("\n--- Raw Data ---")
        print("Thoughts:", react_strategy.thoughts)
        print("Actions:", react_strategy.actions)
        print("Observations:", react_strategy.observations)
        
        # Print the execution trace
        print("\n--- Execution Trace ---")
        execution_trace = react_strategy.get_execution_trace()
        print(f"Trace length: {len(execution_trace)}")
        
        for i, step in enumerate(execution_trace):
            print(f"Step {i+1}:")
            print(f"  Thought: {step.get('thought', 'N/A')}")
            print(f"  Action: {step.get('action', 'N/A')}")
            print(f"  Observation: {step.get('observation', 'N/A')}")
            print()
        
        # Print the final response
        print("--- Final Answer ---")
        print(response)
        print("--------------------\n")
        
        logger.info("ReAct example completed successfully!")
    finally:
        # Clean up the temporary file
        os.unlink(template_path)

if __name__ == "__main__":
    asyncio.run(main()) 