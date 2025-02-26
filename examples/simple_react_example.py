#!/usr/bin/env python3
"""
Simple example demonstrating the ReAct prompt strategy.
This is a simplified implementation to illustrate the core concepts.
"""

import os
import sys
import asyncio
import json
import re
import logging
import datetime

# Add parent directory to path so we can import from bmw_agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bmw_agents.utils.llm_providers import OllamaProvider
from bmw_agents.utils.logger import setup_logger

# Set up logging
logger = setup_logger(level=logging.INFO)

# Define some simple tools
async def get_current_time():
    """Get the current date and time."""
    now = datetime.datetime.now()
    return now.strftime("%B %d, %Y, at %I:%M %p")

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

def extract_thought_and_action(content):
    """
    Extract thought and action from LLM response.
    """
    # Remove <think>...</think> tags if present
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    
    # Extract thought - handle both with and without asterisks
    thought_match = re.search(r'\*{0,2}Thought:{0,2}\s*(.*?)(?:\*{0,2}Action:|$)', content, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""
    
    # Extract action - handle both with and without asterisks
    action_match = re.search(r'\*{0,2}Action:{0,2}\s*(.*?)(?:\*{0,2}Observation:|$)', content, re.DOTALL)
    action_text = action_match.group(1).strip() if action_match else ""
    
    # Parse action JSON
    action = None
    if action_text:
        try:
            # Clean up JSON - sometimes models add extra text
            json_text = action_text
            json_match = re.search(r'({.*})', json_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            
            action = json.loads(json_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse action JSON: {e}")
            logger.error(f"Raw action text: {action_text}")
    
    return thought, action

class SimpleReAct:
    """
    A simplified implementation of the ReAct strategy.
    """
    
    def __init__(self, llm_provider, tools=None, max_iterations=5):
        """Initialize with an LLM provider and tools."""
        self.llm_provider = llm_provider
        self.tools = tools or []
        self.max_iterations = max_iterations
        
        # Store execution trace
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.messages = []
        
    def get_tool_descriptions(self):
        """Get formatted descriptions of available tools."""
        descriptions = []
        for tool in self.tools:
            description = f"Tool: {tool['name']}\nDescription: {tool['description']}\n"
            descriptions.append(description)
        return "\n".join(descriptions)
    
    def find_tool_by_name(self, name):
        """Find a tool by name."""
        for tool in self.tools:
            if tool['name'] == name:
                return tool
        return None
    
    async def execute_action(self, action):
        """Execute a tool action and return the result."""
        if not action:
            return "No action specified"
        
        tool_name = action.get('tool')
        args = action.get('args', {})
        
        tool = self.find_tool_by_name(tool_name)
        if not tool:
            return f"Error: Tool '{tool_name}' not found"
        
        try:
            result = await tool['function'](**args)
            return str(result)
        except Exception as e:
            logger.error(f"Error executing tool: {e}")
            return f"Error executing tool: {str(e)}"
    
    async def run(self, instruction):
        """Run the ReAct strategy with the given instruction."""
        logger.info("Starting ReAct execution")
        
        # Create system message with tools and instructions
        system_message = f"""You are a helpful AI assistant with access to the following tools:

{self.get_tool_descriptions()}

To use a tool, respond with:

Thought: <your reasoning about what to do next>
Action: {{
  "tool": "<tool_name>",
  "args": {{
    "<arg_name>": "<arg_value>"
  }}
}}

The system will respond with the tool's output:
Observation: <result of the tool call>

After you have all the information needed to answer the user's question, use:

FINAL ANSWER: <your final answer>

Remember to:
1. Only use the tools provided
2. First think about what you need to do
3. Then select the appropriate tool and provide valid arguments
4. Wait for the observation before proceeding
5. When you have enough information, provide your final answer
"""

        # Create user message with the instruction
        user_message = instruction
        
        # Initialize conversation
        self.messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Main execution loop
        iteration_count = 0
        final_answer = None
        
        while iteration_count < self.max_iterations and final_answer is None:
            iteration_count += 1
            logger.info(f"Iteration {iteration_count}")
            
            # Get response from model
            response = await self.llm_provider.generate(self.messages, temperature=0.7)
            content = response["content"]
            
            # Check for final answer
            final_answer_match = re.search(r'FINAL ANSWER:\s*(.*)', content, re.DOTALL)
            if final_answer_match:
                final_answer = final_answer_match.group(1).strip()
                break
            
            # Extract thought and action
            thought, action = extract_thought_and_action(content)
            
            # Store in trace
            self.thoughts.append(thought)
            self.actions.append(action)
            
            # Execute action if found
            observation = await self.execute_action(action) if action else "No action to execute"
            self.observations.append(observation)
            
            # Add to conversation
            self.messages.append({"role": "assistant", "content": content})
            self.messages.append({"role": "user", "content": f"Observation: {observation}"})
        
        # Return results
        if final_answer is None:
            final_answer = "Failed to reach a conclusion within the maximum iterations."
        
        return {
            "result": final_answer,
            "trace": {
                "thoughts": self.thoughts,
                "actions": self.actions,
                "observations": self.observations
            }
        }

async def main():
    """Run a simple ReAct example."""
    logger.info("Initializing Ollama provider with deepseek-r1:14b model...")
    
    # Initialize the LLM provider
    llm_provider = OllamaProvider(model_name="deepseek-r1:14b")
    
    # Create tools
    tools = [
        {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "function": get_current_time
        },
        {
            "name": "calculate_sum",
            "description": "Calculate the sum of two numbers",
            "function": calculate_sum
        },
        {
            "name": "calculate_product",
            "description": "Calculate the product of two numbers",
            "function": calculate_product
        },
        {
            "name": "get_random_number",
            "description": "Generate a random number between min_value and max_value",
            "function": get_random_number
        }
    ]
    
    # Create the ReAct strategy
    react = SimpleReAct(
        llm_provider=llm_provider,
        tools=tools,
        max_iterations=5
    )
    
    # Run the strategy with a user query
    user_query = "What is the current time? Then calculate the sum of 123.45 and 678.9, and finally multiply the result by a random number between 1 and 10."
    
    logger.info(f"Running query: {user_query}")
    result = await react.run(user_query)
    
    # Print the execution trace
    print("\n--- Execution Trace ---")
    for i, (thought, action, observation) in enumerate(zip(
        result["trace"]["thoughts"], 
        result["trace"]["actions"], 
        result["trace"]["observations"]
    )):
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