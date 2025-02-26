# BMW Agents - A Framework For Task Automation Through Multi-Agent Collaboration

[![CI](https://github.com/bmwgroup/bmw-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/bmwgroup/bmw-agents/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)

BMW Agents is a flexible agent engineering framework designed for autonomous task automation through multi-agent collaboration. The framework provides a robust foundation for building AI agent applications with careful attention to planning, execution, and verification.

## Features

- **Plan-Execute-Verify Workflow**: Task decomposition, execution, and result verification
- **Multiple Prompt Strategies**: Support for ReAct, PlanReAct, and other iterative strategies
- **Multi-Agent Collaboration**: Various patterns for agent collaboration (Independent, Sequential, Joint, Hierarchical, and Broadcast)
- **Tool Integration**: Scalable and reliable tool usage with refined toolboxes
- **Memory Management**: Short-term (task-specific) and episodic (cross-task) memory capabilities
- **LLM Provider Abstraction**: Support for OpenAI, Anthropic, and Ollama models
- **Configurable Templates**: Customizable prompt templates for different strategies and providers

## Installation

### Prerequisites

- Python 3.8 or newer

### Installing from Source

```bash
git clone https://github.com/bmwgroup/bmw-agents.git
cd bmw-agents
pip install -e .
```

### Development Installation

To install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

Here's a simple example using the ReAct prompt strategy:

```python
import asyncio
from bmw_agents.utils.llm_providers import OpenAIProvider
from bmw_agents.core.prompt_strategies.react import ReActPromptStrategy, Tool

# Define a simple tool
async def get_weather(location: str):
    return f"The weather in {location} is sunny."

async def main():
    # Initialize LLM provider
    llm_provider = OpenAIProvider(model_name="gpt-4")
    
    # Create tools
    tools = [
        Tool(
            name="get_weather",
            description="Get the current weather for a location",
            function=get_weather
        )
    ]
    
    # Create prompt strategy
    prompt_strategy = ReActPromptStrategy(
        llm_provider=llm_provider,
        tools=tools
    )
    
    # Run the strategy
    result = await prompt_strategy.run("What's the weather in Berlin?")
    
    # Print the result
    print(result["result"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Ollama Integration

BMW Agents includes specialized support for [Ollama](https://ollama.ai), allowing you to run the framework with locally hosted open-source models.

### Ollama-Specific Features

1. **Specialized Prompt Strategies**:
   - `OllamaReActPromptStrategy` and `OllamaPlanReActPromptStrategy` (basic versions)
   - `OllamaTracedReAct` and `OllamaTracedPlanReAct` (traced versions)
   - `OllamaSingleResponseReAct` and `OllamaSingleResponsePlanReAct` (optimized for Ollama's generation pattern)

2. **Recommended Approach**: The single-response implementations work best with Ollama models, which tend to generate the complete execution trace in one response.

### Using Ollama with BMW Agents

```python
import asyncio
from bmw_agents.utils.llm_providers import OllamaProvider
from bmw_agents.core.prompt_strategies.ollama_single_response_react import OllamaSingleResponseReAct

async def get_weather(location: str):
    return f"The weather in {location} is sunny."

async def main():
    # Initialize Ollama provider
    llm_provider = OllamaProvider(model_name="deepseek-r1:14b")
    
    # Create tools
    tools = [Tool(name="get_weather", description="Get weather for a location", function=get_weather)]
    
    # Create the Ollama-specific strategy
    prompt_strategy = OllamaSingleResponseReAct(
        llm_provider=llm_provider,
        tools=tools
    )
    
    # Run the strategy
    result = await prompt_strategy.run("What's the weather in Berlin?")
    
    # Print the result and execution trace
    print(f"Result: {result['result']}")
    print(f"Steps: {len(result['trace']['thoughts'])}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Documentation

For more detailed documentation and examples, see:

- [Examples Directory](examples/README.md) - Various example implementations
- [Ollama Integration Guide](examples/OLLAMA_INTEGRATION.md) - Detailed guide for using Ollama with BMW Agents

## Project Structure

```
bmw_agents/
├── core/
│   ├── prompt_strategies/  # Different prompting approaches (ReAct, PlanReAct, etc.)
│   ├── memory/             # Short-term and episodic memory implementations
│   └── toolbox/            # Tool abstractions and management
├── utils/
│   ├── llm_providers.py    # LLM provider abstractions
│   └── logger.py           # Logging utilities
└── configs/
    └── prompt_templates/   # Template files for different prompt strategies
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use BMW Agents in your research, please cite:

```
@article{BMWAgents2024,
  title={BMW Agents - A Framework For Task Automation Through Multi-Agent Collaboration},
  author={Crawford, Noel and Duffy, Edward B. and Evazzade, Iman and Foehr, Torsten and 
          Robbins, Gregory and Saha, Debbrata Kumar and Varma, Jiya and Ziolkowski, Marcin},
  journal={arXiv preprint},
  year={2024}
}
```

You can find the [original research paper](docs/BMW%20Agents%20-%20A%20Framework%20For%20Task%20Automation%20Through%20Multi-Agent%20Collaboration.pdf) in the docs directory of this repository.

## Acknowledgements

This framework was developed by the BMW Group Information Technology Research Center.