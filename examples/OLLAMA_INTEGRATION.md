# Ollama Integration with BMW Agents Framework

This document provides a detailed explanation of the BMW Agents framework integration with Ollama models, our approach, findings, and recommendations.

## Background

[Ollama](https://ollama.ai) is a powerful tool for running open-source large language models (LLMs) locally on various hardware. It offers a simple API that's similar to commercial services, making it a great option for developing and testing agent applications without depending on external providers.

However, integrating Ollama models with agent frameworks designed for commercial APIs presents some unique challenges:

1. **Response format differences**: Ollama models may include markdown formatting or other artifacts in their outputs
2. **JSON parsing complexity**: Some Ollama models struggle with precisely formatted JSON in ReAct-style tool calls
3. **Generation behavior**: Many Ollama models tend to generate the entire Thought/Action/Observation sequence at once, rather than waiting for feedback at each step

## Our Integration Approach

We've developed several implementations to address these challenges, gradually improving our approach based on observations of how Ollama models naturally behave:

### 1. Basic Integration (OllamaReActPromptStrategy)

Our initial implementation focused on improving JSON parsing and adding more flexible regex patterns to handle Ollama's output format variations. This works for simple scenarios but showed limitations with more complex tasks.

### 2. Traced Implementation (OllamaTracedReAct)

We created an enhanced version that explicitly tracks thoughts, actions, and observations, making debugging easier. However, this still relied on the iterative ReAct pattern, which doesn't align well with how Ollama models naturally generate responses.

### 3. Single-Response Implementation (OllamaSingleResponseReAct)

Our most successful approach adapts to Ollama's natural tendency to generate the complete chain of reasoning in a single response. This implementation:

1. Makes a single call to the Ollama model
2. Extracts the entire sequence of Thought/Action/Observation steps from the response
3. Executes any tool calls found in the response
4. Provides a complete execution trace without needing multiple API calls

## Key Components

The integration consists of several important components:

### Enhanced Templates

We've created Ollama-specific templates that:
- Use clearer formatting instructions
- Avoid relying on markdown formatting
- Include explicit examples of the expected response format
- Use double braces for JSON examples to avoid Python formatting issues

### Robust Parsing

Our implementations include:
- Flexible regex patterns to handle variations in formatting
- Multiple fallback mechanisms for JSON parsing
- Clean error handling when formats don't match expectations

### Single-Response Processing

For the most reliable results, our single-response implementation:
- Handles complete Thought/Action/Observation sequences in one response
- Properly extracts steps even when the model includes its own observations
- Executes actual tools when needed, replacing the model's imagined observations

## Recommendations

Based on our findings, we recommend the following approach for integrating Ollama models with the BMW Agents framework:

1. **Use the SingleResponseTracedReAct strategy** - This works best with Ollama models' natural generation behavior
2. **Provide clear formatting instructions** - Be explicit about the expected format in your templates
3. **Focus on simple, clear tool descriptions** - Ensure tools have unambiguous parameter names and descriptions
4. **Monitor outputs for format drift** - Some models may occasionally deviate from the expected format
5. **Consider model selection carefully** - Some Ollama models are better at following structured formats than others
   - The deepseek-r1:14b model performed well in our testing
   - Models with deeper instruct tuning tend to better follow explicit instructions

## Future Improvements

Potential enhancements for the Ollama integration include:

1. **Streaming support** - Adding token streaming for more responsive UIs and real-time updates
2. **More specialized model-specific templates** - Templates optimized for specific Ollama models
3. **Tool verification** - Additional checks to ensure tools are called with valid parameters
4. **Structured output control** - Methods to more reliably extract structured outputs when needed

## Example Usage

The recommended way to use Ollama with BMW Agents is:

```python
from bmw_agents.utils.llm_providers import OllamaProvider
from bmw_agents.core.prompt_strategies.ollama_single_response_react import OllamaSingleResponseReAct

# Initialize the LLM provider
llm_provider = OllamaProvider(model_name="deepseek-r1:14b")

# Create the Ollama-specific strategy
prompt_strategy = OllamaSingleResponseReAct(
    llm_provider=llm_provider,
    tools=your_tools_list
)

# Run the strategy
result = await prompt_strategy.run("Your instruction here")

# Access execution trace
execution_trace = result["trace"]
final_answer = result["result"]
```

See the `ollama_single_response_example.py` file for a complete working example. 