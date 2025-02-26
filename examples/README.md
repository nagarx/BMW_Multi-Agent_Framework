# BMW Agents Framework Examples

This directory contains example scripts demonstrating how to use the BMW Agents framework with different LLM providers.

## Using Ollama with BMW Agents

The following examples show how to use locally running Ollama models with the BMW Agents framework:

### Prerequisites

1. Make sure you have Ollama installed and running. Follow the installation instructions at [Ollama's website](https://ollama.ai).

2. Pull the deepseek-r1:14b model (or any other model you want to use):
   ```bash
   ollama pull deepseek-r1:14b
   ```

3. Make sure Ollama server is running:
   ```bash
   ollama serve
   ```

### Basic Examples

Run the basic examples to test if Ollama is working with the framework:

```bash
python ollama_test.py  # Simple query
python ollama_react_example.py  # Basic ReAct example
python simple_react_example.py  # Simplified ReAct implementation
```

### Improved Ollama Integration

The framework now includes Ollama-specific prompt strategies that are better optimized for the output format of Ollama models:

```bash
python improved_ollama_react_example.py  # Optimized ReAct strategy for Ollama
python improved_ollama_plan_react_example.py  # Optimized PlanReAct strategy for Ollama
```

### Traced Implementations

For better debugging and visualization of the execution steps, we've created traced versions of the ReAct and PlanReAct strategies that explicitly track thoughts, actions, observations, and plans:

```bash
python improved_ollama_traced_react_example.py  # Traced ReAct strategy for Ollama
python improved_ollama_traced_plan_react_example.py  # Traced PlanReAct strategy for Ollama
```

These traced implementations provide complete execution traces, allowing you to see every step of the process, including:
- Plans created by the model (for PlanReAct)
- Thoughts at each step
- Actions taken
- Observations received

### Single-Response Implementation (Recommended for Ollama)

We've found that Ollama models tend to generate a complete execution trace in a single response, rather than iteratively as expected by traditional ReAct implementations. To better handle this behavior, we've created specialized SingleResponseTracedReAct strategies:

```bash
python ollama_single_response_example.py  # SingleResponseTracedReAct strategy for Ollama
python ollama_single_response_plan_react_example.py  # SingleResponseTracedPlanReAct strategy for Ollama
```

These implementations:
1. Make a single call to the LLM rather than iterating
2. Parse the complete thought/action/observation sequence from the response
3. Execute any tools called by the model
4. Provide a full execution trace of all steps

The PlanReAct version also extracts and displays the plan created by the model.

This approach is more reliable with Ollama models since it works with their natural generation behavior, and avoids the need for multiple API calls.

### Key Features of the Ollama Integration

1. **Specialized Prompt Strategies**: The framework includes specialized classes optimized for Ollama models:
   - `OllamaReActPromptStrategy` and `OllamaPlanReActPromptStrategy` (basic versions)
   - `OllamaTracedReAct` and `OllamaTracedPlanReAct` (traced versions with explicit execution tracking)
   - `OllamaSingleResponseReAct` and `OllamaSingleResponsePlanReAct` (optimized for Ollama's single-response behavior)

2. **Enhanced Templates**: Ollama-specific templates that provide clearer formatting instructions and avoid markdown formatting issues.

3. **Improved JSON Parsing**: More robust extraction of thoughts, actions, and plans from Ollama model responses.

4. **Better Error Handling**: Enhanced error recovery when responses don't match the expected format.

5. **Execution Tracing**: Complete tracking of the execution workflow, including all intermediate steps.

## Configuring Ollama

By default, the examples connect to Ollama running on `http://localhost:11434`. If your Ollama instance is running on a different host or port, you can set the `OLLAMA_HOST` environment variable:

```bash
export OLLAMA_HOST="http://your-ollama-host:port"
```

You can also modify the model name in the example scripts if you want to use a different model than `deepseek-r1:14b`.

## Other Examples

Feel free to create more examples and add them to this directory. To use different LLM providers, check the `bmw_agents/utils/llm_providers.py` file for available options. 