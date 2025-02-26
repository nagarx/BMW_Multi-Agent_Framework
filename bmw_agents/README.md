# BMW Agents Framework

This repository contains an implementation of the BMW Agents Framework as described in the paper "BMW Agents - A Framework For Task Automation Through Multi-Agent Collaboration".

## Overview

The BMW Agents Framework provides a flexible architecture for orchestrating multi-agent workflows that can automate complex tasks. The framework is built around three main stages:

1. **Planning** - Decomposition of the input into simple logical steps with a clearly defined order of operations.
2. **Execution** - Completion of the work planned in Step 1 by agents solving simple tasks and creating results.
3. **Verification** - Independent check if the original objective has been achieved in Step 2.

## Key Features

- **Multi-Agent Workflows**: Support for various collaboration patterns (Independent, Sequential, Joint, Hierarchical, Broadcast)
- **Flexible Prompt Strategies**: Non-iterative, ReAct, PlanReAct, and Conversational strategies
- **Memory Systems**: Short-term (within task) and Episodic (across tasks) memory
- **Tool Integration**: Standardized interface for tools with refinement mechanisms
- **Human Feedback**: Support for both intentional and incidental human feedback

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bmw-agents.git
cd bmw-agents

# Install dependencies
pip install -r requirements.txt
```

## Getting Started

```python
from bmw_agents.core.coordinator import Coordinator
from bmw_agents.examples.qa_rag import create_qa_workflow

# Create a workflow
workflow = create_qa_workflow()

# Run the workflow with a user instruction
result = workflow.run("What are the key components of the BMW Agents Framework?")
print(result)
```

## Examples

This repository includes several example applications:

1. **Question & Answer with RAG** - A simple Q&A system using Retrieval Augmented Generation
2. **Document Editing** - An actor/critic workflow for document editing
3. **Coding Tasks** - A software development workflow with multiple specialized agents

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation is based on the paper "BMW Agents - A Framework For Task Automation Through Multi-Agent Collaboration" by Noel Crawford, Edward B. Duffy, Iman Evazzade, Torsten Foehr, Gregory Robbins, Debbrata Kumar Saha, Jiya Varma, and Marcin Ziolkowski. 