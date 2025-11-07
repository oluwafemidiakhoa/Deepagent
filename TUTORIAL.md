# DeepAgent Tutorial

A comprehensive guide to using DeepAgent for autonomous task execution.

## Table of Contents

1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Getting Started](#getting-started)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Integration Guide](#integration-guide)
7. [Best Practices](#best-practices)

## Introduction

DeepAgent is an advanced agentic AI system that implements:

- **End-to-End Reasoning**: Continuous reasoning loop within the model
- **Dynamic Tool Discovery**: On-demand API search and integration
- **Three-Layer Memory**: Episodic, working, and tool memory
- **Tool Policy Optimization**: RL-based optimization for tool usage

Unlike traditional ReAct frameworks (Reason → Act → Observe), DeepAgent keeps the entire loop **inside** the model, enabling more fluid and adaptive reasoning.

## Core Concepts

### 1. Three-Layer Memory System

DeepAgent uses three specialized memory layers inspired by biological cognition:

#### Episodic Memory
- **Purpose**: Long-term storage of task events
- **Capacity**: 1000 entries (configurable)
- **Features**: Importance-based compression, event categorization
- **Use Case**: Track what happened across multiple tasks

```python
from deepagent.core.memory import EpisodicMemory, EpisodicMemoryEntry

memory = EpisodicMemory(max_size=1000)
memory.add(EpisodicMemoryEntry(
    event_type="task_complete",
    content="Successfully analyzed protein structure",
    importance_score=0.9
))
```

#### Working Memory
- **Purpose**: Current subgoals and focused context
- **Capacity**: 5 active subgoals (configurable)
- **Features**: Priority-based focus, automatic status tracking
- **Use Case**: Maintain focus during complex multi-step tasks

```python
from deepagent.core.memory import WorkingMemory

working = WorkingMemory(max_active=5)
entry = working.add_subgoal(
    subgoal="Analyze binding sites",
    content="Find drug-protein interaction sites",
    priority=10
)
working.complete_subgoal(entry)
```

#### Tool Memory
- **Purpose**: Tool usage patterns and outcomes
- **Capacity**: 500 executions (configurable)
- **Features**: Success rate tracking, execution time stats
- **Use Case**: Learn from past tool executions

```python
from deepagent.core.memory import ToolMemory, ToolMemoryEntry

tool_mem = ToolMemory(max_size=500)
success_rate = tool_mem.get_success_rate("search_pubmed")
recommended = tool_mem.get_recommended_tools(top_k=5)
```

### 2. Dense Tool Retrieval

DeepAgent discovers tools dynamically using semantic search:

```python
from deepagent.tools.retrieval import DenseToolRetriever, ToolDefinition

retriever = DenseToolRetriever()

# Search for relevant tools
results = retriever.search(
    query="analyze protein structure",
    top_k=5,
    category_filter="bioinformatics"
)

for tool, similarity in results:
    print(f"{tool.name}: {similarity:.2f}")
```

### 3. End-to-End Reasoning Loop

The reasoning engine orchestrates the entire process:

```python
from deepagent.core.reasoning import ReasoningEngine

engine = ReasoningEngine(max_steps=50, verbose=True)

result = engine.reason(
    task="Your task here",
    context="Current memory context",
    tool_discovery_fn=discover_tools_func,
    tool_execution_fn=execute_tool_func,
    llm_generate_fn=generate_response_func
)
```

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/deepagent.git
cd deepagent

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Quick Start

Run the quick start script:

```bash
python quickstart.py
```

Or use programmatically:

```python
from deepagent import DeepAgent

agent = DeepAgent()
result = agent.run(task="Your task here")
print(result.answer)
```

## Basic Usage

### Example 1: Simple Task

```python
from deepagent import DeepAgent

agent = DeepAgent()

result = agent.run(
    task="Search for information about CRISPR gene editing"
)

print(f"Answer: {result.answer}")
print(f"Tools used: {result.tools_used}")
```

### Example 2: Configured Agent

```python
from deepagent import DeepAgent, AgentConfig

config = AgentConfig(
    max_steps=100,
    max_tools_per_search=10,
    tool_timeout=60.0,
    verbose=True
)

agent = DeepAgent(config)

result = agent.run(
    task="Complex multi-step task",
    max_steps=50  # Override config
)
```

### Example 3: Custom Tools

```python
from deepagent import DeepAgent, ToolDefinition

agent = DeepAgent()

# Define custom tool
custom_tool = ToolDefinition(
    name="my_custom_tool",
    description="Does something specific",
    category="custom",
    parameters=[
        {"name": "input", "type": "str", "description": "Input data"}
    ],
    returns="str",
    examples=["my_custom_tool('test')"]
)

# Implement the tool
def my_tool_impl(input: str) -> str:
    return f"Processed: {input}"

# Register with agent
agent.add_custom_tool(custom_tool, my_tool_impl)

# Use it
result = agent.run(task="Use my custom tool to process data")
```

## Advanced Features

### Memory Persistence

Save and load agent memory:

```python
# Save memory state
agent.save_state("agent_memory.json")

# Create new agent and load state
new_agent = DeepAgent()
# (Load functionality would be added to ThreeLayerMemory)
```

### Reasoning Trace Analysis

Inspect detailed reasoning:

```python
result = agent.run(task="Complex task")

# Analyze each step
for i, trace in enumerate(result.reasoning_trace):
    print(f"\nStep {i+1}:")
    print(f"  Type: {trace.step_type.value}")
    print(f"  Content: {trace.content}")

    if trace.tool_name:
        print(f"  Tool: {trace.tool_name}")
        print(f"  Result: {trace.tool_result}")
```

### Tool Statistics

Monitor tool performance:

```python
stats = agent.get_tool_stats()

print(f"Total tools: {stats['total_tools_available']}")
print(f"Tools executed: {stats['tools_executed']}")
print(f"Success rate: {stats['tool_success_rate']:.2%}")
print(f"Top tools: {stats['top_tools']}")
```

### ToolPO Training

Train the agent to optimize tool usage:

```python
from deepagent.training import ToolPolicyOptimizer

optimizer = ToolPolicyOptimizer(
    learning_rate=3e-4,
    clip_epsilon=0.2
)

# Define training tasks
training_tasks = [
    "Task 1: ...",
    "Task 2: ...",
    "Task 3: ...",
]

# Train
stats = optimizer.train(
    agent=agent,
    tasks=training_tasks,
    num_iterations=100,
    episodes_per_iteration=10
)

# Save training stats
optimizer.save_stats("training_stats.json")
```

## Integration Guide

### GENESIS-AI Integration

For synthetic biology applications:

```python
from deepagent import DeepAgent, ToolDefinition

# Create specialized agent
agent = DeepAgent()

# Add biology-specific tools
bio_tools = [
    ToolDefinition(
        name="design_genetic_circuit",
        description="Design genetic circuits",
        category="synthetic_biology",
        parameters=[...],
        returns="Dict",
        examples=[...]
    ),
    # ... more tools
]

for tool in bio_tools:
    agent.add_custom_tool(tool, implementation)

# Use for complex workflows
result = agent.run(
    task="""Design a metabolic pathway for biofuel production:
    1. Simulate pathway
    2. Optimize genes
    3. Predict yields
    """
)
```

### LLM Integration

Replace mock LLM with real model:

```python
import openai

class RealLLMAgent(DeepAgent):
    def _generate_llm_response(self, prompt: str) -> str:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
```

### API Integration

Connect to real APIs:

```python
def real_search_pubmed(query: str, max_results: int) -> list:
    import requests

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json"
    }

    response = requests.get(url, params=params)
    return response.json()

agent.tool_executor.register_tool("search_pubmed", real_search_pubmed)
```

## Best Practices

### 1. Task Design

Write clear, specific tasks:

```python
# ❌ Vague
result = agent.run("Do some biology stuff")

# ✅ Clear
result = agent.run(
    task="""Analyze BRCA1 protein:
    1. Get structure from PDB
    2. Identify binding sites
    3. Search for known drug interactions
    """
)
```

### 2. Memory Management

Monitor and reset when needed:

```python
# Check memory usage
print(agent.get_memory_summary())

# Reset for new session
agent.reset()
```

### 3. Tool Organization

Group related tools by category:

```python
retriever.search(
    query="protein analysis",
    category_filter="bioinformatics"  # Focus search
)
```

### 4. Error Handling

Handle failures gracefully:

```python
result = agent.run(task="Complex task")

if not result.success:
    print(f"Task failed after {result.total_steps} steps")
    print(f"Reason: {result.metadata.get('termination_reason')}")
```

### 5. Performance Optimization

- Use `verbose=False` for production
- Set appropriate `max_steps` for task complexity
- Filter tool searches by category
- Monitor execution time

### 6. Testing

Test custom tools before integration:

```python
from deepagent.tools.executor import ToolExecutor

executor = ToolExecutor()
executor.register_tool("my_tool", my_implementation)

result = executor.execute("my_tool", {"param": "value"})
assert result.status == ExecutionStatus.SUCCESS
```

## Next Steps

- Explore [examples/basic_usage.py](examples/basic_usage.py)
- Check [examples/genesis_ai_integration.py](examples/genesis_ai_integration.py)
- Read the API documentation
- Join the community discussions

## Support

- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share use cases
- Documentation: Full API reference

---

**Happy Reasoning with DeepAgent!** 🚀
