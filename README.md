# DeepAgent - Advanced Agentic AI System

A production-ready implementation of DeepAgent-inspired architecture with end-to-end reasoning, dynamic tool discovery, and autonomous memory management.

## Key Features

### 1. End-to-End Reasoning Loop
Unlike traditional ReAct frameworks, DeepAgent keeps the entire reasoning loop inside the model:
- **Internal Reasoning**: Continuous thought process without external orchestration
- **Dynamic Tool Discovery**: On-demand API search and selection
- **Adaptive Execution**: Real-time tool chain optimization

### 2. Three-Layer Memory System
Modular memory architecture inspired by biological cognition:
- **Episodic Memory**: Long-term storage of task events and outcomes
- **Working Memory**: Current subgoal and focused context
- **Tool Memory**: Dynamic cache of tool names, parameters, and results

### 3. Dense Tool Retrieval
Dynamic discovery from large-scale API repositories:
- Semantic search over 10,000+ potential tools
- Runtime API discovery and integration
- Intelligent tool ranking and selection

### 4. ToolPO (Tool Policy Optimization)
Reinforcement learning for optimal tool usage:
- Token-level advantage attribution
- Simulated API rollouts for stable training
- PPO-based optimization for tool selection

## Architecture

```
deepagent/
├── core/
│   ├── agent.py           # Main DeepAgent orchestrator
│   ├── memory.py          # Three-layer memory system
│   └── reasoning.py       # End-to-end reasoning loop
├── tools/
│   ├── retrieval.py       # Dense tool retrieval
│   ├── executor.py        # Tool execution engine
│   └── registry.py        # API registry and management
├── training/
│   ├── toolpo.py          # Tool Policy Optimization
│   └── rewards.py         # Reward modeling for RL
├── utils/
│   ├── embeddings.py      # Semantic embedding utilities
│   └── config.py          # Configuration management
└── examples/
    ├── basic_usage.py     # Simple examples
    └── genesis_ai.py      # GENESIS-AI integration example

```

## Quick Start

```python
from deepagent import DeepAgent

# Initialize agent
agent = DeepAgent(
    model="gpt-4",
    max_tools=100,
    memory_layers=3
)

# Execute task with autonomous reasoning
result = agent.run(
    task="Find the protein structure for BRCA1 and analyze binding sites",
    max_steps=50
)

print(result.answer)
print(result.reasoning_trace)
print(result.tools_used)
```

## Installation

```bash
pip install -r requirements.txt
```

## Use Cases

- **Bioinformatics Research**: Dynamic API discovery for genomic databases
- **Scientific Simulation**: Long-horizon experimental workflows
- **Drug Discovery**: Multi-step pathway analysis and compound matching
- **General Purpose**: Any task requiring complex tool orchestration

## Author

**Oluwafemi Idiakhoa**
- Email: Oluwafemidiakhoa@gmail.com
- GitHub: https://github.com/oluwafemidiakhoa
- Repository: https://github.com/oluwafemidiakhoa/Deepagent

## License

MIT License
