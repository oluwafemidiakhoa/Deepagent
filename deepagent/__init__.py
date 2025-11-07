"""
DeepAgent - Advanced Agentic AI System

A production-ready implementation of DeepAgent-inspired architecture.
"""

__version__ = "0.1.0"

from .core.agent import DeepAgent, AgentConfig
from .core.memory import (
    ThreeLayerMemory,
    EpisodicMemory,
    WorkingMemory,
    ToolMemory,
    EpisodicMemoryEntry,
    WorkingMemoryEntry,
    ToolMemoryEntry
)
from .core.reasoning import ReasoningEngine, ReasoningResult, ReasoningTrace
from .tools.retrieval import (
    ToolRegistry,
    ToolDefinition,
    DenseToolRetriever,
    create_sample_tool_registry
)
from .tools.executor import ToolExecutor, ExecutionResult, ExecutionStatus
from .training.toolpo import ToolPolicyOptimizer, RewardModel

__all__ = [
    # Main agent
    "DeepAgent",
    "AgentConfig",

    # Memory system
    "ThreeLayerMemory",
    "EpisodicMemory",
    "WorkingMemory",
    "ToolMemory",
    "EpisodicMemoryEntry",
    "WorkingMemoryEntry",
    "ToolMemoryEntry",

    # Reasoning
    "ReasoningEngine",
    "ReasoningResult",
    "ReasoningTrace",

    # Tools
    "ToolRegistry",
    "ToolDefinition",
    "DenseToolRetriever",
    "create_sample_tool_registry",
    "ToolExecutor",
    "ExecutionResult",
    "ExecutionStatus",

    # Training
    "ToolPolicyOptimizer",
    "RewardModel",
]
