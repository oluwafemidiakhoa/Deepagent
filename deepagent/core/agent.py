"""
DeepAgent Main Class

Integrates memory, tool retrieval, execution, and reasoning
into a unified autonomous agent.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import time

from .memory import (
    ThreeLayerMemory,
    EpisodicMemoryEntry,
    WorkingMemoryEntry,
    ToolMemoryEntry
)
from .reasoning import ReasoningEngine, ReasoningResult
from ..tools.retrieval import ToolRegistry, ToolDefinition, create_sample_tool_registry
from ..tools.executor import ToolExecutor, ExecutionResult, ExecutionStatus
from ..integrations.llm_providers import get_llm_provider, LLMProvider


@dataclass
class AgentConfig:
    """Configuration for DeepAgent"""
    max_steps: int = 50
    max_tools_per_search: int = 5
    tool_timeout: float = 30.0
    verbose: bool = True
    memory_episodic_max: int = 1000
    memory_working_max: int = 5
    memory_tool_max: int = 500
    llm_provider: str = "openai"  # "openai", "anthropic", or "ollama"
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.7
    llm_api_key: Optional[str] = None  # Optional, uses env var if not provided
    use_mock_llm: bool = False  # Set to True to use mock for testing


class DeepAgent:
    """
    DeepAgent - Autonomous AI Agent with End-to-End Reasoning

    Key features:
    - Three-layer memory system (episodic, working, tool)
    - Dense tool retrieval for dynamic API discovery
    - End-to-end reasoning loop
    - Tool policy optimization ready
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

        # Initialize memory system
        self.memory = ThreeLayerMemory(
            episodic_max=self.config.memory_episodic_max,
            working_max_active=self.config.memory_working_max,
            tool_max=self.config.memory_tool_max
        )

        # Initialize tool system
        self.tool_registry = create_sample_tool_registry()
        self.tool_executor = ToolExecutor(timeout=self.config.tool_timeout)

        # Initialize reasoning engine
        self.reasoning_engine = ReasoningEngine(
            max_steps=self.config.max_steps,
            verbose=self.config.verbose
        )

        # Initialize LLM provider (real or mock)
        if self.config.use_mock_llm:
            self.llm_provider = None  # Use mock implementation
        else:
            try:
                self.llm_provider: Optional[LLMProvider] = get_llm_provider(
                    provider_name=self.config.llm_provider,
                    api_key=self.config.llm_api_key,
                    model=self.config.llm_model
                )
            except Exception as e:
                if self.config.verbose:
                    print(f"Warning: Could not initialize LLM provider: {e}")
                    print("Falling back to mock LLM. Set OPENAI_API_KEY or ANTHROPIC_API_KEY to use real LLMs.")
                self.llm_provider = None

    def run(
        self,
        task: str,
        context: Optional[str] = None,
        max_steps: Optional[int] = None
    ) -> ReasoningResult:
        """
        Execute a task with autonomous reasoning

        Args:
            task: Natural language description of the task
            context: Optional additional context
            max_steps: Optional override for max reasoning steps

        Returns:
            ReasoningResult with answer and trace
        """
        if self.config.verbose:
            print("\n" + "=" * 70)
            print("DEEPAGENT EXECUTION")
            print("=" * 70)
            print(f"Task: {task}")
            print("=" * 70 + "\n")

        start_time = time.time()

        # Create working memory entry for this task
        task_entry = self.memory.working.add_subgoal(
            subgoal="Complete main task",
            content=task,
            priority=10
        )

        # Add to episodic memory
        self.memory.episodic.add(EpisodicMemoryEntry(
            event_type="task_start",
            content=f"Started task: {task}",
            importance_score=1.0
        ))

        # Get memory context
        memory_context = self.memory.get_full_context()

        if context:
            memory_context += f"\n\nADDITIONAL CONTEXT:\n{context}"

        # Override max steps if provided
        if max_steps:
            self.reasoning_engine.max_steps = max_steps

        # Execute reasoning loop
        result = self.reasoning_engine.reason(
            task=task,
            context=memory_context,
            tool_discovery_fn=self._discover_tools,
            tool_execution_fn=self._execute_tool,
            llm_generate_fn=self._generate_llm_response
        )

        # Update memories based on result
        if result.success:
            self.memory.working.complete_subgoal(task_entry)
            self.memory.episodic.add(EpisodicMemoryEntry(
                event_type="task_complete",
                content=f"Completed task: {task}",
                importance_score=0.9
            ))
        else:
            self.memory.working.fail_subgoal(task_entry)
            self.memory.episodic.add(EpisodicMemoryEntry(
                event_type="task_failed",
                content=f"Failed task: {task}",
                importance_score=0.8
            ))

        # Add execution time to result
        result.execution_time = time.time() - start_time

        if self.config.verbose:
            print("\n" + "=" * 70)
            print("EXECUTION COMPLETE")
            print("=" * 70)
            print(f"Success: {result.success}")
            print(f"Steps: {result.total_steps}")
            print(f"Tools Used: {', '.join(result.tools_used) if result.tools_used else 'None'}")
            print(f"Time: {result.execution_time:.2f}s")
            print("=" * 70)
            print(f"\nFINAL ANSWER:\n{result.answer}")
            print("=" * 70 + "\n")

        return result

    def _discover_tools(self, query: str, max_tools: int = 5) -> List[ToolDefinition]:
        """Discover relevant tools for a query"""
        tools = self.tool_registry.discover_tools(
            task_description=query,
            max_tools=max_tools
        )

        # Add to episodic memory
        self.memory.episodic.add(EpisodicMemoryEntry(
            event_type="tool_discovery",
            content=f"Discovered {len(tools)} tools for: {query}",
            importance_score=0.5
        ))

        return tools

    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ExecutionResult:
        """Execute a tool and update memories"""
        # Execute tool
        result = self.tool_executor.execute(tool_name, parameters)

        # Add to tool memory
        tool_entry = ToolMemoryEntry(
            tool_name=tool_name,
            parameters=parameters,
            result=result.result,
            success=result.status == ExecutionStatus.SUCCESS,
            execution_time=result.execution_time,
            content=f"Executed {tool_name}"
        )
        self.memory.tool.add(tool_entry)

        # Add to episodic memory
        self.memory.episodic.add(EpisodicMemoryEntry(
            event_type="tool_execution",
            content=f"Executed {tool_name}: {'success' if result.status == ExecutionStatus.SUCCESS else 'failed'}",
            importance_score=0.7 if result.status == ExecutionStatus.SUCCESS else 0.6
        ))

        return result

    def _generate_llm_response(self, prompt: str) -> str:
        """
        Generate next reasoning step using LLM

        Uses configured LLM provider (OpenAI, Anthropic, Ollama) or falls back to mock.
        """
        # Use real LLM if available
        if self.llm_provider is not None:
            try:
                response = self.llm_provider.generate(
                    prompt=prompt,
                    temperature=self.config.llm_temperature,
                    max_tokens=2000
                )
                return response.content
            except Exception as e:
                if self.config.verbose:
                    print(f"LLM API error: {e}. Falling back to mock response.")
                # Fall through to mock implementation

        # Mock implementation for testing/fallback
        if "TASK:" in prompt and prompt.count("\n") < 20:
            return "THINK: Let me break down this task and identify what tools I need."

        elif "THINK:" in prompt and "SEARCH_TOOLS:" not in prompt:
            if "protein" in prompt.lower() or "brca" in prompt.lower():
                return "SEARCH_TOOLS: protein structure retrieval and analysis"
            elif "drug" in prompt.lower():
                return "SEARCH_TOOLS: drug discovery and molecular analysis"
            else:
                return "SEARCH_TOOLS: data analysis and information retrieval"

        elif "AVAILABLE TOOLS:" in prompt and "EXECUTE_TOOL:" not in prompt:
            lines = prompt.split("\n")
            for line in lines:
                if line.strip().startswith("- "):
                    tool_name = line.strip()[2:].split(":")[0]
                    if "protein" in tool_name.lower():
                        return f'EXECUTE_TOOL: {tool_name}\nPARAMETERS: {{"protein_id": "BRCA1"}}'
                    elif "search" in tool_name.lower():
                        return f'EXECUTE_TOOL: {tool_name}\nPARAMETERS: {{"query": "research topic", "max_results": 5}}'
                    else:
                        return f'EXECUTE_TOOL: {tool_name}\nPARAMETERS: {{}}'

        elif "RESULT:" in prompt:
            return "OBSERVE: The tool execution was successful and provided relevant data. Let me analyze these results."

        elif "OBSERVE:" in prompt:
            return "CONCLUDE: Based on the tool execution and analysis, I have gathered sufficient information to answer the task. The results show relevant data that addresses the original question."

        else:
            return "THINK: Continuing to reason about the task."

    def get_memory_summary(self) -> str:
        """Get summary of current memory state"""
        return self.memory.get_full_context()

    def get_tool_stats(self) -> Dict[str, Any]:
        """Get statistics about tool usage"""
        return {
            "total_tools_available": len(self.tool_registry.get_all_tools()),
            "tools_executed": len(self.memory.tool.entries),
            "tool_success_rate": sum(
                1 for e in self.memory.tool.entries if e.success
            ) / max(len(self.memory.tool.entries), 1),
            "top_tools": self.memory.tool.get_recommended_tools(top_k=5)
        }

    def save_state(self, filepath: str) -> None:
        """Save agent state to disk"""
        self.memory.save_to_file(filepath)

    def reset(self) -> None:
        """Reset agent state"""
        self.memory.clear()
        self.reasoning_engine.reasoning_trace.clear()

    def add_custom_tool(self, tool: ToolDefinition, implementation: callable) -> None:
        """Add a custom tool to the agent"""
        self.tool_registry.register_tool(tool)
        self.tool_executor.register_tool(tool.name, implementation)

    def __repr__(self) -> str:
        return f"DeepAgent(model={self.llm_model}, tools={len(self.tool_registry.get_all_tools())})"
