"""
Unit tests for DeepAgent core functionality
"""

import sys
sys.path.insert(0, '..')

import pytest
from deepagent import (
    DeepAgent,
    AgentConfig,
    ToolDefinition,
    ExecutionStatus
)


class TestDeepAgent:
    """Test suite for DeepAgent"""

    def test_agent_initialization(self):
        """Test basic agent initialization"""
        agent = DeepAgent()
        assert agent is not None
        assert agent.memory is not None
        assert agent.tool_registry is not None
        assert agent.tool_executor is not None

    def test_agent_with_config(self):
        """Test agent with custom configuration"""
        config = AgentConfig(
            max_steps=100,
            max_tools_per_search=10,
            verbose=False
        )
        agent = DeepAgent(config)
        assert agent.config.max_steps == 100
        assert agent.config.max_tools_per_search == 10
        assert agent.config.verbose is False

    def test_agent_run(self):
        """Test basic agent execution"""
        agent = DeepAgent(AgentConfig(verbose=False))
        result = agent.run(task="Simple test task", max_steps=10)

        assert result is not None
        assert hasattr(result, 'answer')
        assert hasattr(result, 'success')
        assert hasattr(result, 'reasoning_trace')
        assert hasattr(result, 'tools_used')

    def test_custom_tool_registration(self):
        """Test adding custom tools"""
        agent = DeepAgent()

        custom_tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            category="test",
            parameters=[{"name": "input", "type": "str"}],
            returns="str",
            examples=["test_tool('test')"]
        )

        def test_impl(input: str) -> str:
            return f"Test: {input}"

        agent.add_custom_tool(custom_tool, test_impl)

        # Verify tool was added
        tool = agent.tool_registry.get_tool("test_tool")
        assert tool is not None
        assert tool.name == "test_tool"

    def test_memory_operations(self):
        """Test memory system"""
        agent = DeepAgent()

        # Run a task to populate memory
        result = agent.run(task="Test task", max_steps=5)

        # Check memory has entries
        assert len(agent.memory.episodic.memories) > 0
        assert len(agent.memory.tool.entries) >= 0

        # Test memory summary
        summary = agent.get_memory_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_tool_discovery(self):
        """Test tool discovery"""
        agent = DeepAgent()

        tools = agent._discover_tools("protein structure analysis", max_tools=5)

        assert isinstance(tools, list)
        assert len(tools) <= 5

        if tools:
            assert hasattr(tools[0], 'name')
            assert hasattr(tools[0], 'description')

    def test_tool_execution(self):
        """Test tool execution"""
        agent = DeepAgent()

        # Execute a known tool
        result = agent._execute_tool(
            "fetch_wikipedia",
            {"topic": "DNA"}
        )

        assert result is not None
        assert hasattr(result, 'status')
        assert result.status in [ExecutionStatus.SUCCESS, ExecutionStatus.ERROR]

    def test_agent_reset(self):
        """Test agent reset"""
        agent = DeepAgent(AgentConfig(verbose=False))

        # Run task
        agent.run(task="Test task", max_steps=5)

        # Check memory has data
        assert len(agent.memory.episodic.memories) > 0

        # Reset
        agent.reset()

        # Check memory is cleared
        assert len(agent.memory.episodic.memories) == 0
        assert len(agent.memory.working.entries) == 0

    def test_tool_stats(self):
        """Test tool statistics"""
        agent = DeepAgent(AgentConfig(verbose=False))

        # Run task
        agent.run(task="Test task", max_steps=10)

        # Get stats
        stats = agent.get_tool_stats()

        assert isinstance(stats, dict)
        assert 'total_tools_available' in stats
        assert 'tools_executed' in stats
        assert stats['total_tools_available'] > 0

    def test_multiple_tasks(self):
        """Test running multiple tasks"""
        agent = DeepAgent(AgentConfig(verbose=False))

        result1 = agent.run(task="First task", max_steps=5)
        result2 = agent.run(task="Second task", max_steps=5)

        assert result1 is not None
        assert result2 is not None

        # Check that memory accumulated
        assert len(agent.memory.episodic.memories) >= 2


class TestMemorySystem:
    """Test suite for memory components"""

    def test_episodic_memory(self):
        """Test episodic memory"""
        from deepagent.core.memory import EpisodicMemory, EpisodicMemoryEntry

        memory = EpisodicMemory(max_size=100)

        entry = EpisodicMemoryEntry(
            event_type="test",
            content="Test content",
            importance_score=0.8
        )

        memory.add(entry)
        assert len(memory.memories) == 1

        results = memory.query(event_type="test")
        assert len(results) == 1

    def test_working_memory(self):
        """Test working memory"""
        from deepagent.core.memory import WorkingMemory

        memory = WorkingMemory(max_active=5)

        entry = memory.add_subgoal(
            subgoal="Test goal",
            content="Test content",
            priority=10
        )

        assert memory.current_focus == entry
        assert len(memory.get_active_subgoals()) == 1

        memory.complete_subgoal(entry)
        assert entry.status == "completed"

    def test_tool_memory(self):
        """Test tool memory"""
        from deepagent.core.memory import ToolMemory, ToolMemoryEntry

        memory = ToolMemory(max_size=100)

        entry = ToolMemoryEntry(
            tool_name="test_tool",
            parameters={"param": "value"},
            result="result",
            success=True,
            execution_time=0.5
        )

        memory.add(entry)
        assert len(memory.entries) == 1

        success_rate = memory.get_success_rate("test_tool")
        assert success_rate == 1.0


class TestToolSystem:
    """Test suite for tool components"""

    def test_tool_registry(self):
        """Test tool registry"""
        from deepagent.tools.retrieval import ToolRegistry, ToolDefinition

        registry = ToolRegistry()

        tool = ToolDefinition(
            name="test_tool",
            description="Test tool",
            category="test",
            parameters=[],
            returns="str",
            examples=[]
        )

        registry.register_tool(tool)

        retrieved = registry.get_tool("test_tool")
        assert retrieved is not None
        assert retrieved.name == "test_tool"

    def test_tool_executor(self):
        """Test tool executor"""
        from deepagent.tools.executor import ToolExecutor, ExecutionStatus

        executor = ToolExecutor()

        # Test mock tool
        result = executor.execute(
            "fetch_wikipedia",
            {"topic": "test"}
        )

        assert result.status in [ExecutionStatus.SUCCESS, ExecutionStatus.ERROR]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
