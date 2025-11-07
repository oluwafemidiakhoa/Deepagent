"""
Three-Layer Memory System for DeepAgent

Implements episodic, working, and tool memory to manage context
and prevent overflow during long-horizon reasoning tasks.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


@dataclass
class MemoryEntry:
    """Base class for memory entries"""
    timestamp: datetime = field(default_factory=datetime.now)
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "metadata": self.metadata
        }


@dataclass
class EpisodicMemoryEntry(MemoryEntry):
    """Long-term storage of task events and outcomes"""
    event_type: str = ""  # observation, action, reasoning, outcome
    importance_score: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "event_type": self.event_type,
            "importance_score": self.importance_score
        })
        return data


@dataclass
class WorkingMemoryEntry(MemoryEntry):
    """Current subgoal and focused context"""
    subgoal: str = ""
    priority: int = 0
    status: str = "active"  # active, completed, failed

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "subgoal": self.subgoal,
            "priority": self.priority,
            "status": self.status
        })
        return data


@dataclass
class ToolMemoryEntry(MemoryEntry):
    """Tool names, parameters, and outcomes"""
    tool_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    success: bool = False
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "result": str(self.result) if self.result else None,
            "success": self.success,
            "execution_time": self.execution_time
        })
        return data


class EpisodicMemory:
    """
    Long-term memory for task events
    Implements importance-based compression to prevent overflow
    """

    def __init__(self, max_size: int = 1000, compression_threshold: float = 0.8):
        self.memories: List[EpisodicMemoryEntry] = []
        self.max_size = max_size
        self.compression_threshold = compression_threshold

    def add(self, entry: EpisodicMemoryEntry) -> None:
        """Add new episodic memory with automatic compression"""
        self.memories.append(entry)

        if len(self.memories) > self.max_size * self.compression_threshold:
            self._compress()

    def _compress(self) -> None:
        """Keep only high-importance memories"""
        # Sort by importance and keep top memories
        self.memories.sort(key=lambda x: x.importance_score, reverse=True)
        self.memories = self.memories[:int(self.max_size * 0.7)]

    def query(self, event_type: Optional[str] = None, limit: int = 10) -> List[EpisodicMemoryEntry]:
        """Query memories by type"""
        if event_type:
            filtered = [m for m in self.memories if m.event_type == event_type]
        else:
            filtered = self.memories

        return sorted(filtered, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_summary(self) -> str:
        """Generate compressed summary of episodic memory"""
        if not self.memories:
            return "No episodic memories yet."

        summary_parts = []
        event_types = set(m.event_type for m in self.memories)

        for event_type in event_types:
            events = [m for m in self.memories if m.event_type == event_type]
            summary_parts.append(f"{event_type.upper()}: {len(events)} events")

        return " | ".join(summary_parts)


class WorkingMemory:
    """
    Short-term memory for current subgoals and focus
    Maintains active task context
    """

    def __init__(self, max_active: int = 5):
        self.entries: List[WorkingMemoryEntry] = []
        self.max_active = max_active
        self.current_focus: Optional[WorkingMemoryEntry] = None

    def add_subgoal(self, subgoal: str, content: str, priority: int = 0) -> WorkingMemoryEntry:
        """Add new subgoal to working memory"""
        entry = WorkingMemoryEntry(
            subgoal=subgoal,
            content=content,
            priority=priority,
            status="active"
        )
        self.entries.append(entry)

        # Auto-set focus to highest priority active task
        self._update_focus()

        return entry

    def complete_subgoal(self, entry: WorkingMemoryEntry) -> None:
        """Mark subgoal as completed"""
        entry.status = "completed"
        self._update_focus()

    def fail_subgoal(self, entry: WorkingMemoryEntry) -> None:
        """Mark subgoal as failed"""
        entry.status = "failed"
        self._update_focus()

    def _update_focus(self) -> None:
        """Update current focus to highest priority active task"""
        active = [e for e in self.entries if e.status == "active"]
        if active:
            self.current_focus = max(active, key=lambda x: x.priority)
        else:
            self.current_focus = None

    def get_active_subgoals(self) -> List[WorkingMemoryEntry]:
        """Get all active subgoals"""
        return [e for e in self.entries if e.status == "active"]

    def get_context(self) -> str:
        """Get current working memory context as string"""
        if not self.current_focus:
            return "No active subgoal."

        active = self.get_active_subgoals()
        context = f"CURRENT FOCUS: {self.current_focus.subgoal}\n"
        context += f"Details: {self.current_focus.content}\n"

        if len(active) > 1:
            context += f"\nOTHER ACTIVE SUBGOALS ({len(active)-1}):\n"
            for entry in active:
                if entry != self.current_focus:
                    context += f"  - {entry.subgoal}\n"

        return context


class ToolMemory:
    """
    Memory for tool usage patterns and outcomes
    Enables learning from past tool executions
    """

    def __init__(self, max_size: int = 500):
        self.entries: List[ToolMemoryEntry] = []
        self.max_size = max_size
        self.tool_stats: Dict[str, Dict[str, Any]] = {}

    def add(self, entry: ToolMemoryEntry) -> None:
        """Add tool execution to memory"""
        self.entries.append(entry)

        # Update statistics
        if entry.tool_name not in self.tool_stats:
            self.tool_stats[entry.tool_name] = {
                "total_calls": 0,
                "successes": 0,
                "failures": 0,
                "avg_time": 0.0
            }

        stats = self.tool_stats[entry.tool_name]
        stats["total_calls"] += 1

        if entry.success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1

        # Update average execution time
        stats["avg_time"] = (
            (stats["avg_time"] * (stats["total_calls"] - 1) + entry.execution_time)
            / stats["total_calls"]
        )

        # Maintain size limit
        if len(self.entries) > self.max_size:
            self.entries = self.entries[-self.max_size:]

    def get_tool_history(self, tool_name: str, limit: int = 5) -> List[ToolMemoryEntry]:
        """Get recent executions of a specific tool"""
        history = [e for e in self.entries if e.tool_name == tool_name]
        return history[-limit:]

    def get_success_rate(self, tool_name: str) -> float:
        """Get success rate for a tool"""
        if tool_name not in self.tool_stats:
            return 0.0

        stats = self.tool_stats[tool_name]
        if stats["total_calls"] == 0:
            return 0.0

        return stats["successes"] / stats["total_calls"]

    def get_tool_stats(self, tool_name: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a tool"""
        return self.tool_stats.get(tool_name, {
            "total_calls": 0,
            "successes": 0,
            "failures": 0,
            "avg_time": 0.0
        })

    def get_recommended_tools(self, top_k: int = 10) -> List[str]:
        """Get tools ranked by success rate and usage"""
        ranked = sorted(
            self.tool_stats.items(),
            key=lambda x: (x[1]["successes"] / max(x[1]["total_calls"], 1), x[1]["total_calls"]),
            reverse=True
        )
        return [name for name, _ in ranked[:top_k]]


class ThreeLayerMemory:
    """
    Integrated three-layer memory system
    Coordinates episodic, working, and tool memory
    """

    def __init__(
        self,
        episodic_max: int = 1000,
        working_max_active: int = 5,
        tool_max: int = 500
    ):
        self.episodic = EpisodicMemory(max_size=episodic_max)
        self.working = WorkingMemory(max_active=working_max_active)
        self.tool = ToolMemory(max_size=tool_max)

    def get_full_context(self, include_episodic: bool = True) -> str:
        """
        Generate complete memory context for reasoning
        This is what gets fed to the LLM
        """
        context = []

        # Working memory (always included - this is the focus)
        context.append("=== WORKING MEMORY ===")
        context.append(self.working.get_context())

        # Tool memory summary
        context.append("\n=== TOOL MEMORY ===")
        recommended = self.tool.get_recommended_tools(top_k=5)
        if recommended:
            context.append("Top performing tools:")
            for tool in recommended:
                stats = self.tool.get_tool_stats(tool)
                success_rate = stats["successes"] / max(stats["total_calls"], 1) * 100
                context.append(f"  - {tool}: {success_rate:.1f}% success ({stats['total_calls']} calls)")
        else:
            context.append("No tool history yet.")

        # Episodic memory summary (optional, for long tasks)
        if include_episodic:
            context.append("\n=== EPISODIC MEMORY ===")
            context.append(self.episodic.get_summary())

        return "\n".join(context)

    def save_to_file(self, filepath: str) -> None:
        """Persist memory to disk"""
        data = {
            "episodic": [e.to_dict() for e in self.episodic.memories],
            "working": [e.to_dict() for e in self.working.entries],
            "tool": [e.to_dict() for e in self.tool.entries],
            "tool_stats": self.tool.tool_stats
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def clear(self) -> None:
        """Clear all memory layers"""
        self.episodic.memories.clear()
        self.working.entries.clear()
        self.working.current_focus = None
        self.tool.entries.clear()
        self.tool.tool_stats.clear()
