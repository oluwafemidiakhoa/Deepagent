"""
Behavior Baseline - Foundation #5

Profiles normal agent behavior patterns.
"""

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List
import statistics


@dataclass
class BehaviorProfile:
    """Statistical profile of normal behavior"""
    avg_actions_per_session: float
    common_tools: List[str]
    avg_risk_score: float
    typical_duration_ms: float
    action_type_distribution: Dict[str, float]


class BehaviorBaseline:
    """Learns and maintains behavioral baseline"""

    def __init__(self):
        self.action_counts: List[int] = []
        self.tool_usage: defaultdict = defaultdict(int)
        self.risk_scores: List[float] = []
        self.durations: List[float] = []
        self.action_types: defaultdict = defaultdict(int)

    def record_session(
        self,
        action_count: int,
        tools_used: List[str],
        risk_scores: List[float],
        durations: List[float],
        action_types: List[str]
    ):
        """Record session for baseline learning"""
        self.action_counts.append(action_count)

        for tool in tools_used:
            self.tool_usage[tool] += 1

        self.risk_scores.extend(risk_scores)
        self.durations.extend(durations)

        for action_type in action_types:
            self.action_types[action_type] += 1

    def get_profile(self) -> BehaviorProfile:
        """Get current behavior profile"""
        if not self.action_counts:
            return BehaviorProfile(
                avg_actions_per_session=0,
                common_tools=[],
                avg_risk_score=0,
                typical_duration_ms=0,
                action_type_distribution={}
            )

        # Most common tools
        common_tools = sorted(
            self.tool_usage.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Action type distribution
        total_actions = sum(self.action_types.values())
        distribution = {
            k: v / total_actions
            for k, v in self.action_types.items()
        }

        return BehaviorProfile(
            avg_actions_per_session=statistics.mean(self.action_counts),
            common_tools=[tool for tool, _ in common_tools],
            avg_risk_score=statistics.mean(self.risk_scores) if self.risk_scores else 0,
            typical_duration_ms=statistics.mean(self.durations) if self.durations else 0,
            action_type_distribution=distribution
        )
