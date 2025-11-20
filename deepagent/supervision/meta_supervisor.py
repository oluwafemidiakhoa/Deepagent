"""
Meta Supervisor - Foundation #6

Provides high-level oversight for multi-agent systems.
Monitors agent activities, coordinates policies, and triggers interventions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4


class SupervisionLevel(Enum):
    """Level of supervision intensity"""
    MINIMAL = "minimal"          # Hands-off, only critical violations
    STANDARD = "standard"        # Normal supervision
    HEIGHTENED = "heightened"    # Increased monitoring
    MAXIMUM = "maximum"          # Constant oversight


class AgentStatus(Enum):
    """Status of supervised agent"""
    ACTIVE = "active"
    PAUSED = "paused"
    RESTRICTED = "restricted"
    TERMINATED = "terminated"


@dataclass
class SupervisionConfig:
    """Configuration for meta-supervision"""
    supervision_level: SupervisionLevel = SupervisionLevel.STANDARD
    enable_cross_agent_monitoring: bool = True
    enable_resource_coordination: bool = True
    enable_conflict_detection: bool = True
    max_agents_per_supervisor: int = 10
    intervention_threshold: float = 0.7
    coordination_enabled: bool = True


@dataclass
class AgentState:
    """State of a supervised agent"""
    agent_id: str
    agent_type: str
    status: AgentStatus
    current_task: Optional[str]
    risk_level: str
    resource_usage: Dict[str, float]
    violations: List[str] = field(default_factory=list)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SupervisionResult:
    """Result of supervision check"""
    agent_id: str
    supervision_passed: bool
    issues_detected: List[str]
    interventions_required: List[str]
    recommended_actions: List[str]
    risk_assessment: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CoordinationEvent:
    """Event requiring multi-agent coordination"""
    event_id: str
    event_type: str  # "resource_conflict", "goal_conflict", "dependency"
    involved_agents: List[str]
    description: str
    severity: str
    resolution: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class MetaSupervisor:
    """
    Meta-level supervisor for multi-agent systems.

    Provides:
    - Multi-agent monitoring and coordination
    - Cross-agent policy enforcement
    - Resource conflict detection
    - Intervention orchestration
    """

    def __init__(self, config: Optional[SupervisionConfig] = None):
        self.config = config or SupervisionConfig()
        self.supervised_agents: Dict[str, AgentState] = {}
        self.coordination_events: List[CoordinationEvent] = []
        self.supervision_history: List[SupervisionResult] = []
        self.global_policies: List['MetaPolicy'] = []

    def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register an agent for supervision.

        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent
            metadata: Optional agent metadata

        Returns:
            True if registered successfully
        """
        if len(self.supervised_agents) >= self.config.max_agents_per_supervisor:
            return False

        if agent_id in self.supervised_agents:
            return False

        state = AgentState(
            agent_id=agent_id,
            agent_type=agent_type,
            status=AgentStatus.ACTIVE,
            current_task=None,
            risk_level="LOW",
            resource_usage={},
            metadata=metadata or {}
        )

        self.supervised_agents[agent_id] = state
        return True

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from supervision"""
        if agent_id in self.supervised_agents:
            del self.supervised_agents[agent_id]
            return True
        return False

    def update_agent_state(
        self,
        agent_id: str,
        status: Optional[AgentStatus] = None,
        current_task: Optional[str] = None,
        risk_level: Optional[str] = None,
        resource_usage: Optional[Dict[str, float]] = None
    ) -> bool:
        """Update state of a supervised agent"""
        if agent_id not in self.supervised_agents:
            return False

        agent = self.supervised_agents[agent_id]

        if status is not None:
            agent.status = status
        if current_task is not None:
            agent.current_task = current_task
        if risk_level is not None:
            agent.risk_level = risk_level
        if resource_usage is not None:
            agent.resource_usage.update(resource_usage)

        agent.last_activity = datetime.now()
        return True

    def supervise_agent(self, agent_id: str) -> SupervisionResult:
        """
        Perform supervision check on an agent.

        Args:
            agent_id: Agent to supervise

        Returns:
            SupervisionResult with findings
        """
        if agent_id not in self.supervised_agents:
            return SupervisionResult(
                agent_id=agent_id,
                supervision_passed=False,
                issues_detected=["Agent not registered"],
                interventions_required=[],
                recommended_actions=["Register agent for supervision"],
                risk_assessment="UNKNOWN"
            )

        agent = self.supervised_agents[agent_id]
        issues = []
        interventions = []
        recommendations = []

        # Check agent status
        if agent.status == AgentStatus.TERMINATED:
            issues.append("Agent is terminated")

        # Check for violations
        if agent.violations:
            issues.append(f"Agent has {len(agent.violations)} violations")
            if len(agent.violations) > 3:
                interventions.append("Consider restricting agent")

        # Check risk level
        if agent.risk_level in ["HIGH", "CRITICAL"]:
            issues.append(f"Agent operating at {agent.risk_level} risk")
            interventions.append("Increase monitoring")

        # Check resource usage
        if self.config.enable_resource_coordination:
            for resource, usage in agent.resource_usage.items():
                if usage > 0.8:  # 80% threshold
                    issues.append(f"High {resource} usage: {usage:.1%}")
                    recommendations.append(f"Optimize {resource} usage")

        # Check for conflicts with other agents
        if self.config.enable_conflict_detection:
            conflicts = self._detect_conflicts(agent_id)
            if conflicts:
                issues.extend(conflicts)
                interventions.append("Resolve agent conflicts")

        # Determine if supervision passed
        passed = (
            len(issues) == 0 or
            (len(interventions) == 0 and agent.risk_level not in ["CRITICAL"])
        )

        result = SupervisionResult(
            agent_id=agent_id,
            supervision_passed=passed,
            issues_detected=issues,
            interventions_required=interventions,
            recommended_actions=recommendations,
            risk_assessment=agent.risk_level
        )

        self.supervision_history.append(result)
        return result

    def supervise_all_agents(self) -> List[SupervisionResult]:
        """Supervise all registered agents"""
        results = []
        for agent_id in self.supervised_agents:
            result = self.supervise_agent(agent_id)
            results.append(result)
        return results

    def _detect_conflicts(self, agent_id: str) -> List[str]:
        """Detect conflicts with other agents"""
        conflicts = []
        agent = self.supervised_agents[agent_id]

        for other_id, other in self.supervised_agents.items():
            if other_id == agent_id:
                continue

            # Check resource conflicts
            for resource in agent.resource_usage:
                if resource in other.resource_usage:
                    if (agent.resource_usage[resource] +
                        other.resource_usage[resource]) > 1.0:
                        conflicts.append(
                            f"Resource conflict with {other_id} on {resource}"
                        )

                        # Record coordination event
                        self._record_coordination_event(
                            event_type="resource_conflict",
                            involved_agents=[agent_id, other_id],
                            description=f"Competing for {resource}",
                            severity="MEDIUM"
                        )

        return conflicts

    def _record_coordination_event(
        self,
        event_type: str,
        involved_agents: List[str],
        description: str,
        severity: str
    ) -> None:
        """Record a coordination event"""
        event = CoordinationEvent(
            event_id=f"coord_{uuid4().hex[:12]}",
            event_type=event_type,
            involved_agents=involved_agents,
            description=description,
            severity=severity
        )
        self.coordination_events.append(event)

    def coordinate_agents(
        self,
        agent_ids: List[str],
        coordination_goal: str
    ) -> bool:
        """
        Coordinate multiple agents for a goal.

        Args:
            agent_ids: Agents to coordinate
            coordination_goal: What to coordinate on

        Returns:
            True if coordination successful
        """
        if not self.config.coordination_enabled:
            return False

        # Verify all agents exist
        for agent_id in agent_ids:
            if agent_id not in self.supervised_agents:
                return False

        # Record coordination
        self._record_coordination_event(
            event_type="coordination_requested",
            involved_agents=agent_ids,
            description=coordination_goal,
            severity="LOW"
        )

        return True

    def pause_agent(self, agent_id: str, reason: str) -> bool:
        """Pause an agent"""
        if agent_id not in self.supervised_agents:
            return False

        self.supervised_agents[agent_id].status = AgentStatus.PAUSED
        self.supervised_agents[agent_id].violations.append(f"Paused: {reason}")
        return True

    def restrict_agent(self, agent_id: str, reason: str) -> bool:
        """Restrict an agent's capabilities"""
        if agent_id not in self.supervised_agents:
            return False

        self.supervised_agents[agent_id].status = AgentStatus.RESTRICTED
        self.supervised_agents[agent_id].violations.append(f"Restricted: {reason}")
        return True

    def terminate_agent(self, agent_id: str, reason: str) -> bool:
        """Terminate an agent"""
        if agent_id not in self.supervised_agents:
            return False

        self.supervised_agents[agent_id].status = AgentStatus.TERMINATED
        self.supervised_agents[agent_id].violations.append(f"Terminated: {reason}")
        return True

    def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """Get current state of an agent"""
        return self.supervised_agents.get(agent_id)

    def get_all_agents(self) -> List[AgentState]:
        """Get all supervised agents"""
        return list(self.supervised_agents.values())

    def get_coordination_events(
        self,
        event_type: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> List[CoordinationEvent]:
        """Get coordination events, optionally filtered"""
        events = self.coordination_events

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if agent_id:
            events = [e for e in events if agent_id in e.involved_agents]

        return events

    def get_statistics(self) -> Dict[str, Any]:
        """Get supervision statistics"""
        total_agents = len(self.supervised_agents)

        if total_agents == 0:
            return {
                'total_agents': 0,
                'active_agents': 0,
                'paused_agents': 0,
                'restricted_agents': 0,
                'terminated_agents': 0,
                'coordination_events': 0,
                'total_violations': 0
            }

        status_counts = {
            'active': 0,
            'paused': 0,
            'restricted': 0,
            'terminated': 0
        }

        total_violations = 0

        for agent in self.supervised_agents.values():
            status_counts[agent.status.value] += 1
            total_violations += len(agent.violations)

        return {
            'total_agents': total_agents,
            'active_agents': status_counts['active'],
            'paused_agents': status_counts['paused'],
            'restricted_agents': status_counts['restricted'],
            'terminated_agents': status_counts['terminated'],
            'coordination_events': len(self.coordination_events),
            'total_violations': total_violations,
            'supervision_level': self.config.supervision_level.value
        }
