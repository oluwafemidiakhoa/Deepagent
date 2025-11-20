"""
Capability Limiter - Foundation #8

Restricts agent capabilities based on purpose and risk.
Dynamically limits what tools and actions agents can use.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class CapabilityLevel(Enum):
    """Level of capability restriction"""
    FULL = "full"              # All capabilities
    STANDARD = "standard"      # Most capabilities
    LIMITED = "limited"        # Restricted capabilities
    MINIMAL = "minimal"        # Only essential capabilities


@dataclass
class CapabilitySet:
    """Set of capabilities for an agent"""
    capability_id: str
    name: str
    level: CapabilityLevel
    allowed_tools: Set[str]
    allowed_actions: Set[str]
    denied_tools: Set[str] = field(default_factory=set)
    denied_actions: Set[str] = field(default_factory=set)
    restrictions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityRestriction:
    """Record of a capability restriction"""
    restriction_id: str
    agent_id: str
    tool: str
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    permanent: bool = False


@dataclass
class CapabilityCheckResult:
    """Result of capability check"""
    agent_id: str
    tool: str
    action: str
    allowed: bool
    reason: str
    restrictions: List[str]


class CapabilityLimiter:
    """
    Limits agent capabilities based on purpose and context.

    Provides:
    - Capability set definition
    - Dynamic capability restriction
    - Tool and action allowlisting
    - Temporary and permanent restrictions
    """

    def __init__(self):
        self.capability_sets: Dict[str, CapabilitySet] = {}
        self.agent_capabilities: Dict[str, str] = {}  # agent_id -> capability_id
        self.restrictions: Dict[str, List[CapabilityRestriction]] = {}  # agent_id -> restrictions
        self.check_history: List[CapabilityCheckResult] = []
        self._setup_default_capability_sets()

    def _setup_default_capability_sets(self) -> None:
        """Setup default capability sets"""
        # Full capabilities
        self.create_capability_set(
            capability_id="full",
            name="Full Capabilities",
            level=CapabilityLevel.FULL,
            allowed_tools={'*'},
            allowed_actions={'*'}
        )

        # Standard capabilities
        self.create_capability_set(
            capability_id="standard",
            name="Standard Capabilities",
            level=CapabilityLevel.STANDARD,
            allowed_tools={
                'read_file', 'write_file', 'search', 'query_database',
                'run_analysis', 'generate_report'
            },
            allowed_actions={
                'read', 'write', 'analyze', 'query', 'report'
            },
            denied_tools={'execute_code', 'system_command', 'delete_database'}
        )

        # Limited capabilities
        self.create_capability_set(
            capability_id="limited",
            name="Limited Capabilities",
            level=CapabilityLevel.LIMITED,
            allowed_tools={
                'read_file', 'search', 'query_database'
            },
            allowed_actions={
                'read', 'search', 'query'
            }
        )

        # Minimal capabilities
        self.create_capability_set(
            capability_id="minimal",
            name="Minimal Capabilities",
            level=CapabilityLevel.MINIMAL,
            allowed_tools={'read_file'},
            allowed_actions={'read'}
        )

    def create_capability_set(
        self,
        capability_id: str,
        name: str,
        level: CapabilityLevel,
        allowed_tools: Set[str],
        allowed_actions: Set[str],
        denied_tools: Optional[Set[str]] = None,
        denied_actions: Optional[Set[str]] = None,
        restrictions: Optional[Dict[str, Any]] = None
    ) -> CapabilitySet:
        """Create a capability set"""
        capability_set = CapabilitySet(
            capability_id=capability_id,
            name=name,
            level=level,
            allowed_tools=allowed_tools,
            allowed_actions=allowed_actions,
            denied_tools=denied_tools or set(),
            denied_actions=denied_actions or set(),
            restrictions=restrictions or {}
        )

        self.capability_sets[capability_id] = capability_set
        return capability_set

    def assign_capabilities(self, agent_id: str, capability_id: str) -> bool:
        """Assign a capability set to an agent"""
        if capability_id not in self.capability_sets:
            return False

        self.agent_capabilities[agent_id] = capability_id
        return True

    def check_capability(
        self,
        agent_id: str,
        tool: str,
        action: str
    ) -> CapabilityCheckResult:
        """
        Check if agent has capability for tool/action.

        Args:
            agent_id: Agent to check
            tool: Tool to use
            action: Action to perform

        Returns:
            CapabilityCheckResult
        """
        # Get agent's capability set
        if agent_id not in self.agent_capabilities:
            result = CapabilityCheckResult(
                agent_id=agent_id,
                tool=tool,
                action=action,
                allowed=True,
                reason="No capability restrictions",
                restrictions=[]
            )
            self.check_history.append(result)
            return result

        capability_id = self.agent_capabilities[agent_id]
        capability_set = self.capability_sets[capability_id]

        restrictions = []

        # Check denied lists first
        if tool in capability_set.denied_tools:
            result = CapabilityCheckResult(
                agent_id=agent_id,
                tool=tool,
                action=action,
                allowed=False,
                reason=f"Tool '{tool}' is explicitly denied",
                restrictions=[f"Denied tool: {tool}"]
            )
            self.check_history.append(result)
            return result

        if action in capability_set.denied_actions:
            result = CapabilityCheckResult(
                agent_id=agent_id,
                tool=tool,
                action=action,
                allowed=False,
                reason=f"Action '{action}' is explicitly denied",
                restrictions=[f"Denied action: {action}"]
            )
            self.check_history.append(result)
            return result

        # Check allowed lists
        tool_allowed = (
            '*' in capability_set.allowed_tools or
            tool in capability_set.allowed_tools
        )

        action_allowed = (
            '*' in capability_set.allowed_actions or
            action in capability_set.allowed_actions
        )

        if not tool_allowed:
            restrictions.append(f"Tool '{tool}' not in allowed set")

        if not action_allowed:
            restrictions.append(f"Action '{action}' not in allowed set")

        # Check agent-specific restrictions
        agent_restrictions = self.restrictions.get(agent_id, [])
        for restriction in agent_restrictions:
            if restriction.tool == tool or restriction.tool == '*':
                restrictions.append(restriction.reason)

        allowed = tool_allowed and action_allowed and len(restrictions) == 0

        reason = "Allowed" if allowed else "; ".join(restrictions)

        result = CapabilityCheckResult(
            agent_id=agent_id,
            tool=tool,
            action=action,
            allowed=allowed,
            reason=reason,
            restrictions=restrictions
        )

        self.check_history.append(result)
        return result

    def restrict_tool(
        self,
        agent_id: str,
        tool: str,
        reason: str,
        permanent: bool = False
    ) -> CapabilityRestriction:
        """Add a tool restriction for an agent"""
        from uuid import uuid4

        restriction = CapabilityRestriction(
            restriction_id=f"restriction_{uuid4().hex[:12]}",
            agent_id=agent_id,
            tool=tool,
            reason=reason,
            permanent=permanent
        )

        if agent_id not in self.restrictions:
            self.restrictions[agent_id] = []

        self.restrictions[agent_id].append(restriction)
        return restriction

    def remove_restriction(self, agent_id: str, restriction_id: str) -> bool:
        """Remove a restriction"""
        if agent_id not in self.restrictions:
            return False

        for i, restriction in enumerate(self.restrictions[agent_id]):
            if restriction.restriction_id == restriction_id:
                if restriction.permanent:
                    return False  # Can't remove permanent restrictions

                del self.restrictions[agent_id][i]
                return True

        return False

    def clear_temporary_restrictions(self, agent_id: str) -> int:
        """Clear all temporary restrictions for an agent"""
        if agent_id not in self.restrictions:
            return 0

        original_count = len(self.restrictions[agent_id])
        self.restrictions[agent_id] = [
            r for r in self.restrictions[agent_id]
            if r.permanent
        ]

        return original_count - len(self.restrictions[agent_id])

    def escalate_restrictions(self, agent_id: str) -> bool:
        """
        Escalate restrictions by reducing capability level.

        Args:
            agent_id: Agent to restrict

        Returns:
            True if escalated successfully
        """
        if agent_id not in self.agent_capabilities:
            return False

        current_id = self.agent_capabilities[agent_id]
        current_set = self.capability_sets[current_id]

        # Escalation path: FULL -> STANDARD -> LIMITED -> MINIMAL
        escalation_map = {
            CapabilityLevel.FULL: "standard",
            CapabilityLevel.STANDARD: "limited",
            CapabilityLevel.LIMITED: "minimal",
            CapabilityLevel.MINIMAL: "minimal"  # Already at minimum
        }

        new_capability_id = escalation_map[current_set.level]
        self.agent_capabilities[agent_id] = new_capability_id

        return True

    def relax_restrictions(self, agent_id: str) -> bool:
        """
        Relax restrictions by increasing capability level.

        Args:
            agent_id: Agent to relax

        Returns:
            True if relaxed successfully
        """
        if agent_id not in self.agent_capabilities:
            return False

        current_id = self.agent_capabilities[agent_id]
        current_set = self.capability_sets[current_id]

        # Relaxation path: MINIMAL -> LIMITED -> STANDARD -> FULL
        relaxation_map = {
            CapabilityLevel.MINIMAL: "limited",
            CapabilityLevel.LIMITED: "standard",
            CapabilityLevel.STANDARD: "full",
            CapabilityLevel.FULL: "full"  # Already at maximum
        }

        new_capability_id = relaxation_map[current_set.level]
        self.agent_capabilities[agent_id] = new_capability_id

        return True

    def get_agent_capabilities(self, agent_id: str) -> Optional[CapabilitySet]:
        """Get the capability set for an agent"""
        if agent_id not in self.agent_capabilities:
            return None

        capability_id = self.agent_capabilities[agent_id]
        return self.capability_sets.get(capability_id)

    def get_agent_restrictions(self, agent_id: str) -> List[CapabilityRestriction]:
        """Get all restrictions for an agent"""
        return self.restrictions.get(agent_id, [])

    def get_statistics(self) -> Dict[str, Any]:
        """Get capability restriction statistics"""
        total_agents = len(self.agent_capabilities)
        total_restrictions = sum(len(r) for r in self.restrictions.values())
        total_checks = len(self.check_history)
        denied_checks = sum(1 for c in self.check_history if not c.allowed)

        # Count by capability level
        level_counts = {
            'full': 0,
            'standard': 0,
            'limited': 0,
            'minimal': 0
        }

        for capability_id in self.agent_capabilities.values():
            if capability_id in self.capability_sets:
                level = self.capability_sets[capability_id].level.value
                level_counts[level] += 1

        return {
            'total_agents': total_agents,
            'total_restrictions': total_restrictions,
            'total_capability_checks': total_checks,
            'denied_checks': denied_checks,
            'approval_rate': ((total_checks - denied_checks) / total_checks * 100) if total_checks > 0 else 100,
            'agents_by_capability_level': level_counts,
            'capability_sets': len(self.capability_sets)
        }
