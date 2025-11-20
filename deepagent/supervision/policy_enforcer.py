"""
Policy Enforcer - Foundation #6

Enforces meta-level policies across multiple agents.
Ensures consistent policy application and detects violations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


class PolicyScope(Enum):
    """Scope of policy application"""
    GLOBAL = "global"          # Applies to all agents
    AGENT_TYPE = "agent_type"  # Applies to specific agent types
    AGENT = "agent"            # Applies to specific agent
    TASK = "task"              # Applies to specific tasks


class PolicySeverity(Enum):
    """Severity of policy violations"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetaPolicy:
    """Meta-level policy definition"""
    policy_id: str
    name: str
    description: str
    scope: PolicyScope
    target: Optional[str] = None  # Agent ID, type, or task name
    conditions: Dict[str, Any] = field(default_factory=dict)
    severity: PolicySeverity = PolicySeverity.WARNING
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyViolation:
    """Record of a policy violation"""
    violation_id: str
    policy_id: str
    agent_id: str
    violation_type: str
    description: str
    severity: PolicySeverity
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_note: Optional[str] = None


@dataclass
class PolicyCheckResult:
    """Result of policy check"""
    policy_id: str
    agent_id: str
    passed: bool
    violations: List[PolicyViolation]
    warnings: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class PolicyEnforcer:
    """
    Enforces meta-level policies across agents.

    Provides:
    - Policy definition and management
    - Multi-agent policy enforcement
    - Violation tracking and resolution
    - Policy consistency checking
    """

    def __init__(self):
        self.policies: Dict[str, MetaPolicy] = {}
        self.violations: Dict[str, PolicyViolation] = {}
        self.enforcement_history: List[PolicyCheckResult] = []
        self._policy_checkers: Dict[str, Callable] = {}

    def add_policy(self, policy: MetaPolicy) -> bool:
        """
        Add a meta-level policy.

        Args:
            policy: Policy to add

        Returns:
            True if added successfully
        """
        if policy.policy_id in self.policies:
            return False

        self.policies[policy.policy_id] = policy
        return True

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a policy"""
        if policy_id in self.policies:
            del self.policies[policy_id]
            return True
        return False

    def enable_policy(self, policy_id: str) -> bool:
        """Enable a policy"""
        if policy_id in self.policies:
            self.policies[policy_id].enabled = True
            return True
        return False

    def disable_policy(self, policy_id: str) -> bool:
        """Disable a policy"""
        if policy_id in self.policies:
            self.policies[policy_id].enabled = False
            return True
        return False

    def register_policy_checker(
        self,
        policy_type: str,
        checker: Callable[[Any, MetaPolicy], bool]
    ) -> None:
        """
        Register a custom policy checker function.

        Args:
            policy_type: Type of policy this checker handles
            checker: Function that checks policy compliance
        """
        self._policy_checkers[policy_type] = checker

    def check_agent_policies(
        self,
        agent_id: str,
        agent_type: str,
        current_state: Dict[str, Any]
    ) -> PolicyCheckResult:
        """
        Check all applicable policies for an agent.

        Args:
            agent_id: Agent to check
            agent_type: Type of agent
            current_state: Current agent state

        Returns:
            PolicyCheckResult with findings
        """
        violations = []
        warnings = []

        # Get applicable policies
        applicable_policies = self._get_applicable_policies(agent_id, agent_type)

        for policy in applicable_policies:
            if not policy.enabled:
                continue

            # Check policy compliance
            violation = self._check_policy_compliance(
                policy,
                agent_id,
                current_state
            )

            if violation:
                violations.append(violation)
                self.violations[violation.violation_id] = violation

                if violation.severity in [PolicySeverity.WARNING, PolicySeverity.INFO]:
                    warnings.append(violation.description)

        result = PolicyCheckResult(
            policy_id="all",
            agent_id=agent_id,
            passed=len(violations) == 0,
            violations=violations,
            warnings=warnings
        )

        self.enforcement_history.append(result)
        return result

    def _get_applicable_policies(
        self,
        agent_id: str,
        agent_type: str
    ) -> List[MetaPolicy]:
        """Get policies applicable to an agent"""
        applicable = []

        for policy in self.policies.values():
            if policy.scope == PolicyScope.GLOBAL:
                applicable.append(policy)
            elif policy.scope == PolicyScope.AGENT and policy.target == agent_id:
                applicable.append(policy)
            elif policy.scope == PolicyScope.AGENT_TYPE and policy.target == agent_type:
                applicable.append(policy)

        return applicable

    def _check_policy_compliance(
        self,
        policy: MetaPolicy,
        agent_id: str,
        state: Dict[str, Any]
    ) -> Optional[PolicyViolation]:
        """Check if agent complies with policy"""
        # Check built-in policy conditions
        for condition, expected_value in policy.conditions.items():
            actual_value = state.get(condition)

            if actual_value != expected_value:
                # Violation detected
                from uuid import uuid4
                violation = PolicyViolation(
                    violation_id=f"violation_{uuid4().hex[:12]}",
                    policy_id=policy.policy_id,
                    agent_id=agent_id,
                    violation_type=f"condition_mismatch_{condition}",
                    description=f"Policy '{policy.name}' violated: {condition} = {actual_value} (expected {expected_value})",
                    severity=policy.severity,
                    context={
                        'condition': condition,
                        'expected': expected_value,
                        'actual': actual_value,
                        'state': state
                    }
                )
                return violation

        # Check custom policy checkers
        policy_type = policy.metadata.get('type')
        if policy_type and policy_type in self._policy_checkers:
            checker = self._policy_checkers[policy_type]
            if not checker(state, policy):
                from uuid import uuid4
                violation = PolicyViolation(
                    violation_id=f"violation_{uuid4().hex[:12]}",
                    policy_id=policy.policy_id,
                    agent_id=agent_id,
                    violation_type=f"custom_check_{policy_type}",
                    description=f"Custom policy check failed for '{policy.name}'",
                    severity=policy.severity,
                    context={'state': state}
                )
                return violation

        return None

    def get_violations(
        self,
        agent_id: Optional[str] = None,
        policy_id: Optional[str] = None,
        severity: Optional[PolicySeverity] = None,
        resolved: Optional[bool] = None
    ) -> List[PolicyViolation]:
        """Get violations, optionally filtered"""
        violations = list(self.violations.values())

        if agent_id:
            violations = [v for v in violations if v.agent_id == agent_id]

        if policy_id:
            violations = [v for v in violations if v.policy_id == policy_id]

        if severity:
            violations = [v for v in violations if v.severity == severity]

        if resolved is not None:
            violations = [v for v in violations if v.resolved == resolved]

        return violations

    def resolve_violation(
        self,
        violation_id: str,
        resolution_note: str
    ) -> bool:
        """Mark a violation as resolved"""
        if violation_id not in self.violations:
            return False

        self.violations[violation_id].resolved = True
        self.violations[violation_id].resolution_note = resolution_note
        return True

    def get_policy(self, policy_id: str) -> Optional[MetaPolicy]:
        """Get a specific policy"""
        return self.policies.get(policy_id)

    def get_all_policies(
        self,
        scope: Optional[PolicyScope] = None,
        enabled_only: bool = False
    ) -> List[MetaPolicy]:
        """Get all policies, optionally filtered"""
        policies = list(self.policies.values())

        if scope:
            policies = [p for p in policies if p.scope == scope]

        if enabled_only:
            policies = [p for p in policies if p.enabled]

        return policies

    def get_statistics(self) -> Dict[str, Any]:
        """Get policy enforcement statistics"""
        total_policies = len(self.policies)
        enabled_policies = sum(1 for p in self.policies.values() if p.enabled)
        total_violations = len(self.violations)
        unresolved_violations = sum(1 for v in self.violations.values() if not v.resolved)

        # Count by severity
        severity_counts = {
            'info': 0,
            'warning': 0,
            'error': 0,
            'critical': 0
        }

        for violation in self.violations.values():
            if not violation.resolved:
                severity_counts[violation.severity.value] += 1

        # Count by scope
        scope_counts = {
            'global': 0,
            'agent_type': 0,
            'agent': 0,
            'task': 0
        }

        for policy in self.policies.values():
            if policy.enabled:
                scope_counts[policy.scope.value] += 1

        return {
            'total_policies': total_policies,
            'enabled_policies': enabled_policies,
            'disabled_policies': total_policies - enabled_policies,
            'total_violations': total_violations,
            'unresolved_violations': unresolved_violations,
            'resolved_violations': total_violations - unresolved_violations,
            'violations_by_severity': severity_counts,
            'policies_by_scope': scope_counts
        }

    def create_resource_limit_policy(
        self,
        policy_id: str,
        resource: str,
        max_usage: float,
        scope: PolicyScope = PolicyScope.GLOBAL,
        target: Optional[str] = None
    ) -> MetaPolicy:
        """
        Helper to create resource limit policy.

        Args:
            policy_id: Unique policy ID
            resource: Resource to limit (cpu, memory, etc.)
            max_usage: Maximum allowed usage (0.0-1.0)
            scope: Policy scope
            target: Target agent/type if applicable

        Returns:
            MetaPolicy for resource limiting
        """
        policy = MetaPolicy(
            policy_id=policy_id,
            name=f"{resource.upper()} Limit",
            description=f"Limit {resource} usage to {max_usage:.0%}",
            scope=scope,
            target=target,
            conditions={
                f'resource_usage.{resource}': max_usage
            },
            severity=PolicySeverity.ERROR,
            metadata={'type': 'resource_limit', 'resource': resource}
        )
        return policy

    def create_risk_threshold_policy(
        self,
        policy_id: str,
        max_risk_level: str,
        scope: PolicyScope = PolicyScope.GLOBAL,
        target: Optional[str] = None
    ) -> MetaPolicy:
        """
        Helper to create risk threshold policy.

        Args:
            policy_id: Unique policy ID
            max_risk_level: Maximum allowed risk (LOW, MEDIUM, HIGH)
            scope: Policy scope
            target: Target agent/type if applicable

        Returns:
            MetaPolicy for risk limiting
        """
        policy = MetaPolicy(
            policy_id=policy_id,
            name="Risk Threshold",
            description=f"Prevent agents from exceeding {max_risk_level} risk",
            scope=scope,
            target=target,
            conditions={
                'risk_level': max_risk_level
            },
            severity=PolicySeverity.CRITICAL,
            metadata={'type': 'risk_threshold'}
        )
        return policy
