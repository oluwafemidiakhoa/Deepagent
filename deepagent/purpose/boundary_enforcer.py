"""
Boundary Enforcer - Foundation #8

Enforces boundaries to prevent agents from exceeding their scope.
Detects and prevents boundary violations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4


class BoundaryType(Enum):
    """Type of boundary"""
    RESOURCE = "resource"        # Resource access boundary
    FUNCTIONAL = "functional"    # Functional capability boundary
    TEMPORAL = "temporal"        # Time-based boundary
    SPATIAL = "spatial"          # Location/domain boundary
    DATA = "data"                # Data access boundary


@dataclass
class Boundary:
    """Definition of an agent boundary"""
    boundary_id: str
    name: str
    boundary_type: BoundaryType
    limits: Dict[str, Any]
    description: str
    enforcement_level: str = "strict"  # "strict", "warn", "log"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BoundaryViolation:
    """Record of a boundary violation"""
    violation_id: str
    agent_id: str
    boundary_id: str
    violation_type: BoundaryType
    description: str
    attempted_action: Dict[str, Any]
    severity: str
    timestamp: datetime = field(default_factory=datetime.now)
    blocked: bool = True


@dataclass
class BoundaryCheckResult:
    """Result of boundary check"""
    agent_id: str
    allowed: bool
    violations: List[BoundaryViolation]
    warnings: List[str]
    applied_boundaries: List[str]


class BoundaryEnforcer:
    """
    Enforces boundaries for agent actions.

    Provides:
    - Boundary definition and enforcement
    - Multi-dimensional boundary checking
    - Violation detection and logging
    - Boundary compliance monitoring
    """

    def __init__(self):
        self.boundaries: Dict[str, Boundary] = {}
        self.agent_boundaries: Dict[str, List[str]] = {}  # agent_id -> boundary_ids
        self.violations: List[BoundaryViolation] = []

    def add_boundary(self, boundary: Boundary) -> bool:
        """Add a boundary definition"""
        if boundary.boundary_id in self.boundaries:
            return False

        self.boundaries[boundary.boundary_id] = boundary
        return True

    def remove_boundary(self, boundary_id: str) -> bool:
        """Remove a boundary definition"""
        if boundary_id in self.boundaries:
            del self.boundaries[boundary_id]
            return True
        return False

    def assign_boundary(self, agent_id: str, boundary_id: str) -> bool:
        """Assign a boundary to an agent"""
        if boundary_id not in self.boundaries:
            return False

        if agent_id not in self.agent_boundaries:
            self.agent_boundaries[agent_id] = []

        if boundary_id not in self.agent_boundaries[agent_id]:
            self.agent_boundaries[agent_id].append(boundary_id)

        return True

    def unassign_boundary(self, agent_id: str, boundary_id: str) -> bool:
        """Unassign a boundary from an agent"""
        if agent_id not in self.agent_boundaries:
            return False

        if boundary_id in self.agent_boundaries[agent_id]:
            self.agent_boundaries[agent_id].remove(boundary_id)
            return True

        return False

    def check_boundaries(
        self,
        agent_id: str,
        action: Dict[str, Any]
    ) -> BoundaryCheckResult:
        """
        Check if an action violates any boundaries.

        Args:
            agent_id: Agent performing action
            action: Action to check

        Returns:
            BoundaryCheckResult
        """
        if agent_id not in self.agent_boundaries:
            return BoundaryCheckResult(
                agent_id=agent_id,
                allowed=True,
                violations=[],
                warnings=[],
                applied_boundaries=[]
            )

        violations = []
        warnings = []
        applied = []

        for boundary_id in self.agent_boundaries[agent_id]:
            boundary = self.boundaries.get(boundary_id)
            if not boundary:
                continue

            applied.append(boundary_id)

            # Check boundary
            violation = self._check_boundary(agent_id, boundary, action)

            if violation:
                violations.append(violation)
                self.violations.append(violation)

                if boundary.enforcement_level == "warn":
                    warnings.append(violation.description)

        # Determine if action allowed
        strict_violations = [
            v for v in violations
            if self.boundaries[v.boundary_id].enforcement_level == "strict"
        ]

        allowed = len(strict_violations) == 0

        return BoundaryCheckResult(
            agent_id=agent_id,
            allowed=allowed,
            violations=violations,
            warnings=warnings,
            applied_boundaries=applied
        )

    def _check_boundary(
        self,
        agent_id: str,
        boundary: Boundary,
        action: Dict[str, Any]
    ) -> Optional[BoundaryViolation]:
        """Check a specific boundary"""
        if boundary.boundary_type == BoundaryType.RESOURCE:
            return self._check_resource_boundary(agent_id, boundary, action)
        elif boundary.boundary_type == BoundaryType.FUNCTIONAL:
            return self._check_functional_boundary(agent_id, boundary, action)
        elif boundary.boundary_type == BoundaryType.TEMPORAL:
            return self._check_temporal_boundary(agent_id, boundary, action)
        elif boundary.boundary_type == BoundaryType.SPATIAL:
            return self._check_spatial_boundary(agent_id, boundary, action)
        elif boundary.boundary_type == BoundaryType.DATA:
            return self._check_data_boundary(agent_id, boundary, action)

        return None

    def _check_resource_boundary(
        self,
        agent_id: str,
        boundary: Boundary,
        action: Dict[str, Any]
    ) -> Optional[BoundaryViolation]:
        """Check resource boundary"""
        resource = action.get('resource')

        if not resource:
            return None

        allowed_resources = boundary.limits.get('allowed_resources', [])

        if allowed_resources and resource not in allowed_resources:
            return BoundaryViolation(
                violation_id=f"violation_{uuid4().hex[:12]}",
                agent_id=agent_id,
                boundary_id=boundary.boundary_id,
                violation_type=BoundaryType.RESOURCE,
                description=f"Resource '{resource}' not allowed (boundary: {boundary.name})",
                attempted_action=action,
                severity="HIGH",
                blocked=boundary.enforcement_level == "strict"
            )

        return None

    def _check_functional_boundary(
        self,
        agent_id: str,
        boundary: Boundary,
        action: Dict[str, Any]
    ) -> Optional[BoundaryViolation]:
        """Check functional boundary"""
        function = action.get('function') or action.get('tool')

        if not function:
            return None

        allowed_functions = boundary.limits.get('allowed_functions', [])
        denied_functions = boundary.limits.get('denied_functions', [])

        if denied_functions and function in denied_functions:
            return BoundaryViolation(
                violation_id=f"violation_{uuid4().hex[:12]}",
                agent_id=agent_id,
                boundary_id=boundary.boundary_id,
                violation_type=BoundaryType.FUNCTIONAL,
                description=f"Function '{function}' is denied (boundary: {boundary.name})",
                attempted_action=action,
                severity="HIGH",
                blocked=boundary.enforcement_level == "strict"
            )

        if allowed_functions and function not in allowed_functions:
            return BoundaryViolation(
                violation_id=f"violation_{uuid4().hex[:12]}",
                agent_id=agent_id,
                boundary_id=boundary.boundary_id,
                violation_type=BoundaryType.FUNCTIONAL,
                description=f"Function '{function}' not allowed (boundary: {boundary.name})",
                attempted_action=action,
                severity="MEDIUM",
                blocked=boundary.enforcement_level == "strict"
            )

        return None

    def _check_temporal_boundary(
        self,
        agent_id: str,
        boundary: Boundary,
        action: Dict[str, Any]
    ) -> Optional[BoundaryViolation]:
        """Check temporal boundary"""
        now = datetime.now()

        start_time = boundary.limits.get('start_time')
        end_time = boundary.limits.get('end_time')

        if start_time and now < start_time:
            return BoundaryViolation(
                violation_id=f"violation_{uuid4().hex[:12]}",
                agent_id=agent_id,
                boundary_id=boundary.boundary_id,
                violation_type=BoundaryType.TEMPORAL,
                description=f"Action attempted before allowed time (boundary: {boundary.name})",
                attempted_action=action,
                severity="MEDIUM",
                blocked=boundary.enforcement_level == "strict"
            )

        if end_time and now > end_time:
            return BoundaryViolation(
                violation_id=f"violation_{uuid4().hex[:12]}",
                agent_id=agent_id,
                boundary_id=boundary.boundary_id,
                violation_type=BoundaryType.TEMPORAL,
                description=f"Action attempted after allowed time (boundary: {boundary.name})",
                attempted_action=action,
                severity="MEDIUM",
                blocked=boundary.enforcement_level == "strict"
            )

        return None

    def _check_spatial_boundary(
        self,
        agent_id: str,
        boundary: Boundary,
        action: Dict[str, Any]
    ) -> Optional[BoundaryViolation]:
        """Check spatial/domain boundary"""
        domain = action.get('domain') or action.get('location')

        if not domain:
            return None

        allowed_domains = boundary.limits.get('allowed_domains', [])

        if allowed_domains and domain not in allowed_domains:
            return BoundaryViolation(
                violation_id=f"violation_{uuid4().hex[:12]}",
                agent_id=agent_id,
                boundary_id=boundary.boundary_id,
                violation_type=BoundaryType.SPATIAL,
                description=f"Domain '{domain}' not allowed (boundary: {boundary.name})",
                attempted_action=action,
                severity="HIGH",
                blocked=boundary.enforcement_level == "strict"
            )

        return None

    def _check_data_boundary(
        self,
        agent_id: str,
        boundary: Boundary,
        action: Dict[str, Any]
    ) -> Optional[BoundaryViolation]:
        """Check data access boundary"""
        data_source = action.get('data_source')
        data_type = action.get('data_type')

        allowed_sources = boundary.limits.get('allowed_data_sources', [])
        allowed_types = boundary.limits.get('allowed_data_types', [])

        if data_source and allowed_sources and data_source not in allowed_sources:
            return BoundaryViolation(
                violation_id=f"violation_{uuid4().hex[:12]}",
                agent_id=agent_id,
                boundary_id=boundary.boundary_id,
                violation_type=BoundaryType.DATA,
                description=f"Data source '{data_source}' not allowed (boundary: {boundary.name})",
                attempted_action=action,
                severity="HIGH",
                blocked=boundary.enforcement_level == "strict"
            )

        if data_type and allowed_types and data_type not in allowed_types:
            return BoundaryViolation(
                violation_id=f"violation_{uuid4().hex[:12]}",
                agent_id=agent_id,
                boundary_id=boundary.boundary_id,
                violation_type=BoundaryType.DATA,
                description=f"Data type '{data_type}' not allowed (boundary: {boundary.name})",
                attempted_action=action,
                severity="MEDIUM",
                blocked=boundary.enforcement_level == "strict"
            )

        return None

    def get_agent_boundaries(self, agent_id: str) -> List[Boundary]:
        """Get all boundaries assigned to an agent"""
        if agent_id not in self.agent_boundaries:
            return []

        return [
            self.boundaries[bid]
            for bid in self.agent_boundaries[agent_id]
            if bid in self.boundaries
        ]

    def get_violations(
        self,
        agent_id: Optional[str] = None,
        boundary_type: Optional[BoundaryType] = None,
        severity: Optional[str] = None
    ) -> List[BoundaryViolation]:
        """Get violations, optionally filtered"""
        violations = self.violations

        if agent_id:
            violations = [v for v in violations if v.agent_id == agent_id]

        if boundary_type:
            violations = [v for v in violations if v.violation_type == boundary_type]

        if severity:
            violations = [v for v in violations if v.severity == severity]

        return violations

    def get_statistics(self) -> Dict[str, Any]:
        """Get boundary enforcement statistics"""
        total_boundaries = len(self.boundaries)
        total_violations = len(self.violations)
        blocked_violations = sum(1 for v in self.violations if v.blocked)

        # Count by type
        type_counts = {t.value: 0 for t in BoundaryType}
        for boundary in self.boundaries.values():
            type_counts[boundary.boundary_type.value] += 1

        # Count violations by type
        violation_counts = {t.value: 0 for t in BoundaryType}
        for violation in self.violations:
            violation_counts[violation.violation_type.value] += 1

        return {
            'total_boundaries': total_boundaries,
            'total_violations': total_violations,
            'blocked_violations': blocked_violations,
            'allowed_violations': total_violations - blocked_violations,
            'boundaries_by_type': type_counts,
            'violations_by_type': violation_counts,
            'agents_with_boundaries': len(self.agent_boundaries)
        }
