"""
Foundation #8: Purpose-Bound Agents

Ensures agents stay within their defined purpose and task boundaries.
"""

from .purpose_binder import PurposeBinder, Purpose, PurposeScope
from .boundary_enforcer import BoundaryEnforcer, Boundary, BoundaryViolation
from .capability_limiter import CapabilityLimiter, CapabilitySet, CapabilityRestriction

__all__ = [
    'PurposeBinder',
    'Purpose',
    'PurposeScope',
    'BoundaryEnforcer',
    'Boundary',
    'BoundaryViolation',
    'CapabilityLimiter',
    'CapabilitySet',
    'CapabilityRestriction',
]
