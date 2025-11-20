"""
DeepAgent Safety Framework

Comprehensive security system implementing the 8 Foundations of Agentic AI Safety:
1. Action-Level Safety
2. Memory Firewalls
3. Verified Intent & Identity
4. Secure Tooling & Sandboxed Execution
5. Behavioral Rate-Limiters
6. Supervisory Meta-Agent
7. Immutable Audit Logs
8. Purpose-Bound Autonomy

This framework defends against:
- Prompt injection attacks
- Goal hijacking
- Multi-step attack chains
- Unauthorized tool execution
- Memory poisoning
- Self-modification exploits
"""

from .config import SafetyConfig, SafetyMode, ActionRiskLevel
from .exceptions import (
    SecurityViolationError,
    PromptInjectionDetectedError,
    UnauthorizedActionError,
    DomainViolationError,
    RiskThresholdExceededError
)

# Validation components
from .validation import (
    InputValidator,
    PromptInjectionDetector,
    ContentSanitizer
)

# Authorization components
from .authorization import (
    ActionClassifier,
    ActionPolicy,
    RiskScorer,
    ActionDecision,
    PolicyDecision,
    RiskScore
)

__all__ = [
    # Configuration
    "SafetyConfig",
    "SafetyMode",
    "ActionRiskLevel",

    # Exceptions
    "SecurityViolationError",
    "PromptInjectionDetectedError",
    "UnauthorizedActionError",
    "DomainViolationError",
    "RiskThresholdExceededError",

    # Validation
    "InputValidator",
    "PromptInjectionDetector",
    "ContentSanitizer",

    # Authorization
    "ActionClassifier",
    "ActionPolicy",
    "RiskScorer",
    "ActionDecision",
    "PolicyDecision",
    "RiskScore",
]

__version__ = "1.0.0"
