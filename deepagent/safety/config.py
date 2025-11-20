"""
Safety configuration for DeepAgent

Defines security policies, thresholds, and operational modes.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional


class SafetyMode(Enum):
    """
    Safety operation modes with different security levels

    STRICT: Maximum security, blocks anything suspicious
    BALANCED: Production-ready with reasonable security/usability trade-off
    PERMISSIVE: Lower security for trusted environments
    RESEARCH: Minimal restrictions for research/development
    """
    STRICT = "strict"
    BALANCED = "balanced"
    PERMISSIVE = "permissive"
    RESEARCH = "research"


class ActionRiskLevel(Enum):
    """
    Action risk classification levels
    """
    SAFE = 0          # Read operations, queries
    LOW = 1           # Non-destructive writes
    MEDIUM = 2        # Data modifications
    HIGH = 3          # Code execution, API calls
    CRITICAL = 4      # System modifications, deployments


@dataclass
class InputValidationConfig:
    """Configuration for input validation"""
    enabled: bool = True
    max_input_length: int = 10000
    detect_injection: bool = True
    detect_jailbreak: bool = True
    sanitize_input: bool = True
    block_encoded_payloads: bool = True  # base64, unicode, etc.

    # Injection detection sensitivity (0.0 - 1.0)
    injection_threshold: float = 0.7


@dataclass
class ToolFirewallConfig:
    """Configuration for tool use firewall"""
    enabled: bool = True
    require_authorization: bool = True
    validate_parameters: bool = True
    rate_limit_per_minute: int = 100
    sandbox_execution: bool = True

    # Tool risk categories
    allowed_tool_categories: List[str] = field(default_factory=lambda: [
        "search", "read", "analyze", "query"
    ])

    restricted_tool_categories: List[str] = field(default_factory=lambda: [
        "write", "execute", "modify", "deploy"
    ])


@dataclass
class IntentVerificationConfig:
    """Configuration for intent verification"""
    enabled: bool = True
    alignment_threshold: float = 0.7  # Semantic similarity threshold
    detect_goal_drift: bool = True
    max_drift_distance: float = 0.5  # Maximum allowed semantic drift


@dataclass
class SupervisionConfig:
    """Configuration for supervisory meta-agent"""
    enabled: bool = True
    require_approval_for_high_risk: bool = True
    risk_threshold_for_approval: float = 0.8  # Actions above this need approval
    explain_actions: bool = True
    continuous_monitoring: bool = True


@dataclass
class AuditConfig:
    """Configuration for audit logging"""
    enabled: bool = True
    log_all_actions: bool = True
    log_security_events: bool = True
    log_reasoning_traces: bool = True
    retention_days: int = 90
    encrypt_logs: bool = True


@dataclass
class DomainConfig:
    """Configuration for purpose-bound autonomy"""
    enforce_domain_boundaries: bool = True

    # Allowed domains for DeepAgent
    allowed_domains: List[str] = field(default_factory=lambda: [
        "biology",
        "chemistry",
        "drug_discovery",
        "scientific_literature",
        "data_analysis",
        "protein_structure",
        "gene_editing",
        "molecular_simulation",
        "medical_research"
    ])

    # Explicitly forbidden domains
    forbidden_domains: List[str] = field(default_factory=lambda: [
        "cybersecurity",
        "penetration_testing",
        "exploit_development",
        "malware_analysis",
        "social_engineering",
        "surveillance",
        "weapons",
        "dual_use_research_of_concern"
    ])


@dataclass
class SafetyConfig:
    """
    Master configuration for DeepAgent safety framework

    Implements all 8 Foundations of Agentic AI Safety
    """

    # Overall safety mode
    mode: SafetyMode = SafetyMode.BALANCED

    # Foundation #1: Action-Level Safety
    action_level_safety: bool = True

    # Foundation #2: Memory Firewalls
    memory_firewall: bool = True

    # Foundation #3: Verified Intent & Identity
    verify_identity: bool = True

    # Foundation #4: Secure Tooling & Sandboxed Execution
    sandbox_execution: bool = True

    # Foundation #5: Behavioral Rate-Limiters
    behavioral_monitoring: bool = True

    # Foundation #6: Supervisory Meta-Agent
    supervisor_enabled: bool = True

    # Foundation #7: Immutable Audit Logs
    audit_logging: bool = True

    # Foundation #8: Purpose-Bound Autonomy
    domain_enforcement: bool = True

    # Sub-configurations
    input_validation: InputValidationConfig = field(default_factory=InputValidationConfig)
    tool_firewall: ToolFirewallConfig = field(default_factory=ToolFirewallConfig)
    intent_verification: IntentVerificationConfig = field(default_factory=IntentVerificationConfig)
    supervision: SupervisionConfig = field(default_factory=SupervisionConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)

    # Risk thresholds by mode
    risk_thresholds: Dict[str, float] = field(default_factory=lambda: {
        SafetyMode.STRICT.value: 0.3,      # Block anything above 0.3 risk
        SafetyMode.BALANCED.value: 0.7,    # Block anything above 0.7 risk
        SafetyMode.PERMISSIVE.value: 0.9,  # Block only critical risks
        SafetyMode.RESEARCH.value: 1.0     # Allow all (for research only)
    })

    @classmethod
    def create_strict(cls) -> "SafetyConfig":
        """Create strict security configuration"""
        config = cls(mode=SafetyMode.STRICT)
        config.input_validation.injection_threshold = 0.5  # More sensitive
        config.tool_firewall.rate_limit_per_minute = 50   # More restrictive
        config.supervision.risk_threshold_for_approval = 0.5  # Require approval more often
        return config

    @classmethod
    def create_research(cls) -> "SafetyConfig":
        """Create research configuration (minimal restrictions)"""
        config = cls(mode=SafetyMode.RESEARCH)
        config.input_validation.injection_threshold = 0.9  # Less sensitive
        config.tool_firewall.rate_limit_per_minute = 500  # More permissive
        config.supervision.require_approval_for_high_risk = False  # No approvals
        return config

    def get_risk_threshold(self) -> float:
        """Get risk threshold for current mode"""
        return self.risk_thresholds[self.mode.value]

    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            "mode": self.mode.value,
            "action_level_safety": self.action_level_safety,
            "memory_firewall": self.memory_firewall,
            "verify_identity": self.verify_identity,
            "sandbox_execution": self.sandbox_execution,
            "behavioral_monitoring": self.behavioral_monitoring,
            "supervisor_enabled": self.supervisor_enabled,
            "audit_logging": self.audit_logging,
            "domain_enforcement": self.domain_enforcement,
            "risk_threshold": self.get_risk_threshold()
        }
