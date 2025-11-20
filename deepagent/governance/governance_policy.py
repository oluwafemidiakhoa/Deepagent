"""
Governance Policy - Foundation #12

Organizational policies and escalation rules.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class EscalationLevel(Enum):
    """Escalation levels"""
    NONE = "none"
    SUPERVISOR = "supervisor"
    MANAGER = "manager"
    EXECUTIVE = "executive"


@dataclass
class PolicyRule:
    """Governance policy rule"""
    rule_id: str
    condition: str
    action: str
    escalation_level: EscalationLevel
    description: str


class GovernancePolicy:
    """Manages governance policies"""

    def __init__(self):
        self.rules: List[PolicyRule] = []
        self._initialize_default_rules()

    def _initialize_default_rules(self):
        """Initialize default governance rules"""
        self.rules = [
            PolicyRule(
                rule_id="critical_risk",
                condition="risk_score >= 0.9",
                action="require_executive_approval",
                escalation_level=EscalationLevel.EXECUTIVE,
                description="Critical risk requires executive approval"
            ),
            PolicyRule(
                rule_id="attack_detected",
                condition="attack_detected == True",
                action="block_and_escalate",
                escalation_level=EscalationLevel.MANAGER,
                description="Detected attacks escalate to manager"
            ),
            PolicyRule(
                rule_id="high_value_data",
                condition="data_sensitivity == 'high'",
                action="require_supervisor_approval",
                escalation_level=EscalationLevel.SUPERVISOR,
                description="High-value data requires supervisor approval"
            )
        ]

    def evaluate(
        self,
        risk_score: float,
        attack_detected: bool,
        data_sensitivity: str = "low"
    ) -> Optional[PolicyRule]:
        """Evaluate which policy rule applies"""

        # Simple evaluation (in production, use proper rule engine)
        if risk_score >= 0.9:
            return self.rules[0]  # Critical risk

        if attack_detected:
            return self.rules[1]  # Attack detected

        if data_sensitivity == "high":
            return self.rules[2]  # High-value data

        return None  # No policy applies

    def add_rule(self, rule: PolicyRule):
        """Add custom policy rule"""
        self.rules.append(rule)

    def get_escalation_level(
        self,
        risk_score: float,
        attack_detected: bool
    ) -> EscalationLevel:
        """Get required escalation level"""
        rule = self.evaluate(risk_score, attack_detected)

        if rule:
            return rule.escalation_level

        return EscalationLevel.NONE
