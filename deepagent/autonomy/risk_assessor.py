"""
Risk Assessor - Foundation #11

Real-time risk assessment for adaptive autonomy.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List


class RiskLevel(Enum):
    """Risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskAssessment:
    """Risk assessment result"""
    risk_level: RiskLevel
    risk_score: float  # 0.0 - 1.0
    factors: List[str]
    recommendation: str


class RiskAssessor:
    """Assesses real-time risk"""

    def assess(
        self,
        action_type: str,
        risk_score: float,
        recent_violations: int,
        attack_detected: bool
    ) -> RiskAssessment:
        """Assess current risk level"""

        factors = []

        # Base risk from Phase 1
        if risk_score > 0.8:
            factors.append("High Phase 1 risk score")

        # Recent security violations
        if recent_violations > 2:
            factors.append(f"{recent_violations} recent violations")
            risk_score = min(1.0, risk_score + 0.2)

        # Active attack
        if attack_detected:
            factors.append("Active attack detected")
            risk_score = min(1.0, risk_score + 0.3)

        # Destructive action
        if action_type in ["delete", "modify", "execute"]:
            factors.append(f"Destructive action: {action_type}")
            risk_score = min(1.0, risk_score + 0.1)

        # Determine level
        if risk_score >= 0.8:
            level = RiskLevel.CRITICAL
            recommendation = "Minimize autonomy, require approval"
        elif risk_score >= 0.6:
            level = RiskLevel.HIGH
            recommendation = "Reduce autonomy, increase monitoring"
        elif risk_score >= 0.4:
            level = RiskLevel.MEDIUM
            recommendation = "Moderate autonomy with oversight"
        else:
            level = RiskLevel.LOW
            recommendation = "Full autonomy permitted"

        return RiskAssessment(
            risk_level=level,
            risk_score=risk_score,
            factors=factors,
            recommendation=recommendation
        )
