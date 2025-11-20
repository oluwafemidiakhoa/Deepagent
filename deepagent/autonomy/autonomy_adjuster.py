"""
Autonomy Adjuster - Foundation #11

Dynamically adjusts agent autonomy based on risk.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List


class AutonomyLevel(Enum):
    """Levels of agent autonomy"""
    FULL = "full"              # No restrictions
    SUPERVISED = "supervised"  # Monitoring only
    RESTRICTED = "restricted"  # Limited capabilities
    MINIMAL = "minimal"        # Approval required


@dataclass
class AutonomyAdjustment:
    """Result of autonomy adjustment"""
    new_level: AutonomyLevel
    previous_level: AutonomyLevel
    reason: str
    restrictions: List[str]


class AutonomyAdjuster:
    """Adjusts autonomy based on risk"""

    def __init__(self):
        self.current_level = AutonomyLevel.FULL

    def adjust(self, risk_assessment) -> AutonomyAdjustment:
        """Adjust autonomy based on risk"""
        previous = self.current_level

        # Determine new level based on risk
        if risk_assessment.risk_level.value == "critical":
            new_level = AutonomyLevel.MINIMAL
            restrictions = [
                "All actions require approval",
                "Sandboxing mandatory",
                "Limited tool access"
            ]

        elif risk_assessment.risk_level.value == "high":
            new_level = AutonomyLevel.RESTRICTED
            restrictions = [
                "High-risk actions require approval",
                "Destructive operations blocked",
                "Enhanced monitoring"
            ]

        elif risk_assessment.risk_level.value == "medium":
            new_level = AutonomyLevel.SUPERVISED
            restrictions = [
                "Increased logging",
                "Approval for critical operations"
            ]

        else:  # low
            new_level = AutonomyLevel.FULL
            restrictions = []

        self.current_level = new_level

        return AutonomyAdjustment(
            new_level=new_level,
            previous_level=previous,
            reason=risk_assessment.recommendation,
            restrictions=restrictions
        )
