"""
Anomaly Detector - Foundation #5

Detects deviations from normal behavior baseline.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class Anomaly:
    """Detected behavioral anomaly"""
    anomaly_id: str
    anomaly_type: str
    severity: float  # 0.0 - 1.0
    description: str
    detected_at: datetime
    evidence: dict


@dataclass
class AnomalyScore:
    """Anomaly detection result"""
    is_anomalous: bool
    score: float  # 0.0 - 1.0
    anomalies: List[Anomaly]


class AnomalyDetector:
    """Detects behavioral anomalies"""

    def __init__(self, baseline):
        self.baseline = baseline
        self.threshold = 0.7

    def detect(
        self,
        action_count: int,
        tools_used: List[str],
        avg_risk: float
    ) -> AnomalyScore:
        """Detect anomalies in current session"""
        profile = self.baseline.get_profile()
        anomalies = []

        # Check action count deviation
        if profile.avg_actions_per_session > 0:
            deviation = abs(action_count - profile.avg_actions_per_session) / profile.avg_actions_per_session

            if deviation > 2.0:  # >200% deviation
                anomalies.append(Anomaly(
                    anomaly_id=f"anom_{datetime.now().timestamp()}",
                    anomaly_type="action_count",
                    severity=min(deviation / 3.0, 1.0),
                    description=f"Unusual action count: {action_count} vs typical {profile.avg_actions_per_session:.1f}",
                    detected_at=datetime.now(),
                    evidence={"actual": action_count, "expected": profile.avg_actions_per_session}
                ))

        # Check unusual tools
        for tool in tools_used:
            if tool not in profile.common_tools:
                anomalies.append(Anomaly(
                    anomaly_id=f"anom_{datetime.now().timestamp()}",
                    anomaly_type="unusual_tool",
                    severity=0.5,
                    description=f"Unusual tool used: {tool}",
                    detected_at=datetime.now(),
                    evidence={"tool": tool}
                ))

        # Check risk score deviation
        if profile.avg_risk_score > 0 and avg_risk > profile.avg_risk_score * 2:
            anomalies.append(Anomaly(
                anomaly_id=f"anom_{datetime.now().timestamp()}",
                anomaly_type="high_risk",
                severity=0.8,
                description=f"Elevated risk: {avg_risk:.2f} vs typical {profile.avg_risk_score:.2f}",
                detected_at=datetime.now(),
                evidence={"actual": avg_risk, "expected": profile.avg_risk_score}
            ))

        # Calculate overall score
        if anomalies:
            score = max(a.severity for a in anomalies)
        else:
            score = 0.0

        return AnomalyScore(
            is_anomalous=score >= self.threshold,
            score=score,
            anomalies=anomalies
        )
