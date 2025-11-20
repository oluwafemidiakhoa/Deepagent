"""
Risk Scoring System

Comprehensive risk calculation for actions considering:
- Action metadata (risk level, reversibility)
- Parameters (dangerous values)
- Context (user, history, timing)
- Historical patterns

Part of Foundation #1: Action-Level Safety
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from .action_classifier import ActionMetadata, ActionRiskLevel


@dataclass
class RiskScore:
    """
    Comprehensive risk assessment for an action
    """
    # Overall risk score (0.0 - 1.0)
    total_score: float

    # Component scores
    base_risk: float              # From action classification
    parameter_risk: float         # From parameter analysis
    context_risk: float           # From user/environment context
    historical_risk: float        # From usage patterns
    timing_risk: float            # From timing/frequency

    # Risk level classification
    risk_level: ActionRiskLevel

    # Decision support
    requires_approval: bool
    requires_explanation: bool
    can_proceed: bool

    # Evidence
    risk_factors: List[str]
    mitigation_suggestions: List[str]

    # Metadata
    confidence: float             # Confidence in assessment (0.0 - 1.0)
    timestamp: datetime


class RiskScorer:
    """
    Calculates comprehensive risk scores for actions

    Considers multiple factors:
    1. Base action risk (from classifier)
    2. Parameter risk (dangerous values)
    3. Context risk (who, when, why)
    4. Historical patterns (frequency, anomalies)
    5. Timing risk (off-hours, rapid succession)
    """

    def __init__(self, risk_threshold: float = 0.7):
        """
        Initialize risk scorer

        Args:
            risk_threshold: Threshold for blocking actions (0.0 - 1.0)
        """
        self.risk_threshold = risk_threshold

        # Historical action tracking (simple in-memory for now)
        self.action_history: List[Dict[str, Any]] = []

        # Dangerous parameter patterns
        self.dangerous_patterns = [
            # SQL injection patterns
            r"(?i)(DROP|DELETE|TRUNCATE)\s+(TABLE|DATABASE)",
            r"(?i);\s*DROP",
            r"(?i)--\s*$",  # SQL comment
            r"(?i)UNION\s+SELECT",

            # Command injection patterns
            r"(?i)(;|\||&)\s*(rm|del|format|shutdown)",
            r"(?i)`.*`",    # Command substitution
            r"(?i)\$\(.*\)", # Command substitution

            # Path traversal
            r"\.\./",
            r"\.\.\\",

            # Wildcard abuse
            r"\*\s*$",      # Delete all
            r"/\*",         # Root wildcard

            # Admin/privileged access
            r"(?i)(sudo|admin|root|system)",
        ]

    def score_action(
        self,
        action_metadata: ActionMetadata,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RiskScore:
        """
        Calculate comprehensive risk score

        Args:
            action_metadata: Action classification metadata
            parameters: Action parameters
            context: Execution context (user, task, etc.)

        Returns:
            RiskScore with detailed risk assessment
        """
        context = context or {}
        risk_factors = []
        mitigation_suggestions = []

        # 1. Base risk from action classification
        base_risk = self._calculate_base_risk(action_metadata, risk_factors)

        # 2. Parameter risk
        parameter_risk = self._calculate_parameter_risk(
            parameters,
            action_metadata,
            risk_factors,
            mitigation_suggestions
        )

        # 3. Context risk
        context_risk = self._calculate_context_risk(
            context,
            action_metadata,
            risk_factors
        )

        # 4. Historical risk
        historical_risk = self._calculate_historical_risk(
            action_metadata.tool_name,
            parameters,
            risk_factors
        )

        # 5. Timing risk
        timing_risk = self._calculate_timing_risk(
            action_metadata.tool_name,
            risk_factors
        )

        # Weighted combination
        total_score = (
            base_risk * 0.40 +           # Base action risk is most important
            parameter_risk * 0.25 +      # Parameters can escalate risk
            context_risk * 0.15 +        # Context matters
            historical_risk * 0.10 +     # Patterns over time
            timing_risk * 0.10           # Timing anomalies
        )

        # Ensure total is in range
        total_score = max(0.0, min(1.0, total_score))

        # Determine if action can proceed
        can_proceed = total_score < self.risk_threshold

        # Determine if approval is needed (even if below threshold)
        requires_approval = (
            action_metadata.requires_approval or
            action_metadata.risk_level.value >= ActionRiskLevel.HIGH.value or
            total_score >= 0.5
        )

        # Determine if explanation is needed
        requires_explanation = (
            total_score >= 0.3 or
            len(risk_factors) > 2
        )

        # Confidence in assessment
        confidence = self._calculate_confidence(
            action_metadata,
            parameters,
            context
        )

        # Record in history
        self._record_action(action_metadata.tool_name, parameters, total_score)

        return RiskScore(
            total_score=total_score,
            base_risk=base_risk,
            parameter_risk=parameter_risk,
            context_risk=context_risk,
            historical_risk=historical_risk,
            timing_risk=timing_risk,
            risk_level=action_metadata.risk_level,
            requires_approval=requires_approval,
            requires_explanation=requires_explanation,
            can_proceed=can_proceed,
            risk_factors=risk_factors,
            mitigation_suggestions=mitigation_suggestions,
            confidence=confidence,
            timestamp=datetime.now()
        )

    def _calculate_base_risk(
        self,
        metadata: ActionMetadata,
        risk_factors: List[str]
    ) -> float:
        """Calculate base risk from action metadata"""
        # Map risk level to score
        risk_mapping = {
            ActionRiskLevel.SAFE: 0.0,
            ActionRiskLevel.LOW: 0.2,
            ActionRiskLevel.MEDIUM: 0.5,
            ActionRiskLevel.HIGH: 0.8,
            ActionRiskLevel.CRITICAL: 1.0
        }

        base_score = risk_mapping[metadata.risk_level]

        # Add to risk factors
        if base_score > 0.5:
            risk_factors.append(f"High base risk: {metadata.risk_level.name}")

        if not metadata.reversible:
            risk_factors.append("Action is irreversible")
            base_score = min(1.0, base_score + 0.1)

        if metadata.side_effects:
            risk_factors.append(f"Side effects: {', '.join(metadata.side_effects[:3])}")

        return base_score

    def _calculate_parameter_risk(
        self,
        parameters: Dict[str, Any],
        metadata: ActionMetadata,
        risk_factors: List[str],
        mitigation_suggestions: List[str]
    ) -> float:
        """Analyze parameters for dangerous values"""
        if not parameters:
            return 0.0

        param_risk = 0.0
        param_str = str(parameters)

        # Check for dangerous patterns
        import re
        for pattern in self.dangerous_patterns:
            if re.search(pattern, param_str):
                param_risk = max(param_risk, 0.8)
                risk_factors.append(f"Dangerous parameter pattern detected")
                mitigation_suggestions.append("Review parameter values carefully")
                break

        # Check for large scope operations
        scope_keywords = ['all', '*', 'everything', 'global', 'system']
        if any(keyword in param_str.lower() for keyword in scope_keywords):
            param_risk = max(param_risk, 0.6)
            risk_factors.append("Broad scope operation detected")
            mitigation_suggestions.append("Consider limiting operation scope")

        # Check for sensitive data targets
        sensitive_keywords = ['password', 'secret', 'key', 'token', 'credential', 'user', 'admin']
        if any(keyword in param_str.lower() for keyword in sensitive_keywords):
            param_risk = max(param_risk, 0.5)
            risk_factors.append("Operation targets sensitive data")
            mitigation_suggestions.append("Verify authorization for sensitive data access")

        # Check parameter count (too many can be suspicious)
        if len(parameters) > 10:
            param_risk = max(param_risk, 0.3)
            risk_factors.append("Unusually high parameter count")

        return param_risk

    def _calculate_context_risk(
        self,
        context: Dict[str, Any],
        metadata: ActionMetadata,
        risk_factors: List[str]
    ) -> float:
        """Evaluate risk from execution context"""
        context_risk = 0.0

        # Check for missing context (suspicious)
        if not context:
            context_risk = 0.3
            risk_factors.append("Missing execution context")
            return context_risk

        # User context
        user_role = context.get('user_role', 'unknown')
        if user_role == 'unknown':
            context_risk = max(context_risk, 0.4)
            risk_factors.append("Unknown user role")
        elif user_role == 'guest':
            context_risk = max(context_risk, 0.6)
            risk_factors.append("Guest user attempting action")

        # Task alignment
        original_task = context.get('original_task', '')
        if original_task and metadata.tool_name not in original_task.lower():
            # Tool doesn't obviously relate to task
            context_risk = max(context_risk, 0.3)
            risk_factors.append("Tool selection may not align with task")

        # Environment
        environment = context.get('environment', 'unknown')
        if environment == 'production' and metadata.risk_level.value >= ActionRiskLevel.HIGH.value:
            context_risk = max(context_risk, 0.5)
            risk_factors.append("High-risk action in production environment")

        return context_risk

    def _calculate_historical_risk(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        risk_factors: List[str]
    ) -> float:
        """Analyze historical usage patterns"""
        historical_risk = 0.0

        if not self.action_history:
            return 0.0

        # Get recent history for this tool
        recent_actions = [
            action for action in self.action_history[-100:]  # Last 100 actions
            if action['tool_name'] == tool_name
        ]

        if not recent_actions:
            return 0.0

        # Check frequency (too many recent uses = suspicious)
        recent_time_window = 60  # seconds
        now = datetime.now()
        recent_count = sum(
            1 for action in recent_actions
            if (now - action['timestamp']).total_seconds() < recent_time_window
        )

        if recent_count > 10:
            historical_risk = max(historical_risk, 0.7)
            risk_factors.append(f"Unusual frequency: {recent_count} calls in {recent_time_window}s")
        elif recent_count > 5:
            historical_risk = max(historical_risk, 0.4)
            risk_factors.append(f"Elevated frequency: {recent_count} calls in {recent_time_window}s")

        # Check for pattern anomalies
        avg_risk = sum(action['risk_score'] for action in recent_actions) / len(recent_actions)
        if avg_risk > 0.6:
            historical_risk = max(historical_risk, 0.5)
            risk_factors.append("Tool has history of high-risk usage")

        return historical_risk

    def _calculate_timing_risk(
        self,
        tool_name: str,
        risk_factors: List[str]
    ) -> float:
        """Evaluate timing-based risk"""
        timing_risk = 0.0
        now = datetime.now()

        # Off-hours detection (simple version)
        hour = now.hour
        if hour < 6 or hour > 22:  # Before 6 AM or after 10 PM
            timing_risk = 0.2
            risk_factors.append("Action requested during off-hours")

        # Weekend detection
        if now.weekday() >= 5:  # Saturday or Sunday
            timing_risk = max(timing_risk, 0.1)
            risk_factors.append("Action requested on weekend")

        return timing_risk

    def _calculate_confidence(
        self,
        metadata: ActionMetadata,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """Calculate confidence in risk assessment"""
        confidence = 1.0

        # Lower confidence for unknown tools
        if "unknown_tool" in metadata.side_effects:
            confidence -= 0.3

        # Lower confidence with missing context
        if not context:
            confidence -= 0.2

        # Lower confidence with complex parameters
        if parameters and len(str(parameters)) > 500:
            confidence -= 0.1

        return max(0.0, min(1.0, confidence))

    def _record_action(self, tool_name: str, parameters: Dict[str, Any], risk_score: float):
        """Record action in history for pattern analysis"""
        self.action_history.append({
            'tool_name': tool_name,
            'parameters': parameters,
            'risk_score': risk_score,
            'timestamp': datetime.now()
        })

        # Keep history bounded
        if len(self.action_history) > 1000:
            self.action_history = self.action_history[-500:]

    def explain_risk(self, risk_score: RiskScore) -> str:
        """
        Generate human-readable risk explanation

        Args:
            risk_score: Risk assessment to explain

        Returns:
            Explanation string
        """
        lines = []
        lines.append(f"Risk Assessment for Action")
        lines.append("=" * 60)
        lines.append(f"Overall Risk Score: {risk_score.total_score:.2%} ({risk_score.risk_level.name})")
        lines.append(f"Confidence: {risk_score.confidence:.2%}")
        lines.append("")

        lines.append("Risk Breakdown:")
        lines.append(f"  Base Risk:       {risk_score.base_risk:.2%}")
        lines.append(f"  Parameter Risk:  {risk_score.parameter_risk:.2%}")
        lines.append(f"  Context Risk:    {risk_score.context_risk:.2%}")
        lines.append(f"  Historical Risk: {risk_score.historical_risk:.2%}")
        lines.append(f"  Timing Risk:     {risk_score.timing_risk:.2%}")
        lines.append("")

        if risk_score.risk_factors:
            lines.append("Risk Factors:")
            for factor in risk_score.risk_factors:
                lines.append(f"  - {factor}")
            lines.append("")

        if risk_score.mitigation_suggestions:
            lines.append("Mitigation Suggestions:")
            for suggestion in risk_score.mitigation_suggestions:
                lines.append(f"  - {suggestion}")
            lines.append("")

        lines.append("Decision:")
        if not risk_score.can_proceed:
            lines.append(f"  [BLOCKED] Risk exceeds threshold ({self.risk_threshold:.0%})")
        elif risk_score.requires_approval:
            lines.append("  [APPROVAL REQUIRED] Human review needed")
        else:
            lines.append("  [ALLOWED] Action can proceed")

        return "\n".join(lines)
