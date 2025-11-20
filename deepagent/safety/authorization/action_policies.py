"""
Action Policy System

Enforces security policies and makes authorization decisions.

Part of Foundation #1: Action-Level Safety
"""

from enum import Enum
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

from .action_classifier import ActionClassifier, ActionMetadata, ActionRiskLevel
from .risk_scorer import RiskScorer, RiskScore
from ..exceptions import (
    UnauthorizedActionError,
    RiskThresholdExceededError
)


class ActionDecision(Enum):
    """Possible policy decisions for an action"""
    ALLOW = "allow"                          # Action is safe, proceed
    ALLOW_WITH_LOGGING = "allow_with_logging"  # Allow but log extensively
    REQUIRE_APPROVAL = "require_approval"    # Needs human approval
    BLOCK = "block"                          # Action is blocked
    BLOCK_AND_ALERT = "block_and_alert"      # Block and send security alert


@dataclass
class PolicyDecision:
    """
    Policy decision for an action
    """
    decision: ActionDecision
    action_metadata: ActionMetadata
    risk_score: RiskScore
    reason: str
    requires_user_approval: bool
    requires_explanation: bool
    can_proceed: bool
    timestamp: datetime

    # For approval workflow
    approval_message: Optional[str] = None
    approval_timeout_seconds: int = 300  # 5 minutes default


class ActionPolicy:
    """
    Enforces security policies for actions

    Decision flow:
    1. Classify action (what is it?)
    2. Score risk (how dangerous?)
    3. Apply policy (what should we do?)
    4. Make decision (allow/block/approve)
    """

    def __init__(
        self,
        classifier: Optional[ActionClassifier] = None,
        scorer: Optional[RiskScorer] = None,
        risk_threshold: float = 0.7,
        enable_approval_workflow: bool = True
    ):
        """
        Initialize action policy enforcer

        Args:
            classifier: Action classifier (creates default if None)
            scorer: Risk scorer (creates default if None)
            risk_threshold: Maximum acceptable risk (0.0 - 1.0)
            enable_approval_workflow: Enable human approval for high-risk actions
        """
        self.classifier = classifier or ActionClassifier()
        self.scorer = scorer or RiskScorer(risk_threshold=risk_threshold)
        self.risk_threshold = risk_threshold
        self.enable_approval_workflow = enable_approval_workflow

        # Approval callback (can be set by application)
        self.approval_callback: Optional[Callable[[PolicyDecision], bool]] = None

        # Policy violation logging
        self.violations_log = []

    def evaluate_action(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> PolicyDecision:
        """
        Evaluate an action and make policy decision

        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters
            context: Execution context

        Returns:
            PolicyDecision with authorization decision

        Raises:
            UnauthorizedActionError: If action is not authorized
            RiskThresholdExceededError: If risk exceeds threshold
        """
        # 1. Classify the action
        action_metadata = self.classifier.classify_action(tool_name, parameters)

        # 2. Score the risk
        risk_score = self.scorer.score_action(action_metadata, parameters, context)

        # 3. Make policy decision
        decision = self._make_decision(action_metadata, risk_score, context)

        # 4. Enforce decision
        self._enforce_decision(decision)

        return decision

    def _make_decision(
        self,
        metadata: ActionMetadata,
        risk_score: RiskScore,
        context: Optional[Dict[str, Any]]
    ) -> PolicyDecision:
        """
        Make policy decision based on risk assessment

        Args:
            metadata: Action metadata
            risk_score: Risk assessment
            context: Execution context

        Returns:
            PolicyDecision
        """
        context = context or {}

        # Decision logic
        if risk_score.total_score >= 0.9:
            # Critical risk - block immediately
            decision_type = ActionDecision.BLOCK_AND_ALERT
            reason = f"Critical risk level ({risk_score.total_score:.1%}) - action blocked"
            can_proceed = False
            requires_approval = False

        elif risk_score.total_score >= self.risk_threshold:
            # High risk - block unless approved
            decision_type = ActionDecision.BLOCK
            reason = f"Risk ({risk_score.total_score:.1%}) exceeds threshold ({self.risk_threshold:.1%})"
            can_proceed = False
            requires_approval = False

        elif risk_score.requires_approval and self.enable_approval_workflow:
            # Moderate risk - require approval
            decision_type = ActionDecision.REQUIRE_APPROVAL
            reason = f"Action requires human approval (risk: {risk_score.total_score:.1%})"
            can_proceed = False  # Not until approved
            requires_approval = True

        elif risk_score.total_score >= 0.3:
            # Low-moderate risk - allow with logging
            decision_type = ActionDecision.ALLOW_WITH_LOGGING
            reason = f"Action allowed with enhanced logging (risk: {risk_score.total_score:.1%})"
            can_proceed = True
            requires_approval = False

        else:
            # Low risk - allow
            decision_type = ActionDecision.ALLOW
            reason = f"Action approved (risk: {risk_score.total_score:.1%})"
            can_proceed = True
            requires_approval = False

        # Create approval message if needed
        approval_message = None
        if requires_approval:
            approval_message = self._generate_approval_message(metadata, risk_score, context)

        return PolicyDecision(
            decision=decision_type,
            action_metadata=metadata,
            risk_score=risk_score,
            reason=reason,
            requires_user_approval=requires_approval,
            requires_explanation=risk_score.requires_explanation,
            can_proceed=can_proceed,
            approval_message=approval_message,
            timestamp=datetime.now()
        )

    def _enforce_decision(self, decision: PolicyDecision):
        """
        Enforce policy decision

        Args:
            decision: Policy decision to enforce

        Raises:
            UnauthorizedActionError: If action is blocked
            RiskThresholdExceededError: If risk threshold exceeded
        """
        # Log violation if blocked
        if decision.decision in [ActionDecision.BLOCK, ActionDecision.BLOCK_AND_ALERT]:
            self._log_violation(decision)

        # Raise exception if cannot proceed
        if not decision.can_proceed and not decision.requires_user_approval:
            if decision.decision == ActionDecision.BLOCK_AND_ALERT:
                # Critical violation - alert
                self._send_security_alert(decision)

            if decision.risk_score.total_score >= self.risk_threshold:
                raise RiskThresholdExceededError(
                    decision.reason,
                    risk_score=decision.risk_score.total_score,
                    threshold=self.risk_threshold
                )
            else:
                raise UnauthorizedActionError(
                    decision.reason,
                    action=decision.action_metadata.tool_name
                )

    def request_approval(self, decision: PolicyDecision) -> bool:
        """
        Request human approval for action

        Args:
            decision: Policy decision requiring approval

        Returns:
            True if approved, False if denied

        Raises:
            ValueError: If decision doesn't require approval
        """
        if not decision.requires_user_approval:
            raise ValueError("This decision does not require approval")

        # Call approval callback if set
        if self.approval_callback:
            approved = self.approval_callback(decision)
        else:
            # Default: auto-deny if no callback set
            approved = False

        # Log approval decision
        self._log_approval_decision(decision, approved)

        # Update decision
        if approved:
            decision.can_proceed = True
            decision.decision = ActionDecision.ALLOW_WITH_LOGGING

        return approved

    def _generate_approval_message(
        self,
        metadata: ActionMetadata,
        risk_score: RiskScore,
        context: Dict[str, Any]
    ) -> str:
        """Generate human-readable approval request message"""
        lines = []
        lines.append("="*60)
        lines.append("ACTION APPROVAL REQUIRED")
        lines.append("="*60)
        lines.append("")
        lines.append(f"Tool: {metadata.tool_name}")
        lines.append(f"Category: {metadata.category.value}")
        lines.append(f"Risk Level: {metadata.risk_level.name}")
        lines.append(f"Risk Score: {risk_score.total_score:.1%}")
        lines.append("")
        lines.append(f"Description: {metadata.description}")
        lines.append("")

        if risk_score.risk_factors:
            lines.append("Risk Factors:")
            for factor in risk_score.risk_factors[:5]:  # Top 5
                lines.append(f"  - {factor}")
            lines.append("")

        if not metadata.reversible:
            lines.append("[WARNING] This action is IRREVERSIBLE")
            lines.append("")

        if metadata.side_effects:
            lines.append("Side Effects:")
            for effect in metadata.side_effects:
                lines.append(f"  - {effect}")
            lines.append("")

        lines.append("Do you want to proceed with this action?")
        lines.append("(yes/no)")

        return "\n".join(lines)

    def _log_violation(self, decision: PolicyDecision):
        """Log policy violation"""
        self.violations_log.append({
            'timestamp': decision.timestamp,
            'tool_name': decision.action_metadata.tool_name,
            'decision': decision.decision.value,
            'risk_score': decision.risk_score.total_score,
            'reason': decision.reason,
            'risk_factors': decision.risk_score.risk_factors
        })

        # Keep log bounded
        if len(self.violations_log) > 1000:
            self.violations_log = self.violations_log[-500:]

    def _log_approval_decision(self, decision: PolicyDecision, approved: bool):
        """Log approval decision"""
        # TODO: Implement approval logging (Phase 5: Audit system)
        pass

    def _send_security_alert(self, decision: PolicyDecision):
        """Send security alert for critical violations"""
        # TODO: Implement alerting (Phase 5: Audit system)
        # For now, just print to console
        print(f"\n[SECURITY ALERT] Critical action blocked: {decision.action_metadata.tool_name}")
        print(f"  Risk: {decision.risk_score.total_score:.1%}")
        print(f"  Factors: {', '.join(decision.risk_score.risk_factors[:3])}")

    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of policy violations"""
        if not self.violations_log:
            return {
                'total_violations': 0,
                'recent_violations': []
            }

        return {
            'total_violations': len(self.violations_log),
            'recent_violations': self.violations_log[-10:],  # Last 10
            'most_violated_tools': self._get_most_violated_tools(),
            'average_risk_score': sum(v['risk_score'] for v in self.violations_log) / len(self.violations_log)
        }

    def _get_most_violated_tools(self) -> Dict[str, int]:
        """Get tools with most violations"""
        tool_counts = {}
        for violation in self.violations_log:
            tool = violation['tool_name']
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

        # Sort by count
        return dict(sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:5])

    def explain_decision(self, decision: PolicyDecision) -> str:
        """
        Generate detailed explanation of policy decision

        Args:
            decision: Policy decision to explain

        Returns:
            Human-readable explanation
        """
        lines = []
        lines.append("="*60)
        lines.append("POLICY DECISION EXPLANATION")
        lines.append("="*60)
        lines.append("")
        lines.append(f"Action: {decision.action_metadata.tool_name}")
        lines.append(f"Decision: {decision.decision.value.upper()}")
        lines.append(f"Timestamp: {decision.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append(f"Reason: {decision.reason}")
        lines.append("")

        # Risk assessment details
        lines.append(self.scorer.explain_risk(decision.risk_score))

        # Next steps
        lines.append("")
        lines.append("Next Steps:")
        if decision.decision == ActionDecision.ALLOW:
            lines.append("  - Action will proceed normally")
        elif decision.decision == ActionDecision.ALLOW_WITH_LOGGING:
            lines.append("  - Action will proceed with enhanced logging")
        elif decision.decision == ActionDecision.REQUIRE_APPROVAL:
            lines.append("  - Human approval required before proceeding")
            lines.append("  - Use request_approval() to get user decision")
        elif decision.decision == ActionDecision.BLOCK:
            lines.append("  - Action has been blocked")
            lines.append("  - Review risk factors and modify action if possible")
        elif decision.decision == ActionDecision.BLOCK_AND_ALERT:
            lines.append("  - Action has been blocked")
            lines.append("  - Security team has been alerted")
            lines.append("  - This may require security review")

        return "\n".join(lines)
