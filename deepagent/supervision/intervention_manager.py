"""
Intervention Manager - Foundation #6

Manages corrective interventions for agent violations.
Provides automated and manual intervention capabilities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4


class InterventionType(Enum):
    """Type of intervention"""
    PAUSE = "pause"                    # Temporarily pause agent
    RESTRICT = "restrict"              # Restrict agent capabilities
    REDIRECT = "redirect"              # Redirect to different task
    ROLLBACK = "rollback"              # Rollback recent actions
    TERMINATE = "terminate"            # Terminate agent
    NOTIFY = "notify"                  # Notify human supervisor
    ESCALATE = "escalate"              # Escalate to higher authority


class InterventionStatus(Enum):
    """Status of intervention"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class InterventionAction:
    """Definition of an intervention action"""
    action_type: InterventionType
    parameters: Dict[str, Any]
    description: str
    priority: int = 1  # 1=highest, 5=lowest
    automated: bool = True


@dataclass
class InterventionResult:
    """Result of an intervention"""
    intervention_id: str
    agent_id: str
    action_type: InterventionType
    status: InterventionStatus
    success: bool
    message: str
    side_effects: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0


@dataclass
class InterventionRecord:
    """Record of an intervention"""
    intervention_id: str
    agent_id: str
    trigger: str
    trigger_context: Dict[str, Any]
    actions: List[InterventionAction]
    results: List[InterventionResult] = field(default_factory=list)
    status: InterventionStatus = InterventionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class InterventionPolicy:
    """Policy for automatic interventions"""
    policy_id: str
    name: str
    trigger_condition: str
    actions: List[InterventionAction]
    enabled: bool = True
    priority: int = 1


class InterventionManager:
    """
    Manages corrective interventions for agents.

    Provides:
    - Automated intervention based on triggers
    - Manual intervention capabilities
    - Intervention tracking and history
    - Escalation management
    """

    def __init__(self):
        self.intervention_policies: Dict[str, InterventionPolicy] = {}
        self.active_interventions: Dict[str, InterventionRecord] = {}
        self.intervention_history: List[InterventionRecord] = []
        self._intervention_handlers: Dict[InterventionType, Callable] = {}
        self._setup_default_handlers()

    def _setup_default_handlers(self) -> None:
        """Setup default intervention handlers"""
        self._intervention_handlers[InterventionType.PAUSE] = self._handle_pause
        self._intervention_handlers[InterventionType.RESTRICT] = self._handle_restrict
        self._intervention_handlers[InterventionType.REDIRECT] = self._handle_redirect
        self._intervention_handlers[InterventionType.ROLLBACK] = self._handle_rollback
        self._intervention_handlers[InterventionType.TERMINATE] = self._handle_terminate
        self._intervention_handlers[InterventionType.NOTIFY] = self._handle_notify
        self._intervention_handlers[InterventionType.ESCALATE] = self._handle_escalate

    def add_intervention_policy(self, policy: InterventionPolicy) -> bool:
        """Add an automatic intervention policy"""
        if policy.policy_id in self.intervention_policies:
            return False

        self.intervention_policies[policy.policy_id] = policy
        return True

    def remove_intervention_policy(self, policy_id: str) -> bool:
        """Remove an intervention policy"""
        if policy_id in self.intervention_policies:
            del self.intervention_policies[policy_id]
            return True
        return False

    def trigger_intervention(
        self,
        agent_id: str,
        trigger: str,
        context: Dict[str, Any],
        actions: Optional[List[InterventionAction]] = None
    ) -> InterventionRecord:
        """
        Trigger an intervention for an agent.

        Args:
            agent_id: Agent to intervene on
            trigger: What triggered the intervention
            context: Context information
            actions: Specific actions to take (if not auto-determined)

        Returns:
            InterventionRecord tracking the intervention
        """
        intervention_id = f"intervention_{uuid4().hex[:12]}"

        # Determine actions if not provided
        if actions is None:
            actions = self._determine_intervention_actions(trigger, context)

        record = InterventionRecord(
            intervention_id=intervention_id,
            agent_id=agent_id,
            trigger=trigger,
            trigger_context=context,
            actions=actions
        )

        self.active_interventions[intervention_id] = record
        return record

    def _determine_intervention_actions(
        self,
        trigger: str,
        context: Dict[str, Any]
    ) -> List[InterventionAction]:
        """Determine appropriate intervention actions based on trigger"""
        actions = []

        # Check intervention policies
        for policy in self.intervention_policies.values():
            if not policy.enabled:
                continue

            if self._matches_trigger(trigger, policy.trigger_condition, context):
                actions.extend(policy.actions)

        # Default action if no policy matches
        if not actions:
            actions.append(InterventionAction(
                action_type=InterventionType.NOTIFY,
                parameters={'trigger': trigger},
                description="Default notification",
                priority=3
            ))

        # Sort by priority
        actions.sort(key=lambda a: a.priority)
        return actions

    def _matches_trigger(
        self,
        trigger: str,
        condition: str,
        context: Dict[str, Any]
    ) -> bool:
        """Check if trigger matches condition"""
        # Simple keyword matching for now
        return condition.lower() in trigger.lower()

    def execute_intervention(self, intervention_id: str) -> List[InterventionResult]:
        """
        Execute an intervention.

        Args:
            intervention_id: ID of intervention to execute

        Returns:
            List of results for each action
        """
        if intervention_id not in self.active_interventions:
            return []

        record = self.active_interventions[intervention_id]
        record.status = InterventionStatus.IN_PROGRESS
        results = []

        for action in record.actions:
            result = self._execute_action(
                record.agent_id,
                action,
                intervention_id
            )
            results.append(result)
            record.results.append(result)

        # Update overall status
        if all(r.success for r in results):
            record.status = InterventionStatus.COMPLETED
        else:
            record.status = InterventionStatus.FAILED

        record.completed_at = datetime.now()

        # Move to history
        self.intervention_history.append(record)
        del self.active_interventions[intervention_id]

        return results

    def _execute_action(
        self,
        agent_id: str,
        action: InterventionAction,
        intervention_id: str
    ) -> InterventionResult:
        """Execute a single intervention action"""
        start_time = datetime.now()

        handler = self._intervention_handlers.get(action.action_type)

        if not handler:
            return InterventionResult(
                intervention_id=intervention_id,
                agent_id=agent_id,
                action_type=action.action_type,
                status=InterventionStatus.FAILED,
                success=False,
                message=f"No handler for action type {action.action_type}"
            )

        try:
            success, message, side_effects = handler(agent_id, action.parameters)

            duration = (datetime.now() - start_time).total_seconds() * 1000

            return InterventionResult(
                intervention_id=intervention_id,
                agent_id=agent_id,
                action_type=action.action_type,
                status=InterventionStatus.COMPLETED if success else InterventionStatus.FAILED,
                success=success,
                message=message,
                side_effects=side_effects,
                duration_ms=duration
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000

            return InterventionResult(
                intervention_id=intervention_id,
                agent_id=agent_id,
                action_type=action.action_type,
                status=InterventionStatus.FAILED,
                success=False,
                message=f"Exception: {str(e)}",
                duration_ms=duration
            )

    # Default intervention handlers
    def _handle_pause(
        self,
        agent_id: str,
        parameters: Dict[str, Any]
    ) -> tuple[bool, str, List[str]]:
        """Handle pause intervention"""
        reason = parameters.get('reason', 'Intervention triggered')
        return (
            True,
            f"Agent {agent_id} paused: {reason}",
            ["Agent execution paused"]
        )

    def _handle_restrict(
        self,
        agent_id: str,
        parameters: Dict[str, Any]
    ) -> tuple[bool, str, List[str]]:
        """Handle restrict intervention"""
        restrictions = parameters.get('restrictions', [])
        return (
            True,
            f"Agent {agent_id} restricted: {restrictions}",
            [f"Restricted: {r}" for r in restrictions]
        )

    def _handle_redirect(
        self,
        agent_id: str,
        parameters: Dict[str, Any]
    ) -> tuple[bool, str, List[str]]:
        """Handle redirect intervention"""
        new_task = parameters.get('task', 'safe_task')
        return (
            True,
            f"Agent {agent_id} redirected to: {new_task}",
            ["Current task cancelled", f"Redirected to {new_task}"]
        )

    def _handle_rollback(
        self,
        agent_id: str,
        parameters: Dict[str, Any]
    ) -> tuple[bool, str, List[str]]:
        """Handle rollback intervention"""
        checkpoint = parameters.get('checkpoint', 'latest')
        return (
            True,
            f"Agent {agent_id} rolled back to: {checkpoint}",
            ["State rolled back", "Recent actions undone"]
        )

    def _handle_terminate(
        self,
        agent_id: str,
        parameters: Dict[str, Any]
    ) -> tuple[bool, str, List[str]]:
        """Handle terminate intervention"""
        reason = parameters.get('reason', 'Safety intervention')
        return (
            True,
            f"Agent {agent_id} terminated: {reason}",
            ["Agent terminated", "Resources released"]
        )

    def _handle_notify(
        self,
        agent_id: str,
        parameters: Dict[str, Any]
    ) -> tuple[bool, str, List[str]]:
        """Handle notify intervention"""
        trigger = parameters.get('trigger', 'Unknown')
        return (
            True,
            f"Notification sent for agent {agent_id}: {trigger}",
            ["Human supervisor notified"]
        )

    def _handle_escalate(
        self,
        agent_id: str,
        parameters: Dict[str, Any]
    ) -> tuple[bool, str, List[str]]:
        """Handle escalate intervention"""
        level = parameters.get('level', 'supervisor')
        return (
            True,
            f"Agent {agent_id} escalated to: {level}",
            ["Escalation triggered", f"Escalated to {level}"]
        )

    def register_handler(
        self,
        intervention_type: InterventionType,
        handler: Callable[[str, Dict[str, Any]], tuple[bool, str, List[str]]]
    ) -> None:
        """Register a custom intervention handler"""
        self._intervention_handlers[intervention_type] = handler

    def get_active_interventions(self) -> List[InterventionRecord]:
        """Get all active interventions"""
        return list(self.active_interventions.values())

    def get_intervention_history(
        self,
        agent_id: Optional[str] = None,
        action_type: Optional[InterventionType] = None
    ) -> List[InterventionRecord]:
        """Get intervention history, optionally filtered"""
        history = self.intervention_history

        if agent_id:
            history = [r for r in history if r.agent_id == agent_id]

        if action_type:
            history = [r for r in history
                      if any(a.action_type == action_type for a in r.actions)]

        return history

    def cancel_intervention(self, intervention_id: str) -> bool:
        """Cancel an active intervention"""
        if intervention_id not in self.active_interventions:
            return False

        record = self.active_interventions[intervention_id]
        record.status = InterventionStatus.CANCELLED
        record.completed_at = datetime.now()

        self.intervention_history.append(record)
        del self.active_interventions[intervention_id]

        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get intervention statistics"""
        total_interventions = len(self.intervention_history)
        active_interventions = len(self.active_interventions)

        # Count by status
        status_counts = {
            'completed': 0,
            'failed': 0,
            'cancelled': 0
        }

        # Count by type
        type_counts = {t.value: 0 for t in InterventionType}

        for record in self.intervention_history:
            status_counts[record.status.value] += 1

            for action in record.actions:
                type_counts[action.action_type.value] += 1

        return {
            'total_interventions': total_interventions,
            'active_interventions': active_interventions,
            'completed_interventions': status_counts['completed'],
            'failed_interventions': status_counts['failed'],
            'cancelled_interventions': status_counts['cancelled'],
            'interventions_by_type': type_counts,
            'active_policies': len([p for p in self.intervention_policies.values() if p.enabled])
        }
