"""
Task Sequence Analyzer

Analyzes sequences of actions to detect malicious patterns.
Part of Foundation #2: Memory Firewalls
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


@dataclass
class ActionRecord:
    """
    Record of a single action taken by agent
    """
    timestamp: datetime
    step_number: int
    action_type: str
    tool_name: str
    parameters: Dict[str, Any]
    result: Any
    risk_score: float
    reasoning: Optional[str] = None

    # Metadata
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    def to_tuple(self) -> Tuple[str, Dict[str, Any]]:
        """Convert to (action_type, parameters) tuple for pattern matching"""
        return (self.action_type, self.parameters)


@dataclass
class ActionHistory:
    """
    Tracks all actions taken by agent during execution
    """
    actions: List[ActionRecord] = field(default_factory=list)
    original_task: str = ""
    current_goal: str = ""
    start_time: datetime = field(default_factory=datetime.now)

    # Session metadata
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    def add_action(self, action: ActionRecord):
        """Add action to history"""
        self.actions.append(action)

    def get_recent_sequence(self, window_size: int) -> List[ActionRecord]:
        """Get recent N actions"""
        return self.actions[-window_size:] if len(self.actions) >= window_size else self.actions

    def get_by_timeframe(self, start: datetime, end: datetime) -> List[ActionRecord]:
        """Get actions in timeframe"""
        return [a for a in self.actions if start <= a.timestamp <= end]

    def get_action_types_sequence(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get sequence as (action_type, parameters) tuples"""
        return [a.to_tuple() for a in self.actions]

    def total_actions(self) -> int:
        """Get total number of actions"""
        return len(self.actions)

    def average_risk(self) -> float:
        """Calculate average risk score"""
        if not self.actions:
            return 0.0
        return sum(a.risk_score for a in self.actions) / len(self.actions)


@dataclass
class AlignmentResult:
    """
    Result of goal alignment check
    """
    is_aligned: bool
    confidence: float
    drift_score: float  # How far from original goal (0-1)
    suspicious_actions: List[int]  # Step numbers
    explanation: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DriftResult:
    """
    Result of intent drift detection
    """
    drift_detected: bool
    drift_magnitude: float  # 0-1 scale
    drift_direction: str  # What shifted to
    drift_points: List[int]  # Which steps show drift
    explanation: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EscalationResult:
    """
    Result of privilege escalation detection
    """
    escalation_detected: bool
    escalation_rate: float  # How fast privileges increase
    privilege_trajectory: List[float]  # Privilege level over time
    concerning_steps: List[int]
    explanation: str
    timestamp: datetime = field(default_factory=datetime.now)


class TaskSequenceAnalyzer:
    """
    Analyzes sequences of actions to detect malicious patterns

    Combines:
    - Action history tracking
    - Goal alignment checking
    - Intent drift detection
    - Escalation detection
    """

    def __init__(self):
        """Initialize analyzer"""
        self.action_history: Optional[ActionHistory] = None

        # Thresholds
        self.drift_threshold = 0.6  # Drift score above this is concerning
        self.escalation_rate_threshold = 0.15  # Risk increase per step

    def initialize(self, original_task: str, session_id: Optional[str] = None,
                   user_id: Optional[str] = None):
        """
        Initialize with original task

        Args:
            original_task: The original user task
            session_id: Optional session identifier
            user_id: Optional user identifier
        """
        self.action_history = ActionHistory(
            original_task=original_task,
            current_goal=original_task,
            session_id=session_id,
            user_id=user_id
        )

    def record_action(self, action: ActionRecord):
        """
        Record an action in history

        Args:
            action: Action record to add
        """
        if self.action_history is None:
            raise ValueError("Analyzer not initialized. Call initialize() first.")

        self.action_history.add_action(action)

    def check_goal_alignment(
        self,
        original_task: str,
        action_sequence: List[ActionRecord],
        use_simple_heuristics: bool = True
    ) -> AlignmentResult:
        """
        Check if action sequence aligns with original task

        Args:
            original_task: Original user task
            action_sequence: Sequence of actions to check
            use_simple_heuristics: Use simple keyword-based heuristics

        Returns:
            AlignmentResult with alignment analysis
        """
        if not action_sequence:
            return AlignmentResult(
                is_aligned=True,
                confidence=1.0,
                drift_score=0.0,
                suspicious_actions=[],
                explanation="No actions to analyze"
            )

        if use_simple_heuristics:
            # Simple keyword-based alignment check
            return self._check_alignment_heuristic(original_task, action_sequence)
        else:
            # TODO: Use embeddings/LLM for semantic similarity
            return self._check_alignment_heuristic(original_task, action_sequence)

    def _check_alignment_heuristic(
        self,
        original_task: str,
        action_sequence: List[ActionRecord]
    ) -> AlignmentResult:
        """
        Heuristic-based alignment check using keyword matching

        Args:
            original_task: Original task description
            action_sequence: Actions to analyze

        Returns:
            AlignmentResult
        """
        # Extract keywords from original task
        task_keywords = set(original_task.lower().split())

        # Check each action for keyword overlap
        alignment_scores = []
        suspicious_actions = []

        for i, action in enumerate(action_sequence):
            # Get action context (reasoning + parameters)
            action_text = ""
            if action.reasoning:
                action_text += action.reasoning.lower() + " "

            action_text += " ".join(str(v).lower() for v in action.parameters.values())
            action_keywords = set(action_text.split())

            # Calculate keyword overlap
            overlap = len(task_keywords & action_keywords)
            score = overlap / max(len(task_keywords), 1)
            alignment_scores.append(score)

            # Flag actions with low alignment
            if score < 0.1 and len(action_sequence) > 3:
                suspicious_actions.append(i)

        # Calculate overall alignment
        if alignment_scores:
            avg_alignment = sum(alignment_scores) / len(alignment_scores)
            drift_score = 1.0 - avg_alignment
        else:
            avg_alignment = 0.0
            drift_score = 1.0

        is_aligned = drift_score < self.drift_threshold

        if is_aligned:
            explanation = f"Actions align with original task (drift: {drift_score:.2%})"
        else:
            explanation = f"Significant drift detected from original task (drift: {drift_score:.2%})"
            if suspicious_actions:
                explanation += f". Suspicious steps: {suspicious_actions}"

        return AlignmentResult(
            is_aligned=is_aligned,
            confidence=0.7,  # Heuristic confidence
            drift_score=drift_score,
            suspicious_actions=suspicious_actions,
            explanation=explanation
        )

    def detect_intent_drift(
        self,
        original_task: str,
        action_sequence: List[ActionRecord],
        window_size: int = 3
    ) -> DriftResult:
        """
        Detect drift in agent's intent over time

        Args:
            original_task: Original task
            action_sequence: Actions to analyze
            window_size: Size of sliding window for drift detection

        Returns:
            DriftResult with drift analysis
        """
        if len(action_sequence) < window_size:
            return DriftResult(
                drift_detected=False,
                drift_magnitude=0.0,
                drift_direction="",
                drift_points=[],
                explanation="Insufficient actions for drift detection"
            )

        # Compare sliding windows
        drift_points = []
        drift_magnitudes = []

        for i in range(len(action_sequence) - window_size):
            window1 = action_sequence[i:i+window_size]
            window2 = action_sequence[i+1:i+1+window_size]

            # Compare keyword sets
            keywords1 = self._extract_action_keywords(window1)
            keywords2 = self._extract_action_keywords(window2)

            # Jaccard similarity
            intersection = len(keywords1 & keywords2)
            union = len(keywords1 | keywords2)
            similarity = intersection / union if union > 0 else 0.0

            drift = 1.0 - similarity
            drift_magnitudes.append(drift)

            if drift > 0.5:  # Significant drift
                drift_points.append(i + window_size)

        if drift_magnitudes:
            avg_drift = sum(drift_magnitudes) / len(drift_magnitudes)
        else:
            avg_drift = 0.0

        drift_detected = avg_drift > 0.4 or len(drift_points) > 0

        if drift_detected:
            # Determine drift direction (what topics emerged)
            late_keywords = self._extract_action_keywords(action_sequence[-window_size:])
            task_keywords = set(original_task.lower().split())
            new_keywords = late_keywords - task_keywords

            drift_direction = ", ".join(list(new_keywords)[:5]) if new_keywords else "unknown"

            explanation = f"Intent drift detected (magnitude: {avg_drift:.2%}). New topics: {drift_direction}"
        else:
            drift_direction = ""
            explanation = f"No significant intent drift (magnitude: {avg_drift:.2%})"

        return DriftResult(
            drift_detected=drift_detected,
            drift_magnitude=avg_drift,
            drift_direction=drift_direction,
            drift_points=drift_points,
            explanation=explanation
        )

    def detect_escalation(
        self,
        action_history: ActionHistory
    ) -> EscalationResult:
        """
        Detect privilege escalation patterns

        Args:
            action_history: History of actions

        Returns:
            EscalationResult with escalation analysis
        """
        if len(action_history.actions) < 3:
            return EscalationResult(
                escalation_detected=False,
                escalation_rate=0.0,
                privilege_trajectory=[],
                concerning_steps=[],
                explanation="Insufficient actions for escalation detection"
            )

        # Track risk score trajectory (proxy for privilege level)
        trajectory = [a.risk_score for a in action_history.actions]

        # Calculate escalation rate (average increase per step)
        escalations = []
        for i in range(len(trajectory) - 1):
            increase = trajectory[i+1] - trajectory[i]
            escalations.append(increase)

        avg_escalation_rate = sum(escalations) / len(escalations) if escalations else 0.0

        # Find concerning steps (large jumps)
        concerning_steps = []
        for i, increase in enumerate(escalations):
            if increase > 0.2:  # 20% jump in risk
                concerning_steps.append(i + 1)

        # Detect if privilege is steadily increasing
        escalation_detected = (
            avg_escalation_rate > self.escalation_rate_threshold or
            len(concerning_steps) >= 2
        )

        if escalation_detected:
            explanation = f"Privilege escalation detected (rate: {avg_escalation_rate:.2%} per step)"
            if concerning_steps:
                explanation += f". Large jumps at steps: {concerning_steps}"
        else:
            explanation = f"No escalation detected (rate: {avg_escalation_rate:.2%} per step)"

        return EscalationResult(
            escalation_detected=escalation_detected,
            escalation_rate=avg_escalation_rate,
            privilege_trajectory=trajectory,
            concerning_steps=concerning_steps,
            explanation=explanation
        )

    def _extract_action_keywords(self, actions: List[ActionRecord]) -> set:
        """Extract keywords from action sequence"""
        keywords = set()

        for action in actions:
            # Add from reasoning
            if action.reasoning:
                keywords.update(action.reasoning.lower().split())

            # Add from parameters
            for value in action.parameters.values():
                if isinstance(value, str):
                    keywords.update(value.lower().split())

            # Add action type
            keywords.add(action.action_type.lower())

        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        keywords -= stop_words

        return keywords

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of current analysis

        Returns:
            Dictionary with analysis summary
        """
        if self.action_history is None:
            return {"error": "Analyzer not initialized"}

        # Run all checks
        alignment = self.check_goal_alignment(
            self.action_history.original_task,
            self.action_history.actions
        )

        drift = self.detect_intent_drift(
            self.action_history.original_task,
            self.action_history.actions
        )

        escalation = self.detect_escalation(self.action_history)

        return {
            "total_actions": self.action_history.total_actions(),
            "average_risk": self.action_history.average_risk(),
            "alignment": {
                "is_aligned": alignment.is_aligned,
                "drift_score": alignment.drift_score,
                "suspicious_actions": alignment.suspicious_actions
            },
            "drift": {
                "detected": drift.drift_detected,
                "magnitude": drift.drift_magnitude,
                "drift_points": drift.drift_points
            },
            "escalation": {
                "detected": escalation.escalation_detected,
                "rate": escalation.escalation_rate,
                "concerning_steps": escalation.concerning_steps
            }
        }
