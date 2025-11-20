"""
Intent Tracker - Foundation #9

Tracks global user intent across sessions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional


@dataclass
class GlobalIntent:
    """User's global goal/intent"""
    intent_id: str
    description: str
    created_at: datetime
    updated_at: datetime
    priority: float  # 0.0 - 1.0
    status: str  # "active", "completed", "abandoned"
    sub_goals: List[str] = field(default_factory=list)


@dataclass
class IntentAlignment:
    """Check if action aligns with global intent"""
    aligned: bool
    intent_id: str
    alignment_score: float
    explanation: str


class IntentTracker:
    """Tracks and maintains global user intents"""

    def __init__(self):
        self.intents: Dict[str, GlobalIntent] = {}
        self.active_intent: Optional[str] = None

    def set_global_intent(
        self,
        description: str,
        priority: float = 0.8
    ) -> GlobalIntent:
        """Set or update global intent"""
        from uuid import uuid4

        intent = GlobalIntent(
            intent_id=f"intent_{uuid4().hex[:8]}",
            description=description,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            priority=priority,
            status="active"
        )

        self.intents[intent.intent_id] = intent
        self.active_intent = intent.intent_id

        return intent

    def check_alignment(
        self,
        task_description: str,
        action_type: str
    ) -> IntentAlignment:
        """Check if task aligns with global intent"""
        if not self.active_intent or self.active_intent not in self.intents:
            return IntentAlignment(
                aligned=True,  # No global intent set
                intent_id="",
                alignment_score=1.0,
                explanation="No global intent set"
            )

        intent = self.intents[self.active_intent]

        # Simple keyword matching
        intent_keywords = set(intent.description.lower().split())
        task_keywords = set(task_description.lower().split())

        overlap = len(intent_keywords & task_keywords)
        score = overlap / max(len(intent_keywords), 1)

        aligned = score > 0.3

        return IntentAlignment(
            aligned=aligned,
            intent_id=intent.intent_id,
            alignment_score=score,
            explanation=f"Task {'aligns with' if aligned else 'deviates from'} global intent"
        )
