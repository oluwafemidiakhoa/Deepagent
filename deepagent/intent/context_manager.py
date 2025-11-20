"""
Context Manager - Foundation #9

Manages global context across sessions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class GlobalContext:
    """Global context spanning multiple sessions"""
    context_id: str
    user_id: str
    created_at: datetime
    last_updated: datetime
    state: Dict[str, Any] = field(default_factory=dict)
    session_history: List[str] = field(default_factory=list)


class ContextManager:
    """Maintains global context"""

    def __init__(self):
        self.contexts: Dict[str, GlobalContext] = {}

    def get_or_create_context(self, user_id: str) -> GlobalContext:
        """Get or create global context for user"""
        if user_id in self.contexts:
            return self.contexts[user_id]

        from uuid import uuid4
        context = GlobalContext(
            context_id=f"ctx_{uuid4().hex[:8]}",
            user_id=user_id,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        self.contexts[user_id] = context
        return context

    def update_context(
        self,
        user_id: str,
        session_id: str,
        state_updates: Dict[str, Any]
    ):
        """Update global context"""
        context = self.get_or_create_context(user_id)

        context.state.update(state_updates)
        context.session_history.append(session_id)
        context.last_updated = datetime.now()

    def get_context(self, user_id: str) -> Optional[GlobalContext]:
        """Get global context for user"""
        return self.contexts.get(user_id)
