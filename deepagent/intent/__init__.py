"""
Foundation #9: Global Intent & Context

Maintains global task context across sessions.
"""

from deepagent.intent.intent_tracker import (
    IntentTracker,
    GlobalIntent,
    IntentAlignment
)

from deepagent.intent.context_manager import (
    ContextManager,
    GlobalContext
)

__all__ = [
    "IntentTracker",
    "GlobalIntent",
    "IntentAlignment",
    "ContextManager",
    "GlobalContext"
]
