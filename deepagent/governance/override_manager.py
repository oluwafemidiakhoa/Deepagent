"""
Override Manager - Foundation #12

Human override and intervention system.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class OverrideType(Enum):
    """Types of overrides"""
    ALLOW = "allow"      # Override to allow blocked action
    BLOCK = "block"      # Override to block allowed action
    MODIFY = "modify"    # Modify action parameters
    CANCEL = "cancel"    # Cancel current execution


@dataclass
class OverrideRequest:
    """Request to override system decision"""
    override_id: str
    override_type: OverrideType
    target_action: str
    reason: str
    requested_by: str
    requested_at: datetime


@dataclass
class OverrideResult:
    """Result of override"""
    success: bool
    override_id: str
    applied_at: datetime
    message: str


class OverrideManager:
    """Manages human overrides"""

    def __init__(self):
        self.overrides: dict = {}

    def request_override(
        self,
        override_type: str,
        target_action: str,
        reason: str,
        requested_by: str
    ) -> OverrideRequest:
        """Request an override"""
        from uuid import uuid4

        request = OverrideRequest(
            override_id=f"ovr_{uuid4().hex[:8]}",
            override_type=OverrideType(override_type),
            target_action=target_action,
            reason=reason,
            requested_by=requested_by,
            requested_at=datetime.now()
        )

        self.overrides[request.override_id] = request
        return request

    def apply_override(self, override_id: str) -> OverrideResult:
        """Apply an override"""
        if override_id not in self.overrides:
            return OverrideResult(
                success=False,
                override_id=override_id,
                applied_at=datetime.now(),
                message="Override not found"
            )

        # In production, this would actually override the decision
        return OverrideResult(
            success=True,
            override_id=override_id,
            applied_at=datetime.now(),
            message="Override applied successfully"
        )
