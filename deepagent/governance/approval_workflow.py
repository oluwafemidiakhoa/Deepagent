"""
Approval Workflow - Foundation #12

Interactive approval system for human oversight.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Callable


class ApprovalStatus(Enum):
    """Status of approval request"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class ApprovalRequest:
    """Request for human approval"""
    request_id: str
    action_type: str
    tool_name: str
    parameters: dict
    risk_score: float
    reason: str
    requested_at: datetime
    timeout_seconds: int = 300


@dataclass
class ApprovalDecision:
    """Human approval decision"""
    request_id: str
    status: ApprovalStatus
    approved: bool
    approver_id: Optional[str]
    decided_at: datetime
    comments: Optional[str]


class ApprovalWorkflow:
    """Manages approval workflow"""

    def __init__(self):
        self.pending_requests: dict = {}
        self.decisions: dict = {}
        self.approval_callback: Optional[Callable] = None

    def request_approval(
        self,
        action_type: str,
        tool_name: str,
        parameters: dict,
        risk_score: float,
        reason: str
    ) -> ApprovalRequest:
        """Create approval request"""
        from uuid import uuid4

        request = ApprovalRequest(
            request_id=f"apr_{uuid4().hex[:8]}",
            action_type=action_type,
            tool_name=tool_name,
            parameters=parameters,
            risk_score=risk_score,
            reason=reason,
            requested_at=datetime.now()
        )

        self.pending_requests[request.request_id] = request

        # Callback for interactive approval
        if self.approval_callback:
            try:
                approved = self.approval_callback(request)
                self.record_decision(
                    request.request_id,
                    approved,
                    "system",
                    "Callback approval"
                )
            except Exception:
                # Auto-reject on error
                self.record_decision(
                    request.request_id,
                    False,
                    "system",
                    "Callback error"
                )

        return request

    def record_decision(
        self,
        request_id: str,
        approved: bool,
        approver_id: str,
        comments: Optional[str] = None
    ) -> ApprovalDecision:
        """Record approval decision"""
        decision = ApprovalDecision(
            request_id=request_id,
            status=ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED,
            approved=approved,
            approver_id=approver_id,
            decided_at=datetime.now(),
            comments=comments
        )

        self.decisions[request_id] = decision

        # Remove from pending
        self.pending_requests.pop(request_id, None)

        return decision

    def get_decision(self, request_id: str) -> Optional[ApprovalDecision]:
        """Get approval decision"""
        return self.decisions.get(request_id)

    def set_callback(self, callback: Callable):
        """Set approval callback function"""
        self.approval_callback = callback
