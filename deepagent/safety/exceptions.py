"""
Security exceptions for DeepAgent safety framework
"""

class SecurityViolationError(Exception):
    """Base exception for all security violations"""
    pass


class PromptInjectionDetectedError(SecurityViolationError):
    """Raised when prompt injection attack is detected"""

    def __init__(self, message: str, detected_patterns: list = None):
        super().__init__(message)
        self.detected_patterns = detected_patterns or []


class UnauthorizedActionError(SecurityViolationError):
    """Raised when action is not authorized for current user/context"""

    def __init__(self, message: str, action: str = None, required_permission: str = None):
        super().__init__(message)
        self.action = action
        self.required_permission = required_permission


class DomainViolationError(SecurityViolationError):
    """Raised when task is outside allowed domain"""

    def __init__(self, message: str, requested_domain: str = None, allowed_domains: list = None):
        super().__init__(message)
        self.requested_domain = requested_domain
        self.allowed_domains = allowed_domains or []


class RiskThresholdExceededError(SecurityViolationError):
    """Raised when action risk exceeds acceptable threshold"""

    def __init__(self, message: str, risk_score: float = None, threshold: float = None):
        super().__init__(message)
        self.risk_score = risk_score
        self.threshold = threshold


class IdentityVerificationError(SecurityViolationError):
    """Raised when identity cannot be verified"""
    pass


class IntentMismatchError(SecurityViolationError):
    """Raised when claimed intent doesn't match requested actions"""

    def __init__(self, message: str, claimed_intent: str = None, detected_intent: str = None):
        super().__init__(message)
        self.claimed_intent = claimed_intent
        self.detected_intent = detected_intent


class MemoryPoisoningDetectedError(SecurityViolationError):
    """Raised when memory poisoning attempt is detected"""
    pass


class MultiStepAttackDetectedError(SecurityViolationError):
    """Raised when multi-step attack pattern is detected"""

    def __init__(self, message: str, attack_pattern: str = None, matching_steps: list = None):
        super().__init__(message)
        self.attack_pattern = attack_pattern
        self.matching_steps = matching_steps or []


class BehavioralAnomalyError(SecurityViolationError):
    """Raised when abnormal agent behavior is detected"""

    def __init__(self, message: str, anomaly_type: str = None, severity: str = None):
        super().__init__(message)
        self.anomaly_type = anomaly_type
        self.severity = severity


class SandboxViolationError(SecurityViolationError):
    """Raised when sandbox boundaries are violated"""
    pass
