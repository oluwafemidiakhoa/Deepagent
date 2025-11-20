"""
Memory Firewall - Phase 2 Security Components

Implements Foundation #2: Memory Firewalls
Detects sophisticated multi-step attacks that bypass single-action defenses.
"""

from .attack_patterns import (
    AttackPattern,
    AttackStep,
    AttackPatternDatabase,
    AttackDetectionResult,
    AttackCategory,
    AttackSeverity
)

from .sequence_analyzer import (
    TaskSequenceAnalyzer,
    ActionHistory,
    ActionRecord,
    AlignmentResult,
    DriftResult,
    EscalationResult
)

from .reasoning_monitor import (
    ReasoningMonitor,
    ReasoningAnalysis,
    SequenceAnalysis,
    AnomalyType
)

from .memory_validator import (
    MemoryValidator,
    MemoryEntry,
    ValidationResult,
    ProvenanceChain,
    ProvenanceRecord,
    ProvenanceType,
    IntegrityStatus
)

__all__ = [
    # Attack patterns
    "AttackPattern",
    "AttackStep",
    "AttackPatternDatabase",
    "AttackDetectionResult",
    "AttackCategory",
    "AttackSeverity",

    # Sequence analysis
    "TaskSequenceAnalyzer",
    "ActionHistory",
    "ActionRecord",
    "AlignmentResult",
    "DriftResult",
    "EscalationResult",

    # Reasoning monitoring
    "ReasoningMonitor",
    "ReasoningAnalysis",
    "SequenceAnalysis",
    "AnomalyType",

    # Memory validation
    "MemoryValidator",
    "MemoryEntry",
    "ValidationResult",
    "ProvenanceChain",
    "ProvenanceRecord",
    "ProvenanceType",
    "IntegrityStatus",
]

__version__ = "2.0.0"
