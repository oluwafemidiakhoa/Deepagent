"""
Foundation #7: Audit Logs & Forensics

Comprehensive audit logging and forensic analysis for SafeDeepAgent.
"""

from deepagent.audit.audit_logger import (
    AuditLogger,
    AuditRecord,
    EventType,
    EventSeverity,
    AuditConfig,
    JSONAuditLogger,
    SQLiteAuditLogger,
    CompositeAuditLogger
)

from deepagent.audit.forensic_analyzer import (
    ForensicAnalyzer,
    AttackReconstruction,
    TimelineAnalysis,
    RiskTrajectory,
    PatternCorrelation
)

from deepagent.audit.query_interface import (
    AuditQueryInterface,
    QueryFilters,
    QueryResult
)

__all__ = [
    # Audit Logger
    "AuditLogger",
    "AuditRecord",
    "EventType",
    "EventSeverity",
    "AuditConfig",
    "JSONAuditLogger",
    "SQLiteAuditLogger",
    "CompositeAuditLogger",

    # Forensic Analyzer
    "ForensicAnalyzer",
    "AttackReconstruction",
    "TimelineAnalysis",
    "RiskTrajectory",
    "PatternCorrelation",

    # Query Interface
    "AuditQueryInterface",
    "QueryFilters",
    "QueryResult"
]
