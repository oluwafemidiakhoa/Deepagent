# Foundation #7: Audit Logs & Forensics - Design Document

**Status**: In Development
**Date**: 2025-11-15
**Dependencies**: Foundation #1 (Action-Level Safety), Foundation #2 (Memory Firewalls)

---

## Overview

Foundation #7 provides comprehensive audit logging and forensic analysis capabilities for the SafeDeepAgent framework. It captures all security-relevant events from both Phase 1 and Phase 2, enabling:

- Complete action history tracking
- Attack reconstruction and timeline analysis
- Security incident investigation
- Compliance and accountability
- Performance metrics and optimization insights

---

## Architecture

### 3-Component Design

```
┌─────────────────────────────────────────────────────────────┐
│                    SafeDeepAgent                             │
│  (Phase 1 + Phase 2 + Audit Logging)                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Component 1: AuditLogger                                   │
│  - Structured event logging                                 │
│  - Multi-format persistence (JSON, SQLite)                  │
│  - Real-time and batch logging                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Component 2: ForensicAnalyzer                              │
│  - Attack sequence reconstruction                           │
│  - Timeline analysis                                        │
│  - Incident report generation                               │
│  - Pattern correlation across sessions                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Component 3: AuditQueryInterface                           │
│  - Flexible query API                                       │
│  - Time-range filtering                                     │
│  - Event type filtering                                     │
│  - Risk-based searching                                     │
│  - Export capabilities                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. AuditLogger

**Purpose**: Capture all security events with rich context

**Key Features**:
- Structured logging with consistent schema
- Multi-level events (DEBUG, INFO, WARNING, CRITICAL)
- Session and user tracking
- Performance metrics (timestamp, duration)
- Flexible storage backends (JSON files, SQLite, custom)

**Event Types**:
```python
class EventType(Enum):
    # Phase 1 Events
    INPUT_VALIDATION = "input_validation"
    ACTION_AUTHORIZATION = "action_authorization"
    ACTION_BLOCKED = "action_blocked"
    APPROVAL_REQUIRED = "approval_required"

    # Phase 2 Events
    ATTACK_DETECTED = "attack_detected"
    GOAL_DRIFT = "goal_drift"
    ESCALATION_DETECTED = "escalation_detected"
    MEMORY_TAMPERED = "memory_tampered"
    REASONING_ANOMALY = "reasoning_anomaly"

    # Execution Events
    TOOL_EXECUTION = "tool_execution"
    TOOL_SUCCESS = "tool_success"
    TOOL_FAILURE = "tool_failure"

    # Session Events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
```

**Audit Record Schema**:
```python
@dataclass
class AuditRecord:
    # Identity
    record_id: str
    timestamp: datetime
    session_id: str
    user_id: str

    # Event details
    event_type: EventType
    severity: str  # DEBUG, INFO, WARNING, CRITICAL

    # Action context
    step_number: Optional[int]
    tool_name: Optional[str]
    action_type: Optional[str]
    parameters: Dict[str, Any]

    # Security results
    phase1_risk_score: Optional[float]
    phase1_decision: Optional[str]
    phase2_results: Optional[Dict[str, Any]]

    # Outcome
    allowed: bool
    result: Optional[Any]
    error: Optional[str]

    # Performance
    duration_ms: Optional[float]

    # Metadata
    metadata: Dict[str, Any]
```

**Methods**:
```python
class AuditLogger:
    def log_action(self, action_record, phase1_result, phase2_result) -> AuditRecord
    def log_attack_detection(self, attack_result) -> AuditRecord
    def log_security_event(self, event_type, details) -> AuditRecord
    def log_session_start(self, session_id, user_id, task) -> AuditRecord
    def log_session_end(self, session_id, stats) -> AuditRecord
    def flush() -> None  # Force write to storage
    def get_records(self, filters) -> List[AuditRecord]
```

**Storage Backends**:
- **JSONLogger**: Append-only JSON Lines format (default)
- **SQLiteLogger**: SQLite database with indexes
- **CompositeLogger**: Write to multiple backends simultaneously

---

### 2. ForensicAnalyzer

**Purpose**: Reconstruct and analyze security incidents

**Key Features**:
- Attack sequence reconstruction from audit logs
- Timeline visualization of security events
- Incident report generation (markdown, JSON, HTML)
- Pattern correlation across sessions
- Risk trajectory analysis

**Methods**:
```python
class ForensicAnalyzer:
    def reconstruct_attack_sequence(self, attack_id: str) -> AttackReconstruction
    def analyze_session_timeline(self, session_id: str) -> TimelineAnalysis
    def generate_incident_report(self, attack_id: str, format: str) -> str
    def find_similar_attacks(self, attack_id: str) -> List[AttackMatch]
    def analyze_risk_trajectory(self, session_id: str) -> RiskTrajectory
    def correlate_patterns(self, time_range: Tuple[datetime, datetime]) -> PatternCorrelation
```

**Analysis Results**:
```python
@dataclass
class AttackReconstruction:
    attack_id: str
    session_id: str
    attack_pattern: str
    confidence: float

    # Timeline
    start_time: datetime
    end_time: datetime
    duration: timedelta

    # Sequence
    steps: List[AuditRecord]
    critical_steps: List[int]  # Step numbers that triggered detection

    # Context
    original_task: str
    user_id: str

    # Impact
    blocked: bool
    damage_prevented: str

    # Metadata
    detection_method: str
    false_positive_likelihood: float

@dataclass
class TimelineAnalysis:
    session_id: str
    total_actions: int
    total_duration: timedelta

    # Security events
    security_events: List[AuditRecord]
    attacks_detected: int
    escalations_detected: int
    drifts_detected: int

    # Risk progression
    risk_trajectory: List[Tuple[int, float]]  # (step, risk_score)
    peak_risk_step: int
    peak_risk_score: float

    # Visualization
    timeline_chart: str  # ASCII or data for plotting
```

---

### 3. AuditQueryInterface

**Purpose**: Flexible querying and export of audit logs

**Key Features**:
- SQL-like query API for audit records
- Time-range filtering
- Event type filtering
- Risk-based searching
- User/session filtering
- Aggregation and statistics
- Export to multiple formats (JSON, CSV, markdown)

**Query API**:
```python
class AuditQueryInterface:
    def query(self, filters: QueryFilters) -> QueryResult
    def count(self, filters: QueryFilters) -> int
    def aggregate(self, field: str, operation: str, filters: QueryFilters) -> Any
    def export(self, filters: QueryFilters, format: str, output_path: str) -> None

@dataclass
class QueryFilters:
    # Time range
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Identity
    session_ids: Optional[List[str]] = None
    user_ids: Optional[List[str]] = None

    # Event filtering
    event_types: Optional[List[EventType]] = None
    severity_levels: Optional[List[str]] = None

    # Security filtering
    min_risk_score: Optional[float] = None
    only_blocked: bool = False
    only_attacks: bool = False

    # Pagination
    limit: Optional[int] = None
    offset: int = 0

    # Sorting
    sort_by: str = "timestamp"
    sort_order: str = "desc"
```

**Query Examples**:
```python
# Find all attacks in last 24 hours
query.query(QueryFilters(
    start_time=datetime.now() - timedelta(days=1),
    only_attacks=True
))

# Find high-risk actions by specific user
query.query(QueryFilters(
    user_ids=["user_123"],
    min_risk_score=0.7
))

# Find all blocked actions
query.query(QueryFilters(
    only_blocked=True,
    event_types=[EventType.ACTION_BLOCKED]
))

# Export session to CSV
query.export(
    QueryFilters(session_ids=["session_xyz"]),
    format="csv",
    output_path="session_xyz_audit.csv"
)
```

---

## Integration with SafeDeepAgent

### Hook Points

**1. Session Lifecycle**:
```python
class SafeDeepAgent:
    def run(self, task: str):
        # Start logging
        self.audit_logger.log_session_start(self.session_id, self.user_id, task)

        try:
            result = self._execute_task(task)
            stats = self.get_security_stats()
            self.audit_logger.log_session_end(self.session_id, stats)
            return result
        except Exception as e:
            self.audit_logger.log_security_event(EventType.SESSION_END, {"error": str(e)})
            raise
```

**2. Action Execution** (already integrated in Phase 1 & 2):
```python
def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]):
    # Phase 1: Authorization
    decision = self._authorize_action(tool_name, parameters)

    # Phase 2: Multi-step checks
    phase2_result = self._check_phase2_security(tool_name, parameters, decision.risk_score.total_score)

    # NEW: Audit logging
    audit_record = self.audit_logger.log_action(
        action_record=ActionRecord(...),
        phase1_result=decision,
        phase2_result=phase2_result
    )

    # Execute if authorized
    if decision.authorized:
        result = tool_function(**parameters)
        self.audit_logger.log_security_event(EventType.TOOL_SUCCESS, {...})
        return result
    else:
        self.audit_logger.log_security_event(EventType.ACTION_BLOCKED, {...})
        raise UnauthorizedActionError(...)
```

**3. Attack Detection**:
```python
def _check_phase2_security(...):
    attack_result = self.attack_detector.detect_attacks(sequence)

    if attack_result.attack_detected:
        # NEW: Log attack detection
        self.audit_logger.log_attack_detection(attack_result)
        raise SecurityViolationError(...)
```

---

## Configuration

```python
@dataclass
class AuditConfig:
    # Enable/disable
    enable_audit_logging: bool = True

    # Storage
    storage_backend: str = "json"  # json, sqlite, composite
    log_directory: Path = Path("./audit_logs")

    # Retention
    max_log_age_days: int = 90
    auto_cleanup: bool = True

    # Performance
    async_logging: bool = True
    batch_size: int = 100
    flush_interval_seconds: int = 60

    # Privacy
    redact_parameters: bool = False
    redact_results: bool = False

    # Forensics
    enable_forensic_analysis: bool = True
    enable_pattern_correlation: bool = True
```

---

## File Structure

```
deepagent/
├── audit/
│   ├── __init__.py
│   ├── audit_logger.py       # AuditLogger, AuditRecord, storage backends
│   ├── forensic_analyzer.py  # ForensicAnalyzer, attack reconstruction
│   └── query_interface.py    # AuditQueryInterface, query API

audit_logs/                    # Default log directory
├── session_xyz.jsonl         # JSON Lines format
└── audit.db                  # SQLite database (optional)

examples/
└── foundation7_audit_demo.py # Demonstration

tests/
├── test_audit_logger.py
├── test_forensic_analyzer.py
└── test_query_interface.py
```

---

## Success Criteria

- ✅ AuditLogger captures all Phase 1 & Phase 2 events
- ✅ Multi-format storage (JSON, SQLite)
- ✅ ForensicAnalyzer reconstructs attack sequences
- ✅ Incident report generation (markdown, JSON)
- ✅ AuditQueryInterface supports flexible queries
- ✅ Export to CSV, JSON, markdown
- ✅ Integration with SafeDeepAgent
- ✅ Tests pass at >90% rate
- ✅ Working demonstration examples
- ✅ Complete documentation

---

## Performance Targets

- **Logging overhead**: <20ms per action (async mode)
- **Query performance**: <500ms for typical queries
- **Storage efficiency**: <1KB per action record (JSON)
- **Scalability**: Handle 100,000+ records per session

---

## Security Considerations

**Privacy**:
- Optional parameter/result redaction
- PII handling for user_id
- Secure storage of audit logs

**Integrity**:
- Append-only log files
- Optional cryptographic signing of records
- Tamper detection for audit logs themselves

**Access Control**:
- Read-only access to audit logs
- Role-based access for forensic analysis
- Audit log deletion requires admin privileges

---

## Future Enhancements

**Phase 1 (Current)**:
- Basic logging and querying
- Attack reconstruction
- JSON/SQLite storage

**Phase 2 (Future)**:
- Real-time alerting
- Anomaly detection across sessions
- Machine learning for false positive reduction
- Integration with external SIEM systems
- Cryptographic audit log signing
- Distributed audit log aggregation

---

## Next Steps

1. Implement AuditLogger with JSON backend
2. Implement ForensicAnalyzer for attack reconstruction
3. Implement AuditQueryInterface
4. Integrate with SafeDeepAgent
5. Create comprehensive tests
6. Create demonstration examples
7. Write user documentation

---

**Status**: Design Complete ✅
**Ready for Implementation**: Yes
**Estimated Effort**: ~3,000 lines of code + tests + docs
