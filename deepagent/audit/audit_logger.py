"""
Audit Logger - Foundation #7

Comprehensive audit logging for all security events in SafeDeepAgent.
Supports multiple storage backends (JSON, SQLite) and async logging.
"""

import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4
import threading
import queue


class EventType(Enum):
    """Types of security events that can be logged"""

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


class EventSeverity(Enum):
    """Severity levels for audit events"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AuditRecord:
    """
    Comprehensive audit record for all security-relevant events.

    Captures complete context including identity, action details,
    security results, and performance metrics.
    """

    # Identity
    record_id: str
    timestamp: datetime
    session_id: str
    user_id: str

    # Event details
    event_type: EventType
    severity: EventSeverity

    # Action context (optional for some events)
    step_number: Optional[int] = None
    tool_name: Optional[str] = None
    action_type: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Security results
    phase1_risk_score: Optional[float] = None
    phase1_decision: Optional[str] = None  # "authorized" or "blocked"
    phase2_results: Dict[str, Any] = field(default_factory=dict)

    # Outcome
    allowed: bool = True
    result: Optional[Any] = None
    error: Optional[str] = None

    # Performance
    duration_ms: Optional[float] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert enums to strings
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        # Convert datetime to ISO format
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditRecord':
        """Create from dictionary"""
        # Convert strings back to enums
        data['event_type'] = EventType(data['event_type'])
        data['severity'] = EventSeverity(data['severity'])
        # Convert ISO format to datetime
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class AuditConfig:
    """Configuration for audit logging"""

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


class AuditLogger(ABC):
    """
    Abstract base class for audit loggers.

    Defines the interface that all audit logger backends must implement.
    """

    def __init__(self, config: AuditConfig):
        self.config = config
        self._ensure_directory()

    def _ensure_directory(self):
        """Ensure log directory exists"""
        if not self.config.log_directory.exists():
            self.config.log_directory.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def write_record(self, record: AuditRecord) -> None:
        """Write a single audit record to storage"""
        pass

    @abstractmethod
    def read_records(self, filters: Optional[Dict[str, Any]] = None) -> List[AuditRecord]:
        """Read audit records with optional filtering"""
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered records to storage"""
        pass

    @abstractmethod
    def cleanup_old_records(self, max_age_days: int) -> int:
        """Remove records older than max_age_days, return count deleted"""
        pass

    def log_action(
        self,
        session_id: str,
        user_id: str,
        step_number: int,
        tool_name: str,
        action_type: str,
        parameters: Dict[str, Any],
        phase1_result: Optional[Any] = None,
        phase2_result: Optional[Dict[str, Any]] = None,
        allowed: bool = True,
        result: Optional[Any] = None,
        error: Optional[str] = None,
        duration_ms: Optional[float] = None
    ) -> AuditRecord:
        """Log an action execution with security results"""

        # Determine event type and severity
        if not allowed:
            event_type = EventType.ACTION_BLOCKED
            severity = EventSeverity.WARNING
        elif error:
            event_type = EventType.TOOL_FAILURE
            severity = EventSeverity.WARNING
        else:
            event_type = EventType.TOOL_SUCCESS
            severity = EventSeverity.INFO

        # Extract Phase 1 results
        phase1_risk_score = None
        phase1_decision = None
        if phase1_result:
            if hasattr(phase1_result, 'risk_score'):
                if hasattr(phase1_result.risk_score, 'total_score'):
                    phase1_risk_score = phase1_result.risk_score.total_score
                else:
                    phase1_risk_score = phase1_result.risk_score
            if hasattr(phase1_result, 'authorized'):
                phase1_decision = "authorized" if phase1_result.authorized else "blocked"

        # Redact sensitive data if configured
        if self.config.redact_parameters:
            parameters = {"[REDACTED]": len(parameters)}
        if self.config.redact_results:
            result = "[REDACTED]" if result else None

        # Create audit record
        record = AuditRecord(
            record_id=str(uuid4()),
            timestamp=datetime.now(),
            session_id=session_id,
            user_id=user_id,
            event_type=event_type,
            severity=severity,
            step_number=step_number,
            tool_name=tool_name,
            action_type=action_type,
            parameters=parameters,
            phase1_risk_score=phase1_risk_score,
            phase1_decision=phase1_decision,
            phase2_results=phase2_result or {},
            allowed=allowed,
            result=result,
            error=error,
            duration_ms=duration_ms
        )

        self.write_record(record)
        return record

    def log_attack_detection(
        self,
        session_id: str,
        user_id: str,
        attack_result: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditRecord:
        """Log a Phase 2 attack detection event"""

        # Extract attack details
        attack_details = {
            "attack_detected": True,
            "pattern_id": getattr(attack_result.most_likely_pattern, 'pattern_id', None) if hasattr(attack_result, 'most_likely_pattern') else None,
            "pattern_name": getattr(attack_result.most_likely_pattern, 'name', None) if hasattr(attack_result, 'most_likely_pattern') else None,
            "confidence": getattr(attack_result, 'highest_confidence', None),
            "severity": getattr(attack_result.most_likely_pattern, 'severity', None).value if hasattr(attack_result, 'most_likely_pattern') and hasattr(attack_result.most_likely_pattern, 'severity') else None,
            "category": getattr(attack_result.most_likely_pattern, 'attack_category', None).value if hasattr(attack_result, 'most_likely_pattern') and hasattr(attack_result.most_likely_pattern, 'attack_category') else None,
        }

        record = AuditRecord(
            record_id=str(uuid4()),
            timestamp=datetime.now(),
            session_id=session_id,
            user_id=user_id,
            event_type=EventType.ATTACK_DETECTED,
            severity=EventSeverity.CRITICAL,
            allowed=False,
            phase2_results=attack_details,
            metadata=metadata or {}
        )

        self.write_record(record)
        return record

    def log_security_event(
        self,
        session_id: str,
        user_id: str,
        event_type: EventType,
        severity: EventSeverity,
        details: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditRecord:
        """Log a generic security event"""

        record = AuditRecord(
            record_id=str(uuid4()),
            timestamp=datetime.now(),
            session_id=session_id,
            user_id=user_id,
            event_type=event_type,
            severity=severity,
            metadata={**(metadata or {}), **details}
        )

        self.write_record(record)
        return record

    def log_session_start(
        self,
        session_id: str,
        user_id: str,
        task: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditRecord:
        """Log session start event"""

        record = AuditRecord(
            record_id=str(uuid4()),
            timestamp=datetime.now(),
            session_id=session_id,
            user_id=user_id,
            event_type=EventType.SESSION_START,
            severity=EventSeverity.INFO,
            metadata={**(metadata or {}), "task": task}
        )

        self.write_record(record)
        return record

    def log_session_end(
        self,
        session_id: str,
        user_id: str,
        stats: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditRecord:
        """Log session end event with statistics"""

        record = AuditRecord(
            record_id=str(uuid4()),
            timestamp=datetime.now(),
            session_id=session_id,
            user_id=user_id,
            event_type=EventType.SESSION_END,
            severity=EventSeverity.INFO,
            metadata={**(metadata or {}), "stats": stats}
        )

        self.write_record(record)
        return record


class JSONAuditLogger(AuditLogger):
    """
    JSON Lines audit logger.

    Writes audit records to JSON Lines files (one JSON object per line).
    Fast, simple, and human-readable.
    """

    def __init__(self, config: AuditConfig):
        super().__init__(config)
        self.log_file = self.config.log_directory / "audit.jsonl"
        self._lock = threading.Lock()

        # Async logging support
        if self.config.async_logging:
            self._shutdown = False
            self._queue: queue.Queue = queue.Queue(maxsize=1000)
            self._worker_thread = threading.Thread(target=self._async_worker, daemon=True)
            self._worker_thread.start()

    def write_record(self, record: AuditRecord) -> None:
        """Write audit record to JSON Lines file"""
        if self.config.async_logging:
            self._queue.put(record)
        else:
            self._write_sync(record)

    def _write_sync(self, record: AuditRecord) -> None:
        """Synchronously write record to file"""
        with self._lock:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                json.dump(record.to_dict(), f)
                f.write('\n')

    def _async_worker(self):
        """Background worker for async logging"""
        batch = []
        last_flush = datetime.now()

        while not self._shutdown:
            try:
                # Get record with timeout
                record = self._queue.get(timeout=1.0)
                batch.append(record)

                # Flush if batch is full or interval elapsed
                now = datetime.now()
                should_flush = (
                    len(batch) >= self.config.batch_size or
                    (now - last_flush).total_seconds() >= self.config.flush_interval_seconds
                )

                if should_flush:
                    self._flush_batch(batch)
                    batch.clear()
                    last_flush = now

            except queue.Empty:
                # Timeout - flush any pending records
                if batch:
                    self._flush_batch(batch)
                    batch.clear()
                    last_flush = datetime.now()

        # Final flush on shutdown
        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: List[AuditRecord]):
        """Flush a batch of records to file"""
        with self._lock:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                for record in batch:
                    json.dump(record.to_dict(), f)
                    f.write('\n')

    def read_records(self, filters: Optional[Dict[str, Any]] = None) -> List[AuditRecord]:
        """Read audit records from JSON Lines file"""
        records = []

        if not self.log_file.exists():
            return records

        with self._lock:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        record = AuditRecord.from_dict(data)

                        # Apply filters
                        if filters:
                            if not self._matches_filters(record, filters):
                                continue

                        records.append(record)

        return records

    def _matches_filters(self, record: AuditRecord, filters: Dict[str, Any]) -> bool:
        """Check if record matches all filters"""
        for key, value in filters.items():
            if key == 'session_id' and record.session_id != value:
                return False
            if key == 'user_id' and record.user_id != value:
                return False
            if key == 'event_type' and record.event_type != value:
                return False
            if key == 'min_risk_score' and (record.phase1_risk_score is None or record.phase1_risk_score < value):
                return False
            if key == 'start_time' and record.timestamp < value:
                return False
            if key == 'end_time' and record.timestamp > value:
                return False
        return True

    def flush(self) -> None:
        """Flush any buffered records"""
        if self.config.async_logging:
            # Wait for queue to empty
            self._queue.join()

    def cleanup_old_records(self, max_age_days: int) -> int:
        """Remove records older than max_age_days"""
        cutoff = datetime.now() - timedelta(days=max_age_days)
        records = self.read_records()

        # Keep only recent records
        recent_records = [r for r in records if r.timestamp >= cutoff]
        removed_count = len(records) - len(recent_records)

        # Rewrite file with recent records only
        if removed_count > 0:
            with self._lock:
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    for record in recent_records:
                        json.dump(record.to_dict(), f)
                        f.write('\n')

        return removed_count

    def shutdown(self):
        """Shutdown async logging"""
        if self.config.async_logging:
            self._shutdown = True
            self._worker_thread.join(timeout=5.0)


class SQLiteAuditLogger(AuditLogger):
    """
    SQLite audit logger.

    Writes audit records to SQLite database with indexes for fast querying.
    Better for large-scale deployments and complex queries.
    """

    def __init__(self, config: AuditConfig):
        super().__init__(config)
        self.db_path = self.config.log_directory / "audit.db"
        self._init_database()
        self._lock = threading.Lock()

    def _init_database(self):
        """Initialize SQLite database and tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_records (
                record_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                step_number INTEGER,
                tool_name TEXT,
                action_type TEXT,
                parameters TEXT,
                phase1_risk_score REAL,
                phase1_decision TEXT,
                phase2_results TEXT,
                allowed INTEGER NOT NULL,
                result TEXT,
                error TEXT,
                duration_ms REAL,
                metadata TEXT
            )
        """)

        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session ON audit_records(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user ON audit_records(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_records(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_records(event_type)")

        conn.commit()
        conn.close()

    def write_record(self, record: AuditRecord) -> None:
        """Write audit record to SQLite database"""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO audit_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.record_id,
                record.timestamp.isoformat(),
                record.session_id,
                record.user_id,
                record.event_type.value,
                record.severity.value,
                record.step_number,
                record.tool_name,
                record.action_type,
                json.dumps(record.parameters),
                record.phase1_risk_score,
                record.phase1_decision,
                json.dumps(record.phase2_results),
                1 if record.allowed else 0,
                json.dumps(record.result) if record.result else None,
                record.error,
                record.duration_ms,
                json.dumps(record.metadata)
            ))

            conn.commit()
            conn.close()

    def read_records(self, filters: Optional[Dict[str, Any]] = None) -> List[AuditRecord]:
        """Read audit records from SQLite database"""
        query = "SELECT * FROM audit_records"
        params = []

        # Build WHERE clause from filters
        if filters:
            conditions = []
            if 'session_id' in filters:
                conditions.append("session_id = ?")
                params.append(filters['session_id'])
            if 'user_id' in filters:
                conditions.append("user_id = ?")
                params.append(filters['user_id'])
            if 'event_type' in filters:
                conditions.append("event_type = ?")
                params.append(filters['event_type'].value if isinstance(filters['event_type'], EventType) else filters['event_type'])
            if 'start_time' in filters:
                conditions.append("timestamp >= ?")
                params.append(filters['start_time'].isoformat())
            if 'end_time' in filters:
                conditions.append("timestamp <= ?")
                params.append(filters['end_time'].isoformat())
            if 'min_risk_score' in filters:
                conditions.append("phase1_risk_score >= ?")
                params.append(filters['min_risk_score'])

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp DESC"

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

        # Convert rows to AuditRecord objects
        records = []
        for row in rows:
            record = AuditRecord(
                record_id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                session_id=row[2],
                user_id=row[3],
                event_type=EventType(row[4]),
                severity=EventSeverity(row[5]),
                step_number=row[6],
                tool_name=row[7],
                action_type=row[8],
                parameters=json.loads(row[9]) if row[9] else {},
                phase1_risk_score=row[10],
                phase1_decision=row[11],
                phase2_results=json.loads(row[12]) if row[12] else {},
                allowed=bool(row[13]),
                result=json.loads(row[14]) if row[14] else None,
                error=row[15],
                duration_ms=row[16],
                metadata=json.loads(row[17]) if row[17] else {}
            )
            records.append(record)

        return records

    def flush(self) -> None:
        """SQLite flushes automatically on commit"""
        pass

    def cleanup_old_records(self, max_age_days: int) -> int:
        """Remove records older than max_age_days"""
        cutoff = datetime.now() - timedelta(days=max_age_days)

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM audit_records WHERE timestamp < ?", (cutoff.isoformat(),))
            count = cursor.fetchone()[0]

            cursor.execute("DELETE FROM audit_records WHERE timestamp < ?", (cutoff.isoformat(),))
            conn.commit()
            conn.close()

        return count


class CompositeAuditLogger(AuditLogger):
    """
    Composite audit logger that writes to multiple backends simultaneously.

    Example: Write to both JSON (for human readability) and SQLite (for queries).
    """

    def __init__(self, config: AuditConfig, loggers: List[AuditLogger]):
        super().__init__(config)
        self.loggers = loggers

    def write_record(self, record: AuditRecord) -> None:
        """Write to all loggers"""
        for logger in self.loggers:
            logger.write_record(record)

    def read_records(self, filters: Optional[Dict[str, Any]] = None) -> List[AuditRecord]:
        """Read from first logger (typically SQLite for performance)"""
        if self.loggers:
            return self.loggers[0].read_records(filters)
        return []

    def flush(self) -> None:
        """Flush all loggers"""
        for logger in self.loggers:
            logger.flush()

    def cleanup_old_records(self, max_age_days: int) -> int:
        """Cleanup in all loggers"""
        total_removed = 0
        for logger in self.loggers:
            total_removed += logger.cleanup_old_records(max_age_days)
        return total_removed
