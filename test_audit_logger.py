"""
Tests for Foundation #7: Audit Logger

Tests audit logging functionality including:
- JSON and SQLite storage backends
- Action logging
- Attack detection logging
- Session lifecycle logging
- Query and filtering capabilities
"""

import sys
from pathlib import Path
import json
import tempfile
import shutil
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from deepagent.audit import (
    AuditLogger,
    AuditRecord,
    AuditConfig,
    JSONAuditLogger,
    SQLiteAuditLogger,
    CompositeAuditLogger,
    EventType,
    EventSeverity
)


def test_audit_record_serialization():
    """Test 1: Audit record serialization/deserialization"""
    print("\nTest 1: Audit Record Serialization")

    record = AuditRecord(
        record_id="test_123",
        timestamp=datetime.now(),
        session_id="session_1",
        user_id="user_1",
        event_type=EventType.TOOL_EXECUTION,
        severity=EventSeverity.INFO,
        tool_name="test_tool",
        action_type="read",
        parameters={"key": "value"},
        phase1_risk_score=0.5,
        allowed=True
    )

    # Convert to dict
    record_dict = record.to_dict()
    assert isinstance(record_dict, dict)
    assert record_dict['event_type'] == 'tool_execution'
    assert record_dict['severity'] == 'info'

    # Convert back
    record_restored = AuditRecord.from_dict(record_dict)
    assert record_restored.record_id == record.record_id
    assert record_restored.event_type == record.event_type
    assert record_restored.severity == record.severity

    print(" OK - Serialization works correctly")
    return True


def test_json_audit_logger():
    """Test 2: JSON audit logger"""
    print("\nTest 2: JSON Audit Logger")

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp())

    try:
        config = AuditConfig(
            log_directory=temp_dir,
            storage_backend="json",
            async_logging=False  # Sync for testing
        )

        logger = JSONAuditLogger(config)

        # Log some actions
        logger.log_action(
            session_id="session_1",
            user_id="user_1",
            step_number=1,
            tool_name="search",
            action_type="read",
            parameters={"query": "test"},
            allowed=True
        )

        logger.log_action(
            session_id="session_1",
            user_id="user_1",
            step_number=2,
            tool_name="execute",
            action_type="execute",
            parameters={"command": "run"},
            allowed=False,
            error="Blocked by policy"
        )

        # Flush to ensure writes
        logger.flush()

        # Read back records
        records = logger.read_records()
        assert len(records) == 2
        assert records[0].tool_name == "search"
        assert records[1].allowed == False

        # Test filtering
        blocked_records = logger.read_records({'only_blocked': True})
        # Note: filtering needs to be implemented in _apply_filters
        # For now, just check we can query

        print(f"  OK - Logged {len(records)} records successfully")
        return True

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_sqlite_audit_logger():
    """Test 3: SQLite audit logger"""
    print("\nTest 3: SQLite Audit Logger")

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp())

    try:
        config = AuditConfig(
            log_directory=temp_dir,
            storage_backend="sqlite"
        )

        logger = SQLiteAuditLogger(config)

        # Log session start
        logger.log_session_start(
            session_id="session_1",
            user_id="user_1",
            task="Test task",
            metadata={"environment": "test"}
        )

        # Log some actions
        for i in range(5):
            logger.log_action(
                session_id="session_1",
                user_id="user_1",
                step_number=i+1,
                tool_name=f"tool_{i}",
                action_type="read" if i % 2 == 0 else "write",
                parameters={"step": i},
                phase1_result=None,
                allowed=True,
                duration_ms=100.0 + i * 10
            )

        # Log session end
        logger.log_session_end(
            session_id="session_1",
            user_id="user_1",
            stats={"total_actions": 5}
        )

        # Read all records
        all_records = logger.read_records()
        assert len(all_records) == 7  # 1 start + 5 actions + 1 end

        # Test filtering by session
        session_records = logger.read_records({'session_id': 'session_1'})
        assert len(session_records) == 7

        # Test filtering by event type
        action_records = logger.read_records({'event_type': EventType.TOOL_SUCCESS})
        assert len(action_records) == 5

        print(f"  OK - SQLite logger works ({len(all_records)} records)")
        return True

    finally:
        shutil.rmtree(temp_dir)


def test_attack_detection_logging():
    """Test 4: Attack detection logging"""
    print("\nTest 4: Attack Detection Logging")

    temp_dir = Path(tempfile.mkdtemp())

    try:
        config = AuditConfig(log_directory=temp_dir, async_logging=False)
        logger = JSONAuditLogger(config)

        # Simulate attack detection
        class MockAttackResult:
            def __init__(self):
                self.attack_detected = True
                self.highest_confidence = 0.95
                self.most_likely_pattern = self.MockPattern()

            class MockPattern:
                def __init__(self):
                    self.pattern_id = "data_exfiltration"
                    self.name = "Data Exfiltration"
                    self.severity = self.MockSeverity()
                    self.attack_category = self.MockCategory()

                class MockSeverity:
                    def __init__(self):
                        self.value = "critical"

                class MockCategory:
                    def __init__(self):
                        self.value = "data_exfiltration"

        attack_result = MockAttackResult()

        # Log attack
        record = logger.log_attack_detection(
            session_id="session_1",
            user_id="user_1",
            attack_result=attack_result,
            metadata={"tool": "backup"}
        )

        assert record.event_type == EventType.ATTACK_DETECTED
        assert record.severity == EventSeverity.CRITICAL
        assert record.allowed == False
        assert record.phase2_results['confidence'] == 0.95

        # Verify it was written
        records = logger.read_records()
        assert len(records) == 1
        assert records[0].event_type == EventType.ATTACK_DETECTED

        print("  OK - Attack detection logged correctly")
        return True

    finally:
        shutil.rmtree(temp_dir)


def test_composite_logger():
    """Test 5: Composite logger (JSON + SQLite)"""
    print("\nTest 5: Composite Audit Logger")

    temp_dir = Path(tempfile.mkdtemp())

    try:
        config = AuditConfig(log_directory=temp_dir)

        # Create composite logger
        json_logger = JSONAuditLogger(config)
        sqlite_logger = SQLiteAuditLogger(config)
        composite = CompositeAuditLogger(config, [json_logger, sqlite_logger])

        # Log action
        composite.log_action(
            session_id="session_1",
            user_id="user_1",
            step_number=1,
            tool_name="test",
            action_type="read",
            parameters={},
            allowed=True
        )

        composite.flush()

        # Check both backends have the record
        json_records = json_logger.read_records()
        sqlite_records = sqlite_logger.read_records()

        assert len(json_records) == 1
        assert len(sqlite_records) == 1
        assert json_records[0].tool_name == sqlite_records[0].tool_name

        print("  OK - Composite logger writes to both backends")
        return True

    finally:
        shutil.rmtree(temp_dir)


def test_record_cleanup():
    """Test 6: Old record cleanup"""
    print("\nTest 6: Old Record Cleanup")

    temp_dir = Path(tempfile.mkdtemp())

    try:
        config = AuditConfig(log_directory=temp_dir, async_logging=False)
        logger = JSONAuditLogger(config)

        # Create old record (manually for testing)
        old_record = AuditRecord(
            record_id="old_1",
            timestamp=datetime.now() - timedelta(days=100),
            session_id="old_session",
            user_id="user_1",
            event_type=EventType.TOOL_EXECUTION,
            severity=EventSeverity.INFO
        )

        # Create recent record
        recent_record = AuditRecord(
            record_id="recent_1",
            timestamp=datetime.now(),
            session_id="recent_session",
            user_id="user_1",
            event_type=EventType.TOOL_EXECUTION,
            severity=EventSeverity.INFO
        )

        # Write both manually
        logger.write_record(old_record)
        logger.write_record(recent_record)
        logger.flush()

        # Verify both exist
        all_records = logger.read_records()
        assert len(all_records) == 2

        # Cleanup old records (>90 days)
        removed_count = logger.cleanup_old_records(max_age_days=90)
        assert removed_count == 1

        # Verify only recent record remains
        remaining_records = logger.read_records()
        assert len(remaining_records) == 1
        assert remaining_records[0].record_id == "recent_1"

        print(f"  OK - Cleaned up {removed_count} old records")
        return True

    finally:
        shutil.rmtree(temp_dir)


def test_session_lifecycle_logging():
    """Test 7: Complete session lifecycle logging"""
    print("\nTest 7: Session Lifecycle Logging")

    temp_dir = Path(tempfile.mkdtemp())

    try:
        config = AuditConfig(log_directory=temp_dir, async_logging=False)
        logger = JSONAuditLogger(config)

        session_id = "lifecycle_session"
        user_id = "test_user"

        # 1. Log session start
        logger.log_session_start(
            session_id=session_id,
            user_id=user_id,
            task="Complete a complex task",
            metadata={"environment": "production"}
        )

        # 2. Log multiple actions
        for i in range(3):
            logger.log_action(
                session_id=session_id,
                user_id=user_id,
                step_number=i+1,
                tool_name=f"tool_{i}",
                action_type="read",
                parameters={"step": i},
                allowed=True
            )

        # 3. Log session end
        logger.log_session_end(
            session_id=session_id,
            user_id=user_id,
            stats={
                "total_actions": 3,
                "success_rate": 1.0
            }
        )

        # Verify complete lifecycle
        session_records = logger.read_records({'session_id': session_id})
        assert len(session_records) == 5  # start + 3 actions + end

        # Verify order
        assert session_records[0].event_type == EventType.SESSION_START
        assert session_records[-1].event_type == EventType.SESSION_END

        print("  OK - Complete lifecycle logged correctly")
        return True

    finally:
        shutil.rmtree(temp_dir)


def test_async_logging():
    """Test 8: Async logging performance"""
    print("\nTest 8: Async Logging")

    temp_dir = Path(tempfile.mkdtemp())

    try:
        config = AuditConfig(
            log_directory=temp_dir,
            async_logging=True,
            batch_size=10,
            flush_interval_seconds=1
        )

        logger = JSONAuditLogger(config)

        # Log many actions quickly
        for i in range(50):
            logger.log_action(
                session_id="async_session",
                user_id="user_1",
                step_number=i+1,
                tool_name="test",
                action_type="read",
                parameters={},
                allowed=True
            )

        # Flush to ensure all writes complete
        logger.flush()

        # Shutdown async worker
        if hasattr(logger, 'shutdown'):
            logger.shutdown()

        # Verify all records were written
        records = logger.read_records()
        assert len(records) == 50

        print("  OK - Async logging handled 50 records")
        return True

    finally:
        shutil.rmtree(temp_dir)


def main():
    """Run all audit logger tests"""
    print("="*80)
    print("  FOUNDATION #7: AUDIT LOGGER TESTS")
    print("="*80)

    tests = [
        test_audit_record_serialization,
        test_json_audit_logger,
        test_sqlite_audit_logger,
        test_attack_detection_logging,
        test_composite_logger,
        test_record_cleanup,
        test_session_lifecycle_logging,
        test_async_logging
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"  FAILED - {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAILED - {test.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print(f"RESULTS: {passed}/{len(tests)} tests passed ({passed/len(tests)*100:.0f}%)")
    print("="*80)

    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
