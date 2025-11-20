"""
Tests for Foundation #7: Forensic Analyzer and Query Interface

Tests forensic analysis and query capabilities including:
- Attack sequence reconstruction
- Timeline analysis
- Pattern correlation
- Flexible querying
- Export capabilities
"""

import sys
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from deepagent.audit import (
    AuditConfig,
    JSONAuditLogger,
    ForensicAnalyzer,
    AuditQueryInterface,
    QueryFilters,
    EventType,
    EventSeverity
)


def setup_test_data(logger):
    """Create test audit data for forensic analysis"""
    session_id = "forensic_session_1"
    user_id = "attacker_user"

    # Log session start
    logger.log_session_start(
        session_id=session_id,
        user_id=user_id,
        task="Analyze user data",
        metadata={}
    )

    # Log normal actions
    logger.log_action(
        session_id=session_id,
        user_id=user_id,
        step_number=1,
        tool_name="read_user",
        action_type="read",
        parameters={"user_id": "123"},
        phase1_result=None,
        allowed=True
    )

    logger.log_action(
        session_id=session_id,
        user_id=user_id,
        step_number=2,
        tool_name="backup_data",
        action_type="read",
        parameters={"source": "database"},
        phase1_result=None,
        allowed=True
    )

    logger.log_action(
        session_id=session_id,
        user_id=user_id,
        step_number=3,
        tool_name="export_to_external",
        action_type="write",
        parameters={"destination": "external_server"},
        phase1_result=None,
        allowed=True
    )

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

    # Log attack detection
    attack_record = logger.log_attack_detection(
        session_id=session_id,
        user_id=user_id,
        attack_result=attack_result,
        metadata={"step": 4}
    )

    # Log session end
    logger.log_session_end(
        session_id=session_id,
        user_id=user_id,
        stats={"blocked": True}
    )

    return session_id, attack_record.record_id


def test_attack_reconstruction():
    """Test 1: Attack sequence reconstruction"""
    print("\nTest 1: Attack Sequence Reconstruction")

    temp_dir = Path(tempfile.mkdtemp())

    try:
        config = AuditConfig(log_directory=temp_dir, async_logging=False)
        logger = JSONAuditLogger(config)

        # Setup test data
        session_id, attack_id = setup_test_data(logger)

        # Create forensic analyzer
        analyzer = ForensicAnalyzer(logger)

        # Reconstruct attack
        reconstruction = analyzer.reconstruct_attack_sequence(attack_id)

        assert reconstruction is not None
        assert reconstruction.attack_id == attack_id
        assert reconstruction.session_id == session_id
        assert reconstruction.attack_pattern == "Data Exfiltration"
        assert reconstruction.confidence == 0.95
        assert reconstruction.blocked == True
        assert len(reconstruction.steps) == 3  # 3 actions before attack detection

        print(f"  OK - Reconstructed attack with {len(reconstruction.steps)} steps")
        print(f"       Pattern: {reconstruction.attack_pattern} ({reconstruction.confidence:.0%})")
        return True

    finally:
        shutil.rmtree(temp_dir)


def test_timeline_analysis():
    """Test 2: Session timeline analysis"""
    print("\nTest 2: Timeline Analysis")

    temp_dir = Path(tempfile.mkdtemp())

    try:
        config = AuditConfig(log_directory=temp_dir, async_logging=False)
        logger = JSONAuditLogger(config)

        # Setup test data
        session_id, _ = setup_test_data(logger)

        # Create forensic analyzer
        analyzer = ForensicAnalyzer(logger)

        # Analyze timeline
        timeline = analyzer.analyze_session_timeline(session_id)

        assert timeline is not None
        assert timeline.session_id == session_id
        assert timeline.total_actions == 3
        assert timeline.attacks_detected == 1
        assert timeline.successful_actions == 3  # All allowed before attack blocked

        # Check risk trajectory
        assert timeline.risk_trajectory is not None
        # Note: Risk scores would be 0 since we didn't set phase1_result

        print(f"  OK - Timeline: {timeline.total_actions} actions, {timeline.attacks_detected} attacks")
        return True

    finally:
        shutil.rmtree(temp_dir)


def test_incident_report_generation():
    """Test 3: Incident report generation"""
    print("\nTest 3: Incident Report Generation")

    temp_dir = Path(tempfile.mkdtemp())

    try:
        config = AuditConfig(log_directory=temp_dir, async_logging=False)
        logger = JSONAuditLogger(config)

        # Setup test data
        _, attack_id = setup_test_data(logger)

        # Create forensic analyzer
        analyzer = ForensicAnalyzer(logger)

        # Generate markdown report
        markdown_report = analyzer.generate_incident_report(attack_id, format="markdown")
        assert "Attack Incident Report" in markdown_report
        assert "Data Exfiltration" in markdown_report
        assert "BLOCKED" in markdown_report

        # Generate JSON report
        json_report = analyzer.generate_incident_report(attack_id, format="json")
        assert "attack_id" in json_report
        assert "confidence" in json_report

        # Generate text report
        text_report = analyzer.generate_incident_report(attack_id, format="text")
        assert "Attack ID" in text_report

        print("  OK - Generated reports in markdown, JSON, and text formats")
        return True

    finally:
        shutil.rmtree(temp_dir)


def test_risk_trajectory():
    """Test 4: Risk trajectory analysis"""
    print("\nTest 4: Risk Trajectory Analysis")

    temp_dir = Path(tempfile.mkdtemp())

    try:
        config = AuditConfig(log_directory=temp_dir, async_logging=False)
        logger = JSONAuditLogger(config)

        session_id = "risk_session"

        # Log session with increasing risk
        logger.log_session_start(session_id=session_id, user_id="user_1", task="test")

        # Mock policy decisions with increasing risk
        class MockRiskScore:
            def __init__(self, score):
                self.total_score = score

        class MockDecision:
            def __init__(self, score):
                self.risk_score = MockRiskScore(score)

        for i in range(5):
            risk = 0.1 + (i * 0.2)  # 0.1, 0.3, 0.5, 0.7, 0.9
            decision = MockDecision(risk)

            logger.log_action(
                session_id=session_id,
                user_id="user_1",
                step_number=i+1,
                tool_name=f"tool_{i}",
                action_type="read",
                parameters={},
                phase1_result=decision,
                allowed=True
            )

        logger.log_session_end(session_id=session_id, user_id="user_1", stats={})

        # Analyze risk trajectory
        analyzer = ForensicAnalyzer(logger)
        trajectory = analyzer.analyze_risk_trajectory(session_id)

        assert trajectory is not None
        assert len(trajectory.risk_scores) == 5
        assert trajectory.peak_risk_score == 0.9
        assert trajectory.peak_risk_step == 5
        assert trajectory.escalation_rate > 0  # Positive escalation

        # Generate ASCII chart
        chart = trajectory.to_ascii_chart()
        assert "Risk Trajectory" in chart
        assert "Peak" in chart

        print(f"  OK - Risk trajectory: {trajectory.average_risk:.0%} avg, {trajectory.escalation_rate:+.1%}/step")
        return True

    finally:
        shutil.rmtree(temp_dir)


def test_pattern_correlation():
    """Test 5: Pattern correlation across sessions"""
    print("\nTest 5: Pattern Correlation")

    temp_dir = Path(tempfile.mkdtemp())

    try:
        config = AuditConfig(log_directory=temp_dir, async_logging=False)
        logger = JSONAuditLogger(config)

        # Create multiple sessions with attacks
        for i in range(3):
            session_id = f"session_{i}"
            user_id = f"user_{i % 2}"  # 2 users

            logger.log_session_start(session_id=session_id, user_id=user_id, task="test")

            # Mock attack
            class MockAttackResult:
                def __init__(self, pattern_name):
                    self.attack_detected = True
                    self.highest_confidence = 0.9
                    self.most_likely_pattern = self.MockPattern(pattern_name)

                class MockPattern:
                    def __init__(self, name):
                        self.pattern_id = name.lower().replace(" ", "_")
                        self.name = name
                        self.severity = self.MockSeverity()
                        self.attack_category = self.MockCategory()

                    class MockSeverity:
                        def __init__(self):
                            self.value = "high"

                    class MockCategory:
                        def __init__(self):
                            self.value = "escalation"

            pattern = "Privilege Escalation" if i < 2 else "Data Exfiltration"
            attack_result = MockAttackResult(pattern)

            logger.log_attack_detection(
                session_id=session_id,
                user_id=user_id,
                attack_result=attack_result
            )

            logger.log_session_end(session_id=session_id, user_id=user_id, stats={})

        # Correlate patterns
        analyzer = ForensicAnalyzer(logger)
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now() + timedelta(hours=1)

        correlation = analyzer.correlate_patterns((start_time, end_time))

        assert correlation.total_sessions == 3
        assert correlation.sessions_with_attacks == 3
        assert len(correlation.pattern_counts) == 2  # 2 different patterns
        assert correlation.pattern_counts["Privilege Escalation"] == 2
        assert correlation.pattern_counts["Data Exfiltration"] == 1

        # Check user statistics
        assert len(correlation.users_with_attacks) == 2  # 2 users

        print(f"  OK - Correlated {sum(correlation.pattern_counts.values())} attacks across {correlation.total_sessions} sessions")
        return True

    finally:
        shutil.rmtree(temp_dir)


def test_query_interface_filtering():
    """Test 6: Query interface filtering"""
    print("\nTest 6: Query Interface Filtering")

    temp_dir = Path(tempfile.mkdtemp())

    try:
        config = AuditConfig(log_directory=temp_dir, async_logging=False)
        logger = JSONAuditLogger(config)

        # Create test data
        for i in range(10):
            logger.log_action(
                session_id=f"session_{i % 2}",
                user_id=f"user_{i % 3}",
                step_number=i+1,
                tool_name="test_tool",
                action_type="read",
                parameters={},
                allowed=(i % 2 == 0)  # Half blocked
            )

        # Create query interface
        query = AuditQueryInterface(logger)

        # Test 1: Filter by session
        result = query.query(QueryFilters(session_ids=["session_0"]))
        assert result.total_count == 5

        # Test 2: Filter blocked actions
        result = query.query(QueryFilters(only_blocked=True))
        assert result.total_count == 5

        # Test 3: Pagination
        result = query.query(QueryFilters(limit=3, offset=0))
        assert len(result.records) == 3
        assert result.has_more == True

        # Test 4: Count
        count = query.count(QueryFilters())
        assert count == 10

        print(f"  OK - Query interface filtered {count} total records")
        return True

    finally:
        shutil.rmtree(temp_dir)


def test_query_statistics():
    """Test 7: Query statistics aggregation"""
    print("\nTest 7: Query Statistics")

    temp_dir = Path(tempfile.mkdtemp())

    try:
        config = AuditConfig(log_directory=temp_dir, async_logging=False)
        logger = JSONAuditLogger(config)

        # Create diverse test data
        for i in range(20):
            logger.log_action(
                session_id="stats_session",
                user_id="user_1",
                step_number=i+1,
                tool_name=f"tool_{i % 3}",
                action_type=["read", "write", "execute"][i % 3],
                parameters={},
                allowed=(i < 15),  # 15 allowed, 5 blocked
                duration_ms=50.0 + i * 5
            )

        # Get statistics
        query = AuditQueryInterface(logger)
        stats = query.statistics(QueryFilters())

        assert stats['total_records'] == 20
        assert stats['unique_sessions'] == 1
        assert stats['security']['total_blocked'] == 5
        assert 'performance' in stats
        assert 'top_tools' in stats

        print(f"  OK - Statistics: {stats['total_records']} records, {stats['security']['total_blocked']} blocked")
        return True

    finally:
        shutil.rmtree(temp_dir)


def test_query_export():
    """Test 8: Query export to files"""
    print("\nTest 8: Query Export")

    temp_dir = Path(tempfile.mkdtemp())

    try:
        config = AuditConfig(log_directory=temp_dir, async_logging=False)
        logger = JSONAuditLogger(config)

        # Create test data
        for i in range(5):
            logger.log_action(
                session_id="export_session",
                user_id="user_1",
                step_number=i+1,
                tool_name="test",
                action_type="read",
                parameters={},
                allowed=True
            )

        # Create query interface
        query = AuditQueryInterface(logger)

        # Export to JSON
        json_path = temp_dir / "export.json"
        query.export(QueryFilters(), format="json", output_path=str(json_path))
        assert json_path.exists()

        # Export to CSV
        csv_path = temp_dir / "export.csv"
        query.export(QueryFilters(), format="csv", output_path=str(csv_path))
        assert csv_path.exists()

        # Export to Markdown
        md_path = temp_dir / "export.md"
        query.export(QueryFilters(), format="markdown", output_path=str(md_path))
        assert md_path.exists()

        # Export to Text
        txt_path = temp_dir / "export.txt"
        query.export(QueryFilters(), format="text", output_path=str(txt_path))
        assert txt_path.exists()

        print("  OK - Exported to JSON, CSV, Markdown, and Text")
        return True

    finally:
        shutil.rmtree(temp_dir)


def main():
    """Run all forensic analyzer and query interface tests"""
    print("="*80)
    print("  FOUNDATION #7: FORENSIC ANALYZER & QUERY INTERFACE TESTS")
    print("="*80)

    tests = [
        test_attack_reconstruction,
        test_timeline_analysis,
        test_incident_report_generation,
        test_risk_trajectory,
        test_pattern_correlation,
        test_query_interface_filtering,
        test_query_statistics,
        test_query_export
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
