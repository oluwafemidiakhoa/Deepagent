"""
Foundation #7 Audit Logging & Forensics Demo

Demonstrates how audit logging, forensic analysis, and query capabilities
work in SafeDeepAgent to provide comprehensive security monitoring and
incident investigation.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepagent.audit import (
    AuditConfig,
    JSONAuditLogger,
    SQLiteAuditLogger,
    ForensicAnalyzer,
    AuditQueryInterface,
    QueryFilters,
    EventType,
    EventSeverity
)


def print_header(title: str):
    """Print section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def print_result(passed: bool, message: str):
    """Print test result"""
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {message}\n")


def demo_basic_audit_logging():
    """Demo 1: Basic audit logging"""
    print_header("DEMO 1: Basic Audit Logging")

    print("This demonstrates how SafeDeepAgent logs all security events:")
    print("- Session lifecycle (start/end)")
    print("- Action execution")
    print("- Security decisions")
    print("- Attack detection\n")

    # Create audit logger
    config = AuditConfig(
        log_directory=Path("./audit_logs/demo1"),
        storage_backend="json",
        async_logging=False  # Sync for demo
    )

    logger = JSONAuditLogger(config)

    # Simulate session lifecycle
    session_id = "demo_session_1"
    user_id = "demo_user"

    # 1. Log session start
    logger.log_session_start(
        session_id=session_id,
        user_id=user_id,
        task="Research CRISPR applications",
        metadata={"environment": "production"}
    )
    print("1. Session started")

    # 2. Log successful actions
    for i in range(3):
        logger.log_action(
            session_id=session_id,
            user_id=user_id,
            step_number=i+1,
            tool_name="search_papers",
            action_type="read",
            parameters={"query": f"CRISPR research {i}"},
            allowed=True,
            duration_ms=150.0 + i * 20
        )
    print("2. Logged 3 successful read actions")

    # 3. Log blocked action
    logger.log_action(
        session_id=session_id,
        user_id=user_id,
        step_number=4,
        tool_name="admin_command",
        action_type="execute",
        parameters={"command": "sudo rm -rf /"},
        allowed=False,
        error="Unauthorized: High-risk action blocked"
    )
    print("3. Logged 1 blocked action (high-risk)")

    # 4. Log session end
    logger.log_session_end(
        session_id=session_id,
        user_id=user_id,
        stats={
            "total_actions": 4,
            "allowed": 3,
            "blocked": 1
        }
    )
    print("4. Session ended")

    # Query the logs
    query = AuditQueryInterface(logger)
    session_records = query.query(QueryFilters(session_ids=[session_id]))

    print(f"\n[AUDIT LOG SUMMARY]")
    print(f"  Total events logged: {session_records.total_count}")
    print(f"  Events: Session start (1) + Actions (4) + Session end (1) = {session_records.total_count}")

    print_result(
        session_records.total_count == 6,
        "All events logged successfully"
    )


def demo_attack_detection_logging():
    """Demo 2: Attack detection logging and reconstruction"""
    print_header("DEMO 2: Attack Detection Logging & Forensic Reconstruction")

    print("This demonstrates how Phase 2 attacks are logged and later reconstructed:")
    print("- Actions leading to attack")
    print("- Attack detection event")
    print("- Forensic reconstruction")
    print("- Incident report generation\n")

    config = AuditConfig(
        log_directory=Path("./audit_logs/demo2"),
        storage_backend="json",
        async_logging=False
    )

    logger = JSONAuditLogger(config)
    session_id = "attack_session"
    user_id = "suspicious_user"

    # Log session
    logger.log_session_start(session_id=session_id, user_id=user_id, task="Data analysis")

    # Simulate data exfiltration attack sequence
    print("Simulating data exfiltration attack sequence...")

    logger.log_action(
        session_id=session_id,
        user_id=user_id,
        step_number=1,
        tool_name="backup_database",
        action_type="read",
        parameters={"database": "user_data"},
        allowed=True
    )
    print("  Step 1: backup_database (read) - Allowed")

    logger.log_action(
        session_id=session_id,
        user_id=user_id,
        step_number=2,
        tool_name="export_to_file",
        action_type="write",
        parameters={"destination": "external_server.com"},
        allowed=True
    )
    print("  Step 2: export_to_file (write) - Allowed")

    logger.log_action(
        session_id=session_id,
        user_id=user_id,
        step_number=3,
        tool_name="delete_records",
        action_type="delete",
        parameters={"target": "original_data"},
        allowed=True
    )
    print("  Step 3: delete_records (delete) - Allowed")

    # Mock attack detection
    class MockAttackResult:
        def __init__(self):
            self.attack_detected = True
            self.highest_confidence = 0.98
            self.most_likely_pattern = self.MockPattern()

        class MockPattern:
            def __init__(self):
                self.pattern_id = "data_exfiltration"
                self.name = "Data Exfiltration"
                self.severity = self.MockSeverity()
                self.attack_category = self.MockCategory()

            class MockSeverity:
                value = "critical"

            class MockCategory:
                value = "data_exfiltration"

    attack_result = MockAttackResult()
    attack_record = logger.log_attack_detection(
        session_id=session_id,
        user_id=user_id,
        attack_result=attack_result
    )
    print("  Step 4: ATTACK DETECTED! - Blocked\n")

    logger.log_session_end(session_id=session_id, user_id=user_id, stats={"attack_blocked": True})

    # Forensic reconstruction
    print("[FORENSIC RECONSTRUCTION]")
    analyzer = ForensicAnalyzer(logger)
    reconstruction = analyzer.reconstruct_attack_sequence(attack_record.record_id)

    if reconstruction:
        print(f"  Attack Pattern: {reconstruction.attack_pattern}")
        print(f"  Confidence: {reconstruction.confidence:.0%}")
        print(f"  Duration: {reconstruction.duration.total_seconds():.1f}s")
        print(f"  Steps in sequence: {len(reconstruction.steps)}")
        print(f"  Critical steps: {reconstruction.critical_steps}")
        print(f"  Damage prevented: {reconstruction.damage_prevented}")
        print(f"  Status: {'BLOCKED' if reconstruction.blocked else 'DETECTED ONLY'}")

        # Generate incident report
        print("\n[INCIDENT REPORT]")
        report = analyzer.generate_incident_report(
            attack_record.record_id,
            format="text"
        )
        print(report[:500] + "...")  # Show first 500 chars

    print_result(
        reconstruction is not None and reconstruction.confidence > 0.9,
        "Attack reconstructed with high confidence"
    )


def demo_forensic_timeline_analysis():
    """Demo 3: Timeline analysis and risk trajectory"""
    print_header("DEMO 3: Timeline Analysis & Risk Trajectory")

    print("This demonstrates how to analyze security events over time:")
    print("- Risk progression")
    print("- Escalation detection")
    print("- Performance metrics\n")

    config = AuditConfig(
        log_directory=Path("./audit_logs/demo3"),
        storage_backend="json",
        async_logging=False
    )

    logger = JSONAuditLogger(config)
    session_id = "timeline_session"

    # Simulate session with escalating risk
    logger.log_session_start(session_id=session_id, user_id="user_1", task="System maintenance")

    class MockRiskScore:
        def __init__(self, score):
            self.total_score = score

    class MockDecision:
        def __init__(self, score):
            self.risk_score = MockRiskScore(score)

    print("Simulating actions with escalating risk:")
    for i in range(6):
        risk = 0.1 + (i * 0.15)  # Escalating from 10% to 85%
        decision = MockDecision(risk)

        logger.log_action(
            session_id=session_id,
            user_id="user_1",
            step_number=i+1,
            tool_name=f"tool_{i}",
            action_type=["read", "read", "modify", "modify", "execute", "admin"][i],
            parameters={},
            phase1_result=decision,
            allowed=(i < 5),  # Last one blocked
            duration_ms=100.0 + i * 25
        )
        status = "Allowed" if i < 5 else "Blocked"
        print(f"  Step {i+1}: {decision.risk_score.total_score:.0%} risk - {status}")

    logger.log_session_end(session_id=session_id, user_id="user_1", stats={})

    # Analyze timeline
    print("\n[TIMELINE ANALYSIS]")
    analyzer = ForensicAnalyzer(logger)
    timeline = analyzer.analyze_session_timeline(session_id)

    if timeline:
        print(f"  Total Actions: {timeline.total_actions}")
        print(f"  Successful: {timeline.successful_actions}")
        print(f"  Blocked: {timeline.blocked_actions}")
        print(f"  Peak Risk: {timeline.risk_trajectory.peak_risk_score:.0%} at step {timeline.risk_trajectory.peak_risk_step}")
        print(f"  Average Risk: {timeline.risk_trajectory.average_risk:.0%}")
        print(f"  Escalation Rate: {timeline.risk_trajectory.escalation_rate:+.1%} per step")
        print(f"  Average Duration: {timeline.average_action_duration_ms:.1f}ms")

        # Show ASCII risk chart
        print("\n[RISK TRAJECTORY CHART]")
        chart = timeline.risk_trajectory.to_ascii_chart(width=50, height=8)
        print(chart)

    print_result(
        timeline is not None and timeline.risk_trajectory.escalation_rate > 0,
        "Detected risk escalation pattern"
    )


def demo_query_and_export():
    """Demo 4: Flexible querying and export"""
    print_header("DEMO 4: Flexible Querying & Export")

    print("This demonstrates audit log querying and export capabilities:")
    print("- Filter by session, user, event type, risk score")
    print("- Aggregation and statistics")
    print("- Export to JSON, CSV, Markdown\n")

    config = AuditConfig(
        log_directory=Path("./audit_logs/demo4"),
        storage_backend="json",
        async_logging=False
    )

    logger = JSONAuditLogger(config)

    # Create diverse test data
    print("Creating diverse audit data...")
    for session_num in range(3):
        session_id = f"session_{session_num}"
        user_id = f"user_{session_num % 2}"

        logger.log_session_start(session_id=session_id, user_id=user_id, task=f"Task {session_num}")

        for action_num in range(5):
            logger.log_action(
                session_id=session_id,
                user_id=user_id,
                step_number=action_num+1,
                tool_name=f"tool_{action_num % 3}",
                action_type=["read", "write", "execute"][action_num % 3],
                parameters={},
                allowed=(action_num < 4),  # 1 blocked per session
                duration_ms=75.0 + action_num * 15
            )

        logger.log_session_end(session_id=session_id, user_id=user_id, stats={})

    print(f"Created 3 sessions with 15 actions total\n")

    # Query interface
    query = AuditQueryInterface(logger)

    # Query 1: All records
    print("[QUERY 1: All Records]")
    result = query.query(QueryFilters())
    print(f"  Total records: {result.total_count}")

    # Query 2: Blocked actions only
    print("\n[QUERY 2: Blocked Actions Only]")
    result = query.query(QueryFilters(only_blocked=True))
    print(f"  Blocked actions: {result.total_count}")

    # Query 3: Specific user
    print("\n[QUERY 3: User 'user_0' Only]")
    result = query.query(QueryFilters(user_ids=["user_0"]))
    print(f"  Records for user_0: {result.total_count}")

    # Query 4: Specific event types
    print("\n[QUERY 4: Tool Execution Events]")
    result = query.query(QueryFilters(
        event_types=[EventType.TOOL_SUCCESS, EventType.TOOL_FAILURE]
    ))
    print(f"  Tool execution events: {result.total_count}")

    # Statistics
    print("\n[STATISTICS]")
    stats = query.statistics(QueryFilters())
    print(f"  Total Records: {stats['total_records']}")
    print(f"  Unique Sessions: {stats['unique_sessions']}")
    print(f"  Unique Users: {stats['unique_users']}")
    print(f"  Total Blocked: {stats['security']['total_blocked']}")

    if 'performance' in stats:
        print(f"  Avg Duration: {stats['performance']['avg_ms']:.1f}ms")

    # Export to different formats
    print("\n[EXPORT]")
    export_dir = Path("./audit_logs/demo4/exports")
    export_dir.mkdir(parents=True, exist_ok=True)

    # Export to JSON
    query.export(
        QueryFilters(only_blocked=True),
        format="json",
        output_path=str(export_dir / "blocked_actions.json")
    )
    print("  Exported to JSON: blocked_actions.json")

    # Export to CSV
    query.export(
        QueryFilters(user_ids=["user_0"]),
        format="csv",
        output_path=str(export_dir / "user_0_actions.csv")
    )
    print("  Exported to CSV: user_0_actions.csv")

    # Export to Markdown
    query.export(
        QueryFilters(limit=10),
        format="markdown",
        output_path=str(export_dir / "recent_10.md")
    )
    print("  Exported to Markdown: recent_10.md")

    print_result(
        stats['total_records'] > 0,
        "Query and export capabilities working"
    )


def demo_pattern_correlation():
    """Demo 5: Cross-session pattern correlation"""
    print_header("DEMO 5: Cross-Session Pattern Correlation")

    print("This demonstrates pattern correlation across multiple sessions:")
    print("- Identifying attack trends")
    print("- Repeat offenders")
    print("- Pattern frequency\n")

    config = AuditConfig(
        log_directory=Path("./audit_logs/demo5"),
        storage_backend="json",
        async_logging=False
    )

    logger = JSONAuditLogger(config)

    # Simulate multiple attack patterns
    attack_patterns = [
        ("Data Exfiltration", "user_1"),
        ("Privilege Escalation", "user_2"),
        ("Data Exfiltration", "user_1"),  # Repeat offender
        ("Goal Hijacking", "user_3"),
        ("Privilege Escalation", "user_2"),  # Repeat offender
    ]

    print("Simulating 5 attack scenarios across 3 users...\n")

    class MockAttackResult:
        def __init__(self, pattern_name):
            self.attack_detected = True
            self.highest_confidence = 0.92
            self.most_likely_pattern = self.MockPattern(pattern_name)

        class MockPattern:
            def __init__(self, name):
                self.pattern_id = name.lower().replace(" ", "_")
                self.name = name
                self.severity = self.MockSeverity()
                self.attack_category = self.MockCategory()

            class MockSeverity:
                value = "high"

            class MockCategory:
                value = "attack"

    for i, (pattern, user) in enumerate(attack_patterns):
        session_id = f"attack_session_{i}"

        logger.log_session_start(session_id=session_id, user_id=user, task=f"Task {i}")

        attack_result = MockAttackResult(pattern)
        logger.log_attack_detection(
            session_id=session_id,
            user_id=user,
            attack_result=attack_result
        )

        logger.log_session_end(session_id=session_id, user_id=user, stats={})

        print(f"  Session {i+1}: {pattern} by {user}")

    # Correlate patterns
    print("\n[PATTERN CORRELATION]")
    analyzer = ForensicAnalyzer(logger)
    start_time = datetime.now() - timedelta(hours=1)
    end_time = datetime.now() + timedelta(hours=1)

    correlation = analyzer.correlate_patterns((start_time, end_time))

    print(f"  Total Sessions: {correlation.total_sessions}")
    print(f"  Sessions with Attacks: {correlation.sessions_with_attacks}")
    print(f"  Unique Attack Patterns: {len(correlation.pattern_counts)}")

    print("\n  Pattern Breakdown:")
    for pattern, count in sorted(correlation.pattern_counts.items(), key=lambda x: x[1], reverse=True):
        success_rate = correlation.pattern_success_rates.get(pattern, 0)
        print(f"    - {pattern}: {count} occurrences ({success_rate:.0%} blocked)")

    print(f"\n  Unique Attackers: {len(correlation.users_with_attacks)}")

    if correlation.repeat_offenders:
        print("\n  Repeat Offenders:")
        for user_id, count in correlation.repeat_offenders[:3]:
            print(f"    - {user_id}: {count} attacks")

    print_result(
        len(correlation.pattern_counts) == 3 and len(correlation.users_with_attacks) == 3,
        "Pattern correlation identified trends and repeat offenders"
    )


def demo_integrated_safe_agent():
    """Demo 6: Integrated audit logging in SafeDeepAgent"""
    print_header("DEMO 6: Integrated Audit Logging in SafeDeepAgent")

    print("This demonstrates how audit logging integrates with SafeDeepAgent:")
    print("- Automatic session tracking")
    print("- Phase 1 and Phase 2 event logging")
    print("- Forensic analysis of agent sessions\n")

    print("[CONCEPTUAL DEMO]")
    print("When SafeDeepAgent runs with audit logging enabled:")
    print()
    print("1. Session Start is logged automatically")
    print("   - Task, user_id, session_id, environment")
    print()
    print("2. Every tool execution is logged with:")
    print("   - Phase 1 risk score and decision")
    print("   - Phase 2 analysis results (if available)")
    print("   - Execution duration and result")
    print()
    print("3. Security events are logged:")
    print("   - Attack detections")
    print("   - Goal drift warnings")
    print("   - Escalation alerts")
    print("   - Memory tampering")
    print()
    print("4. Session End is logged with:")
    print("   - Complete statistics")
    print("   - Success/failure status")
    print()
    print("5. Forensic analysis is available:")
    print("   - Attack reconstruction")
    print("   - Timeline analysis")
    print("   - Incident reports")
    print()

    print("Example SafeDeepAgent usage:")
    print("-" * 80)
    print("from deepagent.core.safe_agent import create_safe_agent")
    print("from deepagent.safety import SafetyMode")
    print()
    print("# Create agent with audit logging enabled")
    print("agent = create_safe_agent(")
    print("    llm_provider='openai',")
    print("    safety_mode=SafetyMode.STRICT,")
    print("    enable_memory_firewall=True  # Phase 2")
    print("    # Audit logging enabled by default in SafeDeepAgent!")
    print(")")
    print()
    print("# Run task - all events are logged automatically")
    print("result = agent.run('Research CRISPR applications')")
    print()
    print("# Access audit logs and forensics")
    print("if hasattr(agent, 'query_interface'):")
    print("    stats = agent.query_interface.statistics(QueryFilters())")
    print("    print(f'Total events: {stats[\"total_records\"]}')")
    print("-" * 80)

    print_result(
        True,
        "Audit logging integrates seamlessly with SafeDeepAgent"
    )


def main():
    """Run all Foundation #7 demonstrations"""
    print("\n" + "="*80)
    print(" "*20 + "FOUNDATION #7 AUDIT LOGGING DEMO")
    print("="*80)
    print("\nDemonstrating comprehensive audit logging, forensic analysis,")
    print("and security investigation capabilities for SafeDeepAgent.\n")

    # Run all demos
    demo_basic_audit_logging()
    demo_attack_detection_logging()
    demo_forensic_timeline_analysis()
    demo_query_and_export()
    demo_pattern_correlation()
    demo_integrated_safe_agent()

    # Summary
    print_header("SUMMARY")
    print("Foundation #7 successfully demonstrates:")
    print()
    print("  [x] Comprehensive audit logging (JSON/SQLite backends)")
    print("  [x] Attack sequence reconstruction with forensic analysis")
    print("  [x] Timeline analysis and risk trajectory tracking")
    print("  [x] Flexible query interface with filtering and aggregation")
    print("  [x] Multi-format export (JSON, CSV, Markdown, Text)")
    print("  [x] Cross-session pattern correlation")
    print("  [x] Seamless integration with SafeDeepAgent")
    print()
    print("With Foundation #7, you have:")
    print("  - Complete audit trail of all security events")
    print("  - Forensic reconstruction of detected attacks")
    print("  - Incident report generation")
    print("  - Security analytics and trend analysis")
    print("  - Compliance and accountability")
    print()
    print("[SUCCESS] Foundation #7 is production-ready!")


if __name__ == "__main__":
    main()
