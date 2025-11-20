"""
Forensic Analyzer - Foundation #7

Reconstruct and analyze security incidents from audit logs.
Provides attack sequence reconstruction, timeline analysis, and incident reports.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import statistics

from deepagent.audit.audit_logger import (
    AuditRecord,
    EventType,
    EventSeverity,
    AuditLogger
)


@dataclass
class AttackReconstruction:
    """
    Complete reconstruction of an attack sequence from audit logs.

    Provides all context needed to understand what happened,
    how it was detected, and what damage was prevented.
    """

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

    def to_markdown(self) -> str:
        """Generate markdown incident report"""
        report = f"""# Attack Incident Report

## Summary

- **Attack ID**: {self.attack_id}
- **Session ID**: {self.session_id}
- **User ID**: {self.user_id}
- **Attack Pattern**: {self.attack_pattern}
- **Confidence**: {self.confidence:.0%}
- **Status**: {'BLOCKED' if self.blocked else 'DETECTED'}

## Timeline

- **Start Time**: {self.start_time.isoformat()}
- **End Time**: {self.end_time.isoformat()}
- **Duration**: {self.duration.total_seconds():.1f} seconds

## Context

**Original Task**: {self.original_task}

## Attack Sequence

Total steps: {len(self.steps)}

"""
        for i, step in enumerate(self.steps, 1):
            is_critical = i in self.critical_steps
            marker = "⚠️ CRITICAL" if is_critical else ""
            report += f"### Step {i} {marker}\n\n"
            report += f"- **Time**: {step.timestamp.isoformat()}\n"
            report += f"- **Tool**: {step.tool_name}\n"
            report += f"- **Action**: {step.action_type}\n"
            report += f"- **Risk Score**: {step.phase1_risk_score:.0%}\n" if step.phase1_risk_score else ""
            report += f"- **Allowed**: {step.allowed}\n"
            if step.error:
                report += f"- **Error**: {step.error}\n"
            report += "\n"

        report += f"""## Impact

**Damage Prevented**: {self.damage_prevented}

**Detection Method**: {self.detection_method}

**False Positive Likelihood**: {self.false_positive_likelihood:.0%}

## Recommendations

"""
        if self.blocked:
            report += "- Attack was successfully blocked by security system\n"
            report += "- Review user permissions and access controls\n"
            report += f"- Investigate user {self.user_id} for potential compromise\n"
        else:
            report += "- Attack was detected but not blocked\n"
            report += "- Review detection thresholds\n"
            report += "- Implement additional controls\n"

        return report


@dataclass
class RiskTrajectory:
    """Risk progression over time during a session"""

    session_id: str
    risk_scores: List[Tuple[int, float]]  # (step_number, risk_score)
    peak_risk_step: int
    peak_risk_score: float
    average_risk: float
    escalation_rate: float  # Average change per step

    def to_ascii_chart(self, width: int = 60, height: int = 10) -> str:
        """Generate ASCII chart of risk trajectory"""
        if not self.risk_scores:
            return "No risk data available"

        # Normalize scores to chart height
        max_score = max(score for _, score in self.risk_scores)
        if max_score == 0:
            return "All risk scores are zero"

        chart = []
        chart.append(f"Risk Trajectory (Session: {self.session_id})")
        chart.append("-" * width)

        # Build chart from top to bottom
        for level in range(height, -1, -1):
            threshold = (level / height) * max_score
            line = f"{threshold:4.0%} |"

            for step, score in self.risk_scores:
                if score >= threshold:
                    line += "#"  # Use ASCII instead of Unicode block
                else:
                    line += " "

            chart.append(line)

        # Add x-axis
        chart.append("     +" + "-" * len(self.risk_scores))
        chart.append(f"      Steps: 1 -> {len(self.risk_scores)}")
        chart.append(f"\nPeak: Step {self.peak_risk_step} ({self.peak_risk_score:.0%})")
        chart.append(f"Average: {self.average_risk:.0%}")
        chart.append(f"Escalation Rate: {self.escalation_rate:+.1%} per step")

        return "\n".join(chart)


@dataclass
class TimelineAnalysis:
    """
    Comprehensive timeline analysis of a session.

    Provides statistics, security event summary, and risk trajectory.
    """

    session_id: str
    user_id: str
    original_task: str

    # Timing
    start_time: datetime
    end_time: datetime
    total_duration: timedelta

    # Actions
    total_actions: int
    successful_actions: int
    failed_actions: int
    blocked_actions: int

    # Security events
    security_events: List[AuditRecord]
    attacks_detected: int
    escalations_detected: int
    drifts_detected: int
    anomalies_detected: int

    # Risk progression
    risk_trajectory: RiskTrajectory

    # Performance
    average_action_duration_ms: float

    def summary(self) -> str:
        """Generate summary string"""
        return f"""Timeline Analysis: {self.session_id}

User: {self.user_id}
Task: {self.original_task}
Duration: {self.total_duration.total_seconds():.1f}s

Actions:
  Total: {self.total_actions}
  Successful: {self.successful_actions}
  Failed: {self.failed_actions}
  Blocked: {self.blocked_actions}

Security Events:
  Attacks: {self.attacks_detected}
  Escalations: {self.escalations_detected}
  Drifts: {self.drifts_detected}
  Anomalies: {self.anomalies_detected}

Risk:
  Peak: {self.risk_trajectory.peak_risk_score:.0%} (Step {self.risk_trajectory.peak_risk_step})
  Average: {self.risk_trajectory.average_risk:.0%}
  Escalation Rate: {self.risk_trajectory.escalation_rate:+.1%} per step

Performance:
  Avg Action Duration: {self.average_action_duration_ms:.1f}ms
"""


@dataclass
class PatternCorrelation:
    """
    Correlation of attack patterns across multiple sessions.

    Identifies trends and recurring attack patterns.
    """

    time_range: Tuple[datetime, datetime]
    total_sessions: int
    sessions_with_attacks: int

    # Pattern statistics
    pattern_counts: Dict[str, int]
    pattern_success_rates: Dict[str, float]  # % that were blocked

    # User statistics
    users_with_attacks: List[str]
    repeat_offenders: List[Tuple[str, int]]  # (user_id, attack_count)

    # Trends
    attacks_per_day: List[Tuple[str, int]]  # (date, count)

    def summary(self) -> str:
        """Generate summary of pattern correlations"""
        start, end = self.time_range
        total_attacks = sum(self.pattern_counts.values())

        summary = f"""Pattern Correlation Analysis

Time Range: {start.date()} to {end.date()}

Overview:
  Total Sessions: {self.total_sessions}
  Sessions with Attacks: {self.sessions_with_attacks} ({self.sessions_with_attacks/max(self.total_sessions,1)*100:.1f}%)
  Total Attacks: {total_attacks}
  Unique Users: {len(self.users_with_attacks)}

Top Attack Patterns:
"""
        # Sort patterns by count
        sorted_patterns = sorted(self.pattern_counts.items(), key=lambda x: x[1], reverse=True)
        for pattern, count in sorted_patterns[:5]:
            success_rate = self.pattern_success_rates.get(pattern, 0)
            summary += f"  - {pattern}: {count} ({success_rate:.0%} blocked)\n"

        if self.repeat_offenders:
            summary += "\nRepeat Offenders:\n"
            for user_id, count in self.repeat_offenders[:5]:
                summary += f"  - {user_id}: {count} attacks\n"

        return summary


class ForensicAnalyzer:
    """
    Forensic analysis tool for security incidents.

    Reconstructs attacks, analyzes timelines, and correlates patterns
    from audit logs.
    """

    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    def reconstruct_attack_sequence(self, attack_id: str) -> Optional[AttackReconstruction]:
        """
        Reconstruct complete attack sequence from audit logs.

        Args:
            attack_id: ID of the attack to reconstruct (typically record_id of attack detection event)

        Returns:
            AttackReconstruction with complete context, or None if not found
        """
        # Find the attack detection event
        all_records = self.audit_logger.read_records()
        attack_record = None

        for record in all_records:
            if record.record_id == attack_id and record.event_type == EventType.ATTACK_DETECTED:
                attack_record = record
                break

        if not attack_record:
            return None

        # Get all records from this session
        session_records = [
            r for r in all_records
            if r.session_id == attack_record.session_id
        ]

        # Sort by timestamp
        session_records.sort(key=lambda r: r.timestamp)

        # Find session start for original task
        original_task = "Unknown"
        for record in session_records:
            if record.event_type == EventType.SESSION_START:
                original_task = record.metadata.get('task', 'Unknown')
                break

        # Get action steps leading to attack
        action_steps = [
            r for r in session_records
            if r.tool_name is not None and r.timestamp <= attack_record.timestamp
        ]

        # Identify critical steps (high risk or anomalies)
        critical_steps = []
        for i, step in enumerate(action_steps, 1):
            if step.phase1_risk_score and step.phase1_risk_score >= 0.7:
                critical_steps.append(i)
            elif not step.allowed:
                critical_steps.append(i)

        # Calculate timeline
        if action_steps:
            start_time = action_steps[0].timestamp
            end_time = attack_record.timestamp
            duration = end_time - start_time
        else:
            start_time = attack_record.timestamp
            end_time = attack_record.timestamp
            duration = timedelta(0)

        # Extract attack details
        attack_pattern = attack_record.phase2_results.get('pattern_name', 'Unknown')
        confidence = attack_record.phase2_results.get('confidence', 0.0)
        severity = attack_record.phase2_results.get('severity', 'unknown')

        # Determine damage prevented
        damage_prevented = self._assess_damage_prevented(attack_pattern, action_steps)

        # Assess false positive likelihood
        false_positive_likelihood = self._assess_false_positive(confidence, critical_steps, action_steps)

        return AttackReconstruction(
            attack_id=attack_id,
            session_id=attack_record.session_id,
            attack_pattern=attack_pattern,
            confidence=confidence,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            steps=action_steps,
            critical_steps=critical_steps,
            original_task=original_task,
            user_id=attack_record.user_id,
            blocked=True,  # If we logged it, we blocked it
            damage_prevented=damage_prevented,
            detection_method="Phase 2 Pattern Matching",
            false_positive_likelihood=false_positive_likelihood
        )

    def analyze_session_timeline(self, session_id: str) -> Optional[TimelineAnalysis]:
        """
        Analyze complete timeline of a session.

        Args:
            session_id: Session to analyze

        Returns:
            TimelineAnalysis with statistics and risk trajectory
        """
        # Get all records for this session
        records = self.audit_logger.read_records({'session_id': session_id})

        if not records:
            return None

        # Sort by timestamp
        records.sort(key=lambda r: r.timestamp)

        # Find session metadata
        original_task = "Unknown"
        user_id = records[0].user_id if records else "Unknown"

        for record in records:
            if record.event_type == EventType.SESSION_START:
                original_task = record.metadata.get('task', 'Unknown')
                break

        # Calculate timing
        start_time = records[0].timestamp
        end_time = records[-1].timestamp
        total_duration = end_time - start_time

        # Count actions
        action_records = [r for r in records if r.tool_name is not None]
        total_actions = len(action_records)
        successful_actions = len([r for r in action_records if r.allowed and not r.error])
        failed_actions = len([r for r in action_records if r.error])
        blocked_actions = len([r for r in action_records if not r.allowed])

        # Count security events
        security_events = [r for r in records if r.severity in [EventSeverity.WARNING, EventSeverity.CRITICAL]]
        attacks_detected = len([r for r in records if r.event_type == EventType.ATTACK_DETECTED])
        escalations_detected = len([r for r in records if r.event_type == EventType.ESCALATION_DETECTED])
        drifts_detected = len([r for r in records if r.event_type == EventType.GOAL_DRIFT])
        anomalies_detected = len([r for r in records if r.event_type == EventType.REASONING_ANOMALY])

        # Build risk trajectory
        risk_scores = []
        for i, record in enumerate(action_records, 1):
            if record.phase1_risk_score is not None:
                risk_scores.append((i, record.phase1_risk_score))

        if risk_scores:
            peak_step, peak_score = max(risk_scores, key=lambda x: x[1])
            avg_risk = statistics.mean(score for _, score in risk_scores)

            # Calculate escalation rate
            if len(risk_scores) > 1:
                escalation_rate = (risk_scores[-1][1] - risk_scores[0][1]) / (len(risk_scores) - 1)
            else:
                escalation_rate = 0.0
        else:
            peak_step = 0
            peak_score = 0.0
            avg_risk = 0.0
            escalation_rate = 0.0

        risk_trajectory = RiskTrajectory(
            session_id=session_id,
            risk_scores=risk_scores,
            peak_risk_step=peak_step,
            peak_risk_score=peak_score,
            average_risk=avg_risk,
            escalation_rate=escalation_rate
        )

        # Calculate performance metrics
        durations = [r.duration_ms for r in action_records if r.duration_ms is not None]
        avg_duration = statistics.mean(durations) if durations else 0.0

        return TimelineAnalysis(
            session_id=session_id,
            user_id=user_id,
            original_task=original_task,
            start_time=start_time,
            end_time=end_time,
            total_duration=total_duration,
            total_actions=total_actions,
            successful_actions=successful_actions,
            failed_actions=failed_actions,
            blocked_actions=blocked_actions,
            security_events=security_events,
            attacks_detected=attacks_detected,
            escalations_detected=escalations_detected,
            drifts_detected=drifts_detected,
            anomalies_detected=anomalies_detected,
            risk_trajectory=risk_trajectory,
            average_action_duration_ms=avg_duration
        )

    def generate_incident_report(
        self,
        attack_id: str,
        format: str = "markdown"
    ) -> str:
        """
        Generate incident report for an attack.

        Args:
            attack_id: ID of attack to report on
            format: Output format (markdown, text, json)

        Returns:
            Formatted incident report
        """
        reconstruction = self.reconstruct_attack_sequence(attack_id)

        if not reconstruction:
            return f"Attack {attack_id} not found"

        if format == "markdown":
            return reconstruction.to_markdown()
        elif format == "json":
            import json
            return json.dumps({
                'attack_id': reconstruction.attack_id,
                'session_id': reconstruction.session_id,
                'pattern': reconstruction.attack_pattern,
                'confidence': reconstruction.confidence,
                'blocked': reconstruction.blocked,
                'steps': len(reconstruction.steps),
                'critical_steps': reconstruction.critical_steps,
                'duration_seconds': reconstruction.duration.total_seconds()
            }, indent=2)
        else:  # text
            return f"""Attack Incident Report
=====================

Attack ID: {reconstruction.attack_id}
Session: {reconstruction.session_id}
User: {reconstruction.user_id}
Pattern: {reconstruction.attack_pattern}
Confidence: {reconstruction.confidence:.0%}
Status: {'BLOCKED' if reconstruction.blocked else 'DETECTED'}

Timeline:
  Start: {reconstruction.start_time}
  End: {reconstruction.end_time}
  Duration: {reconstruction.duration.total_seconds():.1f}s

Sequence: {len(reconstruction.steps)} steps
Critical Steps: {', '.join(map(str, reconstruction.critical_steps))}

Damage Prevented: {reconstruction.damage_prevented}
Detection Method: {reconstruction.detection_method}
"""

    def find_similar_attacks(
        self,
        attack_id: str,
        min_similarity: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Find similar attack patterns in audit logs.

        Args:
            attack_id: Reference attack to compare against
            min_similarity: Minimum similarity score (0-1)

        Returns:
            List of (attack_id, similarity_score) tuples
        """
        reconstruction = self.reconstruct_attack_sequence(attack_id)

        if not reconstruction:
            return []

        # Find all other attacks
        all_records = self.audit_logger.read_records()
        attack_records = [r for r in all_records if r.event_type == EventType.ATTACK_DETECTED]

        similar_attacks = []

        for record in attack_records:
            if record.record_id == attack_id:
                continue

            # Compare patterns
            other_pattern = record.phase2_results.get('pattern_name', '')
            if other_pattern == reconstruction.attack_pattern:
                similarity = 1.0
            else:
                similarity = 0.5  # Different pattern but still an attack

            if similarity >= min_similarity:
                similar_attacks.append((record.record_id, similarity))

        # Sort by similarity
        similar_attacks.sort(key=lambda x: x[1], reverse=True)

        return similar_attacks

    def analyze_risk_trajectory(self, session_id: str) -> Optional[RiskTrajectory]:
        """
        Analyze risk progression over a session.

        Args:
            session_id: Session to analyze

        Returns:
            RiskTrajectory with risk progression data
        """
        timeline = self.analyze_session_timeline(session_id)
        return timeline.risk_trajectory if timeline else None

    def correlate_patterns(
        self,
        time_range: Tuple[datetime, datetime]
    ) -> PatternCorrelation:
        """
        Correlate attack patterns across sessions in a time range.

        Args:
            time_range: (start_time, end_time) tuple

        Returns:
            PatternCorrelation with aggregate statistics
        """
        start_time, end_time = time_range

        # Get all records in time range
        all_records = self.audit_logger.read_records({
            'start_time': start_time,
            'end_time': end_time
        })

        # Group by session
        sessions = defaultdict(list)
        for record in all_records:
            sessions[record.session_id].append(record)

        total_sessions = len(sessions)

        # Find sessions with attacks
        sessions_with_attacks = set()
        attack_records = [r for r in all_records if r.event_type == EventType.ATTACK_DETECTED]

        for record in attack_records:
            sessions_with_attacks.add(record.session_id)

        # Count patterns
        pattern_counts = defaultdict(int)
        pattern_blocked = defaultdict(int)

        for record in attack_records:
            pattern = record.phase2_results.get('pattern_name', 'Unknown')
            pattern_counts[pattern] += 1
            if not record.allowed:
                pattern_blocked[pattern] += 1

        # Calculate success rates (% blocked)
        pattern_success_rates = {}
        for pattern, count in pattern_counts.items():
            blocked = pattern_blocked.get(pattern, 0)
            pattern_success_rates[pattern] = blocked / count if count > 0 else 0.0

        # User statistics
        users_with_attacks = list(set(r.user_id for r in attack_records))
        user_attack_counts = defaultdict(int)

        for record in attack_records:
            user_attack_counts[record.user_id] += 1

        repeat_offenders = sorted(user_attack_counts.items(), key=lambda x: x[1], reverse=True)

        # Attacks per day
        attacks_by_date = defaultdict(int)
        for record in attack_records:
            date_key = record.timestamp.date().isoformat()
            attacks_by_date[date_key] += 1

        attacks_per_day = sorted(attacks_by_date.items())

        return PatternCorrelation(
            time_range=time_range,
            total_sessions=total_sessions,
            sessions_with_attacks=len(sessions_with_attacks),
            pattern_counts=dict(pattern_counts),
            pattern_success_rates=pattern_success_rates,
            users_with_attacks=users_with_attacks,
            repeat_offenders=repeat_offenders,
            attacks_per_day=attacks_per_day
        )

    def _assess_damage_prevented(self, attack_pattern: str, steps: List[AuditRecord]) -> str:
        """Assess what damage was prevented by blocking this attack"""
        damage_map = {
            'Data Exfiltration': 'Prevented unauthorized data export and deletion',
            'Privilege Escalation': 'Prevented unauthorized privilege elevation',
            'Goal Hijacking': 'Prevented task manipulation and scope creep',
            'Scope Expansion': 'Prevented gradual expansion beyond authorized scope',
            'Reconnaissance': 'Prevented information gathering for future attacks',
            'Memory Poisoning': 'Prevented injection of false information'
        }

        return damage_map.get(attack_pattern, 'Prevented malicious multi-step attack')

    def _assess_false_positive(
        self,
        confidence: float,
        critical_steps: List[int],
        all_steps: List[AuditRecord]
    ) -> float:
        """Estimate likelihood of false positive"""

        # High confidence = low false positive likelihood
        fp_likelihood = 1.0 - confidence

        # More critical steps = lower false positive likelihood
        if len(all_steps) > 0:
            critical_ratio = len(critical_steps) / len(all_steps)
            fp_likelihood *= (1.0 - critical_ratio * 0.5)

        return max(0.0, min(1.0, fp_likelihood))
