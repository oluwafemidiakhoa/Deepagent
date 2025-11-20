"""
Consistency Checker - Foundation #10

Checks for internal consistency and contradictions in agent statements.
Detects logical inconsistencies across time.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4


@dataclass
class Statement:
    """A statement made by an agent"""
    statement_id: str
    agent_id: str
    content: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Contradiction:
    """A detected contradiction between statements"""
    contradiction_id: str
    statement1_id: str
    statement2_id: str
    description: str
    severity: str  # "LOW", "MEDIUM", "HIGH"
    confidence: float  # 0.0-1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConsistencyResult:
    """Result of consistency check"""
    agent_id: str
    consistent: bool
    contradictions: List[Contradiction]
    warnings: List[str]
    consistency_score: float  # 0.0 (inconsistent) to 1.0 (consistent)
    timestamp: datetime = field(default_factory=datetime.now)


class ConsistencyChecker:
    """
    Checks for consistency and contradictions.

    Provides:
    - Statement tracking across time
    - Contradiction detection
    - Consistency scoring
    - Temporal consistency analysis
    """

    def __init__(self):
        self.statements: Dict[str, Statement] = {}
        self.agent_statements: Dict[str, List[str]] = {}  # agent_id -> statement_ids
        self.contradictions: List[Contradiction] = []
        self.consistency_checks: List[ConsistencyResult] = []

    def add_statement(
        self,
        agent_id: str,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Statement:
        """
        Add a statement to track.

        Args:
            agent_id: Agent making statement
            content: Statement content
            context: Statement context

        Returns:
            Statement object
        """
        statement = Statement(
            statement_id=f"stmt_{uuid4().hex[:12]}",
            agent_id=agent_id,
            content=content,
            timestamp=datetime.now(),
            context=context or {}
        )

        self.statements[statement.statement_id] = statement

        if agent_id not in self.agent_statements:
            self.agent_statements[agent_id] = []

        self.agent_statements[agent_id].append(statement.statement_id)

        return statement

    def check_consistency(self, agent_id: str) -> ConsistencyResult:
        """
        Check consistency of all statements from an agent.

        Args:
            agent_id: Agent to check

        Returns:
            ConsistencyResult
        """
        if agent_id not in self.agent_statements:
            return ConsistencyResult(
                agent_id=agent_id,
                consistent=True,
                contradictions=[],
                warnings=["No statements to check"],
                consistency_score=1.0
            )

        statement_ids = self.agent_statements[agent_id]
        statements = [self.statements[sid] for sid in statement_ids]

        contradictions = []
        warnings = []

        # Check each pair of statements for contradictions
        for i in range(len(statements)):
            for j in range(i + 1, len(statements)):
                stmt1 = statements[i]
                stmt2 = statements[j]

                contradiction = self._detect_contradiction(stmt1, stmt2)
                if contradiction:
                    contradictions.append(contradiction)
                    self.contradictions.append(contradiction)

        # Calculate consistency score
        total_pairs = len(statements) * (len(statements) - 1) / 2
        if total_pairs == 0:
            consistency_score = 1.0
        else:
            consistency_score = 1.0 - (len(contradictions) / total_pairs)

        # Generate warnings
        if len(contradictions) > 0:
            warnings.append(f"Found {len(contradictions)} contradiction(s)")

        if len(statements) > 100:
            warnings.append("Large number of statements may slow consistency checking")

        result = ConsistencyResult(
            agent_id=agent_id,
            consistent=len(contradictions) == 0,
            contradictions=contradictions,
            warnings=warnings,
            consistency_score=consistency_score
        )

        self.consistency_checks.append(result)
        return result

    def _detect_contradiction(
        self,
        stmt1: Statement,
        stmt2: Statement
    ) -> Optional[Contradiction]:
        """
        Detect if two statements contradict each other.

        Note: Simple implementation using keyword matching.
        Production systems would use semantic understanding.
        """
        content1 = stmt1.content.lower()
        content2 = stmt2.content.lower()

        # Look for direct negation patterns
        contradiction = self._check_negation_contradiction(content1, content2)
        if contradiction:
            return Contradiction(
                contradiction_id=f"contra_{uuid4().hex[:12]}",
                statement1_id=stmt1.statement_id,
                statement2_id=stmt2.statement_id,
                description=contradiction,
                severity="HIGH",
                confidence=0.8
            )

        # Look for opposite value claims
        contradiction = self._check_value_contradiction(content1, content2)
        if contradiction:
            return Contradiction(
                contradiction_id=f"contra_{uuid4().hex[:12]}",
                statement1_id=stmt1.statement_id,
                statement2_id=stmt2.statement_id,
                description=contradiction,
                severity="MEDIUM",
                confidence=0.6
            )

        return None

    def _check_negation_contradiction(
        self,
        content1: str,
        content2: str
    ) -> Optional[str]:
        """Check for negation-based contradictions"""
        # Negation indicators
        affirmative = {'is', 'are', 'will', 'can', 'has', 'have', 'should'}
        negative = {'is not', 'are not', 'will not', 'cannot', 'has not',
                   'have not', 'should not', "isn't", "aren't", "won't",
                   "can't", "hasn't", "haven't", "shouldn't"}

        # Check if one affirms and one negates similar content
        has_affirmative1 = any(word in content1 for word in affirmative)
        has_negative1 = any(word in content1 for word in negative)

        has_affirmative2 = any(word in content2 for word in affirmative)
        has_negative2 = any(word in content2 for word in negative)

        # If one is affirmative and other is negative
        if (has_affirmative1 and has_negative2) or (has_negative1 and has_affirmative2):
            # Check for topic similarity
            words1 = set(content1.split())
            words2 = set(content2.split())
            common = words1 & words2

            if len(common) > 3:  # Talking about similar things
                return f"Negation contradiction: one affirms, one negates similar content ({len(common)} common words)"

        return None

    def _check_value_contradiction(
        self,
        content1: str,
        content2: str
    ) -> Optional[str]:
        """Check for contradicting values"""
        # Look for opposite value pairs
        opposites = [
            ('safe', 'unsafe'),
            ('allowed', 'forbidden'),
            ('enabled', 'disabled'),
            ('true', 'false'),
            ('yes', 'no'),
            ('always', 'never'),
            ('high', 'low'),
            ('increase', 'decrease'),
        ]

        for word1, word2 in opposites:
            if (word1 in content1 and word2 in content2) or \
               (word2 in content1 and word1 in content2):
                # Check if talking about same subject
                words1 = set(content1.split())
                words2 = set(content2.split())
                common = words1 & words2

                if len(common) > 2:
                    return f"Value contradiction: opposite values ({word1}/{word2}) for similar subject"

        return None

    def check_temporal_consistency(
        self,
        agent_id: str,
        time_window_hours: int = 24
    ) -> ConsistencyResult:
        """
        Check consistency within a time window.

        Args:
            agent_id: Agent to check
            time_window_hours: Hours to look back

        Returns:
            ConsistencyResult for time window
        """
        if agent_id not in self.agent_statements:
            return ConsistencyResult(
                agent_id=agent_id,
                consistent=True,
                contradictions=[],
                warnings=["No statements to check"],
                consistency_score=1.0
            )

        # Filter statements by time window
        now = datetime.now()
        recent_statements = []

        for stmt_id in self.agent_statements[agent_id]:
            stmt = self.statements[stmt_id]
            hours_ago = (now - stmt.timestamp).total_seconds() / 3600

            if hours_ago <= time_window_hours:
                recent_statements.append(stmt)

        # Check consistency of recent statements
        contradictions = []

        for i in range(len(recent_statements)):
            for j in range(i + 1, len(recent_statements)):
                contradiction = self._detect_contradiction(
                    recent_statements[i],
                    recent_statements[j]
                )
                if contradiction:
                    contradictions.append(contradiction)

        consistency_score = 1.0
        if len(recent_statements) > 1:
            total_pairs = len(recent_statements) * (len(recent_statements) - 1) / 2
            consistency_score = 1.0 - (len(contradictions) / total_pairs)

        return ConsistencyResult(
            agent_id=agent_id,
            consistent=len(contradictions) == 0,
            contradictions=contradictions,
            warnings=[],
            consistency_score=consistency_score
        )

    def get_agent_contradictions(self, agent_id: str) -> List[Contradiction]:
        """Get all contradictions for an agent"""
        if agent_id not in self.agent_statements:
            return []

        statement_ids = set(self.agent_statements[agent_id])

        return [
            c for c in self.contradictions
            if c.statement1_id in statement_ids or c.statement2_id in statement_ids
        ]

    def resolve_contradiction(
        self,
        contradiction_id: str,
        resolution: str
    ) -> bool:
        """Mark a contradiction as resolved"""
        for contradiction in self.contradictions:
            if contradiction.contradiction_id == contradiction_id:
                contradiction.description += f" [Resolved: {resolution}]"
                return True
        return False

    def get_statement(self, statement_id: str) -> Optional[Statement]:
        """Get a specific statement"""
        return self.statements.get(statement_id)

    def get_agent_statements(
        self,
        agent_id: str,
        limit: Optional[int] = None
    ) -> List[Statement]:
        """Get all statements from an agent"""
        if agent_id not in self.agent_statements:
            return []

        statement_ids = self.agent_statements[agent_id]
        statements = [self.statements[sid] for sid in statement_ids]

        # Sort by timestamp (newest first)
        statements.sort(key=lambda s: s.timestamp, reverse=True)

        if limit:
            statements = statements[:limit]

        return statements

    def get_statistics(self) -> Dict[str, Any]:
        """Get consistency checking statistics"""
        total_statements = len(self.statements)
        total_contradictions = len(self.contradictions)
        total_agents = len(self.agent_statements)

        # Count by severity
        severity_counts = {
            'low': 0,
            'medium': 0,
            'high': 0
        }

        for contradiction in self.contradictions:
            severity_counts[contradiction.severity.lower()] += 1

        # Average consistency score
        if self.consistency_checks:
            avg_consistency = sum(c.consistency_score for c in self.consistency_checks) / len(self.consistency_checks)
        else:
            avg_consistency = 1.0

        return {
            'total_statements': total_statements,
            'total_contradictions': total_contradictions,
            'total_agents': total_agents,
            'contradictions_by_severity': severity_counts,
            'average_consistency_score': avg_consistency,
            'total_checks': len(self.consistency_checks)
        }
