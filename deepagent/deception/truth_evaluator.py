"""
Truth Evaluator - Foundation #10

Evaluates truthfulness of agent claims and statements.
Verifies claims against known facts and evidence.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4


class TruthValue(Enum):
    """Truth value of a claim"""
    TRUE = "true"
    FALSE = "false"
    UNCERTAIN = "uncertain"
    UNVERIFIABLE = "unverifiable"


@dataclass
class TruthScore:
    """Score representing likelihood of truth"""
    score: float  # 0.0 (definitely false) to 1.0 (definitely true)
    confidence: float  # 0.0 (no confidence) to 1.0 (full confidence)
    evidence_count: int
    reasoning: str


@dataclass
class ClaimVerification:
    """Result of claim verification"""
    verification_id: str
    claim: str
    truth_value: TruthValue
    truth_score: TruthScore
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    verified_by: str = "TruthEvaluator"


@dataclass
class FactRecord:
    """A known fact for verification"""
    fact_id: str
    statement: str
    source: str
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class TruthEvaluator:
    """
    Evaluates truthfulness of claims.

    Provides:
    - Claim verification against known facts
    - Evidence collection and weighing
    - Truth scoring with confidence
    - Fact base management
    """

    def __init__(self):
        self.known_facts: Dict[str, FactRecord] = {}
        self.verifications: List[ClaimVerification] = []
        self.claim_patterns: Dict[str, float] = {}  # Pattern -> truth probability

    def add_fact(
        self,
        statement: str,
        source: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FactRecord:
        """
        Add a known fact to the fact base.

        Args:
            statement: The factual statement
            source: Source of the fact
            confidence: Confidence in fact (0.0-1.0)
            metadata: Additional metadata

        Returns:
            FactRecord
        """
        fact = FactRecord(
            fact_id=f"fact_{uuid4().hex[:12]}",
            statement=statement,
            source=source,
            confidence=confidence,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        self.known_facts[fact.fact_id] = fact
        return fact

    def verify_claim(
        self,
        claim: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ClaimVerification:
        """
        Verify a claim against known facts.

        Args:
            claim: Claim to verify
            context: Additional context

        Returns:
            ClaimVerification result
        """
        supporting = []
        contradicting = []

        # Check against known facts
        for fact in self.known_facts.values():
            similarity = self._calculate_similarity(claim, fact.statement)

            if similarity > 0.8:  # High similarity = supporting
                supporting.append(f"{fact.statement} (source: {fact.source})")
            elif similarity < 0.2:  # Low similarity might be contradicting
                if self._are_contradictory(claim, fact.statement):
                    contradicting.append(f"{fact.statement} (source: {fact.source})")

        # Calculate truth score
        truth_score = self._calculate_truth_score(
            claim,
            supporting,
            contradicting,
            context
        )

        # Determine truth value
        if truth_score.score >= 0.8 and truth_score.confidence >= 0.7:
            truth_value = TruthValue.TRUE
        elif truth_score.score <= 0.2 and truth_score.confidence >= 0.7:
            truth_value = TruthValue.FALSE
        elif truth_score.confidence < 0.3:
            truth_value = TruthValue.UNVERIFIABLE
        else:
            truth_value = TruthValue.UNCERTAIN

        verification = ClaimVerification(
            verification_id=f"verification_{uuid4().hex[:12]}",
            claim=claim,
            truth_value=truth_value,
            truth_score=truth_score,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting
        )

        self.verifications.append(verification)
        return verification

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.

        Note: This is a simple implementation using word overlap.
        Production systems would use embeddings or semantic similarity models.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _are_contradictory(self, claim1: str, claim2: str) -> bool:
        """
        Check if two claims contradict each other.

        Note: Simple implementation. Production systems would use
        natural language understanding and logic reasoning.
        """
        # Look for negation indicators
        negation_words = {'not', 'no', 'never', 'neither', 'none', 'nobody'}

        words1 = set(claim1.lower().split())
        words2 = set(claim2.lower().split())

        # If one has negation and the other doesn't, might be contradictory
        has_neg1 = bool(words1 & negation_words)
        has_neg2 = bool(words2 & negation_words)

        if has_neg1 != has_neg2:
            # Check if they're talking about similar things
            similarity = self._calculate_similarity(claim1, claim2)
            return similarity > 0.5

        return False

    def _calculate_truth_score(
        self,
        claim: str,
        supporting: List[str],
        contradicting: List[str],
        context: Optional[Dict[str, Any]]
    ) -> TruthScore:
        """Calculate truth score based on evidence"""
        total_evidence = len(supporting) + len(contradicting)

        if total_evidence == 0:
            return TruthScore(
                score=0.5,
                confidence=0.0,
                evidence_count=0,
                reasoning="No evidence available"
            )

        # Calculate score based on evidence ratio
        support_ratio = len(supporting) / total_evidence
        score = support_ratio

        # Calculate confidence based on evidence count
        confidence = min(1.0, total_evidence / 5.0)  # Max confidence at 5+ pieces

        reasoning_parts = []
        if supporting:
            reasoning_parts.append(f"{len(supporting)} supporting evidence(s)")
        if contradicting:
            reasoning_parts.append(f"{len(contradicting)} contradicting evidence(s)")

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Insufficient evidence"

        return TruthScore(
            score=score,
            confidence=confidence,
            evidence_count=total_evidence,
            reasoning=reasoning
        )

    def batch_verify_claims(
        self,
        claims: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[ClaimVerification]:
        """Verify multiple claims"""
        return [self.verify_claim(claim, context) for claim in claims]

    def get_claim_pattern_score(self, claim: str) -> Optional[float]:
        """Get truth probability for claim pattern"""
        # Check if claim matches known patterns
        for pattern, probability in self.claim_patterns.items():
            if pattern.lower() in claim.lower():
                return probability
        return None

    def add_claim_pattern(self, pattern: str, truth_probability: float) -> None:
        """Add a pattern with known truth probability"""
        self.claim_patterns[pattern] = truth_probability

    def get_verification_history(
        self,
        truth_value: Optional[TruthValue] = None,
        min_confidence: Optional[float] = None
    ) -> List[ClaimVerification]:
        """Get verification history, optionally filtered"""
        verifications = self.verifications

        if truth_value:
            verifications = [v for v in verifications if v.truth_value == truth_value]

        if min_confidence:
            verifications = [
                v for v in verifications
                if v.truth_score.confidence >= min_confidence
            ]

        return verifications

    def get_statistics(self) -> Dict[str, Any]:
        """Get truth evaluation statistics"""
        total_verifications = len(self.verifications)

        if total_verifications == 0:
            return {
                'total_verifications': 0,
                'total_facts': len(self.known_facts),
                'verifications_by_truth_value': {},
                'average_confidence': 0.0
            }

        # Count by truth value
        truth_counts = {
            'true': 0,
            'false': 0,
            'uncertain': 0,
            'unverifiable': 0
        }

        total_confidence = 0.0

        for verification in self.verifications:
            truth_counts[verification.truth_value.value] += 1
            total_confidence += verification.truth_score.confidence

        return {
            'total_verifications': total_verifications,
            'total_facts': len(self.known_facts),
            'verifications_by_truth_value': truth_counts,
            'average_confidence': total_confidence / total_verifications,
            'claim_patterns': len(self.claim_patterns)
        }
