"""
Trust Scorer - Foundation #3

Evaluates source credibility and data trustworthiness.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class SourceCredibility:
    """Credibility rating for a source"""
    source_id: str
    credibility_score: float  # 0.0 - 1.0
    reliability: float
    accuracy: float
    freshness: float


@dataclass
class TrustScore:
    """Overall trust score for data"""
    data_id: str
    overall_trust: float  # 0.0 - 1.0
    source_trust: float
    transformation_trust: float
    age_penalty: float


class TrustScorer:
    """Evaluates trust and credibility"""

    def __init__(self):
        self.source_scores: Dict[str, float] = {
            "user": 0.9,
            "database": 0.8,
            "llm": 0.6,
            "web": 0.5,
            "tool": 0.7
        }

    def score_source(self, source_type: str) -> float:
        """Score a source type"""
        return self.source_scores.get(source_type, 0.5)

    def calculate_trust(self, lineage) -> TrustScore:
        """Calculate overall trust score"""
        source_trust = self.score_source(lineage.original_source.source_type)

        # Decay trust through transformations
        transformation_trust = 1.0
        for _ in lineage.transformation_chain:
            transformation_trust *= 0.95

        age_hours = (datetime.now() - lineage.created_at).total_seconds() / 3600
        age_penalty = max(0.5, 1.0 - (age_hours / 168))  # Week decay

        overall = source_trust * transformation_trust * age_penalty

        return TrustScore(
            data_id=lineage.data_id,
            overall_trust=overall,
            source_trust=source_trust,
            transformation_trust=transformation_trust,
            age_penalty=age_penalty
        )


from datetime import datetime
