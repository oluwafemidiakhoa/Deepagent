"""
Foundation #10: Deception Detection

Detects deceptive behavior and verifies claim truthfulness.
"""

from .truth_evaluator import TruthEvaluator, TruthScore, ClaimVerification
from .consistency_checker import ConsistencyChecker, ConsistencyResult, Contradiction
from .deception_scorer import DeceptionScorer, DeceptionScore, DeceptionIndicator

__all__ = [
    'TruthEvaluator',
    'TruthScore',
    'ClaimVerification',
    'ConsistencyChecker',
    'ConsistencyResult',
    'Contradiction',
    'DeceptionScorer',
    'DeceptionScore',
    'DeceptionIndicator',
]
