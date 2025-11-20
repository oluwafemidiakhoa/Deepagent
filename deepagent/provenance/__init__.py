"""
Foundation #3: Identity & Provenance

Complete data lineage tracking, source attribution, and trust evaluation.
"""

from deepagent.provenance.provenance_tracker import (
    ProvenanceTracker,
    DataLineage,
    SourceAttribution
)

from deepagent.provenance.trust_scorer import (
    TrustScorer,
    TrustScore,
    SourceCredibility
)

from deepagent.provenance.signature_manager import (
    SignatureManager,
    DataSignature,
    VerificationResult
)

__all__ = [
    "ProvenanceTracker",
    "DataLineage",
    "SourceAttribution",
    "TrustScorer",
    "TrustScore",
    "SourceCredibility",
    "SignatureManager",
    "DataSignature",
    "VerificationResult"
]
