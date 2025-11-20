"""
Memory Validator

Validates memory entries and tracks data provenance.
Part of Foundation #2: Memory Firewalls
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json


class IntegrityStatus(Enum):
    """Status of memory entry integrity"""
    VALID = "valid"
    TAMPERED = "tampered"
    MISSING_PROVENANCE = "missing_provenance"
    INCONSISTENT = "inconsistent"
    SUSPICIOUS = "suspicious"


class ProvenanceType(Enum):
    """Type of data provenance"""
    USER_INPUT = "user_input"
    TOOL_OUTPUT = "tool_output"
    AGENT_GENERATED = "agent_generated"
    EXTERNAL_SOURCE = "external_source"
    DERIVED = "derived"
    UNKNOWN = "unknown"


@dataclass
class ProvenanceRecord:
    """
    Record of where data came from
    """
    source_type: ProvenanceType
    source_id: str
    timestamp: datetime
    operation: str  # What created this data
    parent_records: List[str] = field(default_factory=list)  # Parent provenance IDs
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for hashing"""
        return {
            "source_type": self.source_type.value,
            "source_id": self.source_id,
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation,
            "parent_records": self.parent_records,
            "metadata": self.metadata
        }


@dataclass
class MemoryEntry:
    """
    Entry in agent's memory with integrity checking
    """
    entry_id: str
    content: Any
    timestamp: datetime
    provenance: ProvenanceRecord

    # Integrity
    content_hash: Optional[str] = None
    provenance_hash: Optional[str] = None

    # Metadata
    tags: Set[str] = field(default_factory=set)
    access_count: int = 0
    last_modified: Optional[datetime] = None

    def __post_init__(self):
        """Compute hashes after initialization"""
        if self.content_hash is None:
            self.content_hash = self._compute_content_hash()
        if self.provenance_hash is None:
            self.provenance_hash = self._compute_provenance_hash()

    def _compute_content_hash(self) -> str:
        """Compute hash of content"""
        content_str = json.dumps(self.content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def _compute_provenance_hash(self) -> str:
        """Compute hash of provenance"""
        prov_str = json.dumps(self.provenance.to_dict(), sort_keys=True)
        return hashlib.sha256(prov_str.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify content and provenance haven't been tampered with"""
        current_content_hash = self._compute_content_hash()
        current_prov_hash = self._compute_provenance_hash()

        return (
            current_content_hash == self.content_hash and
            current_prov_hash == self.provenance_hash
        )


@dataclass
class ValidationResult:
    """
    Result of memory validation
    """
    is_valid: bool
    status: IntegrityStatus
    issues: List[str] = field(default_factory=list)
    risk_score: float = 0.0  # 0-1 scale
    explanation: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ProvenanceChain:
    """
    Complete chain of data provenance
    """
    entry_id: str
    chain: List[ProvenanceRecord]
    is_complete: bool
    missing_links: List[str] = field(default_factory=list)
    depth: int = 0

    def has_untrusted_sources(self) -> bool:
        """Check if chain contains untrusted sources"""
        untrusted = {ProvenanceType.UNKNOWN, ProvenanceType.EXTERNAL_SOURCE}
        return any(record.source_type in untrusted for record in self.chain)

    def get_root_source(self) -> Optional[ProvenanceRecord]:
        """Get the root (original) source"""
        if not self.chain:
            return None
        return self.chain[0]


class MemoryValidator:
    """
    Validates memory entries and tracks provenance

    Detects:
    - Tampered memory entries
    - Inconsistent data
    - Missing provenance chains
    - Suspicious modifications
    """

    def __init__(self):
        """Initialize validator"""
        # Memory storage
        self.memory_entries: Dict[str, MemoryEntry] = {}
        self.provenance_graph: Dict[str, List[str]] = {}  # entry_id -> parent_ids

        # Configuration
        self.max_provenance_depth = 10
        self.require_provenance = True

        # Tracking
        self.validation_history: List[ValidationResult] = []

    def add_entry(
        self,
        entry_id: str,
        content: Any,
        provenance: ProvenanceRecord,
        tags: Optional[Set[str]] = None
    ) -> MemoryEntry:
        """
        Add new memory entry with provenance

        Args:
            entry_id: Unique identifier
            content: Memory content
            provenance: Provenance record
            tags: Optional tags

        Returns:
            Created memory entry
        """
        entry = MemoryEntry(
            entry_id=entry_id,
            content=content,
            timestamp=datetime.now(),
            provenance=provenance,
            tags=tags or set()
        )

        self.memory_entries[entry_id] = entry

        # Update provenance graph
        if provenance.parent_records:
            self.provenance_graph[entry_id] = provenance.parent_records

        return entry

    def validate_entry(self, entry_id: str) -> ValidationResult:
        """
        Validate a memory entry

        Args:
            entry_id: Entry to validate

        Returns:
            ValidationResult with validation status
        """
        if entry_id not in self.memory_entries:
            return ValidationResult(
                is_valid=False,
                status=IntegrityStatus.MISSING_PROVENANCE,
                issues=["Entry not found"],
                risk_score=1.0,
                explanation=f"Entry {entry_id} does not exist"
            )

        entry = self.memory_entries[entry_id]
        issues = []
        risk_score = 0.0

        # Check integrity
        if not entry.verify_integrity():
            issues.append("Content or provenance hash mismatch (tampering detected)")
            risk_score += 0.5

        # Check provenance
        if self.require_provenance:
            prov_result = self._validate_provenance(entry)
            if not prov_result.is_valid:
                issues.extend(prov_result.issues)
                risk_score += prov_result.risk_score

        # Check for suspicious patterns
        suspicious = self._check_suspicious_patterns(entry)
        if suspicious:
            issues.extend(suspicious)
            risk_score += 0.2

        # Determine status
        if not entry.verify_integrity():
            status = IntegrityStatus.TAMPERED
        elif issues:
            if "provenance" in " ".join(issues).lower():
                status = IntegrityStatus.MISSING_PROVENANCE
            elif "inconsistent" in " ".join(issues).lower():
                status = IntegrityStatus.INCONSISTENT
            else:
                status = IntegrityStatus.SUSPICIOUS
        else:
            status = IntegrityStatus.VALID

        is_valid = status == IntegrityStatus.VALID

        if is_valid:
            explanation = f"Entry {entry_id} is valid"
        else:
            explanation = f"Entry {entry_id} has issues: {', '.join(issues[:3])}"

        result = ValidationResult(
            is_valid=is_valid,
            status=status,
            issues=issues,
            risk_score=min(risk_score, 1.0),
            explanation=explanation
        )

        self.validation_history.append(result)
        return result

    def _validate_provenance(self, entry: MemoryEntry) -> ValidationResult:
        """Validate provenance chain"""
        issues = []
        risk_score = 0.0

        # Check if provenance type is known
        if entry.provenance.source_type == ProvenanceType.UNKNOWN:
            issues.append("Unknown provenance source")
            risk_score += 0.3

        # Check parent records exist
        for parent_id in entry.provenance.parent_records:
            if parent_id not in self.memory_entries:
                issues.append(f"Missing parent record: {parent_id}")
                risk_score += 0.2

        # Check for circular references
        if self._has_circular_provenance(entry.entry_id):
            issues.append("Circular provenance chain detected")
            risk_score += 0.4

        return ValidationResult(
            is_valid=len(issues) == 0,
            status=IntegrityStatus.VALID if len(issues) == 0 else IntegrityStatus.MISSING_PROVENANCE,
            issues=issues,
            risk_score=min(risk_score, 1.0)
        )

    def _check_suspicious_patterns(self, entry: MemoryEntry) -> List[str]:
        """Check for suspicious patterns in entry"""
        issues = []

        # Check for excessive modifications
        if entry.access_count > 50:
            issues.append(f"Excessive access count: {entry.access_count}")

        # Check timestamp consistency
        if entry.last_modified and entry.last_modified < entry.timestamp:
            issues.append("Last modified timestamp before creation timestamp")

        # Check for external sources without verification
        if entry.provenance.source_type == ProvenanceType.EXTERNAL_SOURCE:
            if "verified" not in entry.provenance.metadata:
                issues.append("External source without verification metadata")

        return issues

    def get_provenance_chain(self, entry_id: str) -> ProvenanceChain:
        """
        Get complete provenance chain for entry

        Args:
            entry_id: Entry to trace

        Returns:
            ProvenanceChain with full chain
        """
        if entry_id not in self.memory_entries:
            return ProvenanceChain(
                entry_id=entry_id,
                chain=[],
                is_complete=False,
                missing_links=[entry_id],
                depth=0
            )

        chain = []
        visited = set()
        missing = []

        def trace_back(current_id: str, depth: int = 0):
            """Recursively trace provenance"""
            if depth > self.max_provenance_depth:
                return

            if current_id in visited:
                return

            visited.add(current_id)

            if current_id not in self.memory_entries:
                missing.append(current_id)
                return

            entry = self.memory_entries[current_id]
            chain.append(entry.provenance)

            # Trace parents
            for parent_id in entry.provenance.parent_records:
                trace_back(parent_id, depth + 1)

        trace_back(entry_id)

        return ProvenanceChain(
            entry_id=entry_id,
            chain=chain,
            is_complete=len(missing) == 0,
            missing_links=missing,
            depth=len(chain)
        )

    def _has_circular_provenance(self, entry_id: str, visited: Optional[Set[str]] = None) -> bool:
        """Check for circular references in provenance"""
        if visited is None:
            visited = set()

        if entry_id in visited:
            return True

        if entry_id not in self.memory_entries:
            return False

        visited.add(entry_id)
        entry = self.memory_entries[entry_id]

        for parent_id in entry.provenance.parent_records:
            if self._has_circular_provenance(parent_id, visited.copy()):
                return True

        return False

    def detect_tampering(self, entry_id: str) -> bool:
        """
        Check if entry has been tampered with

        Args:
            entry_id: Entry to check

        Returns:
            True if tampering detected
        """
        if entry_id not in self.memory_entries:
            return True

        return not self.memory_entries[entry_id].verify_integrity()

    def validate_all(self) -> Dict[str, ValidationResult]:
        """
        Validate all memory entries

        Returns:
            Dictionary of entry_id -> ValidationResult
        """
        results = {}

        for entry_id in self.memory_entries:
            results[entry_id] = self.validate_entry(entry_id)

        return results

    def get_summary(self) -> Dict[str, Any]:
        """
        Get validation summary

        Returns:
            Dictionary with summary statistics
        """
        if not self.memory_entries:
            return {
                "total_entries": 0,
                "message": "No entries to validate"
            }

        # Validate all
        results = self.validate_all()

        valid_count = sum(1 for r in results.values() if r.is_valid)
        tampered_count = sum(1 for r in results.values() if r.status == IntegrityStatus.TAMPERED)
        missing_prov_count = sum(1 for r in results.values() if r.status == IntegrityStatus.MISSING_PROVENANCE)

        # Calculate average risk
        avg_risk = sum(r.risk_score for r in results.values()) / len(results)

        # Get issues breakdown
        all_issues = []
        for result in results.values():
            all_issues.extend(result.issues)

        return {
            "total_entries": len(self.memory_entries),
            "valid_entries": valid_count,
            "tampered_entries": tampered_count,
            "missing_provenance": missing_prov_count,
            "validation_rate": valid_count / len(self.memory_entries),
            "average_risk": avg_risk,
            "total_issues": len(all_issues),
            "unique_issue_types": len(set(all_issues))
        }

    def clear(self):
        """Clear all memory entries and history"""
        self.memory_entries.clear()
        self.provenance_graph.clear()
        self.validation_history.clear()
