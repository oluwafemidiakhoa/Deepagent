"""
Provenance Tracker - Foundation #3

Tracks complete data lineage from source to usage.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class SourceAttribution:
    """Attribution for a data source"""
    source_id: str
    source_type: str  # "llm", "tool", "user", "database", "web"
    source_name: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataLineage:
    """Complete lineage of a piece of data"""
    data_id: str
    original_source: SourceAttribution
    transformation_chain: List[SourceAttribution]
    current_value: Any
    created_at: datetime
    last_modified: datetime
    access_count: int = 0


class ProvenanceTracker:
    """Tracks data lineage and source attribution"""

    def __init__(self):
        self.lineages: Dict[str, DataLineage] = {}

    def track_data(
        self,
        data: Any,
        source_type: str,
        source_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Track new data with its source"""
        data_id = f"data_{uuid4().hex[:12]}"

        source = SourceAttribution(
            source_id=f"src_{uuid4().hex[:8]}",
            source_type=source_type,
            source_name=source_name,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        lineage = DataLineage(
            data_id=data_id,
            original_source=source,
            transformation_chain=[],
            current_value=data,
            created_at=datetime.now(),
            last_modified=datetime.now()
        )

        self.lineages[data_id] = lineage
        return data_id

    def add_transformation(
        self,
        data_id: str,
        transformer_type: str,
        transformer_name: str,
        new_value: Any
    ) -> bool:
        """Record a data transformation"""
        if data_id not in self.lineages:
            return False

        lineage = self.lineages[data_id]

        transformation = SourceAttribution(
            source_id=f"trans_{uuid4().hex[:8]}",
            source_type=transformer_type,
            source_name=transformer_name,
            timestamp=datetime.now()
        )

        lineage.transformation_chain.append(transformation)
        lineage.current_value = new_value
        lineage.last_modified = datetime.now()

        return True

    def get_lineage(self, data_id: str) -> Optional[DataLineage]:
        """Get complete lineage for data"""
        lineage = self.lineages.get(data_id)
        if lineage:
            lineage.access_count += 1
        return lineage

    def verify_source(self, data_id: str, expected_source_type: str) -> bool:
        """Verify data came from expected source type"""
        lineage = self.get_lineage(data_id)
        if not lineage:
            return False
        return lineage.original_source.source_type == expected_source_type
