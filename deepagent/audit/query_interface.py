"""
Audit Query Interface - Foundation #7

Flexible querying and export of audit logs.
Provides SQL-like query API with filtering, aggregation, and export capabilities.
"""

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from collections import defaultdict
import statistics

from deepagent.audit.audit_logger import (
    AuditLogger,
    AuditRecord,
    EventType,
    EventSeverity
)


@dataclass
class QueryFilters:
    """
    Flexible filters for querying audit records.

    Supports time-range, identity, event type, security, and pagination filtering.
    """

    # Time range
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Identity
    session_ids: Optional[List[str]] = None
    user_ids: Optional[List[str]] = None

    # Event filtering
    event_types: Optional[List[EventType]] = None
    severity_levels: Optional[List[EventSeverity]] = None

    # Security filtering
    min_risk_score: Optional[float] = None
    max_risk_score: Optional[float] = None
    only_blocked: bool = False
    only_attacks: bool = False
    only_errors: bool = False

    # Tool filtering
    tool_names: Optional[List[str]] = None
    action_types: Optional[List[str]] = None

    # Pagination
    limit: Optional[int] = None
    offset: int = 0

    # Sorting
    sort_by: str = "timestamp"
    sort_order: str = "desc"  # asc or desc

    def to_dict(self) -> Dict[str, Any]:
        """Convert filters to dictionary for logging/debugging"""
        return {
            'time_range': f"{self.start_time} to {self.end_time}" if self.start_time or self.end_time else None,
            'sessions': len(self.session_ids) if self.session_ids else None,
            'users': len(self.user_ids) if self.user_ids else None,
            'event_types': [et.value for et in self.event_types] if self.event_types else None,
            'min_risk': self.min_risk_score,
            'only_blocked': self.only_blocked,
            'only_attacks': self.only_attacks,
            'limit': self.limit,
            'offset': self.offset,
            'sort': f"{self.sort_by} {self.sort_order}"
        }


@dataclass
class QueryResult:
    """
    Result of a query operation.

    Contains matched records and metadata about the query.
    """

    # Results
    records: List[AuditRecord]
    total_count: int  # Total matches (before pagination)

    # Metadata
    query_time_ms: float
    filters_used: QueryFilters

    # Pagination
    has_more: bool
    next_offset: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export"""
        return {
            'total_count': self.total_count,
            'returned_count': len(self.records),
            'has_more': self.has_more,
            'next_offset': self.next_offset,
            'query_time_ms': self.query_time_ms,
            'filters': self.filters_used.to_dict(),
            'records': [r.to_dict() for r in self.records]
        }


class AuditQueryInterface:
    """
    Flexible query interface for audit logs.

    Provides filtering, aggregation, statistics, and export capabilities.
    """

    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    def query(self, filters: QueryFilters) -> QueryResult:
        """
        Query audit records with flexible filtering.

        Args:
            filters: QueryFilters specifying what to retrieve

        Returns:
            QueryResult with matched records and metadata
        """
        start_time = datetime.now()

        # Build base filter for audit logger
        base_filters = {}

        if filters.start_time:
            base_filters['start_time'] = filters.start_time
        if filters.end_time:
            base_filters['end_time'] = filters.end_time

        # Get records from audit logger
        all_records = self.audit_logger.read_records(base_filters)

        # Apply additional filters in memory
        filtered_records = self._apply_filters(all_records, filters)

        # Sort records
        sorted_records = self._sort_records(filtered_records, filters.sort_by, filters.sort_order)

        # Get total count before pagination
        total_count = len(sorted_records)

        # Apply pagination
        paginated_records = sorted_records[filters.offset:]
        if filters.limit:
            paginated_records = paginated_records[:filters.limit]

        # Calculate pagination metadata
        has_more = (filters.offset + len(paginated_records)) < total_count
        next_offset = filters.offset + len(paginated_records) if has_more else None

        # Calculate query time
        query_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        return QueryResult(
            records=paginated_records,
            total_count=total_count,
            query_time_ms=query_time_ms,
            filters_used=filters,
            has_more=has_more,
            next_offset=next_offset
        )

    def count(self, filters: QueryFilters) -> int:
        """
        Count records matching filters without retrieving them.

        Args:
            filters: QueryFilters specifying what to count

        Returns:
            Number of matching records
        """
        result = self.query(filters)
        return result.total_count

    def aggregate(
        self,
        field: str,
        operation: str,
        filters: QueryFilters
    ) -> Any:
        """
        Aggregate a field across matching records.

        Args:
            field: Field to aggregate (e.g., 'phase1_risk_score', 'duration_ms')
            operation: Aggregation operation ('sum', 'avg', 'min', 'max', 'count')
            filters: QueryFilters to select records

        Returns:
            Aggregated value
        """
        result = self.query(filters)
        records = result.records

        if not records:
            return None

        # Extract field values
        values = []
        for record in records:
            value = getattr(record, field, None)
            if value is not None:
                values.append(value)

        if not values:
            return None

        # Perform aggregation
        if operation == 'sum':
            return sum(values)
        elif operation == 'avg':
            return statistics.mean(values)
        elif operation == 'min':
            return min(values)
        elif operation == 'max':
            return max(values)
        elif operation == 'count':
            return len(values)
        else:
            raise ValueError(f"Unknown aggregation operation: {operation}")

    def group_by(
        self,
        field: str,
        filters: QueryFilters
    ) -> Dict[Any, List[AuditRecord]]:
        """
        Group records by a field value.

        Args:
            field: Field to group by (e.g., 'user_id', 'tool_name', 'event_type')
            filters: QueryFilters to select records

        Returns:
            Dictionary mapping field values to lists of records
        """
        result = self.query(filters)
        records = result.records

        groups = defaultdict(list)
        for record in records:
            value = getattr(record, field, None)
            # Convert enums to values
            if isinstance(value, (EventType, EventSeverity)):
                value = value.value
            groups[value].append(record)

        return dict(groups)

    def statistics(self, filters: QueryFilters) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for matching records.

        Args:
            filters: QueryFilters to select records

        Returns:
            Dictionary with various statistics
        """
        result = self.query(filters)
        records = result.records

        if not records:
            return {
                'total_records': 0,
                'error': 'No records match filters'
            }

        # Basic counts
        stats = {
            'total_records': len(records),
            'unique_sessions': len(set(r.session_id for r in records)),
            'unique_users': len(set(r.user_id for r in records)),
        }

        # Event type breakdown
        event_counts = defaultdict(int)
        for record in records:
            event_counts[record.event_type.value] += 1
        stats['event_types'] = dict(event_counts)

        # Severity breakdown
        severity_counts = defaultdict(int)
        for record in records:
            severity_counts[record.severity.value] += 1
        stats['severity_levels'] = dict(severity_counts)

        # Risk statistics
        risk_scores = [r.phase1_risk_score for r in records if r.phase1_risk_score is not None]
        if risk_scores:
            stats['risk'] = {
                'min': min(risk_scores),
                'max': max(risk_scores),
                'avg': statistics.mean(risk_scores),
                'median': statistics.median(risk_scores)
            }

        # Performance statistics
        durations = [r.duration_ms for r in records if r.duration_ms is not None]
        if durations:
            stats['performance'] = {
                'min_ms': min(durations),
                'max_ms': max(durations),
                'avg_ms': statistics.mean(durations),
                'median_ms': statistics.median(durations)
            }

        # Security events
        stats['security'] = {
            'total_blocked': len([r for r in records if not r.allowed]),
            'total_errors': len([r for r in records if r.error]),
            'attacks_detected': len([r for r in records if r.event_type == EventType.ATTACK_DETECTED]),
            'escalations': len([r for r in records if r.event_type == EventType.ESCALATION_DETECTED]),
            'drifts': len([r for r in records if r.event_type == EventType.GOAL_DRIFT])
        }

        # Tool usage
        tool_counts = defaultdict(int)
        for record in records:
            if record.tool_name:
                tool_counts[record.tool_name] += 1
        stats['top_tools'] = dict(sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:10])

        return stats

    def export(
        self,
        filters: QueryFilters,
        format: str,
        output_path: str
    ) -> None:
        """
        Export query results to a file.

        Args:
            filters: QueryFilters to select records
            format: Output format ('json', 'csv', 'markdown', 'text')
            output_path: Path to output file
        """
        result = self.query(filters)
        records = result.records

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            self._export_json(records, output_file)
        elif format == 'csv':
            self._export_csv(records, output_file)
        elif format == 'markdown':
            self._export_markdown(records, output_file)
        elif format == 'text':
            self._export_text(records, output_file)
        else:
            raise ValueError(f"Unknown export format: {format}")

    def _apply_filters(
        self,
        records: List[AuditRecord],
        filters: QueryFilters
    ) -> List[AuditRecord]:
        """Apply filters to records in memory"""
        filtered = records

        # Session filtering
        if filters.session_ids:
            filtered = [r for r in filtered if r.session_id in filters.session_ids]

        # User filtering
        if filters.user_ids:
            filtered = [r for r in filtered if r.user_id in filters.user_ids]

        # Event type filtering
        if filters.event_types:
            filtered = [r for r in filtered if r.event_type in filters.event_types]

        # Severity filtering
        if filters.severity_levels:
            filtered = [r for r in filtered if r.severity in filters.severity_levels]

        # Risk score filtering
        if filters.min_risk_score is not None:
            filtered = [r for r in filtered if r.phase1_risk_score is not None and r.phase1_risk_score >= filters.min_risk_score]

        if filters.max_risk_score is not None:
            filtered = [r for r in filtered if r.phase1_risk_score is not None and r.phase1_risk_score <= filters.max_risk_score]

        # Blocked actions only
        if filters.only_blocked:
            filtered = [r for r in filtered if not r.allowed]

        # Attacks only
        if filters.only_attacks:
            filtered = [r for r in filtered if r.event_type == EventType.ATTACK_DETECTED]

        # Errors only
        if filters.only_errors:
            filtered = [r for r in filtered if r.error is not None]

        # Tool filtering
        if filters.tool_names:
            filtered = [r for r in filtered if r.tool_name in filters.tool_names]

        # Action type filtering
        if filters.action_types:
            filtered = [r for r in filtered if r.action_type in filters.action_types]

        return filtered

    def _sort_records(
        self,
        records: List[AuditRecord],
        sort_by: str,
        sort_order: str
    ) -> List[AuditRecord]:
        """Sort records by field"""
        reverse = (sort_order == "desc")

        # Handle special sorting cases
        if sort_by == "timestamp":
            return sorted(records, key=lambda r: r.timestamp, reverse=reverse)
        elif sort_by == "risk_score":
            return sorted(records, key=lambda r: r.phase1_risk_score or 0, reverse=reverse)
        elif sort_by == "duration":
            return sorted(records, key=lambda r: r.duration_ms or 0, reverse=reverse)
        else:
            # Generic attribute sorting
            return sorted(records, key=lambda r: getattr(r, sort_by, ""), reverse=reverse)

    def _export_json(self, records: List[AuditRecord], output_path: Path):
        """Export records to JSON file"""
        data = {
            'total_records': len(records),
            'exported_at': datetime.now().isoformat(),
            'records': [r.to_dict() for r in records]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def _export_csv(self, records: List[AuditRecord], output_path: Path):
        """Export records to CSV file"""
        if not records:
            return

        # Define CSV columns
        columns = [
            'record_id', 'timestamp', 'session_id', 'user_id',
            'event_type', 'severity', 'tool_name', 'action_type',
            'phase1_risk_score', 'phase1_decision', 'allowed',
            'error', 'duration_ms'
        ]

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for record in records:
                row = {
                    'record_id': record.record_id,
                    'timestamp': record.timestamp.isoformat(),
                    'session_id': record.session_id,
                    'user_id': record.user_id,
                    'event_type': record.event_type.value,
                    'severity': record.severity.value,
                    'tool_name': record.tool_name or '',
                    'action_type': record.action_type or '',
                    'phase1_risk_score': record.phase1_risk_score or '',
                    'phase1_decision': record.phase1_decision or '',
                    'allowed': record.allowed,
                    'error': record.error or '',
                    'duration_ms': record.duration_ms or ''
                }
                writer.writerow(row)

    def _export_markdown(self, records: List[AuditRecord], output_path: Path):
        """Export records to Markdown table"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Audit Log Export\n\n")
            f.write(f"**Exported**: {datetime.now().isoformat()}\n\n")
            f.write(f"**Total Records**: {len(records)}\n\n")

            f.write("## Records\n\n")
            f.write("| Timestamp | Session | User | Event | Severity | Tool | Risk | Allowed |\n")
            f.write("|-----------|---------|------|-------|----------|------|------|----------|\n")

            for record in records:
                timestamp = record.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                session = record.session_id[:8]
                user = record.user_id[:10]
                event = record.event_type.value
                severity = record.severity.value
                tool = record.tool_name or '-'
                risk = f"{record.phase1_risk_score:.0%}" if record.phase1_risk_score else '-'
                allowed = 'Yes' if record.allowed else 'No'

                f.write(f"| {timestamp} | {session} | {user} | {event} | {severity} | {tool} | {risk} | {allowed} |\n")

    def _export_text(self, records: List[AuditRecord], output_path: Path):
        """Export records to plain text"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("AUDIT LOG EXPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Exported: {datetime.now().isoformat()}\n")
            f.write(f"Total Records: {len(records)}\n\n")

            for i, record in enumerate(records, 1):
                f.write(f"\nRecord {i}/{len(records)}\n")
                f.write("-" * 80 + "\n")
                f.write(f"ID: {record.record_id}\n")
                f.write(f"Time: {record.timestamp.isoformat()}\n")
                f.write(f"Session: {record.session_id}\n")
                f.write(f"User: {record.user_id}\n")
                f.write(f"Event: {record.event_type.value}\n")
                f.write(f"Severity: {record.severity.value}\n")

                if record.tool_name:
                    f.write(f"Tool: {record.tool_name}\n")
                if record.action_type:
                    f.write(f"Action: {record.action_type}\n")
                if record.phase1_risk_score is not None:
                    f.write(f"Risk Score: {record.phase1_risk_score:.0%}\n")

                f.write(f"Allowed: {'Yes' if record.allowed else 'No'}\n")

                if record.error:
                    f.write(f"Error: {record.error}\n")
                if record.duration_ms:
                    f.write(f"Duration: {record.duration_ms:.1f}ms\n")
