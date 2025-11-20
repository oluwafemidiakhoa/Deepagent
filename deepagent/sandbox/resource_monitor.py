"""
Resource Monitor - Foundation #4

Monitors and enforces resource limits for sandboxed execution.
Tracks CPU, memory, disk, and network usage.
"""

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict


@dataclass
class ResourceUsage:
    """Current resource usage for a sandbox"""

    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    disk_mb: float = 0.0
    network_kb: float = 0.0
    execution_time: float = 0.0
    file_operations: int = 0
    violations: List[str] = field(default_factory=list)


@dataclass
class ResourceViolation:
    """Details of a resource limit violation"""

    sandbox_id: str
    violation_type: str
    limit: float
    actual: float
    timestamp: datetime
    message: str


class ResourceMonitor:
    """
    Monitors resource usage and enforces limits.

    Note: This is a simplified implementation. Production systems would
    use psutil, cgroups, or container metrics for accurate monitoring.
    """

    def __init__(self):
        self._monitoring: Dict[str, bool] = {}
        self._usage: Dict[str, ResourceUsage] = {}
        self._limits: Dict[str, 'ResourceLimits'] = {}
        self._start_times: Dict[str, datetime] = {}
        self._violations: Dict[str, List[ResourceViolation]] = defaultdict(list)
        self._lock = threading.Lock()

    def start_monitoring(
        self,
        sandbox_id: str,
        limits: 'ResourceLimits'
    ) -> None:
        """
        Start monitoring a sandbox.

        Args:
            sandbox_id: ID of sandbox to monitor
            limits: Resource limits to enforce
        """
        with self._lock:
            self._monitoring[sandbox_id] = True
            self._limits[sandbox_id] = limits
            self._usage[sandbox_id] = ResourceUsage()
            self._start_times[sandbox_id] = datetime.now()

    def stop_monitoring(self, sandbox_id: str) -> None:
        """Stop monitoring a sandbox"""
        with self._lock:
            if sandbox_id in self._monitoring:
                self._monitoring[sandbox_id] = False

    def record_file_operation(self, sandbox_id: str) -> None:
        """Record a file operation"""
        with self._lock:
            if sandbox_id in self._usage:
                self._usage[sandbox_id].file_operations += 1

                # Check limit
                if sandbox_id in self._limits:
                    limit = self._limits[sandbox_id].max_file_operations
                    actual = self._usage[sandbox_id].file_operations

                    if actual > limit:
                        violation = ResourceViolation(
                            sandbox_id=sandbox_id,
                            violation_type="file_operations",
                            limit=limit,
                            actual=actual,
                            timestamp=datetime.now(),
                            message=f"File operations ({actual}) exceeded limit ({limit})"
                        )
                        self._violations[sandbox_id].append(violation)
                        self._usage[sandbox_id].violations.append(violation.message)

    def update_usage(
        self,
        sandbox_id: str,
        cpu_percent: Optional[float] = None,
        memory_mb: Optional[float] = None,
        disk_mb: Optional[float] = None,
        network_kb: Optional[float] = None
    ) -> None:
        """
        Update resource usage for a sandbox.

        Args:
            sandbox_id: ID of sandbox
            cpu_percent: CPU usage percentage
            memory_mb: Memory usage in MB
            disk_mb: Disk usage in MB
            network_kb: Network usage in KB
        """
        with self._lock:
            if sandbox_id not in self._usage:
                return

            usage = self._usage[sandbox_id]

            if cpu_percent is not None:
                usage.cpu_percent = cpu_percent
            if memory_mb is not None:
                usage.memory_mb = memory_mb
            if disk_mb is not None:
                usage.disk_mb = disk_mb
            if network_kb is not None:
                usage.network_kb = network_kb

            # Update execution time
            if sandbox_id in self._start_times:
                usage.execution_time = (datetime.now() - self._start_times[sandbox_id]).total_seconds()

            # Check violations
            self._check_violations_internal(sandbox_id, usage)

    def _check_violations_internal(self, sandbox_id: str, usage: ResourceUsage) -> None:
        """Internal method to check for violations (must be called with lock)"""
        if sandbox_id not in self._limits:
            return

        limits = self._limits[sandbox_id]

        # Check CPU
        if usage.cpu_percent > limits.max_cpu_percent:
            violation = ResourceViolation(
                sandbox_id=sandbox_id,
                violation_type="cpu",
                limit=limits.max_cpu_percent,
                actual=usage.cpu_percent,
                timestamp=datetime.now(),
                message=f"CPU usage ({usage.cpu_percent:.1f}%) exceeded limit ({limits.max_cpu_percent}%)"
            )
            if violation.message not in usage.violations:
                self._violations[sandbox_id].append(violation)
                usage.violations.append(violation.message)

        # Check memory
        if usage.memory_mb > limits.max_memory_mb:
            violation = ResourceViolation(
                sandbox_id=sandbox_id,
                violation_type="memory",
                limit=limits.max_memory_mb,
                actual=usage.memory_mb,
                timestamp=datetime.now(),
                message=f"Memory usage ({usage.memory_mb:.1f}MB) exceeded limit ({limits.max_memory_mb}MB)"
            )
            if violation.message not in usage.violations:
                self._violations[sandbox_id].append(violation)
                usage.violations.append(violation.message)

        # Check disk
        if usage.disk_mb > limits.max_disk_mb:
            violation = ResourceViolation(
                sandbox_id=sandbox_id,
                violation_type="disk",
                limit=limits.max_disk_mb,
                actual=usage.disk_mb,
                timestamp=datetime.now(),
                message=f"Disk usage ({usage.disk_mb:.1f}MB) exceeded limit ({limits.max_disk_mb}MB)"
            )
            if violation.message not in usage.violations:
                self._violations[sandbox_id].append(violation)
                usage.violations.append(violation.message)

        # Check network
        if usage.network_kb > limits.max_network_kb:
            violation = ResourceViolation(
                sandbox_id=sandbox_id,
                violation_type="network",
                limit=limits.max_network_kb,
                actual=usage.network_kb,
                timestamp=datetime.now(),
                message=f"Network usage ({usage.network_kb:.1f}KB) exceeded limit ({limits.max_network_kb}KB)"
            )
            if violation.message not in usage.violations:
                self._violations[sandbox_id].append(violation)
                usage.violations.append(violation.message)

        # Check execution time
        if usage.execution_time > limits.max_execution_time:
            violation = ResourceViolation(
                sandbox_id=sandbox_id,
                violation_type="timeout",
                limit=limits.max_execution_time,
                actual=usage.execution_time,
                timestamp=datetime.now(),
                message=f"Execution time ({usage.execution_time:.1f}s) exceeded limit ({limits.max_execution_time}s)"
            )
            if violation.message not in usage.violations:
                self._violations[sandbox_id].append(violation)
                usage.violations.append(violation.message)

    def get_usage(self, sandbox_id: str) -> Optional[ResourceUsage]:
        """Get current resource usage for a sandbox"""
        with self._lock:
            return self._usage.get(sandbox_id)

    def check_violations(self, sandbox_id: str) -> List[str]:
        """
        Check for resource limit violations.

        Returns:
            List of violation messages
        """
        with self._lock:
            if sandbox_id in self._usage:
                return self._usage[sandbox_id].violations.copy()
            return []

    def get_violations(self, sandbox_id: str) -> List[ResourceViolation]:
        """Get all violations for a sandbox"""
        with self._lock:
            return self._violations.get(sandbox_id, []).copy()

    def has_violations(self, sandbox_id: str) -> bool:
        """Check if sandbox has any violations"""
        with self._lock:
            return sandbox_id in self._usage and len(self._usage[sandbox_id].violations) > 0

    def terminate_if_violated(self, sandbox_id: str) -> bool:
        """
        Check if sandbox should be terminated due to violations.

        Returns:
            True if sandbox should be terminated
        """
        return self.has_violations(sandbox_id)

    def cleanup(self, sandbox_id: str) -> None:
        """Clean up monitoring data for a sandbox"""
        with self._lock:
            self._monitoring.pop(sandbox_id, None)
            self._usage.pop(sandbox_id, None)
            self._limits.pop(sandbox_id, None)
            self._start_times.pop(sandbox_id, None)
            self._violations.pop(sandbox_id, None)

    def get_statistics(self, sandbox_id: str) -> Dict[str, any]:
        """Get statistics for a sandbox"""
        with self._lock:
            if sandbox_id not in self._usage:
                return {}

            usage = self._usage[sandbox_id]
            limits = self._limits.get(sandbox_id)

            stats = {
                'usage': {
                    'cpu_percent': usage.cpu_percent,
                    'memory_mb': usage.memory_mb,
                    'disk_mb': usage.disk_mb,
                    'network_kb': usage.network_kb,
                    'execution_time': usage.execution_time,
                    'file_operations': usage.file_operations
                },
                'violations': len(usage.violations),
                'violation_messages': usage.violations.copy()
            }

            if limits:
                stats['limits'] = {
                    'cpu_percent': limits.max_cpu_percent,
                    'memory_mb': limits.max_memory_mb,
                    'disk_mb': limits.max_disk_mb,
                    'network_kb': limits.max_network_kb,
                    'execution_time': limits.max_execution_time,
                    'file_operations': limits.max_file_operations
                }

                stats['utilization'] = {
                    'cpu': (usage.cpu_percent / limits.max_cpu_percent * 100) if limits.max_cpu_percent > 0 else 0,
                    'memory': (usage.memory_mb / limits.max_memory_mb * 100) if limits.max_memory_mb > 0 else 0,
                    'disk': (usage.disk_mb / limits.max_disk_mb * 100) if limits.max_disk_mb > 0 else 0,
                    'network': (usage.network_kb / limits.max_network_kb * 100) if limits.max_network_kb > 0 else 0
                }

            return stats


# Import ResourceLimits from sandbox_manager to avoid circular import
from deepagent.sandbox.sandbox_manager import ResourceLimits
