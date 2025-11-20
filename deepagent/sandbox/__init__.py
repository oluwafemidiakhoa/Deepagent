"""
Foundation #4: Execution Sandboxing

Isolated execution environments with resource limits and rollback capabilities.
"""

from deepagent.sandbox.sandbox_manager import (
    SandboxManager,
    SandboxContext,
    SandboxConfig,
    IsolationLevel,
    SandboxMode
)

from deepagent.sandbox.resource_monitor import (
    ResourceMonitor,
    ResourceLimits,
    ResourceUsage,
    ResourceViolation
)

from deepagent.sandbox.rollback_system import (
    RollbackSystem,
    Checkpoint,
    RollbackResult
)

__all__ = [
    # Sandbox Manager
    "SandboxManager",
    "SandboxContext",
    "SandboxConfig",
    "IsolationLevel",
    "SandboxMode",

    # Resource Monitor
    "ResourceMonitor",
    "ResourceLimits",
    "ResourceUsage",
    "ResourceViolation",

    # Rollback System
    "RollbackSystem",
    "Checkpoint",
    "RollbackResult"
]
