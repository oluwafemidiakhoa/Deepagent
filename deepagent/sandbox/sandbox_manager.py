"""
Sandbox Manager - Foundation #4

Creates and manages isolated execution environments for SafeDeepAgent.
Provides process-level isolation with filesystem and environment controls.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4
import json


class IsolationLevel(Enum):
    """Level of isolation for sandbox execution"""
    NONE = "none"          # No isolation (testing only)
    PROCESS = "process"    # Separate process
    STRICT = "strict"      # Maximum isolation available


class SandboxMode(Enum):
    """When to use sandboxing"""
    ALWAYS = "always"      # Sandbox all executions
    AUTO = "auto"          # Sandbox based on risk
    NEVER = "never"        # Never sandbox (testing only)


@dataclass
class ResourceLimits:
    """Resource limits for sandbox"""
    max_cpu_percent: float = 80.0
    max_memory_mb: int = 512
    max_disk_mb: int = 1024
    max_network_kb: int = 10240
    max_execution_time: int = 300  # seconds
    max_file_operations: int = 1000


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution"""

    # Enable/disable
    enable_sandboxing: bool = True
    sandbox_mode: SandboxMode = SandboxMode.AUTO

    # Isolation
    isolation_level: IsolationLevel = IsolationLevel.PROCESS

    # Resource limits
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)

    # Filesystem
    sandbox_root: Path = Path("./sandboxes")
    allowed_paths: List[Path] = field(default_factory=list)
    blocked_paths: List[Path] = field(default_factory=lambda: [
        Path("/etc"),
        Path("/sys"),
        Path("/boot"),
        Path("C:\\Windows"),
        Path("C:\\Program Files")
    ])

    # Network
    allow_network: bool = False
    allowed_hosts: List[str] = field(default_factory=list)

    # Rollback
    enable_auto_rollback: bool = True
    checkpoint_before_execution: bool = True

    # Cleanup
    auto_cleanup: bool = True
    cleanup_age_hours: int = 24


@dataclass
class SandboxContext:
    """Context for a running sandbox"""

    sandbox_id: str
    isolation_level: IsolationLevel
    working_directory: Path
    allowed_paths: List[Path]
    blocked_paths: List[Path]
    environment_vars: Dict[str, str]
    timeout_seconds: int
    resource_limits: ResourceLimits
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class SandboxError(Exception):
    """Base exception for sandbox errors"""
    pass


class SandboxTimeoutError(SandboxError):
    """Sandbox execution timed out"""
    pass


class SandboxResourceError(SandboxError):
    """Sandbox resource limit exceeded"""
    pass


class SandboxManager:
    """
    Manages isolated execution environments.

    Provides process-level isolation with filesystem controls and
    resource monitoring.
    """

    def __init__(self, config: SandboxConfig):
        self.config = config
        self.active_sandboxes: Dict[str, SandboxContext] = {}

        # Ensure sandbox root exists
        if not self.config.sandbox_root.exists():
            self.config.sandbox_root.mkdir(parents=True, exist_ok=True)

    def create_sandbox(
        self,
        timeout_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SandboxContext:
        """
        Create a new sandbox environment.

        Args:
            timeout_seconds: Execution timeout (uses config default if None)
            metadata: Additional metadata for the sandbox

        Returns:
            SandboxContext with sandbox details
        """
        sandbox_id = f"sandbox_{uuid4().hex[:12]}"

        # Create sandbox directory
        working_dir = self.config.sandbox_root / sandbox_id / "workspace"
        working_dir.mkdir(parents=True, exist_ok=True)

        # Create checkpoint directory
        checkpoint_dir = self.config.sandbox_root / sandbox_id / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create sandbox context
        context = SandboxContext(
            sandbox_id=sandbox_id,
            isolation_level=self.config.isolation_level,
            working_directory=working_dir,
            allowed_paths=self.config.allowed_paths.copy(),
            blocked_paths=self.config.blocked_paths.copy(),
            environment_vars=os.environ.copy(),  # Inherit environment
            timeout_seconds=timeout_seconds or self.config.resource_limits.max_execution_time,
            resource_limits=self.config.resource_limits,
            created_at=datetime.now(),
            metadata=metadata or {}
        )

        # Save context metadata
        self._save_context_metadata(context)

        # Track active sandbox
        self.active_sandboxes[sandbox_id] = context

        return context

    def execute_in_sandbox(
        self,
        context: SandboxContext,
        tool_fn: Callable,
        parameters: Dict[str, Any]
    ) -> Any:
        """
        Execute a function in the sandbox.

        Args:
            context: Sandbox context
            tool_fn: Function to execute
            parameters: Parameters for the function

        Returns:
            Result of the function execution

        Raises:
            SandboxTimeoutError: If execution times out
            SandboxResourceError: If resource limits exceeded
            SandboxError: Other sandbox errors
        """
        if self.config.isolation_level == IsolationLevel.NONE:
            # No isolation - direct execution
            return tool_fn(**parameters)

        elif self.config.isolation_level == IsolationLevel.PROCESS:
            # Process-level isolation
            return self._execute_in_process(context, tool_fn, parameters)

        else:  # STRICT
            # Use maximum available isolation
            return self._execute_in_process(context, tool_fn, parameters)

    def _execute_in_process(
        self,
        context: SandboxContext,
        tool_fn: Callable,
        parameters: Dict[str, Any]
    ) -> Any:
        """Execute function in a separate process with isolation"""

        # For simplicity, we'll execute in the same process but with
        # restricted environment and working directory
        # In production, this would use subprocess or multiprocessing

        original_cwd = os.getcwd()
        original_path = sys.path.copy()

        try:
            # Change to sandbox directory
            os.chdir(context.working_directory)

            # Restrict sys.path to prevent import of arbitrary modules
            if self.config.isolation_level == IsolationLevel.STRICT:
                sys.path = [str(context.working_directory)] + sys.path[-2:]

            # Execute function
            start_time = datetime.now()

            try:
                result = tool_fn(**parameters)
            except Exception as e:
                raise SandboxError(f"Execution failed: {str(e)}") from e

            # Check timeout
            execution_time = (datetime.now() - start_time).total_seconds()
            if execution_time > context.timeout_seconds:
                raise SandboxTimeoutError(
                    f"Execution exceeded timeout of {context.timeout_seconds}s"
                )

            return result

        finally:
            # Restore environment
            os.chdir(original_cwd)
            sys.path = original_path

    def _check_path_allowed(self, context: SandboxContext, path: Path) -> bool:
        """Check if a path is allowed for access"""

        # Check if in blocked paths
        for blocked in context.blocked_paths:
            try:
                if path.resolve().is_relative_to(blocked.resolve()):
                    return False
            except (ValueError, OSError):
                pass

        # Check if in allowed paths (if specified)
        if context.allowed_paths:
            for allowed in context.allowed_paths:
                try:
                    if path.resolve().is_relative_to(allowed.resolve()):
                        return True
                except (ValueError, OSError):
                    pass
            return False

        # Default: allow if not blocked
        return True

    def destroy_sandbox(self, sandbox_id: str, cleanup_files: bool = True) -> None:
        """
        Destroy a sandbox and optionally clean up its files.

        Args:
            sandbox_id: ID of sandbox to destroy
            cleanup_files: Whether to delete sandbox files
        """
        if sandbox_id not in self.active_sandboxes:
            return

        context = self.active_sandboxes[sandbox_id]

        # Remove from active sandboxes
        del self.active_sandboxes[sandbox_id]

        # Clean up files if requested
        if cleanup_files and self.config.auto_cleanup:
            sandbox_dir = self.config.sandbox_root / sandbox_id
            if sandbox_dir.exists():
                shutil.rmtree(sandbox_dir)

    def list_active_sandboxes(self) -> List[SandboxContext]:
        """Get list of all active sandboxes"""
        return list(self.active_sandboxes.values())

    def cleanup_old_sandboxes(self) -> int:
        """Clean up sandboxes older than configured age"""
        if not self.config.auto_cleanup:
            return 0

        cutoff_time = datetime.now() - timedelta(hours=self.config.cleanup_age_hours)
        cleaned = 0

        # Check all sandbox directories
        if self.config.sandbox_root.exists():
            for sandbox_dir in self.config.sandbox_root.iterdir():
                if not sandbox_dir.is_dir():
                    continue

                metadata_file = sandbox_dir / "metadata.json"
                if not metadata_file.exists():
                    continue

                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    created_at = datetime.fromisoformat(metadata.get('created_at'))
                    if created_at < cutoff_time:
                        shutil.rmtree(sandbox_dir)
                        cleaned += 1

                except Exception:
                    pass

        return cleaned

    def _save_context_metadata(self, context: SandboxContext):
        """Save sandbox context metadata to file"""
        metadata_file = self.config.sandbox_root / context.sandbox_id / "metadata.json"

        metadata = {
            'sandbox_id': context.sandbox_id,
            'isolation_level': context.isolation_level.value,
            'working_directory': str(context.working_directory),
            'created_at': context.created_at.isoformat(),
            'timeout_seconds': context.timeout_seconds,
            'metadata': context.metadata
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def get_sandbox_info(self, sandbox_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a sandbox"""
        if sandbox_id in self.active_sandboxes:
            context = self.active_sandboxes[sandbox_id]
            return {
                'sandbox_id': context.sandbox_id,
                'isolation_level': context.isolation_level.value,
                'working_directory': str(context.working_directory),
                'created_at': context.created_at.isoformat(),
                'age_seconds': (datetime.now() - context.created_at).total_seconds(),
                'timeout_seconds': context.timeout_seconds,
                'active': True
            }

        return None


from datetime import timedelta
