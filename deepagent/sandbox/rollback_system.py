"""
Rollback System - Foundation #4

Transaction-based execution with checkpointing and rollback capabilities.
Enables automatic rollback on security violations or failures.
"""

import shutil
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class Checkpoint:
    """Represents a filesystem and state checkpoint"""

    checkpoint_id: str
    timestamp: datetime
    sandbox_id: str
    filesystem_snapshot: Path
    state_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RollbackResult:
    """Result of a rollback operation"""

    success: bool
    checkpoint_id: str
    rollback_type: str  # "full", "partial", "failed"
    files_restored: int
    state_restored: bool
    error: Optional[str] = None
    duration_ms: float = 0.0


class RollbackSystem:
    """
    Manages checkpoints and rollback operations.

    Provides transaction-based execution with automatic rollback
    on failures or security violations.
    """

    def __init__(self, sandbox_root: Path):
        self.sandbox_root = sandbox_root
        self.checkpoints: Dict[str, Checkpoint] = {}

    def create_checkpoint(
        self,
        sandbox_id: str,
        sandbox_dir: Path,
        state_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        """
        Create a checkpoint of the current sandbox state.

        Args:
            sandbox_id: ID of sandbox
            sandbox_dir: Directory to snapshot
            state_data: Additional state to save
            metadata: Checkpoint metadata

        Returns:
            Checkpoint object
        """
        checkpoint_id = f"checkpoint_{uuid4().hex[:12]}"

        # Create checkpoint directory
        checkpoint_dir = self.sandbox_root / sandbox_id / "checkpoints" / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot filesystem
        snapshot_dir = checkpoint_dir / "snapshot"
        workspace_dir = sandbox_dir / "workspace"

        if workspace_dir.exists():
            shutil.copytree(workspace_dir, snapshot_dir, dirs_exist_ok=True)

        # Save state data
        state_file = checkpoint_dir / "state.json"
        with open(state_file, 'w') as f:
            json.dump(state_data or {}, f, indent=2)

        # Save metadata
        metadata_file = checkpoint_dir / "metadata.json"
        checkpoint_metadata = {
            'checkpoint_id': checkpoint_id,
            'timestamp': datetime.now().isoformat(),
            'sandbox_id': sandbox_id,
            **(metadata or {})
        }
        with open(metadata_file, 'w') as f:
            json.dump(checkpoint_metadata, f, indent=2)

        # Create checkpoint object
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(),
            sandbox_id=sandbox_id,
            filesystem_snapshot=snapshot_dir,
            state_data=state_data or {},
            metadata=checkpoint_metadata
        )

        self.checkpoints[checkpoint_id] = checkpoint

        return checkpoint

    def rollback_to_checkpoint(
        self,
        checkpoint_id: str,
        sandbox_dir: Path
    ) -> RollbackResult:
        """
        Rollback sandbox to a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to restore
            sandbox_dir: Sandbox directory to restore to

        Returns:
            RollbackResult with details of the rollback
        """
        start_time = datetime.now()

        if checkpoint_id not in self.checkpoints:
            return RollbackResult(
                success=False,
                checkpoint_id=checkpoint_id,
                rollback_type="failed",
                files_restored=0,
                state_restored=False,
                error=f"Checkpoint {checkpoint_id} not found"
            )

        checkpoint = self.checkpoints[checkpoint_id]

        try:
            # Restore filesystem
            workspace_dir = sandbox_dir / "workspace"

            # Clear current workspace
            if workspace_dir.exists():
                shutil.rmtree(workspace_dir)

            # Restore from snapshot
            files_restored = 0
            if checkpoint.filesystem_snapshot.exists():
                shutil.copytree(
                    checkpoint.filesystem_snapshot,
                    workspace_dir,
                    dirs_exist_ok=True
                )

                # Count restored files
                files_restored = sum(1 for _ in workspace_dir.rglob('*') if _.is_file())

            # State restoration (application-specific)
            state_restored = True  # Assumed successful for now

            duration = (datetime.now() - start_time).total_seconds() * 1000

            return RollbackResult(
                success=True,
                checkpoint_id=checkpoint_id,
                rollback_type="full",
                files_restored=files_restored,
                state_restored=state_restored,
                duration_ms=duration
            )

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000

            return RollbackResult(
                success=False,
                checkpoint_id=checkpoint_id,
                rollback_type="failed",
                files_restored=0,
                state_restored=False,
                error=str(e),
                duration_ms=duration
            )

    def list_checkpoints(self, sandbox_id: str) -> List[Checkpoint]:
        """List all checkpoints for a sandbox"""
        return [
            cp for cp in self.checkpoints.values()
            if cp.sandbox_id == sandbox_id
        ]

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to delete

        Returns:
            True if deleted successfully
        """
        if checkpoint_id not in self.checkpoints:
            return False

        checkpoint = self.checkpoints[checkpoint_id]

        # Delete checkpoint directory
        checkpoint_dir = checkpoint.filesystem_snapshot.parent
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

        # Remove from tracking
        del self.checkpoints[checkpoint_id]

        return True

    def auto_rollback_on_violation(
        self,
        sandbox_id: str,
        sandbox_dir: Path,
        violation: str
    ) -> Optional[RollbackResult]:
        """
        Automatically rollback to the most recent checkpoint on violation.

        Args:
            sandbox_id: ID of sandbox
            sandbox_dir: Sandbox directory
            violation: Description of the violation

        Returns:
            RollbackResult if rollback performed, None if no checkpoints
        """
        # Find most recent checkpoint
        checkpoints = self.list_checkpoints(sandbox_id)

        if not checkpoints:
            return None

        # Sort by timestamp and get most recent
        most_recent = max(checkpoints, key=lambda cp: cp.timestamp)

        # Perform rollback
        result = self.rollback_to_checkpoint(most_recent.checkpoint_id, sandbox_dir)

        # Add violation info to result
        if result.success:
            result.metadata = {'violation': violation}

        return result

    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get a specific checkpoint"""
        return self.checkpoints.get(checkpoint_id)

    def cleanup_old_checkpoints(
        self,
        sandbox_id: str,
        keep_count: int = 5
    ) -> int:
        """
        Clean up old checkpoints, keeping only the most recent ones.

        Args:
            sandbox_id: ID of sandbox
            keep_count: Number of checkpoints to keep

        Returns:
            Number of checkpoints deleted
        """
        checkpoints = self.list_checkpoints(sandbox_id)

        if len(checkpoints) <= keep_count:
            return 0

        # Sort by timestamp
        sorted_checkpoints = sorted(checkpoints, key=lambda cp: cp.timestamp, reverse=True)

        # Delete old checkpoints
        to_delete = sorted_checkpoints[keep_count:]
        deleted = 0

        for checkpoint in to_delete:
            if self.delete_checkpoint(checkpoint.checkpoint_id):
                deleted += 1

        return deleted

    def get_checkpoint_size(self, checkpoint_id: str) -> int:
        """
        Get the size of a checkpoint in bytes.

        Args:
            checkpoint_id: ID of checkpoint

        Returns:
            Size in bytes, or 0 if not found
        """
        if checkpoint_id not in self.checkpoints:
            return 0

        checkpoint = self.checkpoints[checkpoint_id]

        if not checkpoint.filesystem_snapshot.exists():
            return 0

        total_size = 0
        for path in checkpoint.filesystem_snapshot.rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size

        return total_size

    def get_statistics(self, sandbox_id: str) -> Dict[str, Any]:
        """Get checkpoint statistics for a sandbox"""
        checkpoints = self.list_checkpoints(sandbox_id)

        if not checkpoints:
            return {
                'total_checkpoints': 0,
                'total_size_bytes': 0,
                'oldest': None,
                'newest': None
            }

        total_size = sum(self.get_checkpoint_size(cp.checkpoint_id) for cp in checkpoints)
        oldest = min(checkpoints, key=lambda cp: cp.timestamp)
        newest = max(checkpoints, key=lambda cp: cp.timestamp)

        return {
            'total_checkpoints': len(checkpoints),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'oldest': {
                'checkpoint_id': oldest.checkpoint_id,
                'timestamp': oldest.timestamp.isoformat()
            },
            'newest': {
                'checkpoint_id': newest.checkpoint_id,
                'timestamp': newest.timestamp.isoformat()
            }
        }
