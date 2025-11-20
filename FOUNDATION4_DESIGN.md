# Foundation #4: Execution Sandboxing - Design Document

**Status**: In Development
**Date**: 2025-11-15
**Dependencies**: Foundation #1 (Action-Level Safety), Foundation #2 (Memory Firewalls), Foundation #7 (Audit Logs)

---

## Overview

Foundation #4 provides execution sandboxing and containment for SafeDeepAgent. It ensures that even if malicious or risky actions pass through Foundations #1 and #2, damage is contained through:

- Isolated execution environments
- Resource limits and quotas
- Filesystem isolation and snapshots
- Network policy enforcement
- Transaction-based execution with rollback
- Damage containment and recovery

---

## Architecture

### 3-Component Design

```
┌─────────────────────────────────────────────────────────────┐
│  Component 1: SandboxManager                                │
│  - Sandbox creation and lifecycle                           │
│  - Environment isolation                                    │
│  - Context management                                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Component 2: ResourceMonitor                               │
│  - CPU/memory/disk monitoring                               │
│  - Network traffic control                                  │
│  - Execution time limits                                    │
│  - Resource quota enforcement                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Component 3: RollbackSystem                                │
│  - Transaction-based execution                              │
│  - Filesystem snapshots                                     │
│  - State checkpointing                                      │
│  - Automated rollback on violations                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. SandboxManager

**Purpose**: Create and manage isolated execution environments

**Key Features**:
- Multiple isolation levels (process, container, VM)
- Temporary filesystem mounting
- Environment variable isolation
- Working directory isolation
- Import restrictions

**Isolation Levels**:
```python
class IsolationLevel(Enum):
    NONE = "none"              # No isolation (testing only)
    PROCESS = "process"        # Separate process
    CONTAINER = "container"    # Docker container (if available)
    STRICT = "strict"          # Maximum isolation
```

**SandboxContext**:
```python
@dataclass
class SandboxContext:
    sandbox_id: str
    isolation_level: IsolationLevel
    working_directory: Path
    allowed_paths: List[Path]
    blocked_paths: List[Path]
    environment_vars: Dict[str, str]
    timeout_seconds: int
    resource_limits: ResourceLimits
```

**Methods**:
```python
class SandboxManager:
    def create_sandbox(self, config: SandboxConfig) -> SandboxContext
    def execute_in_sandbox(self, context: SandboxContext, tool_fn: Callable, params: Dict) -> ExecutionResult
    def destroy_sandbox(self, sandbox_id: str) -> None
    def list_active_sandboxes() -> List[SandboxContext]
```

---

### 2. ResourceMonitor

**Purpose**: Monitor and enforce resource limits

**Key Features**:
- Real-time resource tracking
- Quota enforcement
- Automatic termination on violations
- Resource usage statistics

**ResourceLimits**:
```python
@dataclass
class ResourceLimits:
    max_cpu_percent: float = 80.0      # % of single core
    max_memory_mb: int = 512           # MB
    max_disk_mb: int = 1024            # MB
    max_network_kb: int = 10240        # KB
    max_execution_time: int = 300      # seconds
    max_file_operations: int = 1000    # number of operations
```

**ResourceUsage**:
```python
@dataclass
class ResourceUsage:
    cpu_percent: float
    memory_mb: float
    disk_mb: float
    network_kb: float
    execution_time: float
    file_operations: int
    violations: List[str]  # List of violated limits
```

**Methods**:
```python
class ResourceMonitor:
    def start_monitoring(self, sandbox_id: str, limits: ResourceLimits) -> None
    def get_usage(self, sandbox_id: str) -> ResourceUsage
    def check_violations(self, sandbox_id: str) -> List[str]
    def terminate_if_violated(self, sandbox_id: str) -> bool
```

---

### 3. RollbackSystem

**Purpose**: Enable transaction-based execution with rollback

**Key Features**:
- Filesystem snapshots before execution
- State checkpointing
- Automatic rollback on failure
- Rollback on security violations
- Partial rollback support

**Checkpoint**:
```python
@dataclass
class Checkpoint:
    checkpoint_id: str
    timestamp: datetime
    sandbox_id: str
    filesystem_snapshot: Path
    state_data: Dict[str, Any]
    metadata: Dict[str, Any]
```

**RollbackResult**:
```python
@dataclass
class RollbackResult:
    success: bool
    checkpoint_id: str
    rollback_type: str  # "full", "partial", "failed"
    files_restored: int
    state_restored: bool
    error: Optional[str]
```

**Methods**:
```python
class RollbackSystem:
    def create_checkpoint(self, sandbox_id: str, metadata: Dict) -> Checkpoint
    def rollback_to_checkpoint(self, checkpoint_id: str) -> RollbackResult
    def list_checkpoints(self, sandbox_id: str) -> List[Checkpoint]
    def delete_checkpoint(self, checkpoint_id: str) -> None
    def auto_rollback_on_violation(self, sandbox_id: str, violation: str) -> RollbackResult
```

---

## Integration with SafeDeepAgent

### Execution Flow with Sandboxing

```python
class SafeDeepAgent:
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]):
        # Phase 1: Authorization
        decision = self._authorize_action(tool_name, parameters)

        # Phase 2: Multi-step checks
        self._check_phase2_security(tool_name, parameters, decision.risk_score)

        # NEW: Phase 3: Sandboxed execution
        if self.safe_config.enable_sandboxing:
            result = self._execute_in_sandbox(tool_name, parameters, decision)
        else:
            result = self._execute_directly(tool_name, parameters)

        # Phase 4: Audit logging
        self._log_execution(tool_name, parameters, result, decision)

        return result
```

### Sandbox Decision Logic

```python
def _should_sandbox(self, tool_name: str, risk_score: float) -> bool:
    """Decide if action should be sandboxed"""

    # Always sandbox high-risk actions
    if risk_score > 0.7:
        return True

    # Always sandbox destructive operations
    destructive_tools = ['delete', 'modify', 'execute', 'admin']
    if any(keyword in tool_name.lower() for keyword in destructive_tools):
        return True

    # Always sandbox network operations
    if 'network' in tool_name.lower() or 'http' in tool_name.lower():
        return True

    # Sandbox based on mode
    if self.safe_config.sandbox_mode == SandboxMode.ALWAYS:
        return True
    elif self.safe_config.sandbox_mode == SandboxMode.NEVER:
        return False
    else:  # AUTO
        return risk_score > 0.5
```

---

## Configuration

```python
@dataclass
class SandboxConfig:
    # Enable/disable
    enable_sandboxing: bool = True
    sandbox_mode: SandboxMode = SandboxMode.AUTO  # ALWAYS, AUTO, NEVER

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
        Path("C:\\Windows"),  # Windows
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
```

---

## Use Cases

### Use Case 1: Sandboxing High-Risk Code Execution

```python
# User requests code execution
agent.run("Execute this Python script: analysis.py")

# SafeDeepAgent flow:
# 1. Phase 1: Risk score = 0.85 (high-risk)
# 2. Phase 2: No attack pattern detected
# 3. Phase 3: HIGH RISK -> Execute in sandbox
#    - Create isolated process
#    - Set resource limits
#    - Create checkpoint
#    - Execute script
#    - Monitor resources
#    - If violation -> rollback
#    - If success -> commit changes
# 4. Phase 4: Log execution and sandbox stats
```

### Use Case 2: Automatic Rollback on Attack Detection

```python
# Attack detected by Phase 2
# SafeDeepAgent flow:
# 1. Phase 2 detects data exfiltration attack
# 2. Sandbox system automatically:
#    - Terminates sandbox
#    - Rolls back to last checkpoint
#    - Restores filesystem
#    - Logs rollback event
# 3. Attack damage contained!
```

### Use Case 3: Resource Limit Enforcement

```python
# Malicious infinite loop or resource exhaustion
# Sandbox flow:
# 1. Monitor detects CPU > 80% for 10 seconds
# 2. Monitor detects memory > 512MB
# 3. Automatic termination triggered
# 4. Rollback to checkpoint
# 5. Report violation
```

---

## File Structure

```
deepagent/
├── sandbox/
│   ├── __init__.py
│   ├── sandbox_manager.py     # SandboxManager, isolation
│   ├── resource_monitor.py    # ResourceMonitor, limits
│   └── rollback_system.py     # RollbackSystem, checkpoints

sandboxes/                      # Sandbox working directories
├── sandbox_abc123/
│   ├── workspace/             # Isolated filesystem
│   ├── checkpoints/           # Snapshots
│   └── metadata.json          # Sandbox config

examples/
└── foundation4_sandbox_demo.py

tests/
├── test_sandbox_manager.py
├── test_resource_monitor.py
└── test_rollback_system.py
```

---

## Success Criteria

- ✅ SandboxManager creates isolated execution environments
- ✅ ResourceMonitor enforces CPU/memory/disk/network limits
- ✅ RollbackSystem creates checkpoints and rolls back on violations
- ✅ Integration with SafeDeepAgent
- ✅ Automatic sandboxing decision based on risk score
- ✅ Automatic rollback on Phase 2 attack detection
- ✅ Filesystem isolation working
- ✅ Tests pass at >90% rate
- ✅ Working demonstration examples
- ✅ Complete documentation

---

## Performance Targets

- **Sandbox creation**: <100ms
- **Checkpoint creation**: <500ms
- **Rollback execution**: <1s
- **Resource monitoring overhead**: <5% CPU
- **Storage overhead**: <10MB per sandbox

---

## Security Considerations

**Isolation**:
- Process-level isolation is minimum (default)
- Container-level for production (recommended)
- VM-level for maximum security (optional)

**Resource Limits**:
- Prevent DoS via resource exhaustion
- Prevent runaway processes
- Configurable per environment

**Filesystem**:
- Read-only system paths
- Writable temporary directories only
- Snapshot before execution

**Network**:
- Default: no network access
- Whitelist-based for allowed operations
- Traffic monitoring and logging

---

## Limitations & Trade-offs

**Performance**:
- Sandboxing adds overhead (~100-500ms per action)
- Checkpointing requires disk I/O
- Acceptable for security-critical operations

**Compatibility**:
- Some tools may not work in sandboxes
- Container isolation requires Docker
- Platform-specific implementations (Windows vs Linux)

**Complexity**:
- Adds operational overhead
- Requires careful configuration
- May cause false positives

---

## Future Enhancements

**Phase 1 (Current)**:
- Process-level isolation
- Basic resource monitoring
- Filesystem snapshots
- Automatic rollback

**Phase 2 (Future)**:
- Docker container integration
- Network traffic inspection
- Cross-platform VM support
- Distributed sandboxing
- GPU resource limits
- Advanced rollback strategies

---

## Next Steps

1. Implement SandboxManager with process isolation
2. Implement ResourceMonitor with psutil
3. Implement RollbackSystem with filesystem snapshots
4. Integrate with SafeDeepAgent
5. Create comprehensive tests
6. Create demonstration examples
7. Write user documentation

---

**Status**: Design Complete ✅
**Ready for Implementation**: Yes
**Estimated Effort**: ~2,500 lines of code + tests + docs
