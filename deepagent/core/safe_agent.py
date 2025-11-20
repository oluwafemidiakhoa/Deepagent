"""
Safe DeepAgent - Security-Hardened Autonomous Agent

Extends DeepAgent with comprehensive security framework.
Implements Foundation #1: Action-Level Safety
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from .agent import DeepAgent, AgentConfig
from .reasoning import ReasoningResult
from ..tools.retrieval import ToolDefinition
from ..tools.executor import ExecutionResult
from ..safety import (
    SafetyConfig,
    SafetyMode,
    InputValidator,
    ActionPolicy,
    PromptInjectionDetectedError,
    UnauthorizedActionError,
    RiskThresholdExceededError
)
from ..safety.memory_firewall import (
    AttackPatternDatabase,
    TaskSequenceAnalyzer,
    ActionRecord,
    ReasoningMonitor,
    MemoryValidator,
    ProvenanceRecord,
    ProvenanceType
)
from ..audit import (
    AuditLogger,
    AuditConfig,
    JSONAuditLogger,
    SQLiteAuditLogger,
    CompositeAuditLogger,
    ForensicAnalyzer,
    AuditQueryInterface
)
from datetime import datetime
from pathlib import Path
from uuid import uuid4


@dataclass
class SafeAgentConfig(AgentConfig):
    """
    Configuration for SafeDeepAgent

    Extends AgentConfig with security settings
    """
    # Security configuration
    safety_config: Optional[SafetyConfig] = None
    safety_mode: SafetyMode = SafetyMode.BALANCED

    # Enable/disable security features (Phase 1)
    enable_input_validation: bool = True
    enable_action_authorization: bool = True
    enable_approval_workflow: bool = True

    # Enable/disable Phase 2 features (Memory Firewalls)
    enable_memory_firewall: bool = True
    enable_attack_detection: bool = True
    enable_sequence_analysis: bool = True
    enable_reasoning_monitor: bool = True
    enable_memory_validation: bool = True

    # Foundation #7: Audit Logs & Forensics
    enable_audit_logging: bool = True
    audit_config: Optional[AuditConfig] = None

    # Risk threshold (0.0 - 1.0)
    risk_threshold: float = 0.7

    # User context (for authorization)
    user_role: str = "user"
    user_id: Optional[str] = None
    environment: str = "production"  # "production", "development", "research"

    # Session tracking
    session_id: Optional[str] = None


class SafeDeepAgent(DeepAgent):
    """
    Security-Hardened DeepAgent

    Implements:
    - Foundation #1: Action-Level Safety
      - Input validation and prompt injection detection
      - Action classification and risk scoring
      - Policy enforcement and approval workflows

    - Foundation #2: Memory Firewalls
      - Multi-step attack pattern detection
      - Goal alignment and drift detection
      - Reasoning anomaly monitoring
      - Memory integrity validation

    - Foundation #7: Audit Logs & Forensics
      - Comprehensive audit logging
      - Attack sequence reconstruction
      - Forensic analysis and incident reports
      - Flexible query and export capabilities

    Usage:
        >>> config = SafeAgentConfig(
        ...     llm_provider="openai",
        ...     safety_mode=SafetyMode.STRICT,
        ...     enable_memory_firewall=True
        ... )
        >>> agent = SafeDeepAgent(config)
        >>> result = agent.run("Search for CRISPR research")
    """

    def __init__(self, config: Optional[SafeAgentConfig] = None):
        """
        Initialize safe agent with security framework

        Args:
            config: Safe agent configuration
        """
        # Use SafeAgentConfig if not provided
        if config is None:
            config = SafeAgentConfig()
        elif not isinstance(config, SafeAgentConfig):
            # Convert AgentConfig to SafeAgentConfig
            safe_config = SafeAgentConfig()
            for key, value in vars(config).items():
                if hasattr(safe_config, key):
                    setattr(safe_config, key, value)
            config = safe_config

        self.safe_config = config

        # Initialize base agent
        super().__init__(config)

        # Initialize safety framework
        self._init_security()

        if self.config.verbose:
            print("[SECURITY] SafeDeepAgent initialized")
            print(f"  Mode: {self.safe_config.safety_mode.value}")
            print(f"  Risk threshold: {self.safe_config.risk_threshold:.0%}")
            print(f"\n[PHASE 1] Action-Level Safety:")
            print(f"  Input validation: {'ON' if self.safe_config.enable_input_validation else 'OFF'}")
            print(f"  Action authorization: {'ON' if self.safe_config.enable_action_authorization else 'OFF'}")
            if self.safe_config.enable_memory_firewall:
                print(f"\n[PHASE 2] Memory Firewalls:")
                print(f"  Attack detection: {'ON' if self.safe_config.enable_attack_detection else 'OFF'}")
                print(f"  Sequence analysis: {'ON' if self.safe_config.enable_sequence_analysis else 'OFF'}")
                print(f"  Reasoning monitor: {'ON' if self.safe_config.enable_reasoning_monitor else 'OFF'}")
                print(f"  Memory validation: {'ON' if self.safe_config.enable_memory_validation else 'OFF'}")

    def _init_security(self):
        """Initialize security components"""
        # Get or create safety config
        if self.safe_config.safety_config:
            safety_config = self.safe_config.safety_config
        else:
            # Create safety config based on mode
            if self.safe_config.safety_mode == SafetyMode.STRICT:
                safety_config = SafetyConfig.create_strict()
            elif self.safe_config.safety_mode == SafetyMode.RESEARCH:
                safety_config = SafetyConfig.create_research()
            else:
                safety_config = SafetyConfig(mode=self.safe_config.safety_mode)

        # Initialize input validator
        if self.safe_config.enable_input_validation:
            self.input_validator = InputValidator(safety_config.input_validation)
        else:
            self.input_validator = None

        # Initialize action policy
        if self.safe_config.enable_action_authorization:
            self.action_policy = ActionPolicy(
                risk_threshold=self.safe_config.risk_threshold,
                enable_approval_workflow=self.safe_config.enable_approval_workflow
            )
        else:
            self.action_policy = None

        # Security statistics (Phase 1)
        self.security_stats = {
            'total_validations': 0,
            'blocked_injections': 0,
            'blocked_actions': 0,
            'approved_actions': 0,
            'total_actions_evaluated': 0
        }

        # Phase 2: Memory Firewalls
        if self.safe_config.enable_memory_firewall:
            self._init_memory_firewall()

        # Foundation #7: Audit Logging
        if self.safe_config.enable_audit_logging:
            self._init_audit_logging()

    def _init_memory_firewall(self):
        """Initialize Phase 2: Memory Firewall components"""
        # Attack pattern database
        if self.safe_config.enable_attack_detection:
            self.attack_detector = AttackPatternDatabase()
        else:
            self.attack_detector = None

        # Sequence analyzer
        if self.safe_config.enable_sequence_analysis:
            self.sequence_analyzer = TaskSequenceAnalyzer()
        else:
            self.sequence_analyzer = None

        # Reasoning monitor
        if self.safe_config.enable_reasoning_monitor:
            self.reasoning_monitor = ReasoningMonitor()
        else:
            self.reasoning_monitor = None

        # Memory validator
        if self.safe_config.enable_memory_validation:
            self.memory_validator = MemoryValidator()
        else:
            self.memory_validator = None

        # Phase 2 statistics
        self.phase2_stats = {
            'attacks_detected': 0,
            'drift_detections': 0,
            'escalation_detections': 0,
            'reasoning_anomalies': 0,
            'memory_tampering_detected': 0,
            'total_actions_analyzed': 0
        }

        # Current step number (for sequence tracking)
        self.current_step = 0

    def _init_audit_logging(self):
        """Initialize Foundation #7: Audit Logging & Forensics"""
        # Create or use provided audit config
        if self.safe_config.audit_config:
            audit_config = self.safe_config.audit_config
        else:
            audit_config = AuditConfig(
                enable_audit_logging=True,
                storage_backend="json",
                log_directory=Path("./audit_logs"),
                async_logging=True
            )

        # Initialize audit logger (default to JSON)
        if audit_config.storage_backend == "json":
            self.audit_logger = JSONAuditLogger(audit_config)
        elif audit_config.storage_backend == "sqlite":
            self.audit_logger = SQLiteAuditLogger(audit_config)
        elif audit_config.storage_backend == "composite":
            # Both JSON and SQLite
            json_logger = JSONAuditLogger(audit_config)
            sqlite_logger = SQLiteAuditLogger(audit_config)
            self.audit_logger = CompositeAuditLogger(audit_config, [json_logger, sqlite_logger])
        else:
            self.audit_logger = JSONAuditLogger(audit_config)

        # Initialize forensic analyzer and query interface
        self.forensic_analyzer = ForensicAnalyzer(self.audit_logger)
        self.query_interface = AuditQueryInterface(self.audit_logger)

        # Generate or use provided session ID
        if not self.safe_config.session_id:
            self.safe_config.session_id = f"session_{uuid4().hex[:8]}"

        # Generate user ID if not provided
        if not self.safe_config.user_id:
            self.safe_config.user_id = f"user_{uuid4().hex[:8]}"

    def run(
        self,
        task: str,
        context: Optional[str] = None,
        max_steps: Optional[int] = None
    ) -> ReasoningResult:
        """
        Execute a task with security validation

        Args:
            task: Natural language task description
            context: Optional additional context
            max_steps: Optional max reasoning steps

        Returns:
            ReasoningResult

        Raises:
            PromptInjectionDetectedError: If task contains injection attack
            SecurityViolationError: If security policy violated
        """
        # SECURITY LAYER 1: Input Validation
        if self.input_validator:
            try:
                validated_task, validation_metadata = self.input_validator.validate(
                    task,
                    context="user_task"
                )
                self.security_stats['total_validations'] += 1

                if self.config.verbose and validation_metadata.get('security_warnings'):
                    print(f"[SECURITY WARNING] {validation_metadata['security_warnings'][0]}")

            except PromptInjectionDetectedError as e:
                self.security_stats['blocked_injections'] += 1
                if self.config.verbose:
                    print(f"\n[SECURITY BLOCKED] Prompt injection detected!")
                    print(f"  Patterns: {e.detected_patterns}")
                raise

            # Use validated task
            task = validated_task

            # Validate context if provided
            if context and self.input_validator:
                validated_context, _ = self.input_validator.validate(
                    context,
                    context="additional_context"
                )
                context = validated_context

        # SECURITY LAYER 2 (Phase 2): Initialize Memory Firewall for this task
        if self.safe_config.enable_memory_firewall:
            # Reset step counter
            self.current_step = 0

            # Initialize sequence analyzer with task
            if self.sequence_analyzer:
                session_id = self.safe_config.session_id or f"session_{datetime.now().timestamp()}"
                self.sequence_analyzer.initialize(
                    original_task=task,
                    session_id=session_id,
                    user_id=self.safe_config.user_id
                )

            # Initialize reasoning monitor with task
            if self.reasoning_monitor:
                self.reasoning_monitor.initialize_baseline(task)

        # Foundation #7: Log session start
        if self.safe_config.enable_audit_logging and hasattr(self, 'audit_logger'):
            self.audit_logger.log_session_start(
                session_id=self.safe_config.session_id,
                user_id=self.safe_config.user_id,
                task=task,
                metadata={
                    'context': context,
                    'max_steps': max_steps,
                    'safety_mode': self.safe_config.safety_mode.value,
                    'environment': self.safe_config.environment
                }
            )

        # Execute with base agent (which will call our secured _execute_tool)
        try:
            result = super().run(task, context, max_steps)

            # Foundation #7: Log session end (success)
            if self.safe_config.enable_audit_logging and hasattr(self, 'audit_logger'):
                stats = self.get_security_stats()
                self.audit_logger.log_session_end(
                    session_id=self.safe_config.session_id,
                    user_id=self.safe_config.user_id,
                    stats=stats,
                    metadata={'success': True}
                )

            return result

        except Exception as e:
            # Foundation #7: Log session end (error)
            if self.safe_config.enable_audit_logging and hasattr(self, 'audit_logger'):
                from ..audit.audit_logger import EventType, EventSeverity
                self.audit_logger.log_security_event(
                    session_id=self.safe_config.session_id,
                    user_id=self.safe_config.user_id,
                    event_type=EventType.SESSION_END,
                    severity=EventSeverity.CRITICAL,
                    details={
                        'success': False,
                        'error': str(e),
                        'error_type': type(e).__name__
                    }
                )
            raise

    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ExecutionResult:
        """
        Execute tool with security authorization and multi-step attack detection

        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters

        Returns:
            ExecutionResult

        Raises:
            UnauthorizedActionError: If action not authorized
            RiskThresholdExceededError: If risk too high
            SecurityViolationError: If multi-step attack detected
        """
        # Increment step counter
        if self.safe_config.enable_memory_firewall:
            self.current_step += 1

        # SECURITY LAYER 2 (Phase 1): Action Authorization
        policy_decision = None
        if self.action_policy:
            # Build security context
            security_context = {
                'user_role': self.safe_config.user_role,
                'user_id': self.safe_config.user_id,
                'environment': self.safe_config.environment,
                'agent_type': 'safe_deep_agent'
            }

            try:
                # Evaluate action with policy
                policy_decision = self.action_policy.evaluate_action(
                    tool_name,
                    parameters,
                    security_context
                )

                self.security_stats['total_actions_evaluated'] += 1

                # Handle decision
                if policy_decision.requires_user_approval and self.safe_config.enable_approval_workflow:
                    # TODO: Implement approval workflow
                    # For now, auto-deny high-risk actions if no approval mechanism
                    if not self.action_policy.approval_callback:
                        self.security_stats['blocked_actions'] += 1
                        raise UnauthorizedActionError(
                            f"Action '{tool_name}' requires approval but no approval mechanism configured",
                            action=tool_name
                        )

                    # Request approval
                    approved = self.action_policy.request_approval(policy_decision)
                    if not approved:
                        self.security_stats['blocked_actions'] += 1
                        raise UnauthorizedActionError(
                            f"Action '{tool_name}' was not approved",
                            action=tool_name
                        )

                self.security_stats['approved_actions'] += 1

                if self.config.verbose and policy_decision.requires_explanation:
                    print(f"\n[SECURITY] Action authorized: {tool_name}")
                    print(f"  Risk score: {policy_decision.risk_score.total_score:.1%}")
                    print(f"  Decision: {policy_decision.decision.value}")

            except (UnauthorizedActionError, RiskThresholdExceededError) as e:
                self.security_stats['blocked_actions'] += 1
                if self.config.verbose:
                    print(f"\n[SECURITY BLOCKED] Action not authorized: {tool_name}")
                    print(f"  Reason: {str(e)}")
                raise

        # SECURITY LAYER 3 (Phase 2): Multi-Step Attack Detection
        if self.safe_config.enable_memory_firewall:
            self._check_phase2_security(
                tool_name,
                parameters,
                policy_decision.risk_score.total_score if policy_decision else 0.0
            )

        # Execute with base implementation
        result = super()._execute_tool(tool_name, parameters)

        # SECURITY LAYER 4 (Phase 2): Record action after successful execution
        if self.safe_config.enable_memory_firewall and self.sequence_analyzer:
            self._record_action(tool_name, parameters, result, policy_decision)

        return result

    def _check_phase2_security(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        risk_score: float
    ):
        """
        Check Phase 2 security: multi-step attacks, drift, escalation

        Args:
            tool_name: Tool being executed
            parameters: Tool parameters
            risk_score: Risk score from Phase 1

        Raises:
            SecurityViolationError: If Phase 2 violation detected
        """
        # Create action record for this planned action
        action_record = ActionRecord(
            timestamp=datetime.now(),
            step_number=self.current_step,
            action_type=self._infer_action_type(tool_name, parameters),
            tool_name=tool_name,
            parameters=parameters,
            result=None,  # Not executed yet
            risk_score=risk_score,
            reasoning="",  # TODO: Extract from reasoning trace
            user_id=self.safe_config.user_id,
            session_id=self.safe_config.session_id
        )

        # Check 1: Multi-step attack pattern detection
        if self.attack_detector and self.sequence_analyzer:
            # Get current sequence plus planned action
            current_actions = self.sequence_analyzer.action_history.actions if self.sequence_analyzer.action_history else []
            planned_sequence = [a.to_tuple() for a in current_actions] + [action_record.to_tuple()]

            # Check for attack patterns
            attack_result = self.attack_detector.detect_attacks(planned_sequence)

            if attack_result.attack_detected:
                self.phase2_stats['attacks_detected'] += 1

                # Foundation #7: Log attack detection
                if self.safe_config.enable_audit_logging and hasattr(self, 'audit_logger'):
                    self.audit_logger.log_attack_detection(
                        session_id=self.safe_config.session_id,
                        user_id=self.safe_config.user_id,
                        attack_result=attack_result,
                        metadata={
                            'step_number': self.current_step,
                            'tool_name': tool_name,
                            'parameters': str(parameters)[:200]  # Truncate for logging
                        }
                    )

                if self.config.verbose:
                    print(f"\n[PHASE 2 BLOCKED] Multi-step attack detected!")
                    print(f"  Pattern: {attack_result.most_likely_pattern.name}")
                    print(f"  Confidence: {attack_result.highest_confidence:.0%}")
                    print(f"  Severity: {attack_result.most_likely_pattern.severity.value}")

                from ..safety import SecurityViolationError
                raise SecurityViolationError(
                    f"Multi-step attack pattern detected: {attack_result.most_likely_pattern.name}",
                    details={
                        'pattern_id': attack_result.most_likely_pattern.pattern_id,
                        'confidence': attack_result.confidence,
                        'severity': attack_result.most_likely_pattern.severity.value,
                        'matching_steps': attack_result.matching_steps
                    }
                )

        # Check 2: Goal alignment and drift
        if self.sequence_analyzer and self.sequence_analyzer.action_history:
            current_actions = self.sequence_analyzer.action_history.actions
            if len(current_actions) > 0:
                # Check alignment with planned action
                actions_to_check = current_actions + [action_record]
                alignment = self.sequence_analyzer.check_goal_alignment(
                    self.sequence_analyzer.action_history.original_task,
                    actions_to_check
                )

                if not alignment.is_aligned and alignment.drift_score > 0.8:
                    self.phase2_stats['drift_detections'] += 1

                    if self.config.verbose:
                        print(f"\n[PHASE 2 WARNING] Goal drift detected!")
                        print(f"  Drift score: {alignment.drift_score:.0%}")
                        print(f"  Explanation: {alignment.explanation}")

                    # Don't block on drift alone, but log it
                    # Could be configured to block in STRICT mode

        # Check 3: Privilege escalation detection
        if self.sequence_analyzer and self.sequence_analyzer.action_history:
            if len(self.sequence_analyzer.action_history.actions) >= 2:
                escalation = self.sequence_analyzer.detect_escalation(
                    self.sequence_analyzer.action_history
                )

                if escalation.escalation_detected:
                    self.phase2_stats['escalation_detections'] += 1

                    if self.config.verbose:
                        print(f"\n[PHASE 2 WARNING] Privilege escalation detected!")
                        print(f"  Escalation rate: {escalation.escalation_rate:.0%} per step")
                        print(f"  Concerning steps: {escalation.concerning_steps}")

                    # Log but don't block (could be legitimate workflow)
                    # Could be configured to block in STRICT mode

        self.phase2_stats['total_actions_analyzed'] += 1

    def _record_action(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: ExecutionResult,
        policy_decision
    ):
        """Record executed action in sequence analyzer and audit log"""
        action_type = self._infer_action_type(tool_name, parameters)

        # Record in sequence analyzer (Phase 2)
        if self.sequence_analyzer:
            action_record = ActionRecord(
                timestamp=datetime.now(),
                step_number=self.current_step,
                action_type=action_type,
                tool_name=tool_name,
                parameters=parameters,
                result=result,
                risk_score=policy_decision.risk_score.total_score if policy_decision else 0.0,
                reasoning="",  # TODO: Extract from reasoning trace
                user_id=self.safe_config.user_id,
                session_id=self.safe_config.session_id
            )
            self.sequence_analyzer.record_action(action_record)

        # Foundation #7: Log action to audit log
        if self.safe_config.enable_audit_logging and hasattr(self, 'audit_logger'):
            self.audit_logger.log_action(
                session_id=self.safe_config.session_id,
                user_id=self.safe_config.user_id,
                step_number=self.current_step,
                tool_name=tool_name,
                action_type=action_type,
                parameters=parameters,
                phase1_result=policy_decision,
                phase2_result=None,  # Could add Phase 2 analysis results here
                allowed=True,  # If we got here, action was allowed
                result=result.result if result else None,
                error=result.error if result and hasattr(result, 'error') else None,
                duration_ms=result.execution_time_ms if result and hasattr(result, 'execution_time_ms') else None
            )

    def _infer_action_type(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Infer action type from tool name and parameters"""
        tool_lower = tool_name.lower()

        # Common action type mappings
        if any(keyword in tool_lower for keyword in ['read', 'get', 'fetch', 'retrieve', 'search', 'query']):
            return 'read'
        elif any(keyword in tool_lower for keyword in ['write', 'create', 'save', 'store', 'insert']):
            return 'write'
        elif any(keyword in tool_lower for keyword in ['delete', 'remove', 'drop', 'purge']):
            return 'delete'
        elif any(keyword in tool_lower for keyword in ['update', 'modify', 'edit', 'change', 'patch']):
            return 'modify'
        elif any(keyword in tool_lower for keyword in ['execute', 'run', 'call', 'invoke']):
            return 'execute'
        elif any(keyword in tool_lower for keyword in ['admin', 'permission', 'role', 'auth']):
            return 'admin'
        elif any(keyword in tool_lower for keyword in ['network', 'connect', 'request', 'http']):
            return 'network'
        else:
            return 'other'

    def get_security_stats(self) -> Dict[str, Any]:
        """
        Get security statistics (Phase 1 + Phase 2)

        Returns:
            Dictionary with security metrics
        """
        stats = {
            'phase1': self.security_stats.copy(),
            'phase2': self.phase2_stats.copy() if hasattr(self, 'phase2_stats') else {}
        }

        # Add policy violation summary if available
        if self.action_policy:
            stats['phase1']['policy_violations'] = self.action_policy.get_violation_summary()

        # Calculate Phase 1 rates
        if stats['phase1']['total_validations'] > 0:
            stats['phase1']['injection_block_rate'] = stats['phase1']['blocked_injections'] / stats['phase1']['total_validations']

        if stats['phase1']['total_actions_evaluated'] > 0:
            stats['phase1']['action_approval_rate'] = stats['phase1']['approved_actions'] / stats['phase1']['total_actions_evaluated']
            stats['phase1']['action_block_rate'] = stats['phase1']['blocked_actions'] / stats['phase1']['total_actions_evaluated']

        # Add Phase 2 component summaries if available
        if self.safe_config.enable_memory_firewall:
            if self.sequence_analyzer and self.sequence_analyzer.action_history:
                stats['phase2']['sequence_summary'] = self.sequence_analyzer.get_summary()

            if self.reasoning_monitor and hasattr(self.reasoning_monitor, 'step_analyses') and self.reasoning_monitor.step_analyses:
                stats['phase2']['reasoning_summary'] = self.reasoning_monitor.get_summary()

            if self.memory_validator:
                stats['phase2']['memory_summary'] = self.memory_validator.get_summary()

        return stats

    def set_approval_callback(self, callback):
        """
        Set callback function for approval workflow

        Args:
            callback: Function that takes PolicyDecision and returns bool
        """
        if self.action_policy:
            self.action_policy.approval_callback = callback


def create_safe_agent(
    llm_provider: str = "openai",
    llm_model: str = "gpt-4",
    safety_mode: SafetyMode = SafetyMode.BALANCED,
    risk_threshold: float = 0.7,
    user_role: str = "user",
    enable_approval: bool = True,
    enable_memory_firewall: bool = True
) -> SafeDeepAgent:
    """
    Factory function to create a safe agent

    Args:
        llm_provider: LLM provider ("openai", "anthropic", "ollama")
        llm_model: Model name
        safety_mode: Security mode (STRICT, BALANCED, PERMISSIVE, RESEARCH)
        risk_threshold: Maximum acceptable risk (0.0 - 1.0)
        user_role: User role for authorization
        enable_approval: Enable approval workflow for high-risk actions
        enable_memory_firewall: Enable Phase 2 memory firewalls (multi-step attack detection)

    Returns:
        SafeDeepAgent instance

    Example:
        >>> # Create agent with full security (Phase 1 + Phase 2)
        >>> agent = create_safe_agent(
        ...     llm_provider="openai",
        ...     safety_mode=SafetyMode.STRICT,
        ...     user_role="researcher",
        ...     enable_memory_firewall=True
        ... )
        >>> result = agent.run("Search for CRISPR research")
    """
    config = SafeAgentConfig(
        llm_provider=llm_provider,
        llm_model=llm_model,
        safety_mode=safety_mode,
        risk_threshold=risk_threshold,
        user_role=user_role,
        enable_approval_workflow=enable_approval,
        enable_memory_firewall=enable_memory_firewall
    )

    return SafeDeepAgent(config)
