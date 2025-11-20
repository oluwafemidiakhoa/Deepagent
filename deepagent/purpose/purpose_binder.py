"""
Purpose Binder - Foundation #8

Binds agents to specific purposes and task scopes.
Ensures agents don't deviate from their intended purpose.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class PurposeScope(Enum):
    """Scope of agent purpose"""
    NARROW = "narrow"        # Single specific task
    FOCUSED = "focused"      # Related set of tasks
    BROAD = "broad"          # Wide range of tasks
    UNRESTRICTED = "unrestricted"  # No purpose restrictions


@dataclass
class Purpose:
    """Definition of agent purpose"""
    purpose_id: str
    name: str
    description: str
    scope: PurposeScope
    allowed_tasks: List[str]
    allowed_tools: List[str]
    allowed_domains: List[str] = field(default_factory=list)
    restrictions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PurposeBinding:
    """Binding of purpose to agent"""
    binding_id: str
    agent_id: str
    purpose: Purpose
    bound_at: datetime
    expires_at: Optional[datetime] = None
    violations: List[str] = field(default_factory=list)
    active: bool = True


@dataclass
class PurposeCheckResult:
    """Result of purpose compliance check"""
    agent_id: str
    purpose_id: str
    compliant: bool
    violations: List[str]
    warnings: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class PurposeBinder:
    """
    Binds agents to specific purposes and verifies compliance.

    Provides:
    - Purpose definition and binding
    - Task scope verification
    - Purpose compliance checking
    - Violation tracking
    """

    def __init__(self):
        self.purposes: Dict[str, Purpose] = {}
        self.bindings: Dict[str, PurposeBinding] = {}
        self.agent_bindings: Dict[str, str] = {}  # agent_id -> binding_id
        self.compliance_history: List[PurposeCheckResult] = []

    def create_purpose(
        self,
        purpose_id: str,
        name: str,
        description: str,
        scope: PurposeScope,
        allowed_tasks: List[str],
        allowed_tools: List[str],
        allowed_domains: Optional[List[str]] = None,
        restrictions: Optional[Dict[str, Any]] = None
    ) -> Purpose:
        """
        Create a new purpose definition.

        Args:
            purpose_id: Unique purpose identifier
            name: Purpose name
            description: Purpose description
            scope: Scope of purpose
            allowed_tasks: Tasks agent can perform
            allowed_tools: Tools agent can use
            allowed_domains: Domains agent can access
            restrictions: Additional restrictions

        Returns:
            Purpose object
        """
        purpose = Purpose(
            purpose_id=purpose_id,
            name=name,
            description=description,
            scope=scope,
            allowed_tasks=allowed_tasks,
            allowed_tools=allowed_tools,
            allowed_domains=allowed_domains or [],
            restrictions=restrictions or {}
        )

        self.purposes[purpose_id] = purpose
        return purpose

    def bind_agent(
        self,
        agent_id: str,
        purpose_id: str,
        expires_at: Optional[datetime] = None
    ) -> Optional[PurposeBinding]:
        """
        Bind an agent to a purpose.

        Args:
            agent_id: Agent to bind
            purpose_id: Purpose to bind to
            expires_at: Optional expiration time

        Returns:
            PurposeBinding if successful, None otherwise
        """
        if purpose_id not in self.purposes:
            return None

        # Check if agent already bound
        if agent_id in self.agent_bindings:
            # Unbind existing
            self.unbind_agent(agent_id)

        from uuid import uuid4
        binding_id = f"binding_{uuid4().hex[:12]}"

        binding = PurposeBinding(
            binding_id=binding_id,
            agent_id=agent_id,
            purpose=self.purposes[purpose_id],
            bound_at=datetime.now(),
            expires_at=expires_at
        )

        self.bindings[binding_id] = binding
        self.agent_bindings[agent_id] = binding_id

        return binding

    def unbind_agent(self, agent_id: str) -> bool:
        """Unbind an agent from its purpose"""
        if agent_id not in self.agent_bindings:
            return False

        binding_id = self.agent_bindings[agent_id]
        if binding_id in self.bindings:
            self.bindings[binding_id].active = False
            del self.bindings[binding_id]

        del self.agent_bindings[agent_id]
        return True

    def check_purpose_compliance(
        self,
        agent_id: str,
        intended_action: Dict[str, Any]
    ) -> PurposeCheckResult:
        """
        Check if an intended action complies with agent's purpose.

        Args:
            agent_id: Agent performing action
            intended_action: Action to check

        Returns:
            PurposeCheckResult with findings
        """
        if agent_id not in self.agent_bindings:
            return PurposeCheckResult(
                agent_id=agent_id,
                purpose_id="none",
                compliant=True,  # No binding = no restrictions
                violations=[],
                warnings=["Agent has no purpose binding"]
            )

        binding_id = self.agent_bindings[agent_id]
        binding = self.bindings[binding_id]

        # Check if binding expired
        if binding.expires_at and datetime.now() > binding.expires_at:
            return PurposeCheckResult(
                agent_id=agent_id,
                purpose_id=binding.purpose.purpose_id,
                compliant=False,
                violations=["Purpose binding expired"],
                warnings=[]
            )

        purpose = binding.purpose
        violations = []
        warnings = []

        # Extract action details
        task = intended_action.get('task')
        tool = intended_action.get('tool')
        domain = intended_action.get('domain')
        parameters = intended_action.get('parameters', {})

        # Check task allowed
        if task and purpose.allowed_tasks:
            if task not in purpose.allowed_tasks and '*' not in purpose.allowed_tasks:
                violations.append(f"Task '{task}' not allowed for purpose '{purpose.name}'")

        # Check tool allowed
        if tool and purpose.allowed_tools:
            if tool not in purpose.allowed_tools and '*' not in purpose.allowed_tools:
                violations.append(f"Tool '{tool}' not allowed for purpose '{purpose.name}'")

        # Check domain allowed
        if domain and purpose.allowed_domains:
            if domain not in purpose.allowed_domains and '*' not in purpose.allowed_domains:
                violations.append(f"Domain '{domain}' not allowed for purpose '{purpose.name}'")

        # Check additional restrictions
        for restriction, value in purpose.restrictions.items():
            if restriction in parameters:
                if parameters[restriction] != value:
                    violations.append(
                        f"Restriction violated: {restriction} must be {value}, got {parameters[restriction]}"
                    )

        # Scope-specific checks
        if purpose.scope == PurposeScope.NARROW:
            if task and purpose.allowed_tasks and len(purpose.allowed_tasks) > 1:
                warnings.append("NARROW scope should have single task")

        # Record violations in binding
        if violations:
            binding.violations.extend(violations)

        result = PurposeCheckResult(
            agent_id=agent_id,
            purpose_id=purpose.purpose_id,
            compliant=len(violations) == 0,
            violations=violations,
            warnings=warnings
        )

        self.compliance_history.append(result)
        return result

    def get_agent_purpose(self, agent_id: str) -> Optional[Purpose]:
        """Get the purpose bound to an agent"""
        if agent_id not in self.agent_bindings:
            return None

        binding_id = self.agent_bindings[agent_id]
        binding = self.bindings.get(binding_id)

        return binding.purpose if binding else None

    def get_agent_binding(self, agent_id: str) -> Optional[PurposeBinding]:
        """Get the binding for an agent"""
        if agent_id not in self.agent_bindings:
            return None

        binding_id = self.agent_bindings[agent_id]
        return self.bindings.get(binding_id)

    def get_purpose(self, purpose_id: str) -> Optional[Purpose]:
        """Get a purpose definition"""
        return self.purposes.get(purpose_id)

    def list_purposes(self) -> List[Purpose]:
        """List all purpose definitions"""
        return list(self.purposes.values())

    def list_bindings(self, active_only: bool = True) -> List[PurposeBinding]:
        """List all purpose bindings"""
        bindings = list(self.bindings.values())

        if active_only:
            bindings = [b for b in bindings if b.active]

        return bindings

    def get_compliance_history(
        self,
        agent_id: Optional[str] = None,
        purpose_id: Optional[str] = None,
        compliant_only: Optional[bool] = None
    ) -> List[PurposeCheckResult]:
        """Get compliance check history, optionally filtered"""
        history = self.compliance_history

        if agent_id:
            history = [h for h in history if h.agent_id == agent_id]

        if purpose_id:
            history = [h for h in history if h.purpose_id == purpose_id]

        if compliant_only is not None:
            history = [h for h in history if h.compliant == compliant_only]

        return history

    def get_statistics(self) -> Dict[str, Any]:
        """Get purpose binding statistics"""
        total_purposes = len(self.purposes)
        active_bindings = sum(1 for b in self.bindings.values() if b.active)
        total_checks = len(self.compliance_history)
        violations = sum(1 for c in self.compliance_history if not c.compliant)

        # Count by scope
        scope_counts = {
            'narrow': 0,
            'focused': 0,
            'broad': 0,
            'unrestricted': 0
        }

        for purpose in self.purposes.values():
            scope_counts[purpose.scope.value] += 1

        return {
            'total_purposes': total_purposes,
            'active_bindings': active_bindings,
            'total_compliance_checks': total_checks,
            'violations': violations,
            'compliance_rate': ((total_checks - violations) / total_checks * 100) if total_checks > 0 else 100,
            'purposes_by_scope': scope_counts
        }

    def create_data_analysis_purpose(
        self,
        purpose_id: str,
        allowed_data_sources: List[str]
    ) -> Purpose:
        """
        Helper to create a data analysis purpose.

        Args:
            purpose_id: Unique purpose ID
            allowed_data_sources: Data sources agent can access

        Returns:
            Purpose for data analysis
        """
        return self.create_purpose(
            purpose_id=purpose_id,
            name="Data Analysis",
            description="Analyze data from specified sources",
            scope=PurposeScope.FOCUSED,
            allowed_tasks=['read_data', 'analyze_data', 'generate_report'],
            allowed_tools=['read_file', 'query_database', 'run_analysis'],
            allowed_domains=allowed_data_sources,
            restrictions={'write_access': False}
        )

    def create_code_review_purpose(self, purpose_id: str) -> Purpose:
        """Helper to create a code review purpose"""
        return self.create_purpose(
            purpose_id=purpose_id,
            name="Code Review",
            description="Review code for quality and security",
            scope=PurposeScope.FOCUSED,
            allowed_tasks=['read_code', 'analyze_code', 'generate_feedback'],
            allowed_tools=['read_file', 'run_linter', 'run_tests'],
            allowed_domains=['code_repository'],
            restrictions={'modify_code': False}
        )
