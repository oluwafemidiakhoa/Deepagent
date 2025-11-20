"""
Attack Pattern Database

Stores and matches known multi-step attack patterns.
Part of Foundation #2: Memory Firewalls
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AttackCategory(Enum):
    """Categories of multi-step attacks"""
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    GOAL_HIJACKING = "goal_hijacking"
    SCOPE_EXPANSION = "scope_expansion"
    INFORMATION_POISONING = "information_poisoning"
    PERSISTENCE = "persistence"
    RECONNAISSANCE = "reconnaissance"
    INFRASTRUCTURE_ATTACK = "infrastructure_attack"


class AttackSeverity(Enum):
    """Severity levels for attacks"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AttackStep:
    """
    Single step in a multi-step attack pattern
    """
    step_number: int
    action_type: str  # "read", "write", "execute", "delete", "modify", etc.
    target_pattern: str  # Regex for parameter matching
    required: bool = True  # Must match or optional

    # Flexibility for matching
    alternatives: List[str] = field(default_factory=list)  # Alternative action types
    position_flexibility: int = 0  # Can be ±N positions from step_number

    def matches_action(self, action_type: str, parameters: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Check if action matches this step

        Args:
            action_type: Type of action being checked
            parameters: Action parameters

        Returns:
            (matches, confidence) tuple
        """
        # Check action type
        action_match = action_type == self.action_type or action_type in self.alternatives

        if not action_match:
            return False, 0.0

        # Check parameter pattern
        param_str = " ".join(str(v) for v in parameters.values())
        pattern_match = bool(re.search(self.target_pattern, param_str, re.IGNORECASE))

        if not pattern_match:
            return False, 0.0

        # Calculate confidence based on match quality
        confidence = 0.9 if action_type == self.action_type else 0.7

        return True, confidence


@dataclass
class AttackPattern:
    """
    Multi-step attack pattern definition
    """
    pattern_id: str
    name: str
    description: str
    steps: List[AttackStep]
    severity: AttackSeverity
    confidence_threshold: float = 0.7

    # Classification
    attack_category: AttackCategory = AttackCategory.DATA_EXFILTRATION

    # Metadata
    references: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_date: datetime = field(default_factory=datetime.now)

    def match_sequence(
        self,
        action_sequence: List[Tuple[str, Dict[str, Any]]]
    ) -> 'AttackMatch':
        """
        Match this pattern against an action sequence

        Args:
            action_sequence: List of (action_type, parameters) tuples

        Returns:
            AttackMatch with details about the match
        """
        if len(action_sequence) < len([s for s in self.steps if s.required]):
            return AttackMatch(
                pattern=self,
                matched=False,
                confidence=0.0,
                matching_steps=[],
                explanation="Sequence too short for pattern"
            )

        # Try to match steps with sliding window + flexibility
        matched_steps = []
        total_confidence = 0.0
        sequence_idx = 0

        for step in self.steps:
            # Search within flexibility range
            search_start = max(0, sequence_idx - step.position_flexibility)
            search_end = min(len(action_sequence), sequence_idx + step.position_flexibility + 1)

            found = False
            best_confidence = 0.0
            best_idx = -1

            for idx in range(search_start, search_end):
                if idx >= len(action_sequence):
                    break

                action_type, parameters = action_sequence[idx]
                matches, confidence = step.matches_action(action_type, parameters)

                if matches and confidence > best_confidence:
                    found = True
                    best_confidence = confidence
                    best_idx = idx

            if found:
                matched_steps.append((step.step_number, best_idx, best_confidence))
                total_confidence += best_confidence
                sequence_idx = best_idx + 1
            elif step.required:
                # Required step not found
                return AttackMatch(
                    pattern=self,
                    matched=False,
                    confidence=0.0,
                    matching_steps=matched_steps,
                    explanation=f"Required step {step.step_number} not found"
                )

        # Calculate overall confidence
        if len(matched_steps) > 0:
            avg_confidence = total_confidence / len(matched_steps)
            # Boost confidence if steps are in order
            in_order = all(
                matched_steps[i][1] < matched_steps[i+1][1]
                for i in range(len(matched_steps)-1)
            )
            if in_order:
                avg_confidence *= 1.2  # 20% boost for correct order

            avg_confidence = min(1.0, avg_confidence)

            matched = avg_confidence >= self.confidence_threshold

            return AttackMatch(
                pattern=self,
                matched=matched,
                confidence=avg_confidence,
                matching_steps=matched_steps,
                explanation=f"Matched {len(matched_steps)}/{len(self.steps)} steps" +
                           (", in order" if in_order else ", out of order")
            )
        else:
            return AttackMatch(
                pattern=self,
                matched=False,
                confidence=0.0,
                matching_steps=[],
                explanation="No steps matched"
            )


@dataclass
class AttackMatch:
    """Result of pattern matching against sequence"""
    pattern: AttackPattern
    matched: bool
    confidence: float
    matching_steps: List[Tuple[int, int, float]]  # (step_num, seq_idx, confidence)
    explanation: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AttackDetectionResult:
    """Result of attack detection analysis"""
    attack_detected: bool
    detected_patterns: List[AttackMatch]
    highest_confidence: float
    most_likely_pattern: Optional[AttackPattern]
    severity: Optional[AttackSeverity]
    recommendation: str
    timestamp: datetime = field(default_factory=datetime.now)


class AttackPatternDatabase:
    """
    Database of known multi-step attack patterns

    Stores patterns and provides matching capabilities
    """

    def __init__(self):
        """Initialize with built-in attack patterns"""
        self.patterns: Dict[str, AttackPattern] = {}
        self._load_builtin_patterns()

    def _load_builtin_patterns(self):
        """Load built-in attack patterns"""

        # Pattern 1: Backup-Export-Delete (Data Exfiltration)
        self.add_pattern(AttackPattern(
            pattern_id="data_exfil_001",
            name="Backup-Export-Delete",
            description="Create backup, export to external location, delete original data",
            severity=AttackSeverity.CRITICAL,
            attack_category=AttackCategory.DATA_EXFILTRATION,
            confidence_threshold=0.7,
            steps=[
                AttackStep(
                    step_number=1,
                    action_type="read",
                    target_pattern=r"(backup|copy|export|dump).*data",
                    alternatives=["search", "list"],
                    position_flexibility=1
                ),
                AttackStep(
                    step_number=2,
                    action_type="write",
                    target_pattern=r"(external|remote|upload|send|transmit)",
                    alternatives=["execute", "call_api"],
                    position_flexibility=2
                ),
                AttackStep(
                    step_number=3,
                    action_type="delete",
                    target_pattern=r"(original|source|data|file)",
                    alternatives=["modify"],
                    position_flexibility=1
                )
            ],
            tags=["exfiltration", "data_theft", "backup"],
            references=["MITRE ATT&CK: T1020"]
        ))

        # Pattern 2: Gradual Permission Increase (Privilege Escalation)
        self.add_pattern(AttackPattern(
            pattern_id="priv_esc_001",
            name="Gradual Permission Increase",
            description="Read permissions, modify role, execute privileged action",
            severity=AttackSeverity.HIGH,
            attack_category=AttackCategory.PRIVILEGE_ESCALATION,
            confidence_threshold=0.75,
            steps=[
                AttackStep(
                    step_number=1,
                    action_type="read",
                    target_pattern=r"(user|permission|role|access)",
                    alternatives=["search", "list"],
                    position_flexibility=1
                ),
                AttackStep(
                    step_number=2,
                    action_type="modify",
                    target_pattern=r"(role|permission|access|privilege|admin)",
                    alternatives=["write", "update"],
                    position_flexibility=2
                ),
                AttackStep(
                    step_number=3,
                    action_type="execute",
                    target_pattern=r"(admin|privileged|root|sudo|elevated)",
                    alternatives=["call_api", "run"],
                    position_flexibility=1
                )
            ],
            tags=["privilege_escalation", "permissions"],
            references=["MITRE ATT&CK: T1068"]
        ))

        # Pattern 3: Task Substitution (Goal Hijacking)
        self.add_pattern(AttackPattern(
            pattern_id="goal_hijack_001",
            name="Task Substitution",
            description="Start normal, conclude original, execute different task",
            severity=AttackSeverity.HIGH,
            attack_category=AttackCategory.GOAL_HIJACKING,
            confidence_threshold=0.7,
            steps=[
                AttackStep(
                    step_number=1,
                    action_type="search",
                    target_pattern=r".*",  # Any search
                    required=False,
                    position_flexibility=2
                ),
                AttackStep(
                    step_number=2,
                    action_type="conclude",
                    target_pattern=r"(forget|ignore|disregard).*(original|previous|task)",
                    alternatives=["think", "reason"],
                    position_flexibility=1
                ),
                AttackStep(
                    step_number=3,
                    action_type="execute",
                    target_pattern=r"(new|different|instead|actually).*task",
                    alternatives=["run", "perform"],
                    position_flexibility=1
                )
            ],
            tags=["goal_hijacking", "task_substitution"],
            references=[]
        ))

        # Pattern 4: Scope Creep (Gradual Scope Expansion)
        self.add_pattern(AttackPattern(
            pattern_id="scope_exp_001",
            name="Gradual Scope Expansion",
            description="Start narrow, progressively expand to unrestricted access",
            severity=AttackSeverity.MEDIUM,
            attack_category=AttackCategory.SCOPE_EXPANSION,
            confidence_threshold=0.7,
            steps=[
                AttackStep(
                    step_number=1,
                    action_type="read",
                    target_pattern=r"(user|single|one|specific).*id.*\d+",
                    alternatives=["search"],
                    position_flexibility=1
                ),
                AttackStep(
                    step_number=2,
                    action_type="read",
                    target_pattern=r"(range|multiple|several|batch)",
                    alternatives=["search", "list"],
                    position_flexibility=2
                ),
                AttackStep(
                    step_number=3,
                    action_type="read",
                    target_pattern=r"(all|entire|complete|\*|full)",
                    alternatives=["search", "list", "dump"],
                    position_flexibility=1
                )
            ],
            tags=["scope_expansion", "data_access"],
            references=[]
        ))

        # Pattern 5: Reconnaissance Chain
        self.add_pattern(AttackPattern(
            pattern_id="recon_001",
            name="Infrastructure Reconnaissance",
            description="Systematic information gathering about system/infrastructure",
            severity=AttackSeverity.MEDIUM,
            attack_category=AttackCategory.RECONNAISSANCE,
            confidence_threshold=0.65,
            steps=[
                AttackStep(
                    step_number=1,
                    action_type="read",
                    target_pattern=r"(system|config|environment|info)",
                    alternatives=["search", "execute"],
                    position_flexibility=2
                ),
                AttackStep(
                    step_number=2,
                    action_type="read",
                    target_pattern=r"(network|connectivity|services|ports)",
                    alternatives=["search", "execute"],
                    position_flexibility=2
                ),
                AttackStep(
                    step_number=3,
                    action_type="read",
                    target_pattern=r"(auth|authentication|credentials|login|endpoint)",
                    alternatives=["search"],
                    position_flexibility=2
                )
            ],
            tags=["reconnaissance", "information_gathering"],
            references=["MITRE ATT&CK: TA0043"]
        ))

        # Pattern 6: Memory Poisoning
        self.add_pattern(AttackPattern(
            pattern_id="mem_poison_001",
            name="False Memory Injection",
            description="Insert false information, use as justification for harmful action",
            severity=AttackSeverity.HIGH,
            attack_category=AttackCategory.INFORMATION_POISONING,
            confidence_threshold=0.75,
            steps=[
                AttackStep(
                    step_number=1,
                    action_type="write",
                    target_pattern=r"(memory|context|policy|rule|guideline)",
                    alternatives=["modify", "add"],
                    position_flexibility=1
                ),
                AttackStep(
                    step_number=2,
                    action_type="read",
                    target_pattern=r"(policy|rule|memory|context)",
                    alternatives=["search"],
                    required=False,
                    position_flexibility=2
                ),
                AttackStep(
                    step_number=3,
                    action_type="execute",
                    target_pattern=r"(based on|according to|policy|rule)",
                    alternatives=["delete", "modify"],
                    position_flexibility=1
                )
            ],
            tags=["memory_poisoning", "false_data"],
            references=[]
        ))

    def add_pattern(self, pattern: AttackPattern):
        """Add attack pattern to database"""
        self.patterns[pattern.pattern_id] = pattern

    def get_pattern(self, pattern_id: str) -> Optional[AttackPattern]:
        """Get pattern by ID"""
        return self.patterns.get(pattern_id)

    def detect_attacks(
        self,
        action_sequence: List[Tuple[str, Dict[str, Any]]],
        min_confidence: float = 0.6
    ) -> AttackDetectionResult:
        """
        Analyze action sequence for attack patterns

        Args:
            action_sequence: List of (action_type, parameters) tuples
            min_confidence: Minimum confidence threshold

        Returns:
            AttackDetectionResult with detected patterns
        """
        detected_patterns = []
        highest_confidence = 0.0
        most_likely_pattern = None

        # Check each pattern
        for pattern in self.patterns.values():
            match = pattern.match_sequence(action_sequence)

            if match.matched and match.confidence >= min_confidence:
                detected_patterns.append(match)

                if match.confidence > highest_confidence:
                    highest_confidence = match.confidence
                    most_likely_pattern = pattern

        # Determine result
        attack_detected = len(detected_patterns) > 0

        if attack_detected:
            severity = most_likely_pattern.severity if most_likely_pattern else AttackSeverity.MEDIUM

            if severity == AttackSeverity.CRITICAL:
                recommendation = "BLOCK immediately and alert security team"
            elif severity == AttackSeverity.HIGH:
                recommendation = "BLOCK and require human approval"
            elif severity == AttackSeverity.MEDIUM:
                recommendation = "Flag for review and enhanced monitoring"
            else:
                recommendation = "Log incident for analysis"
        else:
            severity = None
            recommendation = "No attack detected, continue normal operation"

        return AttackDetectionResult(
            attack_detected=attack_detected,
            detected_patterns=detected_patterns,
            highest_confidence=highest_confidence,
            most_likely_pattern=most_likely_pattern,
            severity=severity,
            recommendation=recommendation
        )

    def get_patterns_by_category(self, category: AttackCategory) -> List[AttackPattern]:
        """Get all patterns in a category"""
        return [p for p in self.patterns.values() if p.attack_category == category]

    def get_patterns_by_severity(self, severity: AttackSeverity) -> List[AttackPattern]:
        """Get all patterns of a severity level"""
        return [p for p in self.patterns.values() if p.severity == severity]

    def get_all_patterns(self) -> List[AttackPattern]:
        """Get all patterns"""
        return list(self.patterns.values())

    def pattern_count(self) -> int:
        """Get total number of patterns"""
        return len(self.patterns)
