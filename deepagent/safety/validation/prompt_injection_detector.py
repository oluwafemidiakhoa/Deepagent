"""
Prompt Injection Detection System

Detects and blocks prompt injection attacks using multiple techniques:
1. Pattern-based detection (regex)
2. Heuristic analysis
3. Structural boundary checks
4. Encoding detection (base64, unicode, zero-width)

Based on research showing 89.6% attack success rate for roleplay dynamics,
this system aims to reduce ASR to <5%.
"""

import re
import base64
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class InjectionDetectionResult:
    """Result of injection detection"""
    is_injection: bool
    confidence: float  # 0.0 - 1.0
    detected_patterns: List[str]
    suspicious_elements: List[str]
    risk_score: float  # 0.0 - 1.0


class PromptInjectionDetector:
    """
    Multi-layer prompt injection detector

    Defends against:
    - Direct prompt injection
    - Indirect injection (via tool results)
    - Roleplay-based attacks
    - Logic traps
    - Encoding tricks
    - Payload splitting
    """

    def __init__(self, threshold: float = 0.7):
        """
        Initialize detector

        Args:
            threshold: Confidence threshold for blocking (0.0 - 1.0)
        """
        self.threshold = threshold

        # Known injection patterns (regex)
        self.injection_patterns = [
            # Direct instruction override
            (r"ignore\s+(all\s+)?previous\s+instructions?", "instruction_override"),
            (r"disregard\s+(all\s+)?previous", "instruction_override"),
            (r"forget\s+(all\s+)?previous", "instruction_override"),

            # Role manipulation
            (r"you\s+are\s+now", "role_manipulation"),
            (r"new\s+role:", "role_manipulation"),
            (r"assume\s+the\s+role", "role_manipulation"),
            (r"pretend\s+(you\s+are|to\s+be)", "role_manipulation"),

            # System prompt leakage attempts
            (r"show\s+(me\s+)?(your\s+)?system\s+prompt", "system_leak"),
            (r"what\s+(are|is)\s+your\s+instructions", "system_leak"),
            (r"reveal\s+your\s+prompt", "system_leak"),

            # Direct command injection
            (r"EXECUTE_TOOL:\s*\w+", "command_injection"),
            (r"CONCLUDE:\s*", "command_injection"),
            (r"SEARCH_TOOLS:\s*", "command_injection"),

            # Safety bypass attempts
            (r"disable\s+safety", "safety_bypass"),
            (r"turn\s+off\s+safety", "safety_bypass"),
            (r"bypass\s+security", "safety_bypass"),
            (r"without\s+safety\s+checks", "safety_bypass"),

            # Jailbreak attempts
            (r"DAN\s+mode", "jailbreak"),
            (r"Developer\s+Mode", "jailbreak"),
            (r"evil\s+mode", "jailbreak"),

            # Logic traps
            (r"if\s+.*\s+then\s+ignore", "logic_trap"),
            (r"when\s+.*\s+disregard", "logic_trap"),

            # Goal hijacking
            (r"your\s+(new\s+)?goal\s+is", "goal_hijacking"),
            (r"new\s+objective:", "goal_hijacking"),
        ]

        # Compile regex patterns
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), name)
            for pattern, name in self.injection_patterns
        ]

        # Suspicious keywords (heuristic scoring)
        self.suspicious_keywords = [
            "ignore", "disregard", "forget", "override", "bypass",
            "disable", "jailbreak", "exploit", "hack", "manipulate",
            "system prompt", "reveal", "show instructions", "admin",
            "root access", "sudo", "execute", "malicious"
        ]

    def detect(self, text: str) -> InjectionDetectionResult:
        """
        Detect prompt injection in text

        Args:
            text: Input text to analyze

        Returns:
            InjectionDetectionResult with detection details
        """
        detected_patterns = []
        suspicious_elements = []
        pattern_scores = []

        # 1. Pattern-based detection
        for pattern, name in self.compiled_patterns:
            matches = pattern.findall(text)
            if matches:
                detected_patterns.append(name)
                pattern_scores.append(0.9)  # High confidence for known patterns

        # 2. Heuristic analysis
        heuristic_score = self._analyze_heuristics(text, suspicious_elements)

        # 3. Encoding detection
        encoding_score = self._detect_encoding_tricks(text, suspicious_elements)

        # 4. Structural analysis
        structural_score = self._analyze_structure(text, suspicious_elements)

        # Calculate overall risk score
        if pattern_scores:
            pattern_risk = max(pattern_scores)
        else:
            pattern_risk = 0.0

        # Weighted combination
        risk_score = (
            pattern_risk * 0.5 +          # Pattern matching is most reliable
            heuristic_score * 0.2 +       # Keyword-based heuristics
            encoding_score * 0.2 +        # Encoding tricks
            structural_score * 0.1        # Structural anomalies
        )

        # Confidence based on evidence strength
        confidence = min(1.0, (
            len(detected_patterns) * 0.3 +
            len(suspicious_elements) * 0.1 +
            risk_score * 0.6
        ))

        is_injection = confidence >= self.threshold

        return InjectionDetectionResult(
            is_injection=is_injection,
            confidence=confidence,
            detected_patterns=detected_patterns,
            suspicious_elements=suspicious_elements,
            risk_score=risk_score
        )

    def _analyze_heuristics(self, text: str, suspicious_elements: List[str]) -> float:
        """
        Analyze text using keyword heuristics

        Returns:
            Risk score (0.0 - 1.0)
        """
        text_lower = text.lower()
        keyword_count = 0

        for keyword in self.suspicious_keywords:
            if keyword in text_lower:
                keyword_count += 1
                suspicious_elements.append(f"keyword: {keyword}")

        # Normalize by text length (to avoid false positives on long texts)
        words = text.split()
        if len(words) == 0:
            return 0.0

        keyword_density = keyword_count / max(len(words), 1)

        # Risk score based on density
        return min(1.0, keyword_density * 10)

    def _detect_encoding_tricks(self, text: str, suspicious_elements: List[str]) -> float:
        """
        Detect encoding-based obfuscation

        Checks for:
        - Base64 encoded payloads
        - Hex encoded strings
        - Unicode tricks
        - Zero-width characters
        """
        risk = 0.0

        # Base64 detection
        base64_pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
        base64_matches = re.findall(base64_pattern, text)
        for match in base64_matches:
            try:
                decoded = base64.b64decode(match).decode('utf-8', errors='ignore')
                # Check if decoded text contains suspicious keywords
                for keyword in self.suspicious_keywords[:5]:  # Check top keywords
                    if keyword in decoded.lower():
                        suspicious_elements.append(f"base64_encoded: {keyword}")
                        risk = max(risk, 0.8)
            except Exception:
                pass

        # Zero-width characters
        zero_width_chars = ['\u200b', '\u200c', '\u200d', '\ufeff']
        for char in zero_width_chars:
            if char in text:
                suspicious_elements.append("zero_width_character")
                risk = max(risk, 0.6)

        # Excessive unicode
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        if len(text) > 0 and non_ascii_count / len(text) > 0.3:
            suspicious_elements.append("excessive_unicode")
            risk = max(risk, 0.4)

        return risk

    def _analyze_structure(self, text: str, suspicious_elements: List[str]) -> float:
        """
        Analyze structural anomalies

        Checks for:
        - Multiple instruction blocks
        - Boundary marker violations
        - Nested instructions
        """
        risk = 0.0

        # Multiple instruction markers
        instruction_markers = ["TASK:", "INSTRUCTION:", "SYSTEM:", "USER:", "ASSISTANT:"]
        marker_count = sum(text.upper().count(marker) for marker in instruction_markers)

        if marker_count > 2:
            suspicious_elements.append(f"multiple_instructions: {marker_count}")
            risk = max(risk, 0.5)

        # Nested brackets/braces (potential JSON injection)
        bracket_depth = 0
        max_depth = 0
        for char in text:
            if char in '{[(':
                bracket_depth += 1
                max_depth = max(max_depth, bracket_depth)
            elif char in '}])':
                bracket_depth -= 1

        if max_depth > 5:
            suspicious_elements.append(f"deep_nesting: {max_depth}")
            risk = max(risk, 0.3)

        return risk

    def explain_detection(self, result: InjectionDetectionResult) -> str:
        """
        Generate human-readable explanation of detection

        Args:
            result: Detection result

        Returns:
            Explanation string
        """
        if not result.is_injection:
            return "No prompt injection detected. Input appears safe."

        explanation = f"[SECURITY ALERT] Prompt injection detected (confidence: {result.confidence:.1%})\n\n"

        if result.detected_patterns:
            explanation += "Detected attack patterns:\n"
            for pattern in set(result.detected_patterns):
                explanation += f"  - {pattern.replace('_', ' ').title()}\n"

        if result.suspicious_elements:
            explanation += "\nSuspicious elements:\n"
            for element in result.suspicious_elements[:5]:  # Show top 5
                explanation += f"  - {element}\n"

        explanation += f"\nRisk score: {result.risk_score:.1%}"
        explanation += "\n\nThis input has been blocked for security reasons."

        return explanation
