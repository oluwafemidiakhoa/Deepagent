"""
Input Validation System

Validates all user inputs before they reach the agent's reasoning system.
Provides multi-layer defense against malicious inputs.
"""

from typing import Tuple, Dict, Any
from ..config import InputValidationConfig
from ..exceptions import PromptInjectionDetectedError
from .prompt_injection_detector import PromptInjectionDetector
from .content_sanitizer import ContentSanitizer


class InputValidator:
    """
    Multi-layer input validation system

    Validates:
    1. Input length
    2. Prompt injection attempts
    3. Content sanitization
    4. Encoding attacks
    """

    def __init__(self, config: InputValidationConfig):
        """
        Initialize input validator

        Args:
            config: Input validation configuration
        """
        self.config = config
        self.injection_detector = PromptInjectionDetector(
            threshold=config.injection_threshold
        )
        self.sanitizer = ContentSanitizer()

    def validate(self, text: str, context: str = "user_input") -> Tuple[str, Dict[str, Any]]:
        """
        Validate and sanitize input text

        Args:
            text: Input text to validate
            context: Context of the input (for logging)

        Returns:
            Tuple of (validated_text, validation_metadata)

        Raises:
            PromptInjectionDetectedError: If injection is detected and blocked
            ValueError: If input is invalid
        """
        if not self.config.enabled:
            return text, {"validation_skipped": True}

        metadata = {
            "context": context,
            "original_length": len(text),
            "validations_applied": [],
            "sanitizations_applied": [],
            "security_warnings": []
        }

        # 1. Length validation
        if len(text) > self.config.max_input_length:
            raise ValueError(
                f"Input exceeds maximum length of {self.config.max_input_length} characters"
            )

        # 2. Injection detection
        if self.config.detect_injection:
            detection_result = self.injection_detector.detect(text)

            metadata["injection_check"] = {
                "risk_score": detection_result.risk_score,
                "confidence": detection_result.confidence,
                "detected_patterns": detection_result.detected_patterns,
                "suspicious_elements": detection_result.suspicious_elements
            }

            if detection_result.is_injection:
                # Log the attempt
                explanation = self.injection_detector.explain_detection(detection_result)

                raise PromptInjectionDetectedError(
                    f"Prompt injection detected in {context}: {explanation}",
                    detected_patterns=detection_result.detected_patterns
                )

            # Warn on suspicious but not blocking
            if detection_result.risk_score > 0.3:
                metadata["security_warnings"].append(
                    f"Moderate risk detected (score: {detection_result.risk_score:.2f})"
                )

        # 3. Content sanitization
        if self.config.sanitize_input:
            sanitized_text, sanitizations = self.sanitizer.sanitize(text)
            metadata["sanitizations_applied"] = sanitizations
            text = sanitized_text

        # 4. Final validation
        metadata["validations_applied"].append("length_check")
        if self.config.detect_injection:
            metadata["validations_applied"].append("injection_detection")
        if self.config.sanitize_input:
            metadata["validations_applied"].append("content_sanitization")

        metadata["final_length"] = len(text)
        metadata["validation_passed"] = True

        return text, metadata

    def validate_batch(self, texts: list) -> list:
        """
        Validate multiple inputs

        Args:
            texts: List of input texts

        Returns:
            List of (validated_text, metadata) tuples
        """
        results = []
        for i, text in enumerate(texts):
            validated, metadata = self.validate(text, context=f"batch_input_{i}")
            results.append((validated, metadata))
        return results

    def is_safe(self, text: str) -> bool:
        """
        Quick safety check without raising exceptions

        Args:
            text: Text to check

        Returns:
            True if safe, False otherwise
        """
        try:
            _, metadata = self.validate(text)
            return metadata.get("validation_passed", False)
        except (PromptInjectionDetectedError, ValueError):
            return False
