"""
Content Sanitization System

Sanitizes and normalizes user inputs to prevent injection attacks.
"""

import re
import html
from typing import Tuple, List


class ContentSanitizer:
    """
    Sanitizes content to remove potentially malicious elements
    """

    def __init__(self):
        """Initialize sanitizer"""
        pass

    def sanitize(self, text: str) -> Tuple[str, List[str]]:
        """
        Sanitize input text

        Args:
            text: Raw input text

        Returns:
            Tuple of (sanitized_text, list_of_applied_sanitizations)
        """
        sanitizations_applied = []
        sanitized = text

        # 1. Remove zero-width characters
        sanitized, removed = self._remove_zero_width(sanitized)
        if removed:
            sanitizations_applied.append("removed_zero_width_characters")

        # 2. Normalize whitespace
        sanitized = self._normalize_whitespace(sanitized)

        # 3. HTML/XML escape
        if self._contains_html(sanitized):
            sanitized = html.escape(sanitized)
            sanitizations_applied.append("html_escaped")

        # 4. Remove control characters
        sanitized = self._remove_control_chars(sanitized)

        # 5. Normalize unicode
        sanitized = self._normalize_unicode(sanitized)

        return sanitized, sanitizations_applied

    def _remove_zero_width(self, text: str) -> Tuple[str, bool]:
        """Remove zero-width characters"""
        zero_width_chars = [
            '\u200b',  # Zero width space
            '\u200c',  # Zero width non-joiner
            '\u200d',  # Zero width joiner
            '\ufeff',  # Zero width no-break space
        ]

        original = text
        for char in zero_width_chars:
            text = text.replace(char, '')

        return text, (text != original)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace to prevent obfuscation"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text

    def _contains_html(self, text: str) -> bool:
        """Check if text contains HTML/XML tags"""
        return bool(re.search(r'<[^>]+>', text))

    def _remove_control_chars(self, text: str) -> str:
        """Remove control characters"""
        # Keep newline, tab, carriage return
        return ''.join(char for char in text if ord(char) >= 32 or char in '\n\t\r')

    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode to prevent homograph attacks"""
        # This is a simplified version
        # Production would use unicodedata.normalize()
        return text
