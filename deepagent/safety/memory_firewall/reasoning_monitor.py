"""
Reasoning Monitor

Monitors agent's reasoning process for security anomalies.
Part of Foundation #2: Memory Firewalls
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AnomalyType(Enum):
    """Types of reasoning anomalies"""
    TOPIC_DRIFT = "topic_drift"
    SENTIMENT_SHIFT = "sentiment_shift"
    GOAL_ABANDONMENT = "goal_abandonment"
    UNUSUAL_PATTERN = "unusual_pattern"
    CONTRADICTORY_REASONING = "contradictory_reasoning"
    EXCESSIVE_COMPLEXITY = "excessive_complexity"


@dataclass
class ReasoningAnalysis:
    """Analysis of a single reasoning step"""
    step_number: int
    is_consistent: bool
    anomalies: List[str] = field(default_factory=list)
    topic_drift: float = 0.0  # 0-1 scale
    sentiment_shift: float = 0.0  # -1 to 1 (negative to positive)
    risk_indicators: List[str] = field(default_factory=list)
    confidence: float = 0.7
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SequenceAnalysis:
    """Analysis of entire reasoning sequence"""
    total_steps: int
    consistent_steps: int
    anomalous_steps: int
    overall_consistency: float  # 0-1 scale
    detected_anomalies: Dict[AnomalyType, int]  # Count by type
    risk_level: str  # "low", "medium", "high", "critical"
    explanation: str
    timestamp: datetime = field(default_factory=datetime.now)


class ReasoningMonitor:
    """
    Monitor reasoning trace for security issues

    Analyzes:
    - Topic consistency
    - Sentiment shifts
    - Goal keyword presence
    - Unusual patterns
    - Contradictions
    """

    def __init__(self):
        """Initialize monitor"""
        # Baseline from original task
        self.baseline_topics: Set[str] = set()
        self.baseline_sentiment: Optional[str] = None
        self.goal_keywords: Set[str] = set()
        self.original_task: str = ""

        # Configuration
        self.topic_drift_threshold = 0.6
        self.sentiment_shift_threshold = 0.5

        # Tracking
        self.step_analyses: List[ReasoningAnalysis] = []

        # Negative sentiment indicators
        self.negative_words = {
            'attack', 'exploit', 'hack', 'breach', 'steal', 'destroy',
            'delete', 'remove', 'bypass', 'evade', 'circumvent', 'override',
            'ignore', 'disregard', 'forget', 'abandon', 'malicious',
            'unauthorized', 'illegal', 'forbidden', 'prohibited'
        }

        # Positive sentiment indicators
        self.positive_words = {
            'help', 'assist', 'support', 'research', 'analyze', 'study',
            'learn', 'understand', 'discover', 'create', 'build', 'develop',
            'improve', 'enhance', 'optimize', 'benefit', 'contribute'
        }

        # Suspicious patterns
        self.suspicious_patterns = {
            'role_change': ['i am now', 'as a', 'pretend', 'roleplay', 'act as'],
            'instruction_override': ['ignore', 'forget', 'disregard', 'override'],
            'urgency': ['immediately', 'urgent', 'asap', 'quickly', 'now'],
            'authority': ['admin', 'root', 'sudo', 'elevated', 'privileged'],
            'obfuscation': ['encode', 'decrypt', 'hidden', 'secret', 'covert']
        }

    def initialize_baseline(self, original_task: str):
        """
        Set baseline from original task

        Args:
            original_task: The original user task
        """
        self.original_task = original_task
        self.baseline_topics = self._extract_topics(original_task)
        self.goal_keywords = self._extract_keywords(original_task)
        self.baseline_sentiment = self._analyze_sentiment(original_task)

    def analyze_step(
        self,
        step_number: int,
        reasoning_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ReasoningAnalysis:
        """
        Analyze single reasoning step

        Args:
            step_number: Step number in sequence
            reasoning_text: The reasoning text to analyze
            context: Optional additional context

        Returns:
            ReasoningAnalysis with step analysis
        """
        context = context or {}

        # Extract topics from this step
        step_topics = self._extract_topics(reasoning_text)

        # Calculate topic drift
        if self.baseline_topics:
            topic_overlap = len(self.baseline_topics & step_topics)
            topic_union = len(self.baseline_topics | step_topics)
            topic_similarity = topic_overlap / topic_union if topic_union > 0 else 0.0
            topic_drift = 1.0 - topic_similarity
        else:
            topic_drift = 0.0

        # Analyze sentiment shift
        step_sentiment = self._analyze_sentiment(reasoning_text)
        sentiment_shift = self._calculate_sentiment_shift(
            self.baseline_sentiment,
            step_sentiment
        )

        # Check for goal keywords
        step_keywords = self._extract_keywords(reasoning_text)
        goal_keyword_present = bool(self.goal_keywords & step_keywords)

        # Detect anomalies
        anomalies = []
        risk_indicators = []

        # Topic drift anomaly
        if topic_drift > self.topic_drift_threshold:
            anomalies.append(f"High topic drift: {topic_drift:.1%}")
            risk_indicators.append("topic_drift")

        # Sentiment shift anomaly
        if abs(sentiment_shift) > self.sentiment_shift_threshold:
            anomalies.append(f"Sentiment shift: {sentiment_shift:+.1%}")
            risk_indicators.append("sentiment_shift")

        # Missing goal keywords
        if not goal_keyword_present and step_number > 1:
            anomalies.append("Goal keywords absent")
            risk_indicators.append("goal_abandonment")

        # Check for suspicious patterns
        for pattern_type, patterns in self.suspicious_patterns.items():
            for pattern in patterns:
                if pattern in reasoning_text.lower():
                    anomalies.append(f"Suspicious pattern: {pattern_type}")
                    risk_indicators.append(pattern_type)
                    break  # One per type

        # Determine consistency
        is_consistent = len(anomalies) == 0

        analysis = ReasoningAnalysis(
            step_number=step_number,
            is_consistent=is_consistent,
            anomalies=anomalies,
            topic_drift=topic_drift,
            sentiment_shift=sentiment_shift,
            risk_indicators=risk_indicators
        )

        self.step_analyses.append(analysis)
        return analysis

    def analyze_sequence(
        self,
        reasoning_steps: List[str]
    ) -> SequenceAnalysis:
        """
        Analyze entire reasoning sequence

        Args:
            reasoning_steps: List of reasoning text for each step

        Returns:
            SequenceAnalysis with overall analysis
        """
        # Analyze each step
        analyses = []
        for i, reasoning_text in enumerate(reasoning_steps, 1):
            analysis = self.analyze_step(i, reasoning_text)
            analyses.append(analysis)

        # Count consistent vs anomalous
        total_steps = len(analyses)
        consistent_steps = sum(1 for a in analyses if a.is_consistent)
        anomalous_steps = total_steps - consistent_steps

        # Calculate overall consistency
        overall_consistency = consistent_steps / total_steps if total_steps > 0 else 1.0

        # Count anomaly types
        anomaly_counts: Dict[AnomalyType, int] = {}
        for analysis in analyses:
            for indicator in analysis.risk_indicators:
                # Map indicator to anomaly type
                if indicator == "topic_drift":
                    anomaly_type = AnomalyType.TOPIC_DRIFT
                elif indicator == "sentiment_shift":
                    anomaly_type = AnomalyType.SENTIMENT_SHIFT
                elif indicator == "goal_abandonment":
                    anomaly_type = AnomalyType.GOAL_ABANDONMENT
                else:
                    anomaly_type = AnomalyType.UNUSUAL_PATTERN

                anomaly_counts[anomaly_type] = anomaly_counts.get(anomaly_type, 0) + 1

        # Determine risk level
        if overall_consistency > 0.8:
            risk_level = "low"
        elif overall_consistency > 0.6:
            risk_level = "medium"
        elif overall_consistency > 0.4:
            risk_level = "high"
        else:
            risk_level = "critical"

        # Generate explanation
        if overall_consistency > 0.8:
            explanation = f"Reasoning is consistent ({overall_consistency:.1%} consistency)"
        else:
            explanation = f"Reasoning anomalies detected ({anomalous_steps}/{total_steps} steps)"
            if anomaly_counts:
                top_anomaly = max(anomaly_counts.items(), key=lambda x: x[1])
                explanation += f". Primary issue: {top_anomaly[0].value} ({top_anomaly[1]} occurrences)"

        return SequenceAnalysis(
            total_steps=total_steps,
            consistent_steps=consistent_steps,
            anomalous_steps=anomalous_steps,
            overall_consistency=overall_consistency,
            detected_anomalies=anomaly_counts,
            risk_level=risk_level,
            explanation=explanation
        )

    def _extract_topics(self, text: str) -> Set[str]:
        """
        Extract topic words from text

        Args:
            text: Text to analyze

        Returns:
            Set of topic words
        """
        # Simple word extraction (remove stop words)
        words = text.lower().split()

        # Stop words to filter
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
            'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

        topics = set()
        for word in words:
            # Remove punctuation
            word = word.strip('.,!?;:()"\'')
            # Keep if not stop word and has substance
            if word and word not in stop_words and len(word) > 2:
                topics.add(word)

        return topics

    def _extract_keywords(self, text: str) -> Set[str]:
        """
        Extract keywords (important nouns/verbs)

        Args:
            text: Text to analyze

        Returns:
            Set of keywords
        """
        # For now, same as topics
        # Could be enhanced with NER or POS tagging
        return self._extract_topics(text)

    def _analyze_sentiment(self, text: str) -> str:
        """
        Analyze sentiment of text

        Args:
            text: Text to analyze

        Returns:
            Sentiment: "positive", "negative", or "neutral"
        """
        words = set(text.lower().split())

        # Count positive and negative words
        positive_count = len(words & self.positive_words)
        negative_count = len(words & self.negative_words)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def _calculate_sentiment_shift(
        self,
        baseline: Optional[str],
        current: str
    ) -> float:
        """
        Calculate sentiment shift

        Args:
            baseline: Baseline sentiment
            current: Current sentiment

        Returns:
            Shift value (-1 to 1)
        """
        if baseline is None:
            return 0.0

        # Map sentiments to values
        sentiment_values = {
            "positive": 1.0,
            "neutral": 0.0,
            "negative": -1.0
        }

        baseline_val = sentiment_values.get(baseline, 0.0)
        current_val = sentiment_values.get(current, 0.0)

        return current_val - baseline_val

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of monitoring

        Returns:
            Dictionary with summary statistics
        """
        if not self.step_analyses:
            return {
                "total_steps": 0,
                "message": "No steps analyzed yet"
            }

        total_steps = len(self.step_analyses)
        consistent_steps = sum(1 for a in self.step_analyses if a.is_consistent)
        anomalous_steps = total_steps - consistent_steps

        # Average drift and shift
        avg_topic_drift = sum(a.topic_drift for a in self.step_analyses) / total_steps
        avg_sentiment_shift = sum(abs(a.sentiment_shift) for a in self.step_analyses) / total_steps

        # Most common anomalies
        all_anomalies = []
        for analysis in self.step_analyses:
            all_anomalies.extend(analysis.anomalies)

        return {
            "total_steps": total_steps,
            "consistent_steps": consistent_steps,
            "anomalous_steps": anomalous_steps,
            "consistency_rate": consistent_steps / total_steps,
            "average_topic_drift": avg_topic_drift,
            "average_sentiment_shift": avg_sentiment_shift,
            "total_anomalies": len(all_anomalies),
            "unique_anomaly_types": len(set(all_anomalies))
        }

    def reset(self):
        """Reset monitor state"""
        self.baseline_topics = set()
        self.baseline_sentiment = None
        self.goal_keywords = set()
        self.original_task = ""
        self.step_analyses = []
