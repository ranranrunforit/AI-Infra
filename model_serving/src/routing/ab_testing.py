"""
A/B Testing Module

Statistical A/B testing framework for comparing model variants with automatic
winner selection based on statistical significance.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import asyncio

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment lifecycle status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class VariantMetrics:
    """Metrics for a single variant."""
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency: float = 0.0
    latencies: List[float] = field(default_factory=list)
    user_feedback_scores: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.request_count == 0:
            return 0.0
        return self.success_count / self.request_count

    @property
    def average_latency(self) -> float:
        """Calculate average latency."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    @property
    def average_feedback(self) -> float:
        """Calculate average user feedback score."""
        if not self.user_feedback_scores:
            return 0.0
        return sum(self.user_feedback_scores) / len(self.user_feedback_scores)


@dataclass
class Variant:
    """
    Represents an A/B test variant (model version).

    Attributes:
        name: Variant identifier (e.g., "control", "treatment")
        model_endpoint: URL or identifier of model endpoint
        traffic_percentage: Percentage of traffic (0-100)
        description: Human-readable description
        metrics: Performance metrics for this variant
    """

    name: str
    model_endpoint: str
    traffic_percentage: float
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: VariantMetrics = field(default_factory=VariantMetrics)

    def __post_init__(self):
        """Validate variant configuration."""
        if not 0 <= self.traffic_percentage <= 100:
            raise ValueError("Traffic percentage must be between 0 and 100")
        if not self.name:
            raise ValueError("Variant name cannot be empty")

    def record_request(
        self,
        success: bool,
        latency: float,
        feedback_score: Optional[float] = None
    ) -> None:
        """
        Record a request result for this variant.

        Args:
            success: Whether request was successful
            latency: Request latency in seconds
            feedback_score: Optional user feedback score (0-1 or 1-5)
        """
        self.metrics.request_count += 1

        if success:
            self.metrics.success_count += 1
        else:
            self.metrics.error_count += 1

        self.metrics.total_latency += latency
        self.metrics.latencies.append(latency)

        if feedback_score is not None:
            self.metrics.user_feedback_scores.append(feedback_score)


@dataclass
class Experiment:
    """
    A/B test experiment configuration.

    Attributes:
        name: Experiment identifier
        variants: List of variants to test
        metric: Primary metric to optimize (success_rate, latency, feedback)
        min_sample_size: Minimum samples before statistical testing
        confidence_level: Confidence level for statistical tests (e.g., 0.95)
        status: Current experiment status
    """

    name: str
    variants: List[Variant]
    metric: str = "success_rate"
    min_sample_size: int = 100
    confidence_level: float = 0.95
    max_duration_seconds: Optional[float] = None
    status: ExperimentStatus = ExperimentStatus.DRAFT
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    winner: Optional[str] = None

    def __post_init__(self):
        """Validate experiment configuration."""
        if len(self.variants) < 2:
            raise ValueError("At least 2 variants required for A/B test")

        total_traffic = sum(v.traffic_percentage for v in self.variants)
        if not 99.9 <= total_traffic <= 100.1:  # Allow small floating point errors
            raise ValueError(
                f"Traffic percentages must sum to 100, got {total_traffic}"
            )

        if self.metric not in ["success_rate", "latency", "feedback"]:
            raise ValueError(f"Invalid metric: {self.metric}")

        if not 0 < self.confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")

    def get_variant_by_name(self, name: str) -> Optional[Variant]:
        """Get variant by name."""
        for variant in self.variants:
            if variant.name == name:
                return variant
        return None


class ABTestRouter:
    """
    A/B testing router with statistical analysis.

    This class manages A/B test experiments, routes traffic to variants,
    collects metrics, and performs statistical significance testing to
    automatically determine winners.

    Example:
        ```python
        # Define variants
        control = Variant(
            name="control",
            model_endpoint="http://model-v1:8000",
            traffic_percentage=50
        )
        treatment = Variant(
            name="treatment",
            model_endpoint="http://model-v2:8000",
            traffic_percentage=50
        )

        # Create experiment
        experiment = Experiment(
            name="llama-vs-mistral",
            variants=[control, treatment],
            metric="success_rate",
            min_sample_size=1000
        )

        # Create router
        router = ABTestRouter(experiment)
        await router.start_experiment()

        # Route requests
        variant = router.select_variant(user_id="user123")
        # ... make request to variant.model_endpoint ...
        router.record_result(variant.name, success=True, latency=0.5)

        # Check for winner
        result = router.analyze_results()
        if result["has_winner"]:
            print(f"Winner: {result['winner']}")
        ```
    """

    def __init__(self, experiment: Experiment):
        """
        Initialize A/B test router.

        Args:
            experiment: Experiment configuration
        """
        self.experiment = experiment
        self._lock = asyncio.Lock()

        logger.info(
            f"ABTestRouter initialized for experiment: {experiment.name} "
            f"with {len(experiment.variants)} variants"
        )

    async def start_experiment(self) -> None:
        """Start the A/B test experiment."""
        if self.experiment.status == ExperimentStatus.RUNNING:
            logger.warning("Experiment already running")
            return

        self.experiment.status = ExperimentStatus.RUNNING
        self.experiment.start_time = time.time()
        self.experiment.winner = None

        logger.info(f"Started experiment: {self.experiment.name}")

    async def stop_experiment(self, reason: str = "Manual stop") -> None:
        """
        Stop the experiment.

        Args:
            reason: Reason for stopping
        """
        self.experiment.status = ExperimentStatus.COMPLETED
        self.experiment.end_time = time.time()

        logger.info(f"Stopped experiment: {self.experiment.name} - {reason}")

    def select_variant(self, user_id: Optional[str] = None) -> Variant:
        """
        Select a variant for the request.

        Uses weighted random selection based on traffic percentages.
        Optionally uses hash-based assignment for user consistency.

        Args:
            user_id: Optional user identifier for consistent assignment

        Returns:
            Selected Variant
        """
        if self.experiment.status != ExperimentStatus.RUNNING:
            raise RuntimeError(
                f"Experiment not running (status: {self.experiment.status})"
            )

        # Hash-based assignment for user consistency
        if user_id:
            import hashlib
            hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            rand = (hash_val % 10000) / 100.0  # 0-100
        else:
            rand = random.uniform(0, 100)

        cumulative = 0.0
        for variant in self.experiment.variants:
            cumulative += variant.traffic_percentage
            if rand <= cumulative:
                return variant

        # Fallback to last variant
        return self.experiment.variants[-1]

    def record_result(
        self,
        variant_name: str,
        success: bool,
        latency: float,
        feedback_score: Optional[float] = None
    ) -> None:
        """
        Record a request result for a variant.

        Args:
            variant_name: Name of the variant
            success: Whether request was successful
            latency: Request latency in seconds
            feedback_score: Optional user feedback score
        """
        variant = self.experiment.get_variant_by_name(variant_name)
        if variant is None:
            logger.warning(f"Variant not found: {variant_name}")
            return

        variant.record_request(success, latency, feedback_score)

    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze experiment results and determine if there's a winner.

        Uses statistical tests to determine if one variant is significantly
        better than others based on the primary metric.

        Returns:
            Dictionary with analysis results including:
                - has_winner: Whether a winner was determined
                - winner: Name of winning variant (if any)
                - confidence: Confidence level of the result
                - p_value: Statistical p-value
                - variant_stats: Statistics for each variant
        """
        # Check if enough samples collected
        min_samples_collected = all(
            v.metrics.request_count >= self.experiment.min_sample_size
            for v in self.experiment.variants
        )

        if not min_samples_collected:
            return {
                "has_winner": False,
                "reason": "Insufficient samples",
                "variant_stats": self._get_variant_stats(),
            }

        # Perform statistical test based on metric
        if self.experiment.metric == "success_rate":
            result = self._test_success_rates()
        elif self.experiment.metric == "latency":
            result = self._test_latencies()
        elif self.experiment.metric == "feedback":
            result = self._test_feedback_scores()
        else:
            raise ValueError(f"Unknown metric: {self.experiment.metric}")

        # Update experiment if winner found
        if result["has_winner"]:
            self.experiment.winner = result["winner"]
            logger.info(
                f"Winner determined: {result['winner']} "
                f"(p={result['p_value']:.4f})"
            )

        result["variant_stats"] = self._get_variant_stats()
        return result

    def _test_success_rates(self) -> Dict[str, Any]:
        """Test success rates using chi-square or proportion z-test."""
        if len(self.experiment.variants) != 2:
            return {"has_winner": False, "reason": "Only supports 2 variants"}

        v1, v2 = self.experiment.variants[:2]

        # Two-proportion z-test
        n1 = v1.metrics.request_count
        n2 = v2.metrics.request_count
        p1 = v1.metrics.success_rate
        p2 = v2.metrics.success_rate

        # Pooled proportion
        p_pool = (v1.metrics.success_count + v2.metrics.success_count) / (n1 + n2)

        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

        if se == 0:
            return {"has_winner": False, "reason": "Zero variance"}

        # Z-statistic
        z = (p1 - p2) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        has_winner = p_value < (1 - self.experiment.confidence_level)
        winner = v1.name if p1 > p2 else v2.name

        return {
            "has_winner": has_winner,
            "winner": winner if has_winner else None,
            "p_value": p_value,
            "confidence": 1 - p_value,
            "test": "two_proportion_z_test",
        }

    def _test_latencies(self) -> Dict[str, Any]:
        """Test latencies using Mann-Whitney U test (non-parametric)."""
        if len(self.experiment.variants) != 2:
            return {"has_winner": False, "reason": "Only supports 2 variants"}

        v1, v2 = self.experiment.variants[:2]

        if not v1.metrics.latencies or not v2.metrics.latencies:
            return {"has_winner": False, "reason": "No latency data"}

        # Mann-Whitney U test (non-parametric, doesn't assume normal distribution)
        statistic, p_value = stats.mannwhitneyu(
            v1.metrics.latencies,
            v2.metrics.latencies,
            alternative='two-sided'
        )

        has_winner = p_value < (1 - self.experiment.confidence_level)
        # Lower latency wins
        winner = v1.name if v1.metrics.average_latency < v2.metrics.average_latency else v2.name

        return {
            "has_winner": has_winner,
            "winner": winner if has_winner else None,
            "p_value": p_value,
            "confidence": 1 - p_value,
            "test": "mann_whitney_u",
        }

    def _test_feedback_scores(self) -> Dict[str, Any]:
        """Test user feedback scores using t-test."""
        if len(self.experiment.variants) != 2:
            return {"has_winner": False, "reason": "Only supports 2 variants"}

        v1, v2 = self.experiment.variants[:2]

        if not v1.metrics.user_feedback_scores or not v2.metrics.user_feedback_scores:
            return {"has_winner": False, "reason": "No feedback data"}

        # Independent samples t-test
        statistic, p_value = stats.ttest_ind(
            v1.metrics.user_feedback_scores,
            v2.metrics.user_feedback_scores
        )

        has_winner = p_value < (1 - self.experiment.confidence_level)
        winner = v1.name if v1.metrics.average_feedback > v2.metrics.average_feedback else v2.name

        return {
            "has_winner": has_winner,
            "winner": winner if has_winner else None,
            "p_value": p_value,
            "confidence": 1 - p_value,
            "test": "independent_t_test",
        }

    def _get_variant_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all variants."""
        stats_list = []
        for variant in self.experiment.variants:
            stats_list.append({
                "name": variant.name,
                "request_count": variant.metrics.request_count,
                "success_rate": variant.metrics.success_rate,
                "average_latency": variant.metrics.average_latency,
                "average_feedback": variant.metrics.average_feedback,
                "error_count": variant.metrics.error_count,
            })
        return stats_list

    async def get_status(self) -> Dict[str, Any]:
        """
        Get current experiment status.

        Returns:
            Dictionary with experiment status and statistics
        """
        duration = 0.0
        if self.experiment.start_time:
            end = self.experiment.end_time or time.time()
            duration = end - self.experiment.start_time

        return {
            "name": self.experiment.name,
            "status": self.experiment.status.value,
            "metric": self.experiment.metric,
            "duration": duration,
            "winner": self.experiment.winner,
            "variants": self._get_variant_stats(),
        }
