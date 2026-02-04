"""
Canary Deployment Module

Progressive traffic shifting with automatic rollback based on error rates and
performance metrics. Enables safe deployment of new model versions.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, Callable

logger = logging.getLogger(__name__)


class DeploymentState(Enum):
    """Canary deployment states."""
    PENDING = "pending"
    STAGE_10 = "stage_10"
    STAGE_25 = "stage_25"
    STAGE_50 = "stage_50"
    STAGE_75 = "stage_75"
    STAGE_100 = "stage_100"
    COMPLETED = "completed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class DeploymentMetrics:
    """Metrics for a deployment version."""
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency: float = 0.0
    latencies: list = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.request_count == 0:
            return 1.0
        return self.success_count / self.request_count

    @property
    def average_latency(self) -> float:
        """Calculate average latency."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    @property
    def p99_latency(self) -> float:
        """Calculate 99th percentile latency."""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        index = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[index]


@dataclass
class CanaryConfig:
    """
    Configuration for canary deployment.

    Attributes:
        baseline_endpoint: Stable/production endpoint
        canary_endpoint: New/test endpoint
        error_threshold: Maximum acceptable error rate (0-1)
        latency_threshold: Maximum acceptable latency increase (multiplier)
        stage_duration: Seconds to wait at each stage before promoting
        min_requests_per_stage: Minimum requests before stage evaluation
        auto_promote: Automatically promote if metrics are good
        auto_rollback: Automatically rollback if metrics are bad
    """

    baseline_endpoint: str
    canary_endpoint: str
    error_threshold: float = 0.05  # 5% error rate
    latency_threshold: float = 1.5  # 1.5x baseline latency
    stage_duration: float = 300.0  # 5 minutes per stage
    min_requests_per_stage: int = 100
    auto_promote: bool = True
    auto_rollback: bool = True
    notification_callback: Optional[Callable] = None

    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.error_threshold <= 1:
            raise ValueError("error_threshold must be between 0 and 1")
        if self.latency_threshold < 1:
            raise ValueError("latency_threshold must be >= 1")
        if self.stage_duration < 0:
            raise ValueError("stage_duration must be non-negative")


class CanaryDeployment:
    """
    Canary deployment manager with progressive traffic shifting.

    This class manages the gradual rollout of a new model version (canary)
    while monitoring error rates and performance. It automatically promotes
    or rolls back based on configured thresholds.

    Deployment stages:
        1. 10% traffic to canary
        2. 25% traffic to canary
        3. 50% traffic to canary
        4. 75% traffic to canary
        5. 100% traffic to canary (complete)

    Example:
        ```python
        config = CanaryConfig(
            baseline_endpoint="http://model-v1:8000",
            canary_endpoint="http://model-v2:8000",
            error_threshold=0.05,
            auto_promote=True
        )

        deployment = CanaryDeployment(config)
        await deployment.start()

        # Route requests
        endpoint = deployment.select_endpoint()
        # ... make request ...
        deployment.record_result(endpoint, success=True, latency=0.5)

        # Check status
        status = await deployment.get_status()
        print(f"State: {status['state']}, Traffic: {status['canary_traffic']}")

        await deployment.stop()
        ```
    """

    STAGE_TRAFFIC = {
        DeploymentState.PENDING: 0.0,
        DeploymentState.STAGE_10: 0.10,
        DeploymentState.STAGE_25: 0.25,
        DeploymentState.STAGE_50: 0.50,
        DeploymentState.STAGE_75: 0.75,
        DeploymentState.STAGE_100: 1.0,
        DeploymentState.COMPLETED: 1.0,
    }

    STAGE_PROGRESSION = [
        DeploymentState.PENDING,
        DeploymentState.STAGE_10,
        DeploymentState.STAGE_25,
        DeploymentState.STAGE_50,
        DeploymentState.STAGE_75,
        DeploymentState.STAGE_100,
        DeploymentState.COMPLETED,
    ]

    def __init__(self, config: CanaryConfig):
        """
        Initialize canary deployment.

        Args:
            config: Canary deployment configuration
        """
        self.config = config
        self.state = DeploymentState.PENDING

        self.baseline_metrics = DeploymentMetrics()
        self.canary_metrics = DeploymentMetrics()

        self._start_time: Optional[float] = None
        self._stage_start_time: Optional[float] = None
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        logger.info(
            f"CanaryDeployment initialized: "
            f"{config.baseline_endpoint} -> {config.canary_endpoint}"
        )

    async def start(self) -> None:
        """Start the canary deployment."""
        if self._running:
            logger.warning("Deployment already running")
            return

        self._running = True
        self._start_time = time.time()
        self._stage_start_time = time.time()
        self.state = DeploymentState.STAGE_10

        # Start monitoring task
        if self.config.auto_promote or self.config.auto_rollback:
            self._monitor_task = asyncio.create_task(self._monitor_loop())

        logger.info("Canary deployment started at 10% traffic")

        if self.config.notification_callback:
            await self.config.notification_callback(
                f"Canary deployment started: {self.state.value}"
            )

    async def stop(self) -> None:
        """Stop the canary deployment."""
        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Canary deployment stopped")

    def select_endpoint(self) -> str:
        """
        Select endpoint based on current traffic split.

        Returns:
            Selected endpoint URL
        """
        if self.state in [DeploymentState.ROLLED_BACK, DeploymentState.FAILED]:
            return self.config.baseline_endpoint

        if self.state == DeploymentState.COMPLETED:
            return self.config.canary_endpoint

        import random
        canary_traffic = self.STAGE_TRAFFIC.get(self.state, 0.0)

        if random.random() < canary_traffic:
            return self.config.canary_endpoint
        else:
            return self.config.baseline_endpoint

    def record_result(
        self,
        endpoint: str,
        success: bool,
        latency: float
    ) -> None:
        """
        Record a request result.

        Args:
            endpoint: Endpoint that handled the request
            success: Whether request was successful
            latency: Request latency in seconds
        """
        if endpoint == self.config.baseline_endpoint:
            metrics = self.baseline_metrics
        elif endpoint == self.config.canary_endpoint:
            metrics = self.canary_metrics
        else:
            logger.warning(f"Unknown endpoint: {endpoint}")
            return

        metrics.request_count += 1
        if success:
            metrics.success_count += 1
        else:
            metrics.error_count += 1

        metrics.total_latency += latency
        metrics.latencies.append(latency)

    async def promote(self) -> bool:
        """
        Promote to next stage.

        Returns:
            True if promoted, False if already at final stage
        """
        async with self._lock:
            current_index = self.STAGE_PROGRESSION.index(self.state)

            if current_index >= len(self.STAGE_PROGRESSION) - 1:
                logger.info("Already at final stage")
                return False

            next_state = self.STAGE_PROGRESSION[current_index + 1]
            self.state = next_state
            self._stage_start_time = time.time()

            # Reset canary metrics for new stage
            self.canary_metrics = DeploymentMetrics()

            logger.info(f"Promoted to {next_state.value}")

            if self.config.notification_callback:
                await self.config.notification_callback(
                    f"Canary promoted to {next_state.value}"
                )

            return True

    async def rollback(self, reason: str = "Manual rollback") -> None:
        """
        Rollback the canary deployment.

        Args:
            reason: Reason for rollback
        """
        async with self._lock:
            self.state = DeploymentState.ROLLING_BACK
            logger.warning(f"Rolling back deployment: {reason}")

            # Simulate rollback process
            await asyncio.sleep(1)

            self.state = DeploymentState.ROLLED_BACK
            logger.info("Rollback completed")

            if self.config.notification_callback:
                await self.config.notification_callback(
                    f"Canary rolled back: {reason}"
                )

    async def _monitor_loop(self) -> None:
        """Background monitoring task."""
        logger.info("Starting canary monitoring loop")

        while self._running:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                # Evaluate current stage
                should_rollback, rollback_reason = self._should_rollback()
                if should_rollback and self.config.auto_rollback:
                    await self.rollback(rollback_reason)
                    break

                should_promote, promote_reason = self._should_promote()
                if should_promote and self.config.auto_promote:
                    promoted = await self.promote()
                    if not promoted:
                        # Reached final stage
                        self.state = DeploymentState.COMPLETED
                        logger.info("Canary deployment completed successfully")
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}", exc_info=True)

        logger.info("Canary monitoring loop stopped")

    def _should_rollback(self) -> tuple[bool, str]:
        """
        Check if deployment should be rolled back.

        Returns:
            (should_rollback, reason)
        """
        if self.canary_metrics.request_count < 10:
            return False, ""

        # Check error rate
        if self.canary_metrics.error_rate > self.config.error_threshold:
            return True, (
                f"Error rate {self.canary_metrics.error_rate:.2%} exceeds "
                f"threshold {self.config.error_threshold:.2%}"
            )

        # Check latency increase
        if (self.baseline_metrics.average_latency > 0 and
            self.canary_metrics.average_latency > 0):

            latency_ratio = (
                self.canary_metrics.average_latency /
                self.baseline_metrics.average_latency
            )

            if latency_ratio > self.config.latency_threshold:
                return True, (
                    f"Latency increased by {latency_ratio:.2f}x, exceeds "
                    f"threshold {self.config.latency_threshold:.2f}x"
                )

        return False, ""

    def _should_promote(self) -> tuple[bool, str]:
        """
        Check if deployment should be promoted.

        Returns:
            (should_promote, reason)
        """
        # Check minimum requirements
        if self.canary_metrics.request_count < self.config.min_requests_per_stage:
            return False, ""

        stage_duration = time.time() - self._stage_start_time
        if stage_duration < self.config.stage_duration:
            return False, ""

        # Check metrics are acceptable
        if self.canary_metrics.error_rate > self.config.error_threshold:
            return False, ""

        if (self.baseline_metrics.average_latency > 0 and
            self.canary_metrics.average_latency > 0):

            latency_ratio = (
                self.canary_metrics.average_latency /
                self.baseline_metrics.average_latency
            )

            if latency_ratio > self.config.latency_threshold:
                return False, ""

        return True, (
            f"Stage requirements met: {self.canary_metrics.request_count} requests, "
            f"{stage_duration:.0f}s duration, "
            f"{self.canary_metrics.error_rate:.2%} error rate"
        )

    async def get_status(self) -> Dict[str, Any]:
        """
        Get current deployment status.

        Returns:
            Dictionary with deployment status and metrics
        """
        duration = 0.0
        if self._start_time:
            duration = time.time() - self._start_time

        stage_duration = 0.0
        if self._stage_start_time:
            stage_duration = time.time() - self._stage_start_time

        canary_traffic = self.STAGE_TRAFFIC.get(self.state, 0.0)

        return {
            "state": self.state.value,
            "canary_traffic": canary_traffic,
            "total_duration": duration,
            "stage_duration": stage_duration,
            "baseline": {
                "endpoint": self.config.baseline_endpoint,
                "requests": self.baseline_metrics.request_count,
                "error_rate": self.baseline_metrics.error_rate,
                "average_latency": self.baseline_metrics.average_latency,
            },
            "canary": {
                "endpoint": self.config.canary_endpoint,
                "requests": self.canary_metrics.request_count,
                "error_rate": self.canary_metrics.error_rate,
                "average_latency": self.canary_metrics.average_latency,
            },
            "thresholds": {
                "error_rate": self.config.error_threshold,
                "latency_multiplier": self.config.latency_threshold,
            },
        }
