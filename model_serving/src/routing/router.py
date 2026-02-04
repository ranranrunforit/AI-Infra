"""
Intelligent Router Module

Provides multiple routing strategies for distributing inference requests across
backend model endpoints with health checking and performance tracking.
"""

import asyncio
import hashlib
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
import httpx

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Available routing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    LEAST_LATENCY = "least_latency"
    HASH_BASED = "hash_based"
    RANDOM = "random"


@dataclass
class EndpointStats:
    """Performance statistics for an endpoint."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    average_latency: float = 0.0
    last_request_time: Optional[float] = None
    consecutive_failures: int = 0


@dataclass
class ModelEndpoint:
    """
    Represents a backend model serving endpoint.

    Attributes:
        url: Base URL of the endpoint
        weight: Routing weight (higher = more traffic)
        healthy: Current health status
        max_consecutive_failures: Failures before marking unhealthy
        health_check_interval: Seconds between health checks
        timeout: Request timeout in seconds
    """

    url: str
    weight: int = 1
    healthy: bool = True
    max_consecutive_failures: int = 3
    health_check_interval: float = 30.0
    timeout: float = 60.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    stats: EndpointStats = field(default_factory=EndpointStats)
    _last_health_check: float = field(default=0.0, init=False)

    def __post_init__(self):
        """Validate endpoint configuration."""
        if self.weight < 0:
            raise ValueError("Weight must be non-negative")
        if not self.url:
            raise ValueError("URL cannot be empty")

        logger.info(f"Created endpoint: {self.url} (weight={self.weight})")

    async def health_check(self) -> bool:
        """
        Perform health check on endpoint.

        Returns:
            True if endpoint is healthy, False otherwise
        """
        now = time.time()

        # Skip if checked recently
        if now - self._last_health_check < self.health_check_interval:
            return self.healthy

        self._last_health_check = now

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.url}/health")
                self.healthy = response.status_code == 200

                if self.healthy:
                    self.stats.consecutive_failures = 0
                    logger.debug(f"Health check passed: {self.url}")
                else:
                    logger.warning(
                        f"Health check failed: {self.url} (status={response.status_code})"
                    )

        except Exception as e:
            self.healthy = False
            logger.warning(f"Health check error for {self.url}: {e}")

        return self.healthy

    def record_success(self, latency: float) -> None:
        """Record a successful request."""
        self.stats.total_requests += 1
        self.stats.successful_requests += 1
        self.stats.total_latency += latency
        self.stats.average_latency = (
            self.stats.total_latency / self.stats.successful_requests
        )
        self.stats.last_request_time = time.time()
        self.stats.consecutive_failures = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self.stats.total_requests += 1
        self.stats.failed_requests += 1
        self.stats.consecutive_failures += 1

        # Mark unhealthy if too many consecutive failures
        if self.stats.consecutive_failures >= self.max_consecutive_failures:
            self.healthy = False
            logger.error(
                f"Endpoint {self.url} marked unhealthy after "
                f"{self.stats.consecutive_failures} consecutive failures"
            )

    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.stats.total_requests == 0:
            return 1.0
        return self.stats.successful_requests / self.stats.total_requests


class IntelligentRouter:
    """
    Intelligent request router with multiple routing strategies.

    This class routes incoming inference requests to backend model endpoints
    using configurable strategies like round-robin, weighted, least-latency,
    and hash-based routing. Includes automatic health checking and failover.

    Attributes:
        endpoints: List of backend model endpoints
        strategy: Routing strategy to use
        health_check_enabled: Enable periodic health checks

    Example:
        ```python
        endpoints = [
            ModelEndpoint(url="http://gpu1:8000", weight=2),
            ModelEndpoint(url="http://gpu2:8000", weight=1),
        ]

        router = IntelligentRouter(
            endpoints=endpoints,
            strategy=RoutingStrategy.WEIGHTED
        )

        await router.start()

        # Route request
        endpoint = await router.route(user_id="user123")
        print(f"Selected: {endpoint.url}")

        await router.stop()
        ```
    """

    def __init__(
        self,
        endpoints: List[ModelEndpoint],
        strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN,
        health_check_enabled: bool = True,
        health_check_interval: float = 30.0,
    ):
        """
        Initialize intelligent router.

        Args:
            endpoints: List of backend endpoints
            strategy: Routing strategy to use
            health_check_enabled: Enable health checking
            health_check_interval: Seconds between health checks
        """
        if not endpoints:
            raise ValueError("At least one endpoint required")

        self.endpoints = endpoints
        self.strategy = strategy
        self.health_check_enabled = health_check_enabled
        self.health_check_interval = health_check_interval

        self._round_robin_index = 0
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            f"IntelligentRouter initialized with {len(endpoints)} endpoints "
            f"using {strategy.value} strategy"
        )

    async def start(self) -> None:
        """Start the router and background health checking."""
        if self._running:
            logger.warning("Router already running")
            return

        self._running = True

        if self.health_check_enabled:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("Health check loop started")

        logger.info("Router started")

    async def stop(self) -> None:
        """Stop the router and cancel background tasks."""
        self._running = False

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        logger.info("Router stopped")

    async def route(
        self,
        user_id: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
    ) -> ModelEndpoint:
        """
        Select an endpoint based on the configured strategy.

        Args:
            user_id: User identifier for hash-based routing
            request_data: Additional request data

        Returns:
            Selected ModelEndpoint

        Raises:
            RuntimeError: If no healthy endpoints available
        """
        healthy_endpoints = [ep for ep in self.endpoints if ep.healthy]

        if not healthy_endpoints:
            logger.error("No healthy endpoints available")
            raise RuntimeError("No healthy endpoints available")

        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            return await self._route_round_robin(healthy_endpoints)
        elif self.strategy == RoutingStrategy.WEIGHTED:
            return await self._route_weighted(healthy_endpoints)
        elif self.strategy == RoutingStrategy.LEAST_LATENCY:
            return await self._route_least_latency(healthy_endpoints)
        elif self.strategy == RoutingStrategy.HASH_BASED:
            return await self._route_hash_based(healthy_endpoints, user_id)
        elif self.strategy == RoutingStrategy.RANDOM:
            return random.choice(healthy_endpoints)
        else:
            raise ValueError(f"Unknown routing strategy: {self.strategy}")

    async def _route_round_robin(
        self,
        endpoints: List[ModelEndpoint]
    ) -> ModelEndpoint:
        """Round-robin routing strategy."""
        async with self._lock:
            endpoint = endpoints[self._round_robin_index % len(endpoints)]
            self._round_robin_index += 1
            return endpoint

    async def _route_weighted(
        self,
        endpoints: List[ModelEndpoint]
    ) -> ModelEndpoint:
        """Weighted random routing strategy."""
        total_weight = sum(ep.weight for ep in endpoints)
        if total_weight == 0:
            return random.choice(endpoints)

        rand = random.uniform(0, total_weight)
        cumulative = 0.0

        for endpoint in endpoints:
            cumulative += endpoint.weight
            if rand <= cumulative:
                return endpoint

        return endpoints[-1]  # Fallback

    async def _route_least_latency(
        self,
        endpoints: List[ModelEndpoint]
    ) -> ModelEndpoint:
        """Route to endpoint with lowest average latency."""
        # Prefer endpoints with successful requests
        endpoints_with_stats = [
            ep for ep in endpoints
            if ep.stats.successful_requests > 0
        ]

        if not endpoints_with_stats:
            # No statistics yet, choose randomly
            return random.choice(endpoints)

        # Select endpoint with minimum latency
        return min(endpoints_with_stats, key=lambda ep: ep.stats.average_latency)

    async def _route_hash_based(
        self,
        endpoints: List[ModelEndpoint],
        user_id: Optional[str]
    ) -> ModelEndpoint:
        """Consistent hash-based routing for session affinity."""
        if user_id is None:
            logger.warning("No user_id provided for hash-based routing, using random")
            return random.choice(endpoints)

        # Hash user_id to consistent endpoint
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        index = hash_value % len(endpoints)
        return endpoints[index]

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checking."""
        logger.info("Starting health check loop")

        while self._running:
            try:
                # Check all endpoints
                tasks = [ep.health_check() for ep in self.endpoints]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                healthy_count = sum(1 for r in results if r is True)
                logger.debug(
                    f"Health check complete: {healthy_count}/{len(self.endpoints)} "
                    "endpoints healthy"
                )

                # Wait before next check
                await asyncio.sleep(self.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}", exc_info=True)
                await asyncio.sleep(5)

        logger.info("Health check loop stopped")

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get routing statistics.

        Returns:
            Dictionary with router statistics
        """
        healthy_count = sum(1 for ep in self.endpoints if ep.healthy)

        endpoint_stats = []
        for ep in self.endpoints:
            endpoint_stats.append({
                "url": ep.url,
                "healthy": ep.healthy,
                "weight": ep.weight,
                "total_requests": ep.stats.total_requests,
                "successful_requests": ep.stats.successful_requests,
                "failed_requests": ep.stats.failed_requests,
                "average_latency": ep.stats.average_latency,
                "success_rate": ep.get_success_rate(),
            })

        return {
            "strategy": self.strategy.value,
            "total_endpoints": len(self.endpoints),
            "healthy_endpoints": healthy_count,
            "endpoints": endpoint_stats,
        }

    def add_endpoint(self, endpoint: ModelEndpoint) -> None:
        """Add a new endpoint to the router."""
        self.endpoints.append(endpoint)
        logger.info(f"Added endpoint: {endpoint.url}")

    def remove_endpoint(self, url: str) -> bool:
        """
        Remove an endpoint by URL.

        Args:
            url: URL of endpoint to remove

        Returns:
            True if removed, False if not found
        """
        for i, ep in enumerate(self.endpoints):
            if ep.url == url:
                self.endpoints.pop(i)
                logger.info(f"Removed endpoint: {url}")
                return True

        logger.warning(f"Endpoint not found: {url}")
        return False
