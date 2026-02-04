"""
Intelligent Router Tests

Comprehensive test suite for the intelligent routing module.
Tests various routing strategies, health checking, endpoint management,
and load balancing functionality.

Test Coverage:
- ModelEndpoint class and health checking
- IntelligentRouter initialization
- Round-robin routing
- Weighted routing
- Least-latency routing
- Hash-based routing
- Random routing
- Health check loop
- Endpoint statistics
- Error handling and failover
"""

import asyncio
import time
from unittest.mock import MagicMock, Mock, patch, AsyncMock

import pytest

from routing.router import (
    IntelligentRouter,
    ModelEndpoint,
    RoutingStrategy,
    EndpointStats,
)


class TestEndpointStats:
    """Test EndpointStats dataclass."""

    def test_default_stats(self):
        """Test default statistics values."""
        stats = EndpointStats()

        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0
        assert stats.total_latency == 0.0
        assert stats.average_latency == 0.0
        assert stats.consecutive_failures == 0


class TestModelEndpoint:
    """Test ModelEndpoint class."""

    def test_endpoint_creation(self):
        """Test creating a model endpoint."""
        endpoint = ModelEndpoint(url="http://localhost:8000", weight=2)

        assert endpoint.url == "http://localhost:8000"
        assert endpoint.weight == 2
        assert endpoint.healthy is True
        assert endpoint.max_consecutive_failures == 3

    def test_endpoint_validation_invalid_weight(self):
        """Test endpoint validation with negative weight."""
        with pytest.raises(ValueError, match="Weight must be non-negative"):
            ModelEndpoint(url="http://localhost:8000", weight=-1)

    def test_endpoint_validation_empty_url(self):
        """Test endpoint validation with empty URL."""
        with pytest.raises(ValueError, match="URL cannot be empty"):
            ModelEndpoint(url="", weight=1)

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        endpoint = ModelEndpoint(url="http://localhost:8000")

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await endpoint.health_check()

            assert result is True
            assert endpoint.healthy is True
            assert endpoint.stats.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed health check."""
        endpoint = ModelEndpoint(url="http://localhost:8000")

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 503
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await endpoint.health_check()

            assert result is False
            assert endpoint.healthy is False

    @pytest.mark.asyncio
    async def test_health_check_exception(self):
        """Test health check with network exception."""
        endpoint = ModelEndpoint(url="http://localhost:8000")

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=Exception("Connection error"))
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await endpoint.health_check()

            assert result is False
            assert endpoint.healthy is False

    @pytest.mark.asyncio
    async def test_health_check_caching(self):
        """Test health check interval caching."""
        endpoint = ModelEndpoint(url="http://localhost:8000", health_check_interval=10.0)

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # First check
            result1 = await endpoint.health_check()
            assert result1 is True

            # Immediate second check (should be cached)
            result2 = await endpoint.health_check()
            assert result2 is True

            # Only one actual health check should have been performed
            assert mock_client.get.call_count == 1

    def test_record_success(self):
        """Test recording successful request."""
        endpoint = ModelEndpoint(url="http://localhost:8000")

        endpoint.record_success(latency=0.05)

        assert endpoint.stats.total_requests == 1
        assert endpoint.stats.successful_requests == 1
        assert endpoint.stats.failed_requests == 0
        assert endpoint.stats.average_latency == 0.05
        assert endpoint.stats.consecutive_failures == 0

    def test_record_multiple_successes(self):
        """Test recording multiple successful requests."""
        endpoint = ModelEndpoint(url="http://localhost:8000")

        endpoint.record_success(latency=0.1)
        endpoint.record_success(latency=0.2)
        endpoint.record_success(latency=0.3)

        assert endpoint.stats.total_requests == 3
        assert endpoint.stats.successful_requests == 3
        assert endpoint.stats.average_latency == 0.2

    def test_record_failure(self):
        """Test recording failed request."""
        endpoint = ModelEndpoint(url="http://localhost:8000")

        endpoint.record_failure()

        assert endpoint.stats.total_requests == 1
        assert endpoint.stats.successful_requests == 0
        assert endpoint.stats.failed_requests == 1
        assert endpoint.stats.consecutive_failures == 1
        assert endpoint.healthy is True  # Still healthy after one failure

    def test_record_consecutive_failures_marking_unhealthy(self):
        """Test that consecutive failures mark endpoint as unhealthy."""
        endpoint = ModelEndpoint(url="http://localhost:8000", max_consecutive_failures=3)

        endpoint.record_failure()
        assert endpoint.healthy is True

        endpoint.record_failure()
        assert endpoint.healthy is True

        endpoint.record_failure()
        assert endpoint.healthy is False  # Should be marked unhealthy

    def test_get_success_rate(self):
        """Test success rate calculation."""
        endpoint = ModelEndpoint(url="http://localhost:8000")

        # No requests yet
        assert endpoint.get_success_rate() == 1.0

        # All successful
        endpoint.record_success(0.1)
        endpoint.record_success(0.1)
        assert endpoint.get_success_rate() == 1.0

        # Mixed
        endpoint.record_failure()
        assert endpoint.get_success_rate() == 2.0 / 3.0


class TestIntelligentRouter:
    """Test IntelligentRouter class."""

    def test_router_initialization(self, mock_endpoints):
        """Test router initialization."""
        router = IntelligentRouter(
            endpoints=mock_endpoints,
            strategy=RoutingStrategy.ROUND_ROBIN,
            health_check_enabled=False
        )

        assert len(router.endpoints) == 3
        assert router.strategy == RoutingStrategy.ROUND_ROBIN

    def test_router_initialization_no_endpoints(self):
        """Test router initialization with no endpoints."""
        with pytest.raises(ValueError, match="At least one endpoint required"):
            IntelligentRouter(endpoints=[], strategy=RoutingStrategy.ROUND_ROBIN)

    @pytest.mark.asyncio
    async def test_router_start_stop(self, mock_endpoints):
        """Test router start and stop."""
        router = IntelligentRouter(
            endpoints=mock_endpoints,
            strategy=RoutingStrategy.ROUND_ROBIN,
            health_check_enabled=False
        )

        await router.start()
        assert router._running is True

        await router.stop()
        assert router._running is False

    @pytest.mark.asyncio
    async def test_round_robin_routing(self, mock_endpoints):
        """Test round-robin routing strategy."""
        router = IntelligentRouter(
            endpoints=mock_endpoints,
            strategy=RoutingStrategy.ROUND_ROBIN,
            health_check_enabled=False
        )

        await router.start()

        # Route multiple requests
        endpoints_selected = []
        for _ in range(6):
            endpoint = await router.route()
            endpoints_selected.append(endpoint.url)

        await router.stop()

        # Should cycle through endpoints
        assert len(set(endpoints_selected)) == 3  # All 3 endpoints used
        # First two cycles should be identical
        assert endpoints_selected[0] == endpoints_selected[3]
        assert endpoints_selected[1] == endpoints_selected[4]
        assert endpoints_selected[2] == endpoints_selected[5]

    @pytest.mark.asyncio
    async def test_weighted_routing(self, mock_endpoints):
        """Test weighted routing strategy."""
        router = IntelligentRouter(
            endpoints=mock_endpoints,
            strategy=RoutingStrategy.WEIGHTED,
            health_check_enabled=False
        )

        await router.start()

        # Route many requests and check distribution
        selected_urls = []
        for _ in range(100):
            endpoint = await router.route()
            selected_urls.append(endpoint.url)

        await router.stop()

        # Count selections
        from collections import Counter
        counts = Counter(selected_urls)

        # gpu1 has weight=2, should get more requests
        # This is probabilistic, so we just check it got selected
        assert len(counts) > 0
        assert "http://gpu1:8000" in counts

    @pytest.mark.asyncio
    async def test_least_latency_routing(self, mock_endpoints):
        """Test least-latency routing strategy."""
        router = IntelligentRouter(
            endpoints=mock_endpoints,
            strategy=RoutingStrategy.LEAST_LATENCY,
            health_check_enabled=False
        )

        await router.start()

        # Record some latencies
        mock_endpoints[0].record_success(0.5)  # High latency
        mock_endpoints[1].record_success(0.1)  # Low latency
        mock_endpoints[2].record_success(0.3)  # Medium latency

        # Route request
        endpoint = await router.route()

        await router.stop()

        # Should select endpoint with lowest latency
        assert endpoint.url == "http://gpu2:8000"  # gpu2 has lowest latency

    @pytest.mark.asyncio
    async def test_hash_based_routing(self, mock_endpoints):
        """Test hash-based routing for session affinity."""
        router = IntelligentRouter(
            endpoints=mock_endpoints,
            strategy=RoutingStrategy.HASH_BASED,
            health_check_enabled=False
        )

        await router.start()

        # Same user should always route to same endpoint
        user_id = "user123"
        endpoints_selected = []
        for _ in range(10):
            endpoint = await router.route(user_id=user_id)
            endpoints_selected.append(endpoint.url)

        await router.stop()

        # All requests for same user should go to same endpoint
        assert len(set(endpoints_selected)) == 1

    @pytest.mark.asyncio
    async def test_hash_based_routing_different_users(self, mock_endpoints):
        """Test hash-based routing distributes different users."""
        router = IntelligentRouter(
            endpoints=mock_endpoints,
            strategy=RoutingStrategy.HASH_BASED,
            health_check_enabled=False
        )

        await router.start()

        # Different users
        users = [f"user{i}" for i in range(30)]
        endpoint_urls = set()

        for user_id in users:
            endpoint = await router.route(user_id=user_id)
            endpoint_urls.add(endpoint.url)

        await router.stop()

        # Should use multiple endpoints
        assert len(endpoint_urls) > 1

    @pytest.mark.asyncio
    async def test_random_routing(self, mock_endpoints):
        """Test random routing strategy."""
        router = IntelligentRouter(
            endpoints=mock_endpoints,
            strategy=RoutingStrategy.RANDOM,
            health_check_enabled=False
        )

        await router.start()

        # Route multiple requests
        endpoints_selected = set()
        for _ in range(30):
            endpoint = await router.route()
            endpoints_selected.add(endpoint.url)

        await router.stop()

        # Should use multiple endpoints
        assert len(endpoints_selected) > 1

    @pytest.mark.asyncio
    async def test_route_with_unhealthy_endpoints(self, mock_endpoints):
        """Test routing skips unhealthy endpoints."""
        router = IntelligentRouter(
            endpoints=mock_endpoints,
            strategy=RoutingStrategy.ROUND_ROBIN,
            health_check_enabled=False
        )

        await router.start()

        # Mark first endpoint as unhealthy
        mock_endpoints[0].healthy = False

        # Route multiple requests
        endpoints_selected = []
        for _ in range(10):
            endpoint = await router.route()
            endpoints_selected.append(endpoint.url)

        await router.stop()

        # Should never select unhealthy endpoint
        assert "http://gpu1:8000" not in endpoints_selected
        assert all(url in ["http://gpu2:8000", "http://gpu3:8000"] for url in endpoints_selected)

    @pytest.mark.asyncio
    async def test_route_no_healthy_endpoints(self, mock_endpoints):
        """Test error when no healthy endpoints available."""
        router = IntelligentRouter(
            endpoints=mock_endpoints,
            strategy=RoutingStrategy.ROUND_ROBIN,
            health_check_enabled=False
        )

        await router.start()

        # Mark all endpoints as unhealthy
        for endpoint in mock_endpoints:
            endpoint.healthy = False

        # Should raise error
        with pytest.raises(RuntimeError, match="No healthy endpoints available"):
            await router.route()

        await router.stop()

    @pytest.mark.asyncio
    async def test_health_check_loop(self, mock_endpoints):
        """Test health check background loop."""
        router = IntelligentRouter(
            endpoints=mock_endpoints,
            strategy=RoutingStrategy.ROUND_ROBIN,
            health_check_enabled=True,
            health_check_interval=0.1  # Fast interval for testing
        )

        # Mock health checks
        for endpoint in mock_endpoints:
            endpoint.health_check = AsyncMock(return_value=True)

        await router.start()

        # Wait for health checks to run
        await asyncio.sleep(0.3)

        await router.stop()

        # Verify health checks were called
        for endpoint in mock_endpoints:
            assert endpoint.health_check.call_count > 0

    @pytest.mark.asyncio
    async def test_get_stats(self, mock_endpoints):
        """Test getting router statistics."""
        router = IntelligentRouter(
            endpoints=mock_endpoints,
            strategy=RoutingStrategy.ROUND_ROBIN,
            health_check_enabled=False
        )

        await router.start()

        # Record some statistics
        mock_endpoints[0].record_success(0.1)
        mock_endpoints[1].record_failure()

        stats = await router.get_stats()

        await router.stop()

        assert stats["strategy"] == "round_robin"
        assert stats["total_endpoints"] == 3
        assert "endpoints" in stats
        assert len(stats["endpoints"]) == 3

    def test_add_endpoint(self, mock_endpoints):
        """Test adding endpoint to router."""
        router = IntelligentRouter(
            endpoints=mock_endpoints,
            strategy=RoutingStrategy.ROUND_ROBIN,
            health_check_enabled=False
        )

        new_endpoint = ModelEndpoint(url="http://gpu4:8000", weight=1)
        router.add_endpoint(new_endpoint)

        assert len(router.endpoints) == 4
        assert router.endpoints[-1].url == "http://gpu4:8000"

    def test_remove_endpoint(self, mock_endpoints):
        """Test removing endpoint from router."""
        router = IntelligentRouter(
            endpoints=mock_endpoints,
            strategy=RoutingStrategy.ROUND_ROBIN,
            health_check_enabled=False
        )

        result = router.remove_endpoint("http://gpu2:8000")

        assert result is True
        assert len(router.endpoints) == 2
        assert not any(ep.url == "http://gpu2:8000" for ep in router.endpoints)

    def test_remove_nonexistent_endpoint(self, mock_endpoints):
        """Test removing non-existent endpoint."""
        router = IntelligentRouter(
            endpoints=mock_endpoints,
            strategy=RoutingStrategy.ROUND_ROBIN,
            health_check_enabled=False
        )

        result = router.remove_endpoint("http://nonexistent:8000")

        assert result is False
        assert len(router.endpoints) == 3


class TestRoutingIntegration:
    """Integration tests for routing functionality."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_routing_with_failover(self):
        """Test routing with automatic failover."""
        endpoints = [
            ModelEndpoint(url="http://primary:8000", weight=2),
            ModelEndpoint(url="http://backup:8000", weight=1),
        ]

        router = IntelligentRouter(
            endpoints=endpoints,
            strategy=RoutingStrategy.WEIGHTED,
            health_check_enabled=False
        )

        await router.start()

        # Initially, both endpoints healthy
        selected = []
        for _ in range(10):
            endpoint = await router.route()
            selected.append(endpoint.url)

        # Should use both endpoints
        assert len(set(selected)) == 2

        # Mark primary as unhealthy
        endpoints[0].healthy = False

        # Now only backup should be used
        selected = []
        for _ in range(10):
            endpoint = await router.route()
            selected.append(endpoint.url)

        await router.stop()

        # Only backup endpoint should be used
        assert all(url == "http://backup:8000" for url in selected)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_routing_performance(self, mock_endpoints):
        """Test routing performance under load."""
        router = IntelligentRouter(
            endpoints=mock_endpoints,
            strategy=RoutingStrategy.ROUND_ROBIN,
            health_check_enabled=False
        )

        await router.start()

        # Route many requests and measure time
        start_time = time.time()
        for _ in range(1000):
            await router.route()
        elapsed = time.time() - start_time

        await router.stop()

        # Should complete quickly (< 1 second for 1000 routes)
        assert elapsed < 1.0
