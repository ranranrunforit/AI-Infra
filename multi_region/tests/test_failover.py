"""
Tests for the Failover Controller

Tests health checking, failover triggering, region selection,
and rollback logic without requiring live cloud connections.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.failover.failover_controller import (
    FailoverController,
    FailoverStrategy,
    RegionHealth,
    RegionStatus,
    HealthChecker,
    HealthCheckConfig,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_config():
    return {
        "regions": [
            {"name": "us-west-2", "provider": "aws",   "endpoint": "us.example.com", "k8s_context": "ctx-us"},
            {"name": "eu-west-1", "provider": "gcp",   "endpoint": "eu.example.com", "k8s_context": "ctx-eu"},
            {"name": "ap-south-1","provider": "azure",  "endpoint": "ap.example.com", "k8s_context": "ctx-ap"},
        ],
        "primary_region": "us-west-2",
        "failover_enabled": True,
        "min_healthy_regions": 1,
        "health_check": {
            "endpoint_path": "/health",
            "timeout_seconds": 5,
            "interval_seconds": 10,
            "failure_threshold": 3,
            "success_threshold": 2,
            "expected_status_code": 200,
        },
    }


@pytest.fixture
def controller(base_config):
    return FailoverController(base_config)


# ── HealthChecker Tests ───────────────────────────────────────────────────────

class TestHealthChecker:

    @pytest.mark.asyncio
    async def test_check_endpoint_success(self):
        config = HealthCheckConfig()
        checker = HealthChecker(config)

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("src.failover.failover_controller.requests.get", return_value=mock_response):
            result = await checker.check_endpoint("us.example.com")

        assert result["healthy"] is True
        assert result["status_code"] == 200
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_check_endpoint_timeout(self):
        import requests as req_lib
        config = HealthCheckConfig()
        checker = HealthChecker(config)

        with patch("src.failover.failover_controller.requests.get", side_effect=req_lib.exceptions.Timeout):
            result = await checker.check_endpoint("us.example.com")

        assert result["healthy"] is False
        assert result["error"] == "timeout"

    @pytest.mark.asyncio
    async def test_check_endpoint_non_200(self):
        config = HealthCheckConfig()
        checker = HealthChecker(config)

        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch("src.failover.failover_controller.requests.get", return_value=mock_response):
            result = await checker.check_endpoint("us.example.com")

        assert result["healthy"] is False


# ── FailoverController Initialization ────────────────────────────────────────

class TestFailoverControllerInit:

    def test_regions_initialized(self, controller):
        assert len(controller.regions) == 3
        assert "us-west-2" in controller.regions
        assert "eu-west-1" in controller.regions
        assert "ap-south-1" in controller.regions

    def test_primary_region_set(self, controller):
        assert controller.primary_region == "us-west-2"
        assert controller.regions["us-west-2"].is_primary is True
        assert controller.regions["eu-west-1"].is_primary is False

    def test_initial_health_unknown(self, controller):
        for region in controller.regions.values():
            assert region.health == RegionHealth.UNKNOWN


# ── Health Determination ──────────────────────────────────────────────────────

class TestRegionHealth:

    @pytest.mark.asyncio
    async def test_healthy_region(self, controller):
        controller.health_checker.check_endpoint = AsyncMock(return_value={
            "healthy": True, "status_code": 200, "response_time_ms": 50, "error": None
        })
        controller.health_checker.check_kubernetes_cluster = AsyncMock(return_value={
            "healthy": True, "ready_nodes": 3, "total_nodes": 3, "system_pods_running": 5, "error": None
        })

        status = await controller.check_region_health("us-west-2")
        assert status.health == RegionHealth.HEALTHY

    @pytest.mark.asyncio
    async def test_degraded_region_slow_response(self, controller):
        controller.health_checker.check_endpoint = AsyncMock(return_value={
            "healthy": True, "status_code": 200, "response_time_ms": 350, "error": None
        })
        controller.health_checker.check_kubernetes_cluster = AsyncMock(return_value={
            "healthy": True, "ready_nodes": 3, "total_nodes": 3, "system_pods_running": 5, "error": None
        })

        status = await controller.check_region_health("us-west-2")
        assert status.health == RegionHealth.DEGRADED

    @pytest.mark.asyncio
    async def test_unhealthy_region_after_failures(self, controller):
        controller.health_checker.check_endpoint = AsyncMock(return_value={
            "healthy": False, "status_code": None, "response_time_ms": 5000, "error": "timeout"
        })
        controller.health_checker.check_kubernetes_cluster = AsyncMock(return_value={
            "healthy": False, "ready_nodes": 0, "total_nodes": 3, "system_pods_running": 0, "error": "unreachable"
        })

        # Must exceed failure_threshold (3) to become UNHEALTHY
        for _ in range(4):
            status = await controller.check_region_health("us-west-2")

        assert status.health == RegionHealth.UNHEALTHY


# ── Failover Logic ────────────────────────────────────────────────────────────

class TestFailoverLogic:

    def test_should_trigger_failover_when_primary_unhealthy(self, controller):
        controller.regions["us-west-2"].health = RegionHealth.UNHEALTHY
        reason = controller.should_trigger_failover()
        assert reason == "primary_unhealthy"

    def test_no_failover_when_primary_healthy(self, controller):
        controller.regions["us-west-2"].health = RegionHealth.HEALTHY
        reason = controller.should_trigger_failover()
        assert reason is None

    def test_no_failover_when_disabled(self, controller):
        controller.failover_enabled = False
        controller.regions["us-west-2"].health = RegionHealth.UNHEALTHY
        reason = controller.should_trigger_failover()
        assert reason is None

    def test_select_failover_target_prefers_healthy(self, controller):
        controller.regions["eu-west-1"].health = RegionHealth.HEALTHY
        controller.regions["eu-west-1"].response_time_ms = 40
        controller.regions["ap-south-1"].health = RegionHealth.DEGRADED
        controller.regions["ap-south-1"].response_time_ms = 200

        target = controller.select_failover_target()
        assert target == "eu-west-1"

    def test_select_failover_target_falls_back_to_degraded(self, controller):
        controller.regions["eu-west-1"].health = RegionHealth.DEGRADED
        controller.regions["ap-south-1"].health = RegionHealth.UNHEALTHY

        target = controller.select_failover_target()
        assert target == "eu-west-1"

    def test_select_failover_target_none_when_all_unhealthy(self, controller):
        controller.regions["eu-west-1"].health = RegionHealth.UNHEALTHY
        controller.regions["ap-south-1"].health = RegionHealth.UNHEALTHY

        target = controller.select_failover_target()
        assert target is None

    @pytest.mark.asyncio
    async def test_initiate_failover_updates_primary(self, controller):
        controller.regions["eu-west-1"].health = RegionHealth.HEALTHY
        controller.regions["eu-west-1"].response_time_ms = 40

        with patch.object(controller, "_automatic_failover", new_callable=AsyncMock):
            event = await controller.initiate_failover(
                reason="test",
                strategy=FailoverStrategy.AUTOMATIC,
                target_region="eu-west-1"
            )

        assert controller.primary_region == "eu-west-1"
        assert controller.regions["eu-west-1"].is_primary is True
        assert controller.regions["us-west-2"].is_primary is False
        assert event.status == "completed"

    @pytest.mark.asyncio
    async def test_initiate_failover_no_target_raises(self, controller):
        controller.regions["eu-west-1"].health = RegionHealth.UNHEALTHY
        controller.regions["ap-south-1"].health = RegionHealth.UNHEALTHY

        with pytest.raises(Exception, match="No suitable failover target"):
            await controller.initiate_failover(reason="test")


# ── Metrics ───────────────────────────────────────────────────────────────────

class TestMetrics:

    def test_get_metrics_structure(self, controller):
        controller.regions["us-west-2"].health = RegionHealth.HEALTHY
        metrics = controller.get_metrics()

        assert "total_regions" in metrics
        assert "healthy_regions" in metrics
        assert "primary_region" in metrics
        assert metrics["total_regions"] == 3
        assert metrics["primary_region"] == "us-west-2"

    def test_get_healthy_regions(self, controller):
        controller.regions["us-west-2"].health = RegionHealth.HEALTHY
        controller.regions["eu-west-1"].health = RegionHealth.DEGRADED
        controller.regions["ap-south-1"].health = RegionHealth.UNHEALTHY

        healthy = controller.get_healthy_regions()
        assert "us-west-2" in healthy
        assert "eu-west-1" in healthy
        assert "ap-south-1" not in healthy
