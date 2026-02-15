"""
Failover Controller

Manages automatic failover between regions based on health checks and performance metrics.
Coordinates with DNS updater and recovery manager for seamless failover.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from enum import Enum

import boto3
from kubernetes import client, config as k8s_config
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegionHealth(Enum):
    """Health status of a region"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class FailoverStrategy(Enum):
    """Failover strategies"""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    GRACEFUL = "graceful"  # Drain connections first
    IMMEDIATE = "immediate"  # Immediate cutover


@dataclass
class RegionStatus:
    """Status information for a region"""
    region_name: str
    provider: str  # aws, gcp, azure
    health: RegionHealth
    endpoint: str
    last_check: str
    response_time_ms: float
    error_rate: float
    active_connections: int
    cpu_utilization: float
    memory_utilization: float
    is_primary: bool = False
    consecutive_failures: int = 0


@dataclass
class FailoverEvent:
    """Represents a failover event"""
    event_id: str
    timestamp: str
    source_region: str
    target_region: str
    reason: str
    strategy: FailoverStrategy
    status: str  # initiated, in_progress, completed, failed, rolled_back
    duration_seconds: Optional[float] = None
    affected_connections: int = 0
    error_message: Optional[str] = None
    automated: bool = True


@dataclass
class HealthCheckConfig:
    """Configuration for health checks"""
    endpoint_path: str = "/health"
    timeout_seconds: int = 5
    interval_seconds: int = 10
    failure_threshold: int = 3
    success_threshold: int = 2
    expected_status_code: int = 200


class HealthChecker:
    """Performs health checks on regional endpoints"""

    def __init__(self, config: HealthCheckConfig):
        self.config = config

    async def check_endpoint(self, endpoint: str) -> Dict:
        """Check health of an endpoint"""
        start_time = datetime.utcnow()

        try:
            url = f"https://{endpoint}{self.config.endpoint_path}"
            response = await asyncio.to_thread(
                requests.get,
                url,
                timeout=self.config.timeout_seconds,
                verify=True
            )

            duration = (datetime.utcnow() - start_time).total_seconds() * 1000

            return {
                'healthy': response.status_code == self.config.expected_status_code,
                'status_code': response.status_code,
                'response_time_ms': duration,
                'error': None
            }

        except requests.exceptions.Timeout:
            duration = self.config.timeout_seconds * 1000
            return {
                'healthy': False,
                'status_code': None,
                'response_time_ms': duration,
                'error': 'timeout'
            }
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds() * 1000
            return {
                'healthy': False,
                'status_code': None,
                'response_time_ms': duration,
                'error': str(e)
            }

    async def check_kubernetes_cluster(self, region_config: Dict) -> Dict:
        """Check health of Kubernetes cluster"""
        try:
            # Load kubeconfig for the region
            k8s_config.load_kube_config(context=region_config.get('k8s_context'))
            v1 = client.CoreV1Api()

            # Check node status
            nodes = await asyncio.to_thread(v1.list_node)
            ready_nodes = sum(
                1 for node in nodes.items
                if any(
                    condition.type == "Ready" and condition.status == "True"
                    for condition in node.status.conditions
                )
            )

            total_nodes = len(nodes.items)

            # Check system pods
            pods = await asyncio.to_thread(
                v1.list_pod_for_all_namespaces,
                label_selector="tier=control-plane"
            )
            healthy_pods = sum(
                1 for pod in pods.items
                if pod.status.phase == "Running"
            )

            cluster_healthy = (
                ready_nodes >= total_nodes * 0.8 and
                healthy_pods > 0
            )

            return {
                'healthy': cluster_healthy,
                'ready_nodes': ready_nodes,
                'total_nodes': total_nodes,
                'system_pods_running': healthy_pods,
                'error': None
            }

        except Exception as e:
            return {
                'healthy': False,
                'ready_nodes': 0,
                'total_nodes': 0,
                'system_pods_running': 0,
                'error': str(e)
            }


class FailoverController:
    """
    Multi-Region Failover Controller

    Monitors regional health and automatically triggers failover when needed.
    Coordinates with DNS updater for traffic management.
    """

    def __init__(self, config: Dict, alert_manager=None):
        self.config = config
        self.regions: Dict[str, RegionStatus] = {}
        self.failover_history: List[FailoverEvent] = []
        self.health_checker = HealthChecker(
            HealthCheckConfig(**config.get('health_check', {}))
        )
        self.primary_region: Optional[str] = config.get('primary_region')
        self.failover_enabled = config.get('failover_enabled', True)
        self.min_healthy_regions = config.get('min_healthy_regions', 1)
        self._running = False
        self._initialize_regions()
        
        self.alert_manager = alert_manager

        # Metrics
        from prometheus_client import Counter, Gauge
        self.registry = config.get('registry')
        
        if self.registry:
            self.failover_counter = Counter(
                'failover_events_total',
                'Total failover events',
                ['source_region', 'target_region', 'reason', 'status'],
                registry=self.registry
            )
            self.region_health_gauge = Gauge(
                'region_health_status',
                'Region health status (1=Healthy, 0=Unhealthy)',
                ['region'],
                registry=self.registry
            )

    def _initialize_regions(self):
        """Initialize region status tracking"""
        for region_config in self.config.get('regions', []):
            region_name = region_config['name']
            self.regions[region_name] = RegionStatus(
                region_name=region_name,
                provider=region_config['provider'],
                health=RegionHealth.UNKNOWN,
                endpoint=region_config['endpoint'],
                last_check=datetime.utcnow().isoformat(),
                response_time_ms=0.0,
                error_rate=0.0,
                active_connections=0,
                cpu_utilization=0.0,
                memory_utilization=0.0,
                is_primary=(region_name == self.primary_region)
            )
            logger.info(f"Initialized region tracking: {region_name}")

    async def check_region_health(self, region_name: str) -> RegionStatus:
        """Check health of a specific region"""
        status = self.regions[region_name]
        region_config = next(
            r for r in self.config['regions'] if r['name'] == region_name
        )

        # Perform endpoint health check
        endpoint_result = await self.health_checker.check_endpoint(status.endpoint)

        # Perform cluster health check
        cluster_result = await self.health_checker.check_kubernetes_cluster(region_config)

        # Update status
        status.response_time_ms = endpoint_result['response_time_ms']
        status.last_check = datetime.utcnow().isoformat()

        # Determine overall health
        endpoint_healthy = endpoint_result['healthy']
        cluster_healthy = cluster_result['healthy']

        if endpoint_healthy and cluster_healthy:
            status.consecutive_failures = 0
            if status.response_time_ms < 200:
                status.health = RegionHealth.HEALTHY
            elif status.response_time_ms < 500:
                status.health = RegionHealth.DEGRADED
            else:
                status.health = RegionHealth.DEGRADED
        else:
            status.consecutive_failures += 1
            if status.consecutive_failures >= self.health_checker.config.failure_threshold:
                status.health = RegionHealth.UNHEALTHY
            else:
                status.health = RegionHealth.DEGRADED

        logger.info(
            f"Region {region_name}: {status.health.value}, "
            f"response_time={status.response_time_ms:.2f}ms, "
            f"failures={status.consecutive_failures}"
        )

        if self.registry:
            is_healthy = 1 if status.health in [RegionHealth.HEALTHY, RegionHealth.DEGRADED] else 0
            self.region_health_gauge.labels(region=region_name).set(is_healthy)

        return status

    async def check_all_regions(self) -> Dict[str, RegionStatus]:
        """Check health of all regions concurrently"""
        tasks = [
            self.check_region_health(region_name)
            for region_name in self.regions.keys()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        return self.regions

    def get_healthy_regions(self) -> List[str]:
        """Get list of healthy regions"""
        return [
            name for name, status in self.regions.items()
            if status.health in [RegionHealth.HEALTHY, RegionHealth.DEGRADED]
        ]

    def should_trigger_failover(self) -> Optional[str]:
        """Determine if failover should be triggered"""
        if not self.failover_enabled:
            return None

        primary_status = self.regions.get(self.primary_region)

        if not primary_status:
            return None

        # Check if primary is unhealthy
        if primary_status.health == RegionHealth.UNHEALTHY:
            logger.warning(f"Primary region {self.primary_region} is unhealthy")
            return "primary_unhealthy"

        # Check if primary is degraded for too long
        if primary_status.health == RegionHealth.DEGRADED:
            if primary_status.consecutive_failures >= 5:
                logger.warning(f"Primary region {self.primary_region} degraded for too long")
                return "primary_degraded"

        # Check error rate
        if primary_status.error_rate > 0.1:  # 10% error rate
            logger.warning(f"Primary region {self.primary_region} has high error rate")
            return "high_error_rate"

        return None

    def select_failover_target(self, exclude_regions: Optional[Set[str]] = None) -> Optional[str]:
        """Select the best region for failover"""
        exclude_regions = exclude_regions or set()

        # Get healthy regions
        candidates = [
            (name, status) for name, status in self.regions.items()
            if status.health == RegionHealth.HEALTHY
            and name not in exclude_regions
            and name != self.primary_region
        ]

        if not candidates:
            # No healthy regions, try degraded
            candidates = [
                (name, status) for name, status in self.regions.items()
                if status.health == RegionHealth.DEGRADED
                and name not in exclude_regions
                and name != self.primary_region
            ]

        if not candidates:
            logger.error("No healthy regions available for failover")
            return None

        # Sort by response time
        candidates.sort(key=lambda x: x[1].response_time_ms)

        target = candidates[0][0]
        logger.info(f"Selected failover target: {target}")
        return target

    async def initiate_failover(
        self,
        reason: str,
        strategy: FailoverStrategy = FailoverStrategy.AUTOMATIC,
        target_region: Optional[str] = None
    ) -> FailoverEvent:
        """Initiate failover to a healthy region"""

        if not target_region:
            target_region = self.select_failover_target()

        if not target_region:
            raise Exception("No suitable failover target available")

        event_id = f"failover-{datetime.utcnow().timestamp()}"
        event = FailoverEvent(
            event_id=event_id,
            timestamp=datetime.utcnow().isoformat(),
            source_region=self.primary_region,
            target_region=target_region,
            reason=reason,
            strategy=strategy,
            status="initiated",
            automated=(strategy == FailoverStrategy.AUTOMATIC)
        )

        self.failover_history.append(event)
        logger.info(f"Initiated failover: {self.primary_region} -> {target_region}")

        if self.alert_manager:
            from ..monitoring.alerting import Alert
            await self.alert_manager.send_alert(Alert(
                alert_id=f"alert-{event_id}",
                name="FailoverInitiated",
                severity="critical",
                region=self.primary_region,
                message=f"Failover initiated from {self.primary_region} to {target_region}. Reason: {reason}",
                triggered_at=datetime.utcnow().isoformat(),
                resolved_at=None,
                labels={"source": self.primary_region or "unknown", "target": target_region}
            ))

        try:
            event.status = "in_progress"
            start_time = datetime.utcnow()

            # Execute failover based on strategy
            if strategy == FailoverStrategy.GRACEFUL:
                await self._graceful_failover(event)
            elif strategy == FailoverStrategy.IMMEDIATE:
                await self._immediate_failover(event)
            else:
                await self._automatic_failover(event)

            # Update primary region
            old_primary = self.primary_region
            self.primary_region = target_region
            self.regions[target_region].is_primary = True
            if old_primary:
                self.regions[old_primary].is_primary = False

            duration = (datetime.utcnow() - start_time).total_seconds()
            event.duration_seconds = duration
            event.status = "completed"

            logger.info(
                f"Failover completed in {duration:.2f}s: "
                f"{event.source_region} -> {event.target_region}"
            )

        except Exception as e:
            event.status = "failed"
            event.error_message = str(e)
            logger.error(f"Failover failed: {e}")

        # Record metrics
        if self.registry:
            self.failover_counter.labels(
                source_region=event.source_region or "unknown",
                target_region=event.target_region,
                reason=event.reason,
                status=event.status
            ).inc()

        return event

    async def _automatic_failover(self, event: FailoverEvent):
        """Execute automatic failover"""
        # Import here to avoid circular dependency
        from .dns_updater import DNSUpdater

        # Update DNS to point to new region
        dns_updater = DNSUpdater(self.config)
        await dns_updater.update_primary_region(
            event.target_region,
            self.regions[event.target_region].endpoint
        )

        # Wait for DNS propagation
        await asyncio.sleep(60)

    async def _graceful_failover(self, event: FailoverEvent):
        """Execute graceful failover with connection draining"""
        # Reduce traffic to source region gradually
        weights = [100, 75, 50, 25, 0]

        from .dns_updater import DNSUpdater
        dns_updater = DNSUpdater(self.config)

        for weight in weights:
            await dns_updater.update_weighted_routing(
                event.source_region,
                weight
            )
            await asyncio.sleep(30)  # Wait 30s between changes

        # Complete cutover
        await self._automatic_failover(event)

    async def _immediate_failover(self, event: FailoverEvent):
        """Execute immediate failover without draining"""
        await self._automatic_failover(event)

    async def rollback_failover(self, event_id: str) -> FailoverEvent:
        """Rollback a failover to the original region"""
        event = next(
            (e for e in self.failover_history if e.event_id == event_id),
            None
        )

        if not event:
            raise ValueError(f"Failover event {event_id} not found")

        if event.status != "completed":
            raise ValueError("Can only rollback completed failovers")

        # Check if source region is healthy again
        source_status = self.regions.get(event.source_region)
        if source_status.health == RegionHealth.UNHEALTHY:
            raise ValueError("Source region is still unhealthy")

        logger.info(f"Rolling back failover {event_id}")

        # Create rollback event
        rollback_event = await self.initiate_failover(
            reason=f"Rollback of {event_id}",
            strategy=FailoverStrategy.GRACEFUL,
            target_region=event.source_region
        )

        event.status = "rolled_back"
        return rollback_event

    async def continuous_monitoring(self):
        """Continuously monitor regions and trigger failover if needed"""
        self._running = True
        check_interval = self.health_checker.config.interval_seconds

        logger.info(f"Starting continuous monitoring (interval: {check_interval}s)")

        while self._running:
            try:
                # Check all regions
                await self.check_all_regions()

                # Determine if failover is needed
                failover_reason = self.should_trigger_failover()

                if failover_reason:
                    logger.warning(f"Triggering automatic failover: {failover_reason}")
                    await self.initiate_failover(
                        reason=failover_reason,
                        strategy=FailoverStrategy.AUTOMATIC
                    )

                # Check minimum healthy regions
                healthy_count = len(self.get_healthy_regions())
                if healthy_count < self.min_healthy_regions:
                    logger.critical(
                        f"Only {healthy_count} healthy regions, "
                        f"minimum is {self.min_healthy_regions}"
                    )

                await asyncio.sleep(check_interval)

            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(check_interval)

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self._running = False
        logger.info("Stopped continuous monitoring")

    def get_region_status(self, region_name: Optional[str] = None) -> Dict:
        """Get status of regions"""
        if region_name:
            return self.regions.get(region_name)

        return self.regions

    def get_failover_history(
        self,
        limit: Optional[int] = None,
        status: Optional[str] = None
    ) -> List[FailoverEvent]:
        """Get failover event history"""
        events = self.failover_history

        if status:
            events = [e for e in events if e.status == status]

        if limit:
            events = events[-limit:]

        return events

    async def simulate_failover(self, target_region: str) -> FailoverEvent:
        """Simulate a failover for testing purposes"""
        logger.info(f"Simulating failover to {target_region}")
        return await self.initiate_failover(
            reason="simulation",
            strategy=FailoverStrategy.GRACEFUL,
            target_region=target_region
        )

    def get_metrics(self) -> Dict:
        """Get current metrics across all regions"""
        healthy_regions = self.get_healthy_regions()

        return {
            'total_regions': len(self.regions),
            'healthy_regions': len(healthy_regions),
            'primary_region': self.primary_region,
            'primary_health': self.regions[self.primary_region].health.value if self.primary_region else None,
            'failover_enabled': self.failover_enabled,
            'total_failovers': len(self.failover_history),
            'recent_failovers': len([
                e for e in self.failover_history
                if datetime.fromisoformat(e.timestamp) > datetime.utcnow() - timedelta(hours=24)
            ]),
            'avg_response_time_ms': sum(
                r.response_time_ms for r in self.regions.values()
            ) / len(self.regions) if self.regions else 0
        }
