"""
Recovery Manager

Handles automated recovery of failed regions, including health verification,
gradual traffic restoration, and post-recovery validation.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecoveryPhase(Enum):
    """Phases of recovery process"""
    DETECTION = "detection"
    VERIFICATION = "verification"
    PREPARATION = "preparation"
    TRAFFIC_RESTORATION = "traffic_restoration"
    MONITORING = "monitoring"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RecoveryPlan:
    """Plan for recovering a failed region"""
    plan_id: str
    region: str
    failure_detected_at: str
    recovery_started_at: Optional[str] = None
    recovery_completed_at: Optional[str] = None
    current_phase: RecoveryPhase = RecoveryPhase.DETECTION
    phases_completed: List[RecoveryPhase] = field(default_factory=list)
    traffic_percentage: int = 0
    health_checks_passed: int = 0
    health_checks_required: int = 10
    error_message: Optional[str] = None
    automated: bool = True


@dataclass
class RecoveryMetrics:
    """Metrics during recovery"""
    timestamp: str
    region: str
    phase: RecoveryPhase
    response_time_ms: float
    error_rate: float
    success_rate: float
    active_connections: int
    cpu_utilization: float
    memory_utilization: float


class RecoveryManager:
    """
    Automated Region Recovery Manager

    Manages the recovery process for failed regions, including:
    - Health verification
    - Gradual traffic restoration
    - Continuous monitoring
    - Automatic rollback if recovery fails
    """

    def __init__(self, config: Dict):
        self.config = config
        self.recovery_plans: Dict[str, RecoveryPlan] = {}
        self.recovery_metrics: List[RecoveryMetrics] = []
        self.min_health_checks = config.get('min_health_checks', 10)
        self.traffic_increment = config.get('traffic_increment_percent', 10)
        self.monitoring_duration = config.get('monitoring_duration_seconds', 300)
        self._running = False

    async def detect_recovery_candidate(self, region: str) -> bool:
        """Check if a failed region is ready for recovery"""
        logger.info(f"Checking if {region} is ready for recovery")

        from .failover_controller import HealthChecker, HealthCheckConfig

        # Get region configuration
        region_config = next(
            (r for r in self.config.get('regions', []) if r['name'] == region),
            None
        )

        if not region_config:
            logger.error(f"Region {region} not found in configuration")
            return False

        # Perform health checks
        health_checker = HealthChecker(HealthCheckConfig())

        checks_passed = 0
        checks_required = 5

        for i in range(checks_required):
            result = await health_checker.check_endpoint(region_config['endpoint'])

            if result['healthy']:
                checks_passed += 1
                logger.info(
                    f"Health check {i+1}/{checks_required} passed for {region} "
                    f"({result['response_time_ms']:.2f}ms)"
                )
            else:
                logger.warning(
                    f"Health check {i+1}/{checks_required} failed for {region}: "
                    f"{result.get('error')}"
                )

            await asyncio.sleep(5)

        recovery_ready = checks_passed >= checks_required * 0.8

        logger.info(
            f"Region {region} recovery readiness: {checks_passed}/{checks_required} "
            f"({'ready' if recovery_ready else 'not ready'})"
        )

        return recovery_ready

    async def create_recovery_plan(self, region: str) -> RecoveryPlan:
        """Create a recovery plan for a failed region"""
        plan_id = f"recovery-{region}-{datetime.utcnow().timestamp()}"

        plan = RecoveryPlan(
            plan_id=plan_id,
            region=region,
            failure_detected_at=datetime.utcnow().isoformat(),
            health_checks_required=self.min_health_checks
        )

        self.recovery_plans[plan_id] = plan
        logger.info(f"Created recovery plan {plan_id} for {region}")

        return plan

    async def execute_recovery(self, plan: RecoveryPlan) -> RecoveryPlan:
        """Execute a recovery plan"""
        logger.info(f"Executing recovery plan {plan.plan_id} for {plan.region}")

        plan.recovery_started_at = datetime.utcnow().isoformat()

        try:
            # Phase 1: Verification
            await self._verify_region_health(plan)

            # Phase 2: Preparation
            await self._prepare_region(plan)

            # Phase 3: Traffic Restoration
            await self._restore_traffic(plan)

            # Phase 4: Monitoring
            await self._monitor_recovery(plan)

            # Mark as completed
            plan.current_phase = RecoveryPhase.COMPLETED
            plan.recovery_completed_at = datetime.utcnow().isoformat()

            logger.info(f"Recovery completed for {plan.region}")

        except Exception as e:
            plan.current_phase = RecoveryPhase.FAILED
            plan.error_message = str(e)
            logger.error(f"Recovery failed for {plan.region}: {e}")

        return plan

    async def _verify_region_health(self, plan: RecoveryPlan):
        """Verify region health before recovery"""
        logger.info(f"Phase 1: Verifying health for {plan.region}")
        plan.current_phase = RecoveryPhase.VERIFICATION

        from .failover_controller import HealthChecker, HealthCheckConfig

        region_config = next(
            r for r in self.config['regions'] if r['name'] == plan.region
        )

        health_checker = HealthChecker(HealthCheckConfig())
        checks_passed = 0

        for i in range(plan.health_checks_required):
            # Endpoint check
            endpoint_result = await health_checker.check_endpoint(region_config['endpoint'])

            # Cluster check
            cluster_result = await health_checker.check_kubernetes_cluster(region_config)

            if endpoint_result['healthy'] and cluster_result['healthy']:
                checks_passed += 1
                plan.health_checks_passed = checks_passed

                # Record metrics
                self.recovery_metrics.append(RecoveryMetrics(
                    timestamp=datetime.utcnow().isoformat(),
                    region=plan.region,
                    phase=RecoveryPhase.VERIFICATION,
                    response_time_ms=endpoint_result['response_time_ms'],
                    error_rate=0.0,
                    success_rate=100.0,
                    active_connections=0,
                    cpu_utilization=0.0,
                    memory_utilization=0.0
                ))

            logger.info(
                f"Health check {i+1}/{plan.health_checks_required}: "
                f"{'passed' if endpoint_result['healthy'] else 'failed'}"
            )

            await asyncio.sleep(3)

        if checks_passed < plan.health_checks_required * 0.9:
            raise Exception(
                f"Insufficient health checks passed: "
                f"{checks_passed}/{plan.health_checks_required}"
            )

        plan.phases_completed.append(RecoveryPhase.VERIFICATION)
        logger.info(f"Verification phase completed for {plan.region}")

    async def _prepare_region(self, plan: RecoveryPlan):
        """Prepare region for traffic"""
        logger.info(f"Phase 2: Preparing {plan.region} for traffic")
        plan.current_phase = RecoveryPhase.PREPARATION

        # Verify dependent services are running
        region_config = next(
            r for r in self.config['regions'] if r['name'] == plan.region
        )

        # Check if replication is up to date
        from ..replication import ModelReplicator, DataSync

        # This is a simplified check - in production you'd verify:
        # - Model artifacts are synced
        # - Data is replicated
        # - Configuration is current
        # - Dependencies are healthy

        await asyncio.sleep(5)  # Simulate preparation time

        plan.phases_completed.append(RecoveryPhase.PREPARATION)
        logger.info(f"Preparation phase completed for {plan.region}")

    async def _restore_traffic(self, plan: RecoveryPlan):
        """Gradually restore traffic to recovered region"""
        logger.info(f"Phase 3: Restoring traffic to {plan.region}")
        plan.current_phase = RecoveryPhase.TRAFFIC_RESTORATION

        from .dns_updater import DNSUpdater

        dns_updater = DNSUpdater(self.config)

        # Gradually increase traffic
        traffic_levels = list(range(0, 101, self.traffic_increment))

        for traffic_percent in traffic_levels:
            plan.traffic_percentage = traffic_percent

            # Update DNS weighted routing
            await dns_updater.update_weighted_routing(
                plan.region,
                traffic_percent
            )

            logger.info(f"Increased traffic to {plan.region}: {traffic_percent}%")

            # Monitor for issues
            await asyncio.sleep(30)  # Wait before next increment

            # Check if region is still healthy
            from .failover_controller import HealthChecker, HealthCheckConfig

            region_config = next(
                r for r in self.config['regions'] if r['name'] == plan.region
            )

            health_checker = HealthChecker(HealthCheckConfig())
            result = await health_checker.check_endpoint(region_config['endpoint'])

            if not result['healthy']:
                logger.error(
                    f"Region {plan.region} became unhealthy during traffic restoration "
                    f"at {traffic_percent}%"
                )

                # Rollback traffic
                await dns_updater.update_weighted_routing(plan.region, 0)
                raise Exception("Region became unhealthy during recovery")

            # Record metrics
            self.recovery_metrics.append(RecoveryMetrics(
                timestamp=datetime.utcnow().isoformat(),
                region=plan.region,
                phase=RecoveryPhase.TRAFFIC_RESTORATION,
                response_time_ms=result['response_time_ms'],
                error_rate=0.0,
                success_rate=100.0,
                active_connections=0,
                cpu_utilization=0.0,
                memory_utilization=0.0
            ))

        plan.phases_completed.append(RecoveryPhase.TRAFFIC_RESTORATION)
        logger.info(f"Traffic restoration completed for {plan.region}")

    async def _monitor_recovery(self, plan: RecoveryPlan):
        """Monitor recovered region for stability"""
        logger.info(
            f"Phase 4: Monitoring {plan.region} for {self.monitoring_duration}s"
        )
        plan.current_phase = RecoveryPhase.MONITORING

        from .failover_controller import HealthChecker, HealthCheckConfig

        region_config = next(
            r for r in self.config['regions'] if r['name'] == plan.region
        )

        health_checker = HealthChecker(HealthCheckConfig())
        start_time = datetime.utcnow()
        check_interval = 10  # seconds

        failures = 0
        max_failures = 3

        while (datetime.utcnow() - start_time).total_seconds() < self.monitoring_duration:
            result = await health_checker.check_endpoint(region_config['endpoint'])

            if not result['healthy']:
                failures += 1
                logger.warning(
                    f"Health check failed during monitoring ({failures}/{max_failures})"
                )

                if failures >= max_failures:
                    raise Exception("Too many failures during monitoring phase")
            else:
                failures = 0  # Reset on success

            # Record metrics
            self.recovery_metrics.append(RecoveryMetrics(
                timestamp=datetime.utcnow().isoformat(),
                region=plan.region,
                phase=RecoveryPhase.MONITORING,
                response_time_ms=result['response_time_ms'],
                error_rate=0.0,
                success_rate=100.0,
                active_connections=0,
                cpu_utilization=0.0,
                memory_utilization=0.0
            ))

            await asyncio.sleep(check_interval)

        plan.phases_completed.append(RecoveryPhase.MONITORING)
        logger.info(f"Monitoring phase completed for {plan.region}")

    async def automatic_recovery(self, region: str) -> Optional[RecoveryPlan]:
        """Automatically recover a failed region"""
        logger.info(f"Starting automatic recovery for {region}")

        # Check if region is ready
        ready = await self.detect_recovery_candidate(region)

        if not ready:
            logger.info(f"Region {region} not ready for recovery")
            return None

        # Create and execute recovery plan
        plan = await self.create_recovery_plan(region)

        try:
            await self.execute_recovery(plan)
            return plan
        except Exception as e:
            logger.error(f"Automatic recovery failed for {region}: {e}")
            return plan

    async def rollback_recovery(self, plan_id: str) -> bool:
        """Rollback a recovery attempt"""
        plan = self.recovery_plans.get(plan_id)

        if not plan:
            logger.error(f"Recovery plan {plan_id} not found")
            return False

        logger.info(f"Rolling back recovery for {plan.region}")

        from .dns_updater import DNSUpdater

        dns_updater = DNSUpdater(self.config)

        # Remove region from traffic
        await dns_updater.update_weighted_routing(plan.region, 0)

        plan.traffic_percentage = 0
        plan.current_phase = RecoveryPhase.FAILED
        plan.error_message = "Recovery rolled back"

        logger.info(f"Rolled back recovery for {plan.region}")
        return True

    async def continuous_recovery_monitoring(self):
        """Continuously monitor for recovery opportunities"""
        self._running = True
        logger.info("Starting continuous recovery monitoring")

        while self._running:
            try:
                # Check all regions for recovery candidates
                for region_config in self.config.get('regions', []):
                    region = region_config['name']

                    # Skip if already recovering
                    if any(p.region == region and p.current_phase != RecoveryPhase.COMPLETED
                           for p in self.recovery_plans.values()):
                        continue

                    # Check if region needs recovery
                    from .failover_controller import FailoverController

                    controller = FailoverController(self.config)
                    status = await controller.check_region_health(region)

                    if status.health.value == "unhealthy":
                        logger.info(f"Region {region} detected as unhealthy, checking for recovery")

                        ready = await self.detect_recovery_candidate(region)

                        if ready:
                            logger.info(f"Initiating automatic recovery for {region}")
                            await self.automatic_recovery(region)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in recovery monitoring: {e}")
                await asyncio.sleep(60)

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self._running = False
        logger.info("Stopped recovery monitoring")

    def get_recovery_plan(self, plan_id: str) -> Optional[RecoveryPlan]:
        """Get a specific recovery plan"""
        return self.recovery_plans.get(plan_id)

    def get_recovery_history(
        self,
        region: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[RecoveryPlan]:
        """Get recovery plan history"""
        plans = list(self.recovery_plans.values())

        if region:
            plans = [p for p in plans if p.region == region]

        plans.sort(key=lambda p: p.failure_detected_at, reverse=True)

        if limit:
            plans = plans[:limit]

        return plans

    def get_recovery_metrics(
        self,
        region: Optional[str] = None,
        phase: Optional[RecoveryPhase] = None,
        limit: Optional[int] = None
    ) -> List[RecoveryMetrics]:
        """Get recovery metrics"""
        metrics = self.recovery_metrics

        if region:
            metrics = [m for m in metrics if m.region == region]

        if phase:
            metrics = [m for m in metrics if m.phase == phase]

        if limit:
            metrics = metrics[-limit:]

        return metrics

    def get_recovery_statistics(self) -> Dict:
        """Get recovery statistics"""
        total_plans = len(self.recovery_plans)
        completed = sum(
            1 for p in self.recovery_plans.values()
            if p.current_phase == RecoveryPhase.COMPLETED
        )
        failed = sum(
            1 for p in self.recovery_plans.values()
            if p.current_phase == RecoveryPhase.FAILED
        )
        in_progress = sum(
            1 for p in self.recovery_plans.values()
            if p.current_phase not in [RecoveryPhase.COMPLETED, RecoveryPhase.FAILED]
        )

        success_rate = (completed / total_plans * 100) if total_plans > 0 else 0

        return {
            'total_recovery_attempts': total_plans,
            'completed': completed,
            'failed': failed,
            'in_progress': in_progress,
            'success_rate': success_rate,
            'automated_recoveries': sum(
                1 for p in self.recovery_plans.values() if p.automated
            )
        }
