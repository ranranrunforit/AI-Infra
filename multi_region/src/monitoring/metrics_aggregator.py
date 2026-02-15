"""
Metrics Aggregator

Collects and aggregates metrics from all regions using Prometheus.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RegionMetrics:
    """Metrics for a region"""
    region: str
    timestamp: str
    request_rate: float
    error_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    cpu_utilization: float
    memory_utilization: float
    active_pods: int
    total_pods: int


class MetricsAggregator:
    """
    Multi-Region Metrics Aggregator

    Collects metrics from Prometheus instances in each region
    and provides unified view.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.metrics_cache: Dict[str, RegionMetrics] = {}
        self.registry = CollectorRegistry()
        self._initialize_metrics()

    def _initialize_metrics(self):
        """Initialize Prometheus metrics"""
        self.region_request_rate = Gauge(
            'multiregion_request_rate',
            'Request rate per region',
            ['region'],
            registry=self.registry
        )
        self.region_error_rate = Gauge(
            'multiregion_error_rate',
            'Error rate per region',
            ['region'],
            registry=self.registry
        )
        self.region_latency = Gauge(
            'multiregion_latency_ms',
            'Latency per region',
            ['region', 'percentile'],
            registry=self.registry
        )

    async def query_prometheus(
        self,
        prometheus_url: str,
        query: str
    ) -> Optional[Dict]:
        """Query Prometheus endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{prometheus_url}/api/v1/query"
                params = {'query': query}

                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('data', {})
                    else:
                        logger.error(f"Prometheus query failed: {response.status}")
                        return None

        except Exception as e:
            logger.error(f"Failed to query Prometheus: {e}")
            return None

    async def collect_region_metrics(self, region_config: Dict) -> Optional[RegionMetrics]:
        """Collect metrics from a specific region"""
        region = region_config['name']
        prometheus_url = region_config.get('prometheus_url')

        if not prometheus_url:
            logger.warning(f"No Prometheus URL for {region}")
            return None

        logger.info(f"Collecting metrics from {region}")

        # Query request rate
        request_rate_data = await self.query_prometheus(
            prometheus_url,
            'sum(rate(http_requests_total[5m]))'
        )

        # Query error rate
        error_rate_data = await self.query_prometheus(
            prometheus_url,
            'sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))'
        )

        # Query latency percentiles
        p50_data = await self.query_prometheus(
            prometheus_url,
            'histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))'
        )

        p95_data = await self.query_prometheus(
            prometheus_url,
            'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))'
        )

        p99_data = await self.query_prometheus(
            prometheus_url,
            'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))'
        )

        # Query resource utilization
        cpu_data = await self.query_prometheus(
            prometheus_url,
            'avg(rate(container_cpu_usage_seconds_total[5m]))'
        )

        memory_data = await self.query_prometheus(
            prometheus_url,
            'avg(container_memory_usage_bytes / container_spec_memory_limit_bytes)'
        )

        # Query pod counts
        pods_data = await self.query_prometheus(
            prometheus_url,
            'count(kube_pod_info)'
        )

        # Extract values (simplified)
        def extract_value(data, default=0.0):
            if data and 'result' in data and len(data['result']) > 0:
                return float(data['result'][0]['value'][1])
            return default

        metrics = RegionMetrics(
            region=region,
            timestamp=datetime.utcnow().isoformat(),
            request_rate=extract_value(request_rate_data),
            error_rate=extract_value(error_rate_data),
            p50_latency_ms=extract_value(p50_data) * 1000,
            p95_latency_ms=extract_value(p95_data) * 1000,
            p99_latency_ms=extract_value(p99_data) * 1000,
            cpu_utilization=extract_value(cpu_data) * 100,
            memory_utilization=extract_value(memory_data) * 100,
            active_pods=int(extract_value(pods_data)),
            total_pods=int(extract_value(pods_data))
        )

        self.metrics_cache[region] = metrics

        # Update Prometheus metrics
        self.region_request_rate.labels(region=region).set(metrics.request_rate)
        self.region_error_rate.labels(region=region).set(metrics.error_rate)
        self.region_latency.labels(region=region, percentile='p50').set(metrics.p50_latency_ms)
        self.region_latency.labels(region=region, percentile='p95').set(metrics.p95_latency_ms)
        self.region_latency.labels(region=region, percentile='p99').set(metrics.p99_latency_ms)

        return metrics

    async def collect_all_metrics(self) -> Dict[str, RegionMetrics]:
        """Collect metrics from all regions"""
        tasks = [
            self.collect_region_metrics(region_config)
            for region_config in self.config.get('regions', [])
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        metrics = {}
        for region_config, result in zip(self.config['regions'], results):
            if isinstance(result, Exception):
                logger.error(f"Failed to collect metrics from {region_config['name']}: {result}")
            elif result:
                metrics[result.region] = result

        return metrics

    def get_global_metrics(self) -> Dict:
        """Get aggregated global metrics"""
        if not self.metrics_cache:
            return {}

        total_request_rate = sum(m.request_rate for m in self.metrics_cache.values())
        avg_error_rate = sum(m.error_rate for m in self.metrics_cache.values()) / len(self.metrics_cache)
        avg_p99_latency = sum(m.p99_latency_ms for m in self.metrics_cache.values()) / len(self.metrics_cache)

        return {
            'total_request_rate': total_request_rate,
            'average_error_rate': avg_error_rate,
            'average_p99_latency_ms': avg_p99_latency,
            'regions': len(self.metrics_cache),
            'timestamp': datetime.utcnow().isoformat()
        }

    async def continuous_collection(self, interval_seconds: int = 60):
        """Continuously collect metrics"""
        logger.info(f"Starting continuous metrics collection (interval: {interval_seconds}s)")

        while True:
            try:
                await self.collect_all_metrics()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in continuous collection: {e}")
                await asyncio.sleep(interval_seconds)
