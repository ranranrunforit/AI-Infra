"""
Alert Manager

Manages alerts for multi-region platform using Prometheus Alertmanager.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Alert definition"""
    alert_id: str
    name: str
    severity: str  # critical, warning, info
    region: Optional[str]
    message: str
    triggered_at: str
    resolved_at: Optional[str]
    labels: Dict[str, str]


class AlertManager:
    """
    Multi-Region Alert Manager

    Manages alerting rules and notifications across all regions.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []

    def generate_alert_rules(self) -> str:
        """Generate Prometheus alert rules YAML"""
        rules = """
groups:
  - name: multi_region_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: sum(rate(http_requests_total{status=~"5.."}[5m])) by (region) / sum(rate(http_requests_total[5m])) by (region) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate in {{ $labels.region }}"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: HighLatency
        expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High latency in {{ $labels.region }}"
          description: "P99 latency is {{ $value }}s"

      - alert: RegionDown
        expr: up{job="ml-serving"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Region {{ $labels.region }} is down"
          description: "ML serving endpoint is unreachable"

      - alert: HighCPUUtilization
        expr: avg(rate(container_cpu_usage_seconds_total[5m])) by (region) > 0.8
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "High CPU utilization in {{ $labels.region }}"
          description: "CPU utilization is {{ $value | humanizePercentage }}"

      - alert: HighMemoryUtilization
        expr: avg(container_memory_usage_bytes / container_spec_memory_limit_bytes) by (region) > 0.85
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High memory utilization in {{ $labels.region }}"
          description: "Memory utilization is {{ $value | humanizePercentage }}"

      - alert: BudgetExceeded
        expr: multiregion_monthly_cost > multiregion_monthly_budget
        for: 1h
        labels:
          severity: critical
        annotations:
          summary: "Monthly budget exceeded"
          description: "Current spend: ${{ $value }}"
"""
        return rules

    async def send_alert(self, alert: Alert):
        """Send alert notification"""
        logger.warning(f"ALERT [{alert.severity}]: {alert.name} - {alert.message}")

        # Send to alertmanager
        alertmanager_url = self.config.get('alertmanager_url')

        if alertmanager_url:
            try:
                async with aiohttp.ClientSession() as session:
                    url = f"{alertmanager_url}/api/v2/alerts"
                    payload = [{
                        "labels": {
                            "alertname": alert.name,
                            "severity": alert.severity,
                            "region": alert.region or "global",
                            **alert.labels
                        },
                        "annotations": {
                            "summary": alert.message
                        }
                    }]

                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            logger.info(f"Alert sent successfully: {alert.name}")
                        else:
                            logger.error(f"Failed to send alert: {response.status}")

            except Exception as e:
                logger.error(f"Error sending alert: {e}")

        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)

    async def resolve_alert(self, alert_id: str):
        """Mark alert as resolved"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved_at = datetime.utcnow().isoformat()
            del self.active_alerts[alert_id]
            logger.info(f"Alert resolved: {alert.name}")

    def get_active_alerts(self, severity: Optional[str] = None) -> List[Alert]:
        """Get active alerts"""
        alerts = list(self.active_alerts.values())

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts
