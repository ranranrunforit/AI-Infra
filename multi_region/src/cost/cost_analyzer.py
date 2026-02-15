"""
Cost Analyzer

Multi-cloud cost analysis across AWS, GCP, and Azure.
Tracks spending, identifies anomalies, and provides cost breakdowns.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from decimal import Decimal

import boto3
from google.cloud import billing_v1
from azure.mgmt.costmanagement import CostManagementClient
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CostData:
    """Cost data for a specific resource/service"""
    service: str
    region: str
    provider: str  # aws, gcp, azure
    amount: Decimal
    currency: str
    start_date: str
    end_date: str
    unit: str  # hours, GB, requests, etc.
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class CostReport:
    """Comprehensive cost report"""
    report_id: str
    start_date: str
    end_date: str
    total_cost: Decimal
    currency: str
    costs_by_provider: Dict[str, Decimal]
    costs_by_region: Dict[str, Decimal]
    costs_by_service: Dict[str, Decimal]
    daily_costs: List[Tuple[str, Decimal]]
    anomalies: List[Dict]
    trends: Dict[str, float]


@dataclass
class BudgetAlert:
    """Budget alert"""
    alert_id: str
    budget_name: str
    threshold_percent: float
    current_spend: Decimal
    budget_amount: Decimal
    period: str
    triggered_at: str
    severity: str  # warning, critical


class CostAnalyzer:
    """
    Multi-Cloud Cost Analyzer

    Aggregates and analyzes costs across AWS, GCP, and Azure.
    Provides detailed breakdowns, trend analysis, and anomaly detection.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.cost_data: List[CostData] = []
        self.budgets: Dict[str, Dict] = config.get('budgets', {})
        self.budget_alerts: List[BudgetAlert] = []
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize cloud provider cost management clients"""
        # AWS Cost Explorer
        self.aws_ce = boto3.client(
            'ce',
            region_name=self.config.get('aws_region', 'us-west-2')
        )

        # GCP Billing (requires service account)
        try:
            self.gcp_billing = billing_v1.CloudBillingClient()
        except Exception as e:
            logger.warning(f"Failed to initialize GCP billing client: {e}")
            self.gcp_billing = None

        # Azure Cost Management
        try:
            self.azure_credential = DefaultAzureCredential()
            self.azure_cost = CostManagementClient(
                self.azure_credential,
                self.config.get('azure_subscription_id', '')
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Azure cost client: {e}")
            self.azure_cost = None

        logger.info("Initialized cost management clients")

    async def get_aws_costs(
        self,
        start_date: str,
        end_date: str
    ) -> List[CostData]:
        """Retrieve AWS costs from Cost Explorer"""
        logger.info(f"Fetching AWS costs from {start_date} to {end_date}")

        try:
            response = await asyncio.to_thread(
                self.aws_ce.get_cost_and_usage,
                TimePeriod={
                    'Start': start_date,
                    'End': end_date
                },
                Granularity='DAILY',
                Metrics=['UnblendedCost'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'},
                    {'Type': 'DIMENSION', 'Key': 'REGION'}
                ]
            )

            costs = []

            for result in response.get('ResultsByTime', []):
                start = result['TimePeriod']['Start']
                end = result['TimePeriod']['End']

                for group in result.get('Groups', []):
                    service = group['Keys'][0]
                    region = group['Keys'][1] if len(group['Keys']) > 1 else 'global'
                    amount = Decimal(group['Metrics']['UnblendedCost']['Amount'])

                    if amount > 0:
                        costs.append(CostData(
                            service=service,
                            region=region,
                            provider='aws',
                            amount=amount,
                            currency='USD',
                            start_date=start,
                            end_date=end,
                            unit='USD'
                        ))

            logger.info(f"Retrieved {len(costs)} AWS cost entries")
            return costs

        except Exception as e:
            logger.error(f"Failed to retrieve AWS costs: {e}")
            return []

    async def get_gcp_costs(
        self,
        start_date: str,
        end_date: str
    ) -> List[CostData]:
        """Retrieve GCP costs from Cloud Billing"""
        if not self.gcp_billing:
            logger.warning("GCP billing client not available")
            return []

        logger.info(f"Fetching GCP costs from {start_date} to {end_date}")

        try:
            # GCP BigQuery query would go here
            # This is a simplified version - in production you'd query BigQuery
            # export of billing data

            costs = []
            # Placeholder for GCP cost data
            logger.info("Retrieved 0 GCP cost entries (requires BigQuery setup)")
            return costs

        except Exception as e:
            logger.error(f"Failed to retrieve GCP costs: {e}")
            return []

    async def get_azure_costs(
        self,
        start_date: str,
        end_date: str
    ) -> List[CostData]:
        """Retrieve Azure costs from Cost Management"""
        if not self.azure_cost:
            logger.warning("Azure cost management client not available")
            return []

        logger.info(f"Fetching Azure costs from {start_date} to {end_date}")

        try:
            # Azure Cost Management query
            costs = []
            # Placeholder for Azure cost data
            logger.info("Retrieved 0 Azure cost entries (requires proper configuration)")
            return costs

        except Exception as e:
            logger.error(f"Failed to retrieve Azure costs: {e}")
            return []

    async def aggregate_costs(
        self,
        start_date: str,
        end_date: str
    ) -> List[CostData]:
        """Aggregate costs from all providers"""
        logger.info("Aggregating costs from all providers")

        # Fetch costs concurrently
        aws_costs_task = self.get_aws_costs(start_date, end_date)
        gcp_costs_task = self.get_gcp_costs(start_date, end_date)
        azure_costs_task = self.get_azure_costs(start_date, end_date)

        aws_costs, gcp_costs, azure_costs = await asyncio.gather(
            aws_costs_task,
            gcp_costs_task,
            azure_costs_task,
            return_exceptions=True
        )

        # Handle exceptions
        if isinstance(aws_costs, Exception):
            logger.error(f"AWS costs failed: {aws_costs}")
            aws_costs = []
        if isinstance(gcp_costs, Exception):
            logger.error(f"GCP costs failed: {gcp_costs}")
            gcp_costs = []
        if isinstance(azure_costs, Exception):
            logger.error(f"Azure costs failed: {azure_costs}")
            azure_costs = []

        all_costs = aws_costs + gcp_costs + azure_costs
        self.cost_data = all_costs

        logger.info(f"Aggregated {len(all_costs)} cost entries")
        return all_costs

    def generate_cost_report(
        self,
        start_date: str,
        end_date: str,
        costs: Optional[List[CostData]] = None
    ) -> CostReport:
        """Generate comprehensive cost report"""
        if costs is None:
            costs = self.cost_data

        report_id = f"cost-report-{datetime.utcnow().timestamp()}"

        # Calculate totals
        total_cost = sum(c.amount for c in costs)

        # Costs by provider
        costs_by_provider = {}
        for cost in costs:
            provider = cost.provider
            costs_by_provider[provider] = costs_by_provider.get(provider, Decimal('0')) + cost.amount

        # Costs by region
        costs_by_region = {}
        for cost in costs:
            region = f"{cost.provider}/{cost.region}"
            costs_by_region[region] = costs_by_region.get(region, Decimal('0')) + cost.amount

        # Costs by service
        costs_by_service = {}
        for cost in costs:
            service = cost.service
            costs_by_service[service] = costs_by_service.get(service, Decimal('0')) + cost.amount

        # Daily costs
        daily_costs_dict = {}
        for cost in costs:
            date = cost.start_date
            daily_costs_dict[date] = daily_costs_dict.get(date, Decimal('0')) + cost.amount

        daily_costs = sorted(daily_costs_dict.items())

        # Detect anomalies
        anomalies = self._detect_anomalies(costs)

        # Calculate trends
        trends = self._calculate_trends(daily_costs)

        report = CostReport(
            report_id=report_id,
            start_date=start_date,
            end_date=end_date,
            total_cost=total_cost,
            currency='USD',
            costs_by_provider=costs_by_provider,
            costs_by_region=costs_by_region,
            costs_by_service=costs_by_service,
            daily_costs=daily_costs,
            anomalies=anomalies,
            trends=trends
        )

        logger.info(
            f"Generated cost report: total=${total_cost:.2f}, "
            f"{len(anomalies)} anomalies detected"
        )

        return report

    def _detect_anomalies(self, costs: List[CostData]) -> List[Dict]:
        """Detect cost anomalies"""
        anomalies = []

        # Group by service
        service_costs = {}
        for cost in costs:
            service = cost.service
            if service not in service_costs:
                service_costs[service] = []
            service_costs[service].append(cost.amount)

        # Detect outliers (simple std deviation method)
        for service, amounts in service_costs.items():
            if len(amounts) < 3:
                continue

            avg = sum(amounts) / len(amounts)
            variance = sum((x - avg) ** 2 for x in amounts) / len(amounts)
            std_dev = variance ** Decimal('0.5')

            for amount in amounts:
                if abs(amount - avg) > 2 * std_dev:
                    anomalies.append({
                        'service': service,
                        'amount': float(amount),
                        'average': float(avg),
                        'std_dev': float(std_dev),
                        'severity': 'high' if abs(amount - avg) > 3 * std_dev else 'medium'
                    })

        return anomalies

    def _calculate_trends(self, daily_costs: List[Tuple[str, Decimal]]) -> Dict[str, float]:
        """Calculate cost trends"""
        if len(daily_costs) < 2:
            return {'daily_change': 0.0, 'weekly_change': 0.0}

        # Calculate daily change
        recent_costs = [float(c[1]) for c in daily_costs[-7:]]
        if len(recent_costs) >= 2:
            daily_change = ((recent_costs[-1] - recent_costs[0]) / recent_costs[0] * 100)
        else:
            daily_change = 0.0

        # Calculate weekly trend
        if len(daily_costs) >= 14:
            last_week = sum(float(c[1]) for c in daily_costs[-7:])
            prev_week = sum(float(c[1]) for c in daily_costs[-14:-7])
            weekly_change = ((last_week - prev_week) / prev_week * 100) if prev_week > 0 else 0.0
        else:
            weekly_change = 0.0

        return {
            'daily_change': daily_change,
            'weekly_change': weekly_change
        }

    async def check_budgets(self) -> List[BudgetAlert]:
        """Check if any budgets are exceeded"""
        alerts = []

        # Get current month costs
        start_date = datetime.utcnow().replace(day=1).strftime('%Y-%m-%d')
        end_date = datetime.utcnow().strftime('%Y-%m-%d')

        await self.aggregate_costs(start_date, end_date)
        report = self.generate_cost_report(start_date, end_date)

        # Check each budget
        for budget_name, budget_config in self.budgets.items():
            budget_amount = Decimal(str(budget_config['amount']))
            thresholds = budget_config.get('thresholds', [80, 100])

            # Determine scope (provider, region, service)
            scope = budget_config.get('scope', 'total')
            current_spend = Decimal('0')

            if scope == 'total':
                current_spend = report.total_cost
            elif scope.startswith('provider:'):
                provider = scope.split(':')[1]
                current_spend = report.costs_by_provider.get(provider, Decimal('0'))
            elif scope.startswith('region:'):
                region = scope.split(':')[1]
                current_spend = report.costs_by_region.get(region, Decimal('0'))
            elif scope.startswith('service:'):
                service = scope.split(':')[1]
                current_spend = report.costs_by_service.get(service, Decimal('0'))

            # Check thresholds
            for threshold in thresholds:
                threshold_amount = budget_amount * Decimal(str(threshold / 100))

                if current_spend >= threshold_amount:
                    severity = 'critical' if threshold >= 100 else 'warning'

                    alert = BudgetAlert(
                        alert_id=f"budget-alert-{budget_name}-{datetime.utcnow().timestamp()}",
                        budget_name=budget_name,
                        threshold_percent=float(threshold),
                        current_spend=current_spend,
                        budget_amount=budget_amount,
                        period='monthly',
                        triggered_at=datetime.utcnow().isoformat(),
                        severity=severity
                    )

                    alerts.append(alert)
                    logger.warning(
                        f"Budget alert: {budget_name} at {threshold}% "
                        f"(${current_spend:.2f} / ${budget_amount:.2f})"
                    )

        self.budget_alerts.extend(alerts)
        return alerts

    def get_cost_forecast(
        self,
        days_ahead: int = 30
    ) -> Dict[str, Decimal]:
        """Forecast future costs based on trends"""
        if not self.cost_data:
            logger.warning("No cost data available for forecasting")
            return {}

        # Group costs by day
        daily_costs = {}
        for cost in self.cost_data:
            date = cost.start_date
            daily_costs[date] = daily_costs.get(date, Decimal('0')) + cost.amount

        if len(daily_costs) < 7:
            logger.warning("Insufficient data for forecasting")
            return {}

        # Simple linear regression forecast
        sorted_days = sorted(daily_costs.items())
        recent_days = sorted_days[-30:]  # Use last 30 days

        # Calculate average daily cost
        avg_daily_cost = sum(c[1] for c in recent_days) / len(recent_days)

        # Calculate trend
        x_values = list(range(len(recent_days)))
        y_values = [float(c[1]) for c in recent_days]

        n = len(recent_days)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)

        # Linear regression: y = mx + b
        m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        b = (sum_y - m * sum_x) / n

        # Forecast
        forecast = {}
        last_date = datetime.fromisoformat(sorted_days[-1][0])

        for day in range(1, days_ahead + 1):
            forecast_date = last_date + timedelta(days=day)
            forecast_value = Decimal(str(m * (n + day) + b))
            forecast[forecast_date.strftime('%Y-%m-%d')] = max(forecast_value, Decimal('0'))

        logger.info(f"Generated {days_ahead}-day cost forecast")
        return forecast

    def get_top_cost_drivers(self, top_n: int = 10) -> List[Dict]:
        """Identify top cost drivers"""
        service_costs = {}

        for cost in self.cost_data:
            key = f"{cost.provider}/{cost.service}/{cost.region}"
            if key not in service_costs:
                service_costs[key] = {
                    'provider': cost.provider,
                    'service': cost.service,
                    'region': cost.region,
                    'total_cost': Decimal('0')
                }
            service_costs[key]['total_cost'] += cost.amount

        # Sort by cost
        sorted_costs = sorted(
            service_costs.values(),
            key=lambda x: x['total_cost'],
            reverse=True
        )

        return sorted_costs[:top_n]

    async def compare_provider_costs(
        self,
        start_date: str,
        end_date: str
    ) -> Dict[str, Dict]:
        """Compare costs across providers"""
        await self.aggregate_costs(start_date, end_date)

        comparison = {}

        for provider in ['aws', 'gcp', 'azure']:
            provider_costs = [c for c in self.cost_data if c.provider == provider]
            total = sum(c.amount for c in provider_costs)

            # Calculate per-service breakdown
            service_breakdown = {}
            for cost in provider_costs:
                service = cost.service
                service_breakdown[service] = service_breakdown.get(service, Decimal('0')) + cost.amount

            comparison[provider] = {
                'total_cost': total,
                'percentage': float(total / sum(c.amount for c in self.cost_data) * 100) if self.cost_data else 0,
                'service_breakdown': service_breakdown,
                'num_services': len(service_breakdown)
            }

        return comparison
