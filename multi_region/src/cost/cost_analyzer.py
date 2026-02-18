"""
Cost Analyzer - FIXED VERSION

Multi-cloud cost analysis across AWS, GCP, and Azure.
Gracefully handles missing credentials for GCP-only setup.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from decimal import Decimal

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


class CostAnalyzer:
    """
    Multi-Cloud Cost Analyzer

    Aggregates and analyzes costs across AWS, GCP, and Azure.
    Gracefully handles missing cloud provider credentials.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.cost_data: List[CostData] = []
        self.budgets: Dict[str, Dict] = config.get('budgets', {})
        
        # Initialize client attributes
        self.aws_ce = None
        self.gcp_billing = None
        self.azure_cost = None
        self.azure_credential = None
        
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize cloud provider cost management clients"""
        # AWS Cost Explorer
        try:
            import boto3
            self.aws_ce = boto3.client(
                'ce',
                region_name=self.config.get('aws_region', 'us-west-2')
            )
            logger.info("AWS Cost Explorer client initialized")
        except Exception as e:
            logger.warning(f"AWS client not available: {e}")
            self.aws_ce = None

        # GCP Billing
        try:
            from google.cloud import billing_v1
            self.gcp_billing = billing_v1.CloudBillingClient()
            logger.info("GCP Billing client initialized")
        except Exception as e:
            logger.warning(f"GCP billing client not available: {e}")
            self.gcp_billing = None

        # Azure Cost Management
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.costmanagement import CostManagementClient
            
            self.azure_credential = DefaultAzureCredential()
            self.azure_cost = CostManagementClient(
                self.azure_credential,
                self.config.get('azure_subscription_id', '')
            )
            logger.info("Azure Cost Management client initialized")
        except Exception as e:
            logger.warning(f"Azure client not available: {e}")
            self.azure_cost = None

        logger.info("Initialized cost management clients")

    async def get_aws_costs(
        self,
        start_date: str,
        end_date: str
    ) -> List[CostData]:
        """Retrieve AWS costs from Cost Explorer"""
        if not self.aws_ce:
            logger.info("AWS client not available, skipping AWS costs")
            return []
            
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
        """Retrieve GCP costs from BigQuery Billing Export"""
        table_variable = self.config.get('gcp_billing_table')
        if not table_variable:
            logger.info("GCP billing table not configured, skipping GCP costs")
            return []

        logger.info(f"Fetching GCP costs from BigQuery from {start_date} to {end_date}")

        try:
            from google.cloud import bigquery
            client = bigquery.Client(project=self.config.get('gcp_project_id'))

            query = f"""
                SELECT
                    service.description as service,
                    location.location as region,
                    SUM(cost) as amount,
                    currency
                FROM `{table_variable}`
                WHERE usage_start_time >= TIMESTAMP('{start_date}')
                AND usage_end_time <= TIMESTAMP('{end_date}')
                GROUP BY 1, 2, 4
            """

            query_job = await asyncio.to_thread(client.query, query)
            results = await asyncio.to_thread(query_job.result)

            costs = []
            for row in results:
                amount = Decimal(str(row.amount))
                if amount > 0:
                    costs.append(CostData(
                        service=row.service,
                        region=row.region or 'global',
                        provider='gcp',
                        amount=amount,
                        currency=row.currency,
                        start_date=start_date,
                        end_date=end_date,
                        unit='USD'
                    ))

            logger.info(f"Retrieved {len(costs)} GCP cost entries")
            return costs

        except Exception as e:
            logger.error(f"Failed to retrieve GCP costs: {e}")
            return []

    async def get_azure_costs(
        self,
        start_date: str,
        end_date: str
    ) -> List[CostData]:
        """Retrieve Azure costs from Cost Management API"""
        if not self.azure_cost:
            logger.info("Azure client not available, skipping Azure costs")
            return []
            
        logger.info(f"Fetching Azure costs from {start_date} to {end_date}")

        try:
            from azure.mgmt.costmanagement.models import QueryDefinition, TimeframeType
            
            query = QueryDefinition(
                type="ActualCost",
                timeframe=TimeframeType.CUSTOM,
                time_period={
                    "from": start_date,
                    "to": end_date
                },
                dataset={
                    "granularity": "Daily",
                    "aggregation": {
                        "totalCost": {
                            "name": "Cost",
                            "function": "Sum"
                        }
                    },
                    "grouping": [
                        {"type": "Dimension", "name": "ServiceName"},
                        {"type": "Dimension", "name": "ResourceLocation"}
                    ]
                }
            )

            scope = f"/subscriptions/{self.config.get('azure_subscription_id')}"
            result = await asyncio.to_thread(
                self.azure_cost.query.usage,
                scope,
                query
            )

            costs = []
            for row in result.rows:
                amount = Decimal(str(row[0]))
                service = row[2] if len(row) > 2 else "Unknown"
                region = row[3] if len(row) > 3 else "global"

                if amount > 0:
                    costs.append(CostData(
                        service=service,
                        region=region,
                        provider='azure',
                        amount=amount,
                        currency='USD',
                        start_date=start_date,
                        end_date=end_date,
                        unit='USD'
                    ))

            logger.info(f"Retrieved {len(costs)} Azure cost entries")
            return costs

        except Exception as e:
            logger.error(f"Failed to retrieve Azure costs: {e}")
            return []

    async def get_costs(
        self,
        start_date: str,
        end_date: str
    ) -> List[CostData]:
        """
        ADDED: Get costs from all providers (main aggregation method)
        This is the method that main.py calls
        """
        logger.info(f"Aggregating costs from all providers: {start_date} to {end_date}")
        
        # Gather costs from all providers in parallel
        aws_task = self.get_aws_costs(start_date, end_date)
        gcp_task = self.get_gcp_costs(start_date, end_date)
        azure_task = self.get_azure_costs(start_date, end_date)
        
        aws_costs, gcp_costs, azure_costs = await asyncio.gather(
            aws_task, gcp_task, azure_task,
            return_exceptions=True
        )
        
        # Handle exceptions
        all_costs = []
        if not isinstance(aws_costs, Exception):
            all_costs.extend(aws_costs)
        if not isinstance(gcp_costs, Exception):
            all_costs.extend(gcp_costs)
        if not isinstance(azure_costs, Exception):
            all_costs.extend(azure_costs)
        
        self.cost_data = all_costs
        logger.info(f"Total cost entries retrieved: {len(all_costs)}")
        
        return all_costs

    async def aggregate_costs(
        self,
        start_date: str,
        end_date: str
    ) -> List[CostData]:
        """Aggregate costs from all providers (legacy method)"""
        return await self.get_costs(start_date, end_date)

    def generate_cost_report(
        self,
        start_date: str,
        end_date: str
    ) -> CostReport:
        """Generate comprehensive cost report"""
        # Calculate totals
        total_cost = sum(cost.amount for cost in self.cost_data)

        # Group by provider
        costs_by_provider = {}
        for cost in self.cost_data:
            provider = cost.provider
            costs_by_provider[provider] = costs_by_provider.get(
                provider, Decimal('0')
            ) + cost.amount

        # Group by region
        costs_by_region = {}
        for cost in self.cost_data:
            region = cost.region
            costs_by_region[region] = costs_by_region.get(
                region, Decimal('0')
            ) + cost.amount

        # Group by service
        costs_by_service = {}
        for cost in self.cost_data:
            service = cost.service
            costs_by_service[service] = costs_by_service.get(
                service, Decimal('0')
            ) + cost.amount

        # Daily costs
        daily_costs_dict = {}
        for cost in self.cost_data:
            date = cost.start_date
            daily_costs_dict[date] = daily_costs_dict.get(
                date, Decimal('0')
            ) + cost.amount

        daily_costs = sorted(daily_costs_dict.items())

        # Detect anomalies
        anomalies = self._detect_anomalies(self.cost_data)

        # Calculate trends
        trends = self._calculate_trends(daily_costs)

        report = CostReport(
            report_id=f"cost-report-{datetime.utcnow().timestamp()}",
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

    async def generate_report(
        self,
        start_date: str,
        end_date: str
    ) -> CostReport:
        """Generate report (calls get_costs first)"""
        await self.get_costs(start_date, end_date)
        return self.generate_cost_report(start_date, end_date)

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
            daily_change = ((recent_costs[-1] - recent_costs[0]) / recent_costs[0] * 100) if recent_costs[0] > 0 else 0
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
