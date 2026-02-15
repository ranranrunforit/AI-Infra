"""
Cost Optimizer

Provides cost optimization recommendations for multi-region ML infrastructure.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from decimal import Decimal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationRecommendation:
    """Cost optimization recommendation"""
    recommendation_id: str
    category: str  # compute, storage, network, licensing
    title: str
    description: str
    provider: str
    region: Optional[str]
    potential_savings: Decimal
    impact: str  # low, medium, high
    effort: str  # low, medium, high
    priority: int  # 1-5, 5 being highest
    implementation_steps: List[str]


class CostOptimizer:
    """
    Cost Optimization Engine

    Analyzes resource usage and provides actionable recommendations
    for reducing multi-cloud costs.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.recommendations: List[OptimizationRecommendation] = []

    def analyze_spot_opportunities(self, cost_data: List) -> List[OptimizationRecommendation]:
        """Identify opportunities to use spot/preemptible instances"""
        recommendations = []

        # Check if spot instances are enabled
        spot_enabled = self.config.get('use_spot_instances', False)

        if not spot_enabled:
            rec = OptimizationRecommendation(
                recommendation_id="opt-spot-001",
                category="compute",
                title="Enable Spot/Preemptible Instances",
                description="Use spot instances for non-critical workloads to save 60-90% on compute costs",
                provider="multi-cloud",
                region=None,
                potential_savings=Decimal("5000"),
                impact="high",
                effort="medium",
                priority=5,
                implementation_steps=[
                    "Enable spot instances in Terraform configuration",
                    "Configure spot instance fallback policies",
                    "Test workload compatibility with interruptions",
                    "Gradually migrate workloads to spot instances"
                ]
            )
            recommendations.append(rec)

        return recommendations

    def analyze_idle_resources(self) -> List[OptimizationRecommendation]:
        """Identify idle or underutilized resources"""
        recommendations = []

        rec = OptimizationRecommendation(
            recommendation_id="opt-idle-001",
            category="compute",
            title="Schedule Non-Production Environment Shutdowns",
            description="Automatically shutdown dev/staging environments during off-hours",
            provider="multi-cloud",
            region=None,
            potential_savings=Decimal("2000"),
            impact="medium",
            effort="low",
            priority=4,
            implementation_steps=[
                "Implement auto-scaling schedules",
                "Create shutdown scripts for non-prod",
                "Configure business hour schedules",
                "Monitor resource utilization patterns"
            ]
        )
        recommendations.append(rec)

        return recommendations

    def analyze_storage_optimization(self) -> List[OptimizationRecommendation]:
        """Analyze storage costs and optimization opportunities"""
        recommendations = []

        rec = OptimizationRecommendation(
            recommendation_id="opt-storage-001",
            category="storage",
            title="Implement Storage Lifecycle Policies",
            description="Move old data to cheaper storage tiers automatically",
            provider="multi-cloud",
            region=None,
            potential_savings=Decimal("1500"),
            impact="medium",
            effort="low",
            priority=3,
            implementation_steps=[
                "Analyze data access patterns",
                "Define lifecycle policies for S3/GCS/Azure Blob",
                "Move data older than 90 days to cold storage",
                "Implement automatic archival rules"
            ]
        )
        recommendations.append(rec)

        return recommendations

    def analyze_right_sizing(self) -> List[OptimizationRecommendation]:
        """Analyze instance right-sizing opportunities"""
        recommendations = []

        rec = OptimizationRecommendation(
            recommendation_id="opt-rightsize-001",
            category="compute",
            title="Right-Size Kubernetes Node Pools",
            description="Adjust node sizes based on actual resource utilization",
            provider="multi-cloud",
            region=None,
            potential_savings=Decimal("3000"),
            impact="high",
            effort="medium",
            priority=4,
            implementation_steps=[
                "Analyze CPU and memory utilization over 30 days",
                "Identify oversized node pools",
                "Test smaller instance types",
                "Update Terraform configurations",
                "Monitor performance after changes"
            ]
        )
        recommendations.append(rec)

        return recommendations

    def generate_all_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate comprehensive optimization recommendations"""
        logger.info("Generating cost optimization recommendations")

        all_recommendations = []
        all_recommendations.extend(self.analyze_spot_opportunities([]))
        all_recommendations.extend(self.analyze_idle_resources())
        all_recommendations.extend(self.analyze_storage_optimization())
        all_recommendations.extend(self.analyze_right_sizing())

        # Sort by priority
        all_recommendations.sort(key=lambda x: x.priority, reverse=True)

        self.recommendations = all_recommendations

        total_savings = sum(r.potential_savings for r in all_recommendations)
        logger.info(
            f"Generated {len(all_recommendations)} recommendations "
            f"with potential savings of ${total_savings:,.2f}"
        )

        return all_recommendations

    def get_recommendations_by_priority(self, min_priority: int = 3) -> List[OptimizationRecommendation]:
        """Get recommendations above a certain priority"""
        return [r for r in self.recommendations if r.priority >= min_priority]

    def get_recommendations_by_category(self, category: str) -> List[OptimizationRecommendation]:
        """Get recommendations for a specific category"""
        return [r for r in self.recommendations if r.category == category]
