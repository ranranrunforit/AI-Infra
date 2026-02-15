"""
Cost Reporter

Generates cost reports in various formats (JSON, HTML, CSV).
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from decimal import Decimal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder for Decimal types"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


class CostReporter:
    """
    Cost Report Generator

    Creates formatted cost reports for stakeholders.
    """

    def __init__(self, config: Dict):
        self.config = config

    def generate_json_report(self, cost_report, recommendations: List) -> str:
        """Generate JSON format report"""
        report_data = {
            'report_id': cost_report.report_id,
            'generated_at': datetime.utcnow().isoformat(),
            'period': {
                'start_date': cost_report.start_date,
                'end_date': cost_report.end_date
            },
            'summary': {
                'total_cost': float(cost_report.total_cost),
                'currency': cost_report.currency
            },
            'breakdown': {
                'by_provider': {k: float(v) for k, v in cost_report.costs_by_provider.items()},
                'by_region': {k: float(v) for k, v in cost_report.costs_by_region.items()},
                'by_service': {k: float(v) for k, v in cost_report.costs_by_service.items()}
            },
            'anomalies': cost_report.anomalies,
            'trends': cost_report.trends,
            'recommendations': [
                {
                    'title': r.title,
                    'description': r.description,
                    'potential_savings': float(r.potential_savings),
                    'priority': r.priority
                }
                for r in recommendations
            ]
        }

        return json.dumps(report_data, indent=2, cls=DecimalEncoder)

    def generate_html_report(self, cost_report, recommendations: List) -> str:
        """Generate HTML format report"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Cost Report - {cost_report.report_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .summary {{ background-color: #f2f2f2; padding: 15px; margin: 20px 0; }}
        .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffc107; }}
    </style>
</head>
<body>
    <h1>Multi-Region ML Platform Cost Report</h1>
    <p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
    <p>Period: {cost_report.start_date} to {cost_report.end_date}</p>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Cost:</strong> ${cost_report.total_cost:,.2f} {cost_report.currency}</p>
    </div>

    <h2>Costs by Provider</h2>
    <table>
        <tr><th>Provider</th><th>Cost</th><th>Percentage</th></tr>
"""

        for provider, cost in cost_report.costs_by_provider.items():
            percentage = (cost / cost_report.total_cost * 100) if cost_report.total_cost > 0 else 0
            html += f"        <tr><td>{provider.upper()}</td><td>${cost:,.2f}</td><td>{percentage:.1f}%</td></tr>\n"

        html += """    </table>

    <h2>Cost Optimization Recommendations</h2>
"""

        for rec in recommendations[:5]:  # Top 5 recommendations
            html += f"""
    <div class="recommendation">
        <h3>{rec.title}</h3>
        <p>{rec.description}</p>
        <p><strong>Potential Savings:</strong> ${rec.potential_savings:,.2f}</p>
        <p><strong>Priority:</strong> {rec.priority}/5 | <strong>Effort:</strong> {rec.effort}</p>
    </div>
"""

        html += """
</body>
</html>
"""
        return html

    def generate_csv_report(self, cost_report) -> str:
        """Generate CSV format report"""
        csv_lines = [
            "Service,Region,Provider,Cost",
        ]

        # This would require the original cost_data, simplified here
        csv_lines.append(f"Total,All,All,{cost_report.total_cost}")

        for provider, cost in cost_report.costs_by_provider.items():
            csv_lines.append(f"Provider Total,All,{provider},{cost}")

        return "\n".join(csv_lines)
