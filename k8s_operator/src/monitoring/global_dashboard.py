"""
Global Dashboard

Provides unified dashboard configuration for Grafana showing all regions.
"""

import json
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlobalDashboard:
    """
    Global Dashboard Generator

    Creates Grafana dashboard JSON for multi-region monitoring.
    """

    def __init__(self, config: Dict):
        self.config = config

    def generate_dashboard(self) -> Dict:
        """Generate Grafana dashboard JSON"""
        dashboard = {
            "title": "Multi-Region ML Platform",
            "tags": ["multi-region", "ml-platform"],
            "timezone": "UTC",
            "panels": []
        }

        # Add panels
        dashboard["panels"].extend(self._create_overview_panels())
        dashboard["panels"].extend(self._create_region_panels())
        dashboard["panels"].extend(self._create_cost_panels())

        return dashboard

    def _create_overview_panels(self) -> List[Dict]:
        """Create overview panels"""
        return [
            {
                "id": 1,
                "title": "Global Request Rate",
                "type": "graph",
                "targets": [{
                    "expr": "sum(multiregion_request_rate)",
                    "legendFormat": "Total Requests/sec"
                }]
            },
            {
                "id": 2,
                "title": "Global Error Rate",
                "type": "graph",
                "targets": [{
                    "expr": "sum(multiregion_error_rate) / sum(multiregion_request_rate)",
                    "legendFormat": "Error Rate"
                }]
            }
        ]

    def _create_region_panels(self) -> List[Dict]:
        """Create per-region panels"""
        panels = []
        panel_id = 10

        for region in self.config.get('regions', []):
            region_name = region['name']

            panels.append({
                "id": panel_id,
                "title": f"{region_name} - Latency",
                "type": "graph",
                "targets": [
                    {
                        "expr": f'multiregion_latency_ms{{region="{region_name}"}}',
                        "legendFormat": "{{percentile}}"
                    }
                ]
            })

            panel_id += 1

        return panels

    def _create_cost_panels(self) -> List[Dict]:
        """Create cost tracking panels"""
        return [
            {
                "id": 100,
                "title": "Daily Cost by Provider",
                "type": "bargauge",
                "targets": [{
                    "expr": "multiregion_daily_cost",
                    "legendFormat": "{{provider}}"
                }]
            }
        ]

    def export_to_file(self, output_path: str):
        """Export dashboard to file"""
        dashboard = self.generate_dashboard()

        with open(output_path, 'w') as f:
            json.dump(dashboard, f, indent=2)

        logger.info(f"Exported dashboard to {output_path}")
