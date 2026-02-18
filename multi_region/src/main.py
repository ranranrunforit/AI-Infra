import asyncio
import logging
import os
import signal
import sys
from typing import Dict, Any
import yaml

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.failover.failover_controller import FailoverController
from src.replication.model_replicator import ModelReplicator
from src.replication.data_sync import DataSync
from src.cost.cost_analyzer import CostAnalyzer
from src.monitoring.metrics_aggregator import MetricsAggregator
from src.monitoring.alerting import AlertManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MultiRegionPlatform")

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file or environment"""
    config = {
        # Regions must be dicts with 'name', 'provider', 'endpoint', 'k8s_context'
        "regions": [
            {
                "name": os.getenv("REGION_1_NAME", "us-west-2"),
                "provider": "aws",
                "endpoint": os.getenv("REGION_1_ENDPOINT", "localhost"),
                "k8s_context": os.getenv("REGION_1_K8S_CONTEXT", "local"),
                "prometheus_url": os.getenv("REGION_1_PROMETHEUS_URL", "http://localhost:9091"),
            },
            {
                "name": os.getenv("REGION_2_NAME", "eu-west-1"),
                "provider": "gcp",
                "endpoint": os.getenv("REGION_2_ENDPOINT", "localhost"),
                "k8s_context": os.getenv("REGION_2_K8S_CONTEXT", "local"),
                "prometheus_url": os.getenv("REGION_2_PROMETHEUS_URL", "http://localhost:9091"),
            },
            {
                "name": os.getenv("REGION_3_NAME", "ap-south-1"),
                "provider": "azure",
                "endpoint": os.getenv("REGION_3_ENDPOINT", "localhost"),
                "k8s_context": os.getenv("REGION_3_K8S_CONTEXT", "local"),
                "prometheus_url": os.getenv("REGION_3_PROMETHEUS_URL", "http://localhost:9091"),
            },
        ],
        "primary_region": os.getenv("PRIMARY_REGION", "us-west-2"),
        "failover_enabled": os.getenv("FAILOVER_ENABLED", "true").lower() == "true",
        "aws_region": os.getenv("AWS_REGION", "us-west-2"),
        "gcp_project_id": os.getenv("GCP_PROJECT_ID", ""),
        "gcp_billing_table": os.getenv("GCP_BILLING_TABLE", ""),
        "azure_subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID", ""),
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    config.update(file_config)
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            
    return config

async def shutdown(signal, loop):
    """Cleanup tasks tied to the service's shutdown."""
    logger.info(f"Received exit signal {signal.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    
    [task.cancel() for task in tasks]
    
    logger.info(f"Cancelling {len(tasks)} outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

async def main():
    config = load_config()
    logger.info("Starting Multi-Region ML Platform Services...")

    # Initialize Services
    try:
        # 5. Metrics Aggregator (Initialize first to get registry)
        metrics_aggregator = MetricsAggregator(config)
        
        # Share registry with other services
        config['registry'] = metrics_aggregator.registry
        
        # 6. Alert Manager
        alert_manager = AlertManager(config)

        # 1. FailoverController (with AlertManager)
        failover_controller = FailoverController(config, alert_manager=alert_manager)
        
        # 2. Model Replicator
        model_replicator = ModelReplicator(config, registry=metrics_aggregator.registry)
        
        # 3. Data Sync
        data_sync = DataSync(config)
        
        # 4. Cost Analyzer
        cost_analyzer = CostAnalyzer(config)
        
    except Exception as e:
        logger.critical(f"Failed to initialize services: {e}")
        return

    # Create background tasks
    tasks = [
        asyncio.create_task(failover_controller.continuous_monitoring(), name="FailoverMonitor"),
        asyncio.create_task(model_replicator.continuous_replication(), name="ModelReplication"),
        asyncio.create_task(data_sync.continuous_sync(), name="DataSync"),
        # Cost analyzer periodic report
        asyncio.create_task(periodic_cost_report(cost_analyzer), name="CostReport"),
        asyncio.create_task(metrics_aggregator.continuous_collection(), name="MetricsCollection"),
        # Start Web Servers for Health (8080) & Metrics (9090)
        asyncio.create_task(start_web_servers(metrics_aggregator), name="WebServers")
    ]

    # Handle shutdown signals
    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, lambda s=s: asyncio.create_task(shutdown(s, loop)))
        except NotImplementedError:
            pass

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        logger.info("Services stopped.")

async def start_web_servers(metrics_aggregator: MetricsAggregator, health_port: int = 8080, metrics_port: int = 9090):
    """Start web servers for health checks and metrics"""
    from aiohttp import web
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

    async def health_check(request):
        return web.Response(text="OK", status=200)

    async def metrics(request):
        data = generate_latest(metrics_aggregator.registry)
        return web.Response(body=data, content_type=CONTENT_TYPE_LATEST)

    # Health Server
    health_app = web.Application()
    health_app.router.add_get('/health', health_check)
    health_app.router.add_get('/ready', health_check)
    
    health_runner = web.AppRunner(health_app)
    await health_runner.setup()
    health_site = web.TCPSite(health_runner, '0.0.0.0', health_port)
    logger.info(f"Starting health server on port {health_port}")
    await health_site.start()

    # Metrics Server
    metrics_app = web.Application()
    metrics_app.router.add_get('/metrics', metrics)
    
    metrics_runner = web.AppRunner(metrics_app)
    await metrics_runner.setup()
    metrics_site = web.TCPSite(metrics_runner, '0.0.0.0', metrics_port)
    logger.info(f"Starting metrics server on port {metrics_port}")
    await metrics_site.start()

    # Keep running until cancelled
    try:
        while True:
            await asyncio.sleep(3600)
    finally:
        await health_runner.cleanup()
        await metrics_runner.cleanup()

async def periodic_cost_report(analyzer: CostAnalyzer):
    """Generate a daily cost report"""
    while True:
        try:
            logger.info("Generating daily cost report...")
            # Use yesterday as range
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            costs = await analyzer.get_costs(start_date, end_date)
            report = await analyzer.generate_report(start_date, end_date)
            logger.info(f"Estimated daily cost: ${report.total_cost}")
            
            # Wait 24 hours (or less for demo)
            await asyncio.sleep(86400) 
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cost report error: {e}")
            await asyncio.sleep(3600)

if __name__ == "__main__":
    if os.name == 'nt':
        # Windows specific event loop policy
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
