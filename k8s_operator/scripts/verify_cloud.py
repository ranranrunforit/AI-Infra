import asyncio
import os
import sys
import logging
import uuid
from datetime import datetime
from typing import Dict
import json

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.cost.cost_analyzer import CostAnalyzer, CostData
from src.replication.model_replicator import ModelReplicator
from src.failover.dns_updater import DNSUpdater

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CloudVerification")

def get_verification_config() -> Dict:
    """Get config from env vars"""
    config = {
        "aws_region": os.getenv("AWS_REGION", "us-west-2"),
        "gcp_project_id": os.getenv("GCP_PROJECT_ID"),
        "gcp_billing_table": os.getenv("GCP_BILLING_TABLE"),
        "azure_subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
        # Add bucket names if you have them, or expect them in config setup
    }
    
    # Check for missing critical env vars
    missing = []
    if not config["gcp_project_id"]: missing.append("GCP_PROJECT_ID")
    if not config["azure_subscription_id"]: missing.append("AZURE_SUBSCRIPTION_ID")
    
    if missing:
        logger.warning(f"Missing environment variables: {', '.join(missing)}. Some tests will fail.")
        
    return config

async def verify_costs(config: Dict):
    """Verify cost retrieval from valid cloud providers"""
    logger.info("--- Verifying Cost Analysis (Real Cloud APIs) ---")
    analyzer = CostAnalyzer(config)
    
    start_date = datetime.now().strftime('%Y-%m-%01') # Start of current month
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        logger.info(f"Fetching costs from {start_date} to {end_date}...")
        costs = await analyzer.get_costs(start_date, end_date)
        
        gcp_count = len([c for c in costs if c.provider == 'gcp'])
        azure_count = len([c for c in costs if c.provider == 'azure'])
        aws_count = len([c for c in costs if c.provider == 'aws'])
        
        logger.info(f"Cost records found: AWS={aws_count}, GCP={gcp_count}, Azure={azure_count}")
        
        if not costs:
            logger.warning("No cost data returned. Check credentials or if the account has usage.")
        else:
            logger.info("Successfully retrieved cost data!")
            
    except Exception as e:
        logger.error(f"Cost verification failed: {e}")

async def verify_replication_setup(config: Dict):
    """Check if replication adapters can list objects"""
    logger.info("--- Verifying Replication Access ---")
    try:
        replicator = ModelReplicator(config)
        
        if not replicator.adapters:
            logger.warning("No replication adapters configured. Check 'regions' config.")
            return

        logger.info("Checking adapter connectivity...")
        for region, adapter in replicator.adapters.items():
            try:
                # Try to list root or models/
                # Note: This might fail if bucket doesn't exist, which is expected for verification
                logger.info(f"[{region}] Checking access to bucket: {adapter.bucket_name}...")
                objs = await adapter.list_objects("models/")
                logger.info(f"[{region}] Connection successful. Found {len(objs)} objects.")
            except Exception as e:
                logger.warning(f"[{region}] Failed to list objects (check credentials/bucket existence): {e}")
                
    except Exception as e:
        logger.error(f"Replication setup verification failed: {e}")

async def verify_dns_access(config: Dict):
    """Verify Route53 access"""
    logger.info("--- Verifying DNS (Route53) Access ---")
    
    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        logger.warning("Skipping DNS check: AWS credentials not found.")
        return

    try:
        # Require a domain name for DNSUpdater
        test_config = config.copy()
        if 'domain_name' not in test_config:
            test_config['domain_name'] = 'example.com' # Placeholder for init
            
        updater = DNSUpdater(test_config)
        
        # specific check: list hosted zones
        logger.info("Listing hosted zones...")
        zones = await asyncio.to_thread(updater.route53.list_hosted_zones)
        zone_count = len(zones.get('HostedZones', []))
        logger.info(f"Successfully connected to Route53. Found {zone_count} hosted zones.")
        
    except Exception as e:
        logger.error(f"DNS verification failed: {e}")

async def main():
    logger.info("Starting Cloud Verification Script")
    logger.info("Ensure you have AWS, GCP (GOOGLE_APPLICATION_CREDENTIALS), and Azure (AZURE_SUBSCRIPTION_ID + Identity) credentials set.")
    
    config = get_verification_config()
    
    await verify_costs(config)
    await verify_replication_setup(config)
    await verify_dns_access(config)
    
    logger.info("Verification Complete.")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
