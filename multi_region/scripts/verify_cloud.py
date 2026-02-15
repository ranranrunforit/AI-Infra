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
# Import other necessary classes if needed

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
        
        # Test listing from configured adapters
        # Note: This relies on adapters being correctly configured in replicator.__init__ 
        # which usually requires specific bucket names in the config.
        # Since we might not have 'bucket_names' in the basic config, this might skip or fail gracefully.
        
        logger.info("Checking adapter connectivity...")
        for region, adapter in replicator.adapters.items():
            try:
                # Try to list root or models/
                objs = await adapter.list_objects("models/")
                logger.info(f"[{region}] Connection successful. Found {len(objs)} objects.")
            except Exception as e:
                logger.warning(f"[{region}] Failed to list objects: {e}")
                
    except Exception as e:
        logger.error(f"Replication setup verification failed: {e}")

async def main():
    logger.info("Starting Cloud Verification Script")
    logger.info("Ensure you have AWS, GCP (GOOGLE_APPLICATION_CREDENTIALS), and Azure (AZURE_SUBSCRIPTION_ID + Identity) credentials set.")
    
    config = get_verification_config()
    
    await verify_costs(config)
    await verify_replication_setup(config)
    
    logger.info("Verification Complete.")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
