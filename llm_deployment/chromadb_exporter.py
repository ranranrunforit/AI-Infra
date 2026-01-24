"""
ChromaDB Prometheus Exporter (Simplified)
Exposes ChromaDB metrics in Prometheus format using HTTP API
"""
import time
import requests
from prometheus_client import start_http_server, Gauge, Info
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define metrics
chromadb_up = Gauge('chromadb_up', 'ChromaDB is up and reachable')
chromadb_collections_total = Gauge('chromadb_collections_total', 'Total number of collections')
chromadb_documents_total = Gauge('chromadb_documents_total', 'Total number of documents', ['collection'])
chromadb_heartbeat = Gauge('chromadb_heartbeat', 'ChromaDB heartbeat timestamp')
chromadb_info = Info('chromadb', 'ChromaDB information')

class ChromaDBExporter:
    def __init__(self, chromadb_host='chromadb', chromadb_port=8000):
        self.base_url = f"http://{chromadb_host}:{chromadb_port}"
        self.api_url = f"{self.base_url}/api/v1"
        
    def check_health(self):
        """Check if ChromaDB is reachable"""
        try:
            response = requests.get(f"{self.api_url}/heartbeat", timeout=5)
            if response.status_code == 200:
                chromadb_up.set(1)
                heartbeat_data = response.json()
                chromadb_heartbeat.set(heartbeat_data.get('nanosecond_heartbeat', 0))
                logger.debug(f"ChromaDB heartbeat: {heartbeat_data}")
                return True
            else:
                chromadb_up.set(0)
                return False
        except Exception as e:
            logger.error(f"Failed to reach ChromaDB: {e}")
            chromadb_up.set(0)
            return False
    
    def get_version(self):
        """Get ChromaDB version info"""
        try:
            response = requests.get(f"{self.api_url}/version", timeout=5)
            if response.status_code == 200:
                version_info = response.json()
                chromadb_info.info({'version': str(version_info)})
                logger.debug(f"ChromaDB version: {version_info}")
        except Exception as e:
            logger.warning(f"Could not get version info: {e}")
    
    def get_collections(self):
        """Get list of collections"""
        try:
            response = requests.get(f"{self.api_url}/collections", timeout=10)
            if response.status_code == 200:
                collections = response.json()
                chromadb_collections_total.set(len(collections))
                logger.debug(f"Found {len(collections)} collections")
                return collections
            return []
        except Exception as e:
            logger.error(f"Failed to get collections: {e}")
            return []
    
    def get_collection_count(self, collection_id):
        """Get document count for a collection"""
        try:
            response = requests.get(
                f"{self.api_url}/collections/{collection_id}/count",
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return 0
        except Exception as e:
            logger.warning(f"Failed to get count for collection {collection_id}: {e}")
            return 0
    
    def collect_metrics(self):
        """Collect all metrics from ChromaDB"""
        try:
            # Check if ChromaDB is up
            if not self.check_health():
                logger.warning("ChromaDB is not reachable")
                return
            
            # Get version info (once)
            self.get_version()
            
            # Get collections
            collections = self.get_collections()
            
            # Get counts for each collection
            for collection in collections:
                try:
                    collection_name = collection.get('name', 'unknown')
                    collection_id = collection.get('id', '')
                    
                    if collection_id:
                        count = self.get_collection_count(collection_id)
                        chromadb_documents_total.labels(collection=collection_name).set(count)
                        logger.debug(f"Collection {collection_name}: {count} documents")
                except Exception as e:
                    logger.error(f"Error processing collection {collection.get('name', 'unknown')}: {e}")
                    
            logger.info(f"Successfully collected metrics for {len(collections)} collections")
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            chromadb_up.set(0)

def main():
    import os
    
    # Get configuration from environment
    chromadb_host = os.getenv('CHROMADB_HOST', 'chromadb')
    chromadb_port = int(os.getenv('CHROMADB_PORT', '8000'))
    exporter_port = int(os.getenv('EXPORTER_PORT', '9091'))
    scrape_interval = int(os.getenv('SCRAPE_INTERVAL', '15'))
    
    logger.info(f"Starting ChromaDB exporter on port {exporter_port}")
    logger.info(f"Connecting to ChromaDB at {chromadb_host}:{chromadb_port}")
    logger.info(f"Scrape interval: {scrape_interval} seconds")
    
    # Start Prometheus metrics server
    start_http_server(exporter_port)
    logger.info(f"Metrics available at http://0.0.0.0:{exporter_port}/metrics")
    
    # Create exporter
    exporter = ChromaDBExporter(chromadb_host, chromadb_port)
    
    # Collect metrics periodically
    while True:
        try:
            exporter.collect_metrics()
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
        
        time.sleep(scrape_interval)

if __name__ == '__main__':
    main()