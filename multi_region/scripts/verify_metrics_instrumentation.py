import asyncio
import sys
import unittest
from unittest.mock import MagicMock, AsyncMock

# Add src to path
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from prometheus_client import CollectorRegistry
from src.replication.model_replicator import ModelReplicator
from src.replication.data_sync import DataSync
from src.failover.failover_controller import FailoverController, FailoverEvent

class TestMetricsInstrumentation(unittest.TestCase):
    def setUp(self):
        self.registry = CollectorRegistry()
        self.config = {
            'aws_region': 'us-west-2',
            'gcp_project_id': 'test-project',
            'azure_subscription_id': 'test-sub',
            'registry': self.registry
        }

    def test_model_replicator_metrics(self):
        # Initialize
        replicator = ModelReplicator(self.config, registry=self.registry)
        
        # Check metrics exist in registry
        metrics = [m.name for m in self.registry.collect()]
        self.assertIn('model_replication_total', metrics)
        self.assertIn('model_replication_duration_seconds', metrics)
        
        # Simulate metric increment
        replicator.replication_counter.labels(
            source_region='us-west-2',
            target_region='us-east-1',
            status='success'
        ).inc()
        
        # Verify value
        val = self.registry.get_sample_value(
            'model_replication_total_total',
            {'source_region': 'us-west-2', 'target_region': 'us-east-1', 'status': 'success'}
        )
        self.assertEqual(val, 1.0)
        print("ModelReplicator metrics verified.")

    def test_data_sync_metrics(self):
        # Initialize
        syncer = DataSync(self.config)
        
        # Check metrics exist
        metrics = [m.name for m in self.registry.collect()]
        self.assertIn('data_sync_jobs_total', metrics)
        self.assertIn('data_files_synced_total', metrics)
        
        # Simulate metric increment
        syncer.sync_jobs_counter.labels(
            dataset_id='test-dataset',
            status='success'
        ).inc()
        
        val = self.registry.get_sample_value(
            'data_sync_jobs_total_total',
            {'dataset_id': 'test-dataset', 'status': 'success'}
        )
        self.assertEqual(val, 1.0)
        print("DataSync metrics verified.")

    def test_failover_controller_metrics(self):
        # Initialize
        controller = FailoverController(self.config)
        
        # Check metrics exist
        metrics = [m.name for m in self.registry.collect()]
        self.assertIn('failover_events_total', metrics)
        self.assertIn('region_health_status', metrics)
        
        # Simulate metric updates
        controller.failover_counter.labels(
            source_region='us-west-2',
            target_region='us-east-1',
            reason='outage',
            status='started'
        ).inc()
        
        controller.region_health_gauge.labels(region='us-west-2').set(1)
        
        # Verify values
        failover_val = self.registry.get_sample_value(
            'failover_events_total_total',
            {'source_region': 'us-west-2', 'target_region': 'us-east-1', 'reason': 'outage', 'status': 'started'}
        )
        self.assertEqual(failover_val, 1.0)
        
        health_val = self.registry.get_sample_value(
            'region_health_status',
            {'region': 'us-west-2'}
        )
        self.assertEqual(health_val, 1.0)
        print("FailoverController metrics verified.")

if __name__ == '__main__':
    unittest.main()
