import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from src.controllers.status_controller import StatusController

class TestStatusController(unittest.TestCase):
    def setUp(self):
        self.mock_k8s_client = MagicMock()
        self.controller = StatusController(k8s_client=self.mock_k8s_client)
        self.logger = MagicMock()

    def test_extract_training_metrics(self):
        # Mock pod list
        mock_pod = MagicMock()
        mock_pod.metadata.name = 'worker-0'
        mock_pod.metadata.namespace = 'default'
        mock_pods = MagicMock()
        mock_pods.items = [mock_pod]

        # Mock logs
        logs = """
INFO:root:Starting training
{"epoch": 1, "loss": 0.5, "accuracy": 0.8}
INFO:root:Epoch 1 completed
{"epoch": 2, "loss": 0.3, "accuracy": 0.9, "progress": 50.0}
        """
        self.mock_k8s_client.get_pod_logs.return_value = logs

        metrics = self.controller._extract_training_metrics(mock_pods, self.logger)

        self.assertEqual(metrics['epoch'], 2)
        self.assertEqual(metrics['loss'], 0.3)
        self.assertEqual(metrics['accuracy'], 0.9)
        self.assertEqual(metrics['progress'], 50.0)

    def test_extract_metrics_no_logs(self):
        mock_pods = MagicMock()
        mock_pods.items = [MagicMock()]
        self.mock_k8s_client.get_pod_logs.return_value = ""

        metrics = self.controller._extract_training_metrics(mock_pods, self.logger)
        self.assertEqual(metrics['epoch'], 0)

if __name__ == '__main__':
    unittest.main()
