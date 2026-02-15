"""
StatusController manages status updates for training jobs.
"""

from typing import Dict, Any
from datetime import datetime
import logging
import kopf

from ..utils import get_k8s_client, metrics

logger = logging.getLogger(__name__)


class StatusController:
    """
    Controller for updating TrainingJob status based on actual training progress.
    """

    def __init__(self, k8s_client=None):
        """
        Initialize the StatusController.
        
        Args:
            k8s_client: Optional K8sClient instance
        """
        self.k8s_client = k8s_client or get_k8s_client()

    async def update_training_status(
        self,
        name: str,
        namespace: str,
        spec: Dict[str, Any],
        status: Dict[str, Any],
        klogger: kopf.Logger
    ) -> Dict[str, Any]:
        """
        Update training status by monitoring job progress.

        This method:
        1. Queries the Kubernetes Job status
        2. Retrieves metrics from training pods (via logs or metrics endpoint)
        3. Updates TrainingJob status with current progress
        4. Publishes metrics to Prometheus

        Args:
            name: TrainingJob name
            namespace: Namespace
            spec: TrainingJob spec
            status: Current status
            klogger: Kopf logger

        Returns:
            Updated status
        """
        try:
            job_name = f"{name}-training"
            k8s_job = self.k8s_client.get_job(job_name, namespace)

            if not k8s_job:
                klogger.warning(f"Job {job_name} not found")
                return status

            # Get pods for the job
            label_selector = f"training-job={name}"
            pods = self.k8s_client.list_pods(namespace, label_selector=label_selector)

            # Extract metrics from pods
            # In production, this would parse logs or query metrics endpoints
            training_metrics = self._extract_training_metrics(pods, klogger)

            # Update status with metrics
            new_status = status.copy()

            if training_metrics:
                new_status['metrics'] = training_metrics
                new_status['progress'] = f"{training_metrics.get('progress', 0)}%"
                new_status['currentEpoch'] = training_metrics.get('epoch', 0)

                # Update Prometheus metrics
                metrics.update_training_progress(
                    namespace,
                    name,
                    training_metrics.get('progress', 0),
                    training_metrics.get('epoch', 0)
                )

                metrics.update_training_metrics(
                    namespace,
                    name,
                    loss=training_metrics.get('loss'),
                    accuracy=training_metrics.get('accuracy'),
                    gpu_util=training_metrics.get('gpuUtilization')
                )

            # Update duration
            start_time = status.get('startTime')
            if start_time:
                new_status['duration'] = self._calculate_duration(start_time)

            return new_status

        except Exception as e:
            klogger.error(f"Failed to update status: {e}", exc_info=True)
            return status

    def _extract_training_metrics(
        self,
        pods: Any,
        klogger: kopf.Logger
    ) -> Dict[str, Any]:
        """
        Extract training metrics from pod logs or metrics endpoints.

        In production, this would:
        1. Query metrics endpoint on each pod (e.g., /metrics)
        2. Parse structured logs for training progress
        3. Aggregate metrics across all workers
        4. Return average or sum as appropriate
        """
        import json
        import re

        metrics_data = {
            'epoch': 0,
            'progress': 0,
            'loss': 0.0,
            'accuracy': 0.0,
            'gpuUtilization': 0.0,
        }

        if not pods or not pods.items:
            return metrics_data

        # We'll look at the first worker's logs for simplicity in this implementation
        # In a real distributed setting, we might aggregate or look at rank-0
        pod = pods.items[0]
        pod_name = pod.metadata.name
        namespace = pod.metadata.namespace

        try:
            # Get recent logs
            logs = self.k8s_client.get_pod_logs(
                name=pod_name,
                namespace=namespace,
                tail_lines=50
            )

            if not logs:
                return metrics_data

            # Parse logs for JSON metrics
            # Expected format: {"loss": 0.5, "accuracy": 0.9, "epoch": 1, "progress": 10, "gpu_util": 85}
            for line in reversed(logs.splitlines()):
                try:
                    # Look for JSON-like structure
                    json_match = re.search(r'\{.*\}', line)
                    if json_match:
                        data = json.loads(json_match.group(0))
                        
                        # Update metrics if keys exist
                        if 'epoch' in data:
                            metrics_data['epoch'] = int(data['epoch'])
                        if 'progress' in data:
                            metrics_data['progress'] = float(data['progress'])
                        if 'loss' in data:
                            metrics_data['loss'] = float(data['loss'])
                        if 'accuracy' in data:
                            metrics_data['accuracy'] = float(data['accuracy'])
                        if 'gpu_util' in data:
                            metrics_data['gpuUtilization'] = float(data['gpu_util'])
                        
                        # Once we find the latest metric line, break (assuming logs are chronological)
                        # Since we iterate reversed, the first match is the latest
                        break
                except (json.JSONDecodeError, ValueError):
                    continue

        except Exception as e:
            klogger.warning(f"Failed to extract metrics from pod {pod_name}: {e}")

        return metrics_data

    def _calculate_duration(self, start_time: str) -> str:
        """
        Calculate training duration from start time.

        Args:
            start_time: ISO format timestamp

        Returns:
            Duration string (e.g., "2h 30m")
        """
        from dateutil import parser

        start = parser.isoparse(start_time)
        now = datetime.utcnow()
        delta = now - start.replace(tzinfo=None)

        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
