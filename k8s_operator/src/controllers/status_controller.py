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

    def __init__(self):
        """Initialize the StatusController."""
        self.k8s_client = get_k8s_client()

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

        Args:
            pods: Pod list
            klogger: Kopf logger

        Returns:
            Training metrics dictionary
        """
        # Placeholder implementation
        # In production, parse actual logs or query metrics endpoints
        metrics = {
            'epoch': 0,
            'progress': 0,
            'loss': 0.0,
            'accuracy': 0.0,
            'gpuUtilization': 0.0,
        }

        return metrics

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
