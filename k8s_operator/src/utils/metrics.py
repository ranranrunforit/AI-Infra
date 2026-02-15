"""
Prometheus metrics for the operator.
"""

from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, generate_latest
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class OperatorMetrics:
    """
    Prometheus metrics for the TrainingJob operator.
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize operator metrics.

        Args:
            registry: Prometheus registry to use (default: global registry)
        """
        self.registry = registry

        # Reconciliation metrics
        self.reconciliation_total = Counter(
            'trainingjob_reconciliation_total',
            'Total number of reconciliations',
            ['namespace', 'training_job', 'result'],
            registry=registry
        )

        self.reconciliation_duration = Histogram(
            'trainingjob_reconciliation_duration_seconds',
            'Reconciliation duration in seconds',
            ['namespace', 'training_job'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=registry
        )

        self.reconciliation_errors = Counter(
            'trainingjob_reconciliation_errors_total',
            'Total number of reconciliation errors',
            ['namespace', 'training_job', 'error_type'],
            registry=registry
        )

        # TrainingJob status metrics
        self.training_jobs_total = Gauge(
            'trainingjob_total',
            'Total number of training jobs',
            ['namespace', 'state'],
            registry=registry
        )

        self.training_job_progress = Gauge(
            'trainingjob_progress_percent',
            'Training job progress percentage',
            ['namespace', 'training_job'],
            registry=registry
        )

        self.training_job_epoch = Gauge(
            'trainingjob_current_epoch',
            'Current training epoch',
            ['namespace', 'training_job'],
            registry=registry
        )

        self.training_job_duration = Gauge(
            'trainingjob_duration_seconds',
            'Training job duration in seconds',
            ['namespace', 'training_job', 'state'],
            registry=registry
        )

        # Resource allocation metrics
        self.allocated_gpus = Gauge(
            'trainingjob_allocated_gpus',
            'Number of GPUs allocated to training jobs',
            ['namespace', 'training_job'],
            registry=registry
        )

        self.allocated_workers = Gauge(
            'trainingjob_allocated_workers',
            'Number of workers allocated to training jobs',
            ['namespace', 'training_job'],
            registry=registry
        )

        # Training metrics
        self.training_loss = Gauge(
            'trainingjob_loss',
            'Current training loss',
            ['namespace', 'training_job'],
            registry=registry
        )

        self.training_accuracy = Gauge(
            'trainingjob_accuracy',
            'Current training accuracy',
            ['namespace', 'training_job'],
            registry=registry
        )

        self.gpu_utilization = Gauge(
            'trainingjob_gpu_utilization_percent',
            'GPU utilization percentage',
            ['namespace', 'training_job'],
            registry=registry
        )

        # Checkpoint metrics
        self.checkpoint_created = Counter(
            'trainingjob_checkpoint_created_total',
            'Total number of checkpoints created',
            ['namespace', 'training_job'],
            registry=registry
        )

        self.checkpoint_size_bytes = Gauge(
            'trainingjob_checkpoint_size_bytes',
            'Size of the latest checkpoint in bytes',
            ['namespace', 'training_job'],
            registry=registry
        )

        # Job lifecycle metrics
        self.job_created = Counter(
            'trainingjob_created_total',
            'Total number of training jobs created',
            ['namespace'],
            registry=registry
        )

        self.job_completed = Counter(
            'trainingjob_completed_total',
            'Total number of training jobs completed',
            ['namespace', 'result'],
            registry=registry
        )

        self.job_failed = Counter(
            'trainingjob_failed_total',
            'Total number of training jobs failed',
            ['namespace', 'reason'],
            registry=registry
        )

        self.job_restarted = Counter(
            'trainingjob_restarted_total',
            'Total number of training jobs restarted',
            ['namespace', 'training_job'],
            registry=registry
        )

        # Kubernetes resource metrics
        self.k8s_jobs_created = Counter(
            'trainingjob_k8s_jobs_created_total',
            'Total number of Kubernetes Jobs created',
            ['namespace'],
            registry=registry
        )

        self.k8s_services_created = Counter(
            'trainingjob_k8s_services_created_total',
            'Total number of Kubernetes Services created',
            ['namespace'],
            registry=registry
        )

        # Operator health metrics
        self.operator_up = Gauge(
            'trainingjob_operator_up',
            'Operator is running',
            registry=registry
        )

        self.operator_watch_events = Counter(
            'trainingjob_operator_watch_events_total',
            'Total number of watch events processed',
            ['event_type'],
            registry=registry
        )

        logger.info("Initialized operator metrics")

    def record_reconciliation(
        self,
        namespace: str,
        training_job: str,
        duration: float,
        result: str
    ) -> None:
        """
        Record a reconciliation event.

        Args:
            namespace: Namespace of the training job
            training_job: Name of the training job
            duration: Duration in seconds
            result: Result of reconciliation (success, error, skip)
        """
        self.reconciliation_total.labels(
            namespace=namespace,
            training_job=training_job,
            result=result
        ).inc()

        self.reconciliation_duration.labels(
            namespace=namespace,
            training_job=training_job
        ).observe(duration)

    def record_reconciliation_error(
        self,
        namespace: str,
        training_job: str,
        error_type: str
    ) -> None:
        """
        Record a reconciliation error.

        Args:
            namespace: Namespace of the training job
            training_job: Name of the training job
            error_type: Type of error
        """
        self.reconciliation_errors.labels(
            namespace=namespace,
            training_job=training_job,
            error_type=error_type
        ).inc()

    def update_training_job_count(self, namespace: str, state: str, count: int) -> None:
        """
        Update the count of training jobs in a given state.

        Args:
            namespace: Namespace
            state: Job state
            count: Number of jobs
        """
        self.training_jobs_total.labels(namespace=namespace, state=state).set(count)

    def update_training_progress(
        self,
        namespace: str,
        training_job: str,
        progress: float,
        epoch: int
    ) -> None:
        """
        Update training progress metrics.

        Args:
            namespace: Namespace
            training_job: Training job name
            progress: Progress percentage (0-100)
            epoch: Current epoch
        """
        self.training_job_progress.labels(
            namespace=namespace,
            training_job=training_job
        ).set(progress)

        self.training_job_epoch.labels(
            namespace=namespace,
            training_job=training_job
        ).set(epoch)

    def update_training_metrics(
        self,
        namespace: str,
        training_job: str,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        gpu_util: Optional[float] = None
    ) -> None:
        """
        Update training metrics.

        Args:
            namespace: Namespace
            training_job: Training job name
            loss: Current loss value
            accuracy: Current accuracy value
            gpu_util: GPU utilization percentage
        """
        if loss is not None:
            self.training_loss.labels(
                namespace=namespace,
                training_job=training_job
            ).set(loss)

        if accuracy is not None:
            self.training_accuracy.labels(
                namespace=namespace,
                training_job=training_job
            ).set(accuracy)

        if gpu_util is not None:
            self.gpu_utilization.labels(
                namespace=namespace,
                training_job=training_job
            ).set(gpu_util)

    def record_checkpoint_created(self, namespace: str, training_job: str) -> None:
        """
        Record checkpoint creation.

        Args:
            namespace: Namespace
            training_job: Training job name
        """
        self.checkpoint_created.labels(
            namespace=namespace,
            training_job=training_job
        ).inc()

    def record_job_created(self, namespace: str) -> None:
        """
        Record training job creation.

        Args:
            namespace: Namespace
        """
        self.job_created.labels(namespace=namespace).inc()

    def record_job_completed(self, namespace: str, result: str) -> None:
        """
        Record training job completion.

        Args:
            namespace: Namespace
            result: Completion result (success, failed)
        """
        self.job_completed.labels(namespace=namespace, result=result).inc()

    def set_operator_up(self, up: bool = True) -> None:
        """
        Set operator health status.

        Args:
            up: True if operator is healthy, False otherwise
        """
        self.operator_up.set(1 if up else 0)


# Global metrics instance
metrics = OperatorMetrics()
