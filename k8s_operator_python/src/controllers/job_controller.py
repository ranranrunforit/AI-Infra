"""
JobController manages Kubernetes Job resources for training jobs.
"""

from typing import Dict, Any, Optional
from kubernetes import client
from datetime import datetime
import logging
import kopf

from ..utils import get_k8s_client, metrics
from ..resources.job_builder import JobBuilder

logger = logging.getLogger(__name__)


class JobController:
    """
    Controller for managing Kubernetes Job resources for ML training.
    """

    def __init__(self, k8s_client=None):
        """
        Initialize the JobController.
        
        Args:
            k8s_client: Optional K8sClient instance (for testing/DI)
        """
        self.k8s_client = k8s_client or get_k8s_client()
        self.job_builder = JobBuilder()

    async def create_training_resources(
        self,
        name: str,
        namespace: str,
        spec: Dict[str, Any],
        status: Dict[str, Any],
        klogger: kopf.Logger
    ) -> Dict[str, Any]:
        """
        Create Kubernetes resources for a training job.

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
            klogger.info(f"Creating training resources for {namespace}/{name}")

            # Build Kubernetes Job
            k8s_job = self.job_builder.build_job(name, namespace, spec)

            # Create the Job
            created_job = self.k8s_client.create_job(namespace, k8s_job)
            klogger.info(f"Created Kubernetes Job: {created_job.metadata.name}")

            # Create headless service for distributed training
            if spec.get('numWorkers', 1) > 1:
                service = self._build_headless_service(name, namespace, spec)
                self.k8s_client.create_service(namespace, service)
                klogger.info(f"Created headless service for distributed training")

            # Create ConfigMap with training configuration
            config_map = self._build_config_map(name, namespace, spec)
            self.k8s_client.create_config_map(namespace, config_map)

            # Update metrics
            metrics.k8s_jobs_created.labels(namespace=namespace).inc()
            metrics.allocated_workers.labels(
                namespace=namespace,
                training_job=name
            ).set(spec.get('numWorkers', 1))
            metrics.allocated_gpus.labels(
                namespace=namespace,
                training_job=name
            ).set(spec.get('numWorkers', 1) * spec.get('gpusPerWorker', 1))

            # Update status
            new_status = status.copy()
            new_status['state'] = 'Initializing'
            new_status['conditions'] = status.get('conditions', []) + [
                {
                    'type': 'ResourcesCreated',
                    'status': 'True',
                    'reason': 'JobCreated',
                    'message': f'Kubernetes Job {created_job.metadata.name} created',
                    'lastTransitionTime': datetime.utcnow().isoformat() + 'Z',
                }
            ]
            new_status['resources'] = {
                'allocatedGPUs': spec.get('numWorkers', 1) * spec.get('gpusPerWorker', 1),
                'allocatedNodes': 0,  # Will be updated when pods are scheduled
            }

            return new_status

        except Exception as e:
            klogger.error(f"Failed to create training resources: {e}", exc_info=True)
            new_status = status.copy()
            new_status['state'] = 'Failed'
            new_status['failureReason'] = 'ResourceCreationFailed'
            new_status['failureMessage'] = str(e)
            new_status['conditions'] = status.get('conditions', []) + [
                {
                    'type': 'Failed',
                    'status': 'True',
                    'reason': 'ResourceCreationFailed',
                    'message': str(e),
                    'lastTransitionTime': datetime.utcnow().isoformat() + 'Z',
                }
            ]
            return new_status

    async def check_resources_ready(
        self,
        name: str,
        namespace: str,
        spec: Dict[str, Any],
        status: Dict[str, Any],
        klogger: kopf.Logger
    ) -> Dict[str, Any]:
        """
        Check if training resources are ready.

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

            # Check job status
            job_status = k8s_job.status

            # Count active, succeeded, and failed pods
            active = job_status.active or 0
            succeeded = job_status.succeeded or 0
            failed = job_status.failed or 0

            klogger.info(f"Job status: active={active}, succeeded={succeeded}, failed={failed}")

            # Update status
            new_status = status.copy()
            new_status['workers'] = {
                'active': active,
                'succeeded': succeeded,
                'failed': failed,
                'pending': max(0, spec.get('numWorkers', 1) - active - succeeded - failed),
            }

            # Check if all workers are active (training started)
            if active >= spec.get('numWorkers', 1):
                new_status['state'] = 'Running'
                new_status['conditions'] = status.get('conditions', []) + [
                    {
                        'type': 'Running',
                        'status': 'True',
                        'reason': 'TrainingStarted',
                        'message': 'All workers are active, training has started',
                        'lastTransitionTime': datetime.utcnow().isoformat() + 'Z',
                    }
                ]
                klogger.info(f"Training started for {namespace}/{name}")
                metrics.update_training_job_count(namespace, 'Initializing', -1)
                metrics.update_training_job_count(namespace, 'Running', 1)

            # Check for failures
            if failed > 0:
                backoff_limit = spec.get('failurePolicy', {}).get('backoffLimit', 3)
                if failed >= backoff_limit:
                    new_status['state'] = 'Failed'
                    new_status['failureReason'] = 'TooManyFailures'
                    new_status['failureMessage'] = f'{failed} workers failed (backoff limit: {backoff_limit})'
                    new_status['conditions'] = status.get('conditions', []) + [
                        {
                            'type': 'Failed',
                            'status': 'True',
                            'reason': 'TooManyFailures',
                            'message': f'{failed} workers failed',
                            'lastTransitionTime': datetime.utcnow().isoformat() + 'Z',
                        }
                    ]
                    metrics.job_failed.labels(namespace=namespace, reason='TooManyFailures').inc()

            return new_status

        except Exception as e:
            klogger.error(f"Failed to check resources: {e}", exc_info=True)
            return status

    async def delete_training_resources(
        self,
        name: str,
        namespace: str,
        klogger: kopf.Logger
    ) -> None:
        """
        Delete Kubernetes resources for a training job.

        Args:
            name: TrainingJob name
            namespace: Namespace
            klogger: Kopf logger
        """
        try:
            job_name = f"{name}-training"
            service_name = f"{name}-headless"
            config_map_name = f"{name}-config"

            # Delete Job
            klogger.info(f"Deleting Job {job_name}")
            self.k8s_client.delete_job(job_name, namespace)

            # Delete Service
            klogger.info(f"Deleting Service {service_name}")
            self.k8s_client.delete_service(service_name, namespace)

            # Delete ConfigMap
            klogger.info(f"Deleting ConfigMap {config_map_name}")
            self.k8s_client.delete_config_map(config_map_name, namespace)

            klogger.info(f"Deleted all resources for {namespace}/{name}")

        except Exception as e:
            klogger.error(f"Failed to delete resources: {e}", exc_info=True)
            raise

    def _build_headless_service(
        self,
        name: str,
        namespace: str,
        spec: Dict[str, Any]
    ) -> client.V1Service:
        """
        Build a headless service for distributed training.

        Args:
            name: TrainingJob name
            namespace: Namespace
            spec: TrainingJob spec

        Returns:
            Service object
        """
        service_name = f"{name}-headless"
        labels = {
            'app': 'training-job',
            'training-job': name,
        }

        master_port = spec.get('networking', {}).get('masterPort', 29500)

        service = client.V1Service(
            api_version='v1',
            kind='Service',
            metadata=client.V1ObjectMeta(
                name=service_name,
                namespace=namespace,
                labels=labels,
            ),
            spec=client.V1ServiceSpec(
                cluster_ip='None',  # Headless service
                selector=labels,
                ports=[
                    client.V1ServicePort(
                        name='master',
                        port=master_port,
                        target_port=master_port,
                        protocol='TCP',
                    ),
                ],
            )
        )

        return service

    def _build_config_map(
        self,
        name: str,
        namespace: str,
        spec: Dict[str, Any]
    ) -> client.V1ConfigMap:
        """
        Build a ConfigMap with training configuration.

        Args:
            name: TrainingJob name
            namespace: Namespace
            spec: TrainingJob spec

        Returns:
            ConfigMap object
        """
        import json

        config_map_name = f"{name}-config"
        labels = {
            'app': 'training-job',
            'training-job': name,
        }

        # Extract configuration
        config_data = {
            'model': spec.get('model', ''),
            'dataset': spec.get('dataset', ''),
            'num_workers': str(spec.get('numWorkers', 1)),
            'gpus_per_worker': str(spec.get('gpusPerWorker', 1)),
            'framework': spec.get('framework', 'pytorch'),
        }

        # Add hyperparameters
        hyperparameters = spec.get('hyperparameters', {})
        if hyperparameters:
            config_data['hyperparameters'] = json.dumps(hyperparameters)

        # Add networking config
        networking = spec.get('networking', {})
        if networking:
            config_data['backend'] = networking.get('backend', 'nccl')
            config_data['master_port'] = str(networking.get('masterPort', 29500))

        config_map = client.V1ConfigMap(
            api_version='v1',
            kind='ConfigMap',
            metadata=client.V1ObjectMeta(
                name=config_map_name,
                namespace=namespace,
                labels=labels,
            ),
            data=config_data
        )

        return config_map
