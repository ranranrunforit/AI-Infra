"""
JobBuilder constructs Kubernetes Job resources for ML training.
"""

from typing import Dict, Any, List
from kubernetes import client
import logging

logger = logging.getLogger(__name__)


class JobBuilder:
    """
    Builder for Kubernetes Job resources for ML training.
    """

    def build_job(
        self,
        name: str,
        namespace: str,
        spec: Dict[str, Any]
    ) -> client.V1Job:
        """
        Build a Kubernetes Job for ML training.

        Args:
            name: TrainingJob name
            namespace: Namespace
            spec: TrainingJob spec

        Returns:
            Kubernetes Job object
        """
        job_name = f"{name}-training"
        num_workers = spec.get('numWorkers', 1)

        labels = {
            'app': 'training-job',
            'training-job': name,
            'component': 'worker',
        }

        # Build pod template
        pod_template = self._build_pod_template(name, namespace, spec)

        # Build job spec
        job_spec = client.V1JobSpec(
            parallelism=num_workers,
            completions=num_workers,
            backoff_limit=spec.get('failurePolicy', {}).get('backoffLimit', 3),
            active_deadline_seconds=spec.get('failurePolicy', {}).get('activeDeadlineSeconds'),
            template=pod_template,
        )

        # Build job
        job = client.V1Job(
            api_version='batch/v1',
            kind='Job',
            metadata=client.V1ObjectMeta(
                name=job_name,
                namespace=namespace,
                labels=labels,
            ),
            spec=job_spec
        )

        logger.info(f"Built Kubernetes Job {job_name} with {num_workers} workers")

        return job

    def _build_pod_template(
        self,
        name: str,
        namespace: str,
        spec: Dict[str, Any]
    ) -> client.V1PodTemplateSpec:
        """
        Build pod template for training workers.

        Args:
            name: TrainingJob name
            namespace: Namespace
            spec: TrainingJob spec

        Returns:
            Pod template spec
        """
        labels = {
            'app': 'training-job',
            'training-job': name,
            'component': 'worker',
        }

        # Build container
        container = self._build_container(name, spec)

        # Build pod spec
        pod_spec = client.V1PodSpec(
            containers=[container],
            restart_policy=spec.get('failurePolicy', {}).get('restartPolicy', 'OnFailure'),
            node_selector=spec.get('scheduling', {}).get('nodeSelector'),
            affinity=self._build_affinity(spec),
            tolerations=self._build_tolerations(spec),
            priority_class_name=spec.get('scheduling', {}).get('priority'),
            volumes=self._build_volumes(name, spec),
        )

        # Build pod template
        pod_template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(
                labels=labels,
            ),
            spec=pod_spec
        )

        return pod_template

    def _build_container(
        self,
        name: str,
        spec: Dict[str, Any]
    ) -> client.V1Container:
        """
        Build training container.

        Args:
            name: TrainingJob name
            spec: TrainingJob spec

        Returns:
            Container spec
        """
        # Get container image
        image = spec.get('image', 'pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime')

        # Build command and args
        command = spec.get('command')
        args = spec.get('args')

        if not command:
            # Default command for distributed training
            framework = spec.get('framework', 'pytorch')
            if framework == 'pytorch':
                command = ['python', '-m', 'torch.distributed.run']
                if not args:
                    args = [
                        f"--nproc_per_node={spec.get('gpusPerWorker', 1)}",
                        '--nnodes=$(NUM_WORKERS)',
                        '--node_rank=$(WORKER_RANK)',
                        '--master_addr=$(MASTER_ADDR)',
                        '--master_port=$(MASTER_PORT)',
                        'train.py',
                    ]

        # Build environment variables
        env = self._build_env_vars(name, spec)

        # Build resources
        resources = self._build_resources(spec)

        # Build volume mounts
        volume_mounts = self._build_volume_mounts(name, spec)

        # Build container
        container = client.V1Container(
            name='training',
            image=image,
            command=command,
            args=args,
            env=env,
            resources=resources,
            volume_mounts=volume_mounts,
        )

        return container

    def _build_env_vars(
        self,
        name: str,
        spec: Dict[str, Any]
    ) -> List[client.V1EnvVar]:
        """
        Build environment variables for the container.

        Args:
            name: TrainingJob name
            spec: TrainingJob spec

        Returns:
            List of environment variables
        """
        env_vars = []

        # Add user-specified env vars
        user_env = spec.get('env', [])
        for env in user_env:
            env_vars.append(client.V1EnvVar(
                name=env['name'],
                value=env['value']
            ))

        # Add distributed training env vars
        service_name = f"{name}-headless"
        env_vars.extend([
            client.V1EnvVar(
                name='NUM_WORKERS',
                value=str(spec.get('numWorkers', 1))
            ),
            client.V1EnvVar(
                name='GPUS_PER_WORKER',
                value=str(spec.get('gpusPerWorker', 1))
            ),
            client.V1EnvVar(
                name='MASTER_ADDR',
                value=service_name
            ),
            client.V1EnvVar(
                name='MASTER_PORT',
                value=str(spec.get('networking', {}).get('masterPort', 29500))
            ),
            client.V1EnvVar(
                name='NCCL_BACKEND',
                value=spec.get('networking', {}).get('backend', 'nccl').upper()
            ),
            # Pod-specific info
            client.V1EnvVar(
                name='POD_NAME',
                value_from=client.V1EnvVarSource(
                    field_ref=client.V1ObjectFieldSelector(field_path='metadata.name')
                )
            ),
            client.V1EnvVar(
                name='POD_NAMESPACE',
                value_from=client.V1EnvVarSource(
                    field_ref=client.V1ObjectFieldSelector(field_path='metadata.namespace')
                )
            ),
        ])

        # Add monitoring env vars
        monitoring = spec.get('monitoring', {})
        if monitoring.get('enabled', True):
            if monitoring.get('mlflowTrackingUri'):
                env_vars.append(client.V1EnvVar(
                    name='MLFLOW_TRACKING_URI',
                    value=monitoring['mlflowTrackingUri']
                ))
            if monitoring.get('wandbProject'):
                env_vars.append(client.V1EnvVar(
                    name='WANDB_PROJECT',
                    value=monitoring['wandbProject']
                ))

        return env_vars

    def _build_resources(
        self,
        spec: Dict[str, Any]
    ) -> client.V1ResourceRequirements:
        """
        Build resource requirements.

        Args:
            spec: TrainingJob spec

        Returns:
            Resource requirements
        """
        resources_spec = spec.get('resources', {})

        # Default resources if not specified
        gpus_per_worker = spec.get('gpusPerWorker', 1)

        requests = resources_spec.get('requests', {})
        limits = resources_spec.get('limits', {})

        # Ensure GPU resources are specified
        if gpus_per_worker > 0:
            if 'nvidia.com/gpu' not in requests:
                requests['nvidia.com/gpu'] = str(gpus_per_worker)
            if 'nvidia.com/gpu' not in limits:
                limits['nvidia.com/gpu'] = str(gpus_per_worker)

        # Set default CPU and memory if not specified
        if 'cpu' not in requests:
            requests['cpu'] = '4'
        if 'memory' not in requests:
            requests['memory'] = '16Gi'

        if 'cpu' not in limits:
            limits['cpu'] = '8'
        if 'memory' not in limits:
            limits['memory'] = '32Gi'

        return client.V1ResourceRequirements(
            requests=requests,
            limits=limits
        )

    def _build_affinity(
        self,
        spec: Dict[str, Any]
    ) -> Optional[client.V1Affinity]:
        """
        Build affinity rules.

        Args:
            spec: TrainingJob spec

        Returns:
            Affinity rules or None
        """
        scheduling = spec.get('scheduling', {})
        affinity_spec = scheduling.get('affinity')

        if affinity_spec:
            # Convert dict to Affinity object
            # In production, this would need proper conversion logic
            return affinity_spec

        return None

    def _build_tolerations(
        self,
        spec: Dict[str, Any]
    ) -> Optional[List[client.V1Toleration]]:
        """
        Build tolerations.

        Args:
            spec: TrainingJob spec

        Returns:
            List of tolerations or None
        """
        scheduling = spec.get('scheduling', {})
        tolerations = scheduling.get('tolerations')

        if tolerations:
            # Convert dicts to Toleration objects
            # In production, this would need proper conversion logic
            return tolerations

        return None

    def _build_volumes(
        self,
        name: str,
        spec: Dict[str, Any]
    ) -> List[client.V1Volume]:
        """
        Build volumes.

        Args:
            name: TrainingJob name
            spec: TrainingJob spec

        Returns:
            List of volumes
        """
        volumes = []

        # Add config volume
        config_map_name = f"{name}-config"
        volumes.append(client.V1Volume(
            name='config',
            config_map=client.V1ConfigMapVolumeSource(
                name=config_map_name
            )
        ))

        # Add checkpoint volume if enabled
        checkpoint = spec.get('checkpoint', {})
        if checkpoint.get('enabled', True):
            storage = checkpoint.get('storage', {})
            storage_type = storage.get('type', 'pvc')

            if storage_type == 'pvc':
                pvc_name = storage.get('pvcName', f"{name}-checkpoints")
                volumes.append(client.V1Volume(
                    name='checkpoints',
                    persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                        claim_name=pvc_name
                    )
                ))
            elif storage_type == 'nfs':
                volumes.append(client.V1Volume(
                    name='checkpoints',
                    nfs=client.V1NFSVolumeSource(
                        server=storage.get('nfsServer'),
                        path=storage.get('nfsPath', f'/checkpoints/{name}')
                    )
                ))

        return volumes

    def _build_volume_mounts(
        self,
        name: str,
        spec: Dict[str, Any]
    ) -> List[client.V1VolumeMount]:
        """
        Build volume mounts.

        Args:
            name: TrainingJob name
            spec: TrainingJob spec

        Returns:
            List of volume mounts
        """
        mounts = []

        # Mount config
        mounts.append(client.V1VolumeMount(
            name='config',
            mount_path='/etc/training-config',
            read_only=True
        ))

        # Mount checkpoints if enabled
        checkpoint = spec.get('checkpoint', {})
        if checkpoint.get('enabled', True):
            mounts.append(client.V1VolumeMount(
                name='checkpoints',
                mount_path='/checkpoints'
            ))

        return mounts
