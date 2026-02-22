"""
JobBuilder constructs Kubernetes Job resources for ML training.
"""

from typing import Dict, Any, List, Optional
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
        # ... (existing code) ...
        return job

    # ... (existing methods until _build_volumes) ...

    def _build_node_selector(
        self,
        node_selector_spec: Dict[str, Any]
    ) -> client.V1NodeSelector:
        """
        Build NodeSelector.
        
        Args:
            node_selector_spec: Node selector spec
            
        Returns:
            V1NodeSelector
        """
        from kubernetes.client import ApiClient
        api_client = ApiClient()
        try:
            return api_client._ApiClient__deserialize(node_selector_spec, 'V1NodeSelector')
        except Exception as e:
            logger.warning(f"Failed to deserialize node selector: {e}")
            return None

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

        if not affinity_spec:
            return None

        from kubernetes.client import ApiClient
        api_client = ApiClient()
        try:
             return api_client._ApiClient__deserialize(affinity_spec, 'V1Affinity')
        except Exception as e:
            logger.warning(f"Failed to deserialize affinity spec: {e}")
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
        tolerations_spec = scheduling.get('tolerations')

        if not tolerations_spec:
            return None

        from kubernetes.client import ApiClient
        api_client = ApiClient()
        tolerations = []
        
        for tol in tolerations_spec:
            try:
                tolerations.append(api_client._ApiClient__deserialize(tol, 'V1Toleration'))
            except Exception as e:
                logger.warning(f"Failed to deserialize toleration: {e}")
        
        return tolerations

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

