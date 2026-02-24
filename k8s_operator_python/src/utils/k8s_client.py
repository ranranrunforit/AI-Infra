"""
Kubernetes client wrapper for the operator.
"""

from typing import Dict, List, Optional, Any
import os
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import logging

logger = logging.getLogger(__name__)


class K8sClient:
    """
    Wrapper around Kubernetes client for common operations.
    """

    def __init__(self):
        """Initialize Kubernetes client."""
        try:
            # Try loading in-cluster config first
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes configuration")
        except config.ConfigException:
            # Fall back to local kubeconfig
            config.load_kube_config()
            logger.info("Loaded kubeconfig from local filesystem")

        self.api_client = client.ApiClient()
        self.core_v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.batch_v1 = client.BatchV1Api()
        self.custom_objects = client.CustomObjectsApi()
        self.rbac_v1 = client.RbacAuthorizationV1Api()

    def create_job(self, namespace: str, job: client.V1Job) -> client.V1Job:
        """
        Create a Kubernetes Job.

        Args:
            namespace: Namespace to create the job in
            job: Job specification

        Returns:
            Created Job object
        """
        try:
            return self.batch_v1.create_namespaced_job(namespace=namespace, body=job)
        except ApiException as e:
            logger.error(f"Failed to create job: {e}", exc_info=True)
            raise

    def get_job(self, name: str, namespace: str) -> Optional[client.V1Job]:
        """
        Get a Kubernetes Job.

        Args:
            name: Job name
            namespace: Namespace

        Returns:
            Job object or None if not found
        """
        try:
            return self.batch_v1.read_namespaced_job(name=name, namespace=namespace)
        except ApiException as e:
            if e.status == 404:
                return None
            logger.error(f"Failed to get job {name}: {e}", exc_info=True)
            raise

    def delete_job(self, name: str, namespace: str, propagation_policy: str = 'Background') -> None:
        """
        Delete a Kubernetes Job.

        Args:
            name: Job name
            namespace: Namespace
            propagation_policy: Deletion propagation policy
        """
        try:
            self.batch_v1.delete_namespaced_job(
                name=name,
                namespace=namespace,
                propagation_policy=propagation_policy
            )
            logger.info(f"Deleted job {name} in namespace {namespace}")
        except ApiException as e:
            if e.status != 404:
                logger.error(f"Failed to delete job {name}: {e}", exc_info=True)
                raise

    def create_service(self, namespace: str, service: client.V1Service) -> client.V1Service:
        """
        Create a Kubernetes Service.

        Args:
            namespace: Namespace to create the service in
            service: Service specification

        Returns:
            Created Service object
        """
        try:
            return self.core_v1.create_namespaced_service(namespace=namespace, body=service)
        except ApiException as e:
            logger.error(f"Failed to create service: {e}", exc_info=True)
            raise

    def get_service(self, name: str, namespace: str) -> Optional[client.V1Service]:
        """
        Get a Kubernetes Service.

        Args:
            name: Service name
            namespace: Namespace

        Returns:
            Service object or None if not found
        """
        try:
            return self.core_v1.read_namespaced_service(name=name, namespace=namespace)
        except ApiException as e:
            if e.status == 404:
                return None
            logger.error(f"Failed to get service {name}: {e}", exc_info=True)
            raise

    def delete_service(self, name: str, namespace: str) -> None:
        """
        Delete a Kubernetes Service.

        Args:
            name: Service name
            namespace: Namespace
        """
        try:
            self.core_v1.delete_namespaced_service(name=name, namespace=namespace)
            logger.info(f"Deleted service {name} in namespace {namespace}")
        except ApiException as e:
            if e.status != 404:
                logger.error(f"Failed to delete service {name}: {e}", exc_info=True)
                raise

    def create_config_map(self, namespace: str, config_map: client.V1ConfigMap) -> client.V1ConfigMap:
        """
        Create a ConfigMap.

        Args:
            namespace: Namespace to create the ConfigMap in
            config_map: ConfigMap specification

        Returns:
            Created ConfigMap object
        """
        try:
            return self.core_v1.create_namespaced_config_map(namespace=namespace, body=config_map)
        except ApiException as e:
            logger.error(f"Failed to create config map: {e}", exc_info=True)
            raise

    def get_config_map(self, name: str, namespace: str) -> Optional[client.V1ConfigMap]:
        """
        Get a ConfigMap.

        Args:
            name: ConfigMap name
            namespace: Namespace

        Returns:
            ConfigMap object or None if not found
        """
        try:
            return self.core_v1.read_namespaced_config_map(name=name, namespace=namespace)
        except ApiException as e:
            if e.status == 404:
                return None
            logger.error(f"Failed to get config map {name}: {e}", exc_info=True)
            raise

    def delete_config_map(self, name: str, namespace: str) -> None:
        """
        Delete a ConfigMap.

        Args:
            name: ConfigMap name
            namespace: Namespace
        """
        try:
            self.core_v1.delete_namespaced_config_map(name=name, namespace=namespace)
            logger.info(f"Deleted config map {name} in namespace {namespace}")
        except ApiException as e:
            if e.status != 404:
                logger.error(f"Failed to delete config map {name}: {e}", exc_info=True)
                raise

    def list_pods(self, namespace: str, label_selector: Optional[str] = None) -> client.V1PodList:
        """
        List Pods in a namespace.

        Args:
            namespace: Namespace
            label_selector: Label selector for filtering

        Returns:
            List of Pods
        """
        try:
            return self.core_v1.list_namespaced_pod(
                namespace=namespace,
                label_selector=label_selector
            )
        except ApiException as e:
            logger.error(f"Failed to list pods: {e}", exc_info=True)
            raise

    def get_pod_logs(
        self,
        name: str,
        namespace: str,
        container: Optional[str] = None,
        tail_lines: Optional[int] = None
    ) -> str:
        """
        Get logs from a Pod.

        Args:
            name: Pod name
            namespace: Namespace
            container: Container name (if multi-container pod)
            tail_lines: Number of lines from the end to return

        Returns:
            Pod logs as string
        """
        try:
            return self.core_v1.read_namespaced_pod_log(
                name=name,
                namespace=namespace,
                container=container,
                tail_lines=tail_lines
            )
        except ApiException as e:
            logger.error(f"Failed to get pod logs: {e}", exc_info=True)
            raise

    def create_event(
        self,
        name: str,
        namespace: str,
        message: str,
        reason: str,
        type_: str,
        involved_object: client.V1ObjectReference
    ) -> client.CoreV1Event:
        """
        Create a Kubernetes Event.

        Args:
            name: Event name
            namespace: Namespace
            message: Event message
            reason: Event reason
            type_: Event type (Normal, Warning, Error)
            involved_object: Reference to the object this event is about

        Returns:
            Created Event object
        """
        from datetime import datetime

        event = client.CoreV1Event(
            metadata=client.V1ObjectMeta(name=name, namespace=namespace),
            message=message,
            reason=reason,
            type=type_,
            involved_object=involved_object,
            first_timestamp=datetime.utcnow(),
            last_timestamp=datetime.utcnow(),
            count=1
        )

        try:
            return self.core_v1.create_namespaced_event(namespace=namespace, body=event)
        except ApiException as e:
            logger.error(f"Failed to create event: {e}", exc_info=True)
            raise

    def patch_custom_object_status(
        self,
        group: str,
        version: str,
        plural: str,
        name: str,
        namespace: str,
        status: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Patch the status subresource of a custom object.

        Args:
            group: API group
            version: API version
            plural: Resource plural name
            name: Object name
            namespace: Namespace
            status: Status data to patch

        Returns:
            Patched object
        """
        try:
            return self.custom_objects.patch_namespaced_custom_object_status(
                group=group,
                version=version,
                namespace=namespace,
                plural=plural,
                name=name,
                body={'status': status}
            )
        except ApiException as e:
            logger.error(f"Failed to patch status for {name}: {e}", exc_info=True)
            raise


# Global client instance
_k8s_client: Optional[K8sClient] = None


def get_k8s_client() -> K8sClient:
    """
    Get or create the global Kubernetes client instance.

    Returns:
        K8sClient instance
    """
    global _k8s_client
    if _k8s_client is None:
        _k8s_client = K8sClient()
    return _k8s_client
