import kopf
import logging
import os
from typing import Dict, Any
from datetime import datetime
import time
from prometheus_client import start_http_server

from ..utils import get_logger, setup_logging, metrics, K8sClient
from ..controllers.job_controller import JobController
from ..controllers.status_controller import StatusController
from ..controllers.checkpoint_controller import CheckpointController

# Setup logging
log_level = os.getenv('LOG_LEVEL', 'INFO')
structured_logging = os.getenv('STRUCTURED_LOGGING', 'true').lower() == 'true'
setup_logging(level=log_level, structured=structured_logging)

logger = get_logger(__name__)

# Initialize K8s client and controllers
k8s_client = None
job_controller = None
status_controller = None
checkpoint_controller = None

# Operator configuration
OPERATOR_NAMESPACE = os.getenv('OPERATOR_NAMESPACE', 'ml-training')
GROUP = 'ml.example.com'
VERSION = 'v1'
PLURAL = 'trainingjobs'


@kopf.on.startup()
async def startup(settings: kopf.OperatorSettings, **kwargs):
    """
    Configure the operator on startup.
    """
    logger.info("Starting TrainingJob operator")
    
    # Start Prometheus metrics server
    try:
        # Start on port 9090 to avoid conflict with Kopf healthz (8080)
        start_http_server(9090)
        logger.info("Prometheus metrics server started on port 9090")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")

    # Initialize K8s client
    global k8s_client, job_controller, status_controller, checkpoint_controller
    try:
        k8s_client = K8sClient()
        # Initialize controllers with dependency injection
        job_controller = JobController(k8s_client)
        status_controller = StatusController(k8s_client)
        checkpoint_controller = CheckpointController(k8s_client)
        
        logger.info("Operator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize operator: {e}")
        raise kopf.PermanentError(f"Failed to initialize operator: {e}")

    # Configure kopf settings
    settings.posting.level = logging.WARNING
    settings.watching.connect_timeout = 1 * 60
    settings.watching.server_timeout = 10 * 60
    settings.persistence.finalizer = f'{GROUP}/trainingjob-finalizer'

    # Set operator as healthy
    metrics.set_operator_up(True)

    logger.info(f"Operator configured. Watching {GROUP}/{VERSION}/{PLURAL}")


@kopf.on.cleanup()
def cleanup(**_):
    """
    Cleanup handler when operator is shutting down.
    """
    logger.info("Shutting down TrainingJob operator")
    metrics.set_operator_up(False)


@kopf.on.create(GROUP, VERSION, PLURAL)
async def create_handler(
    spec: Dict[str, Any],
    meta: Dict[str, Any],
    status: Dict[str, Any],
    namespace: str,
    name: str,
    logger: kopf.Logger,
    **kwargs
) -> Dict[str, Any]:
    """
    Handler for TrainingJob creation events.

    Args:
        spec: TrainingJob spec
        meta: Resource metadata
        status: Current status
        namespace: Namespace
        name: Resource name
        logger: Kopf logger

    Returns:
        Status update dictionary
    """
    start_time = time.time()

    try:
        logger.info(f"Creating TrainingJob {namespace}/{name}")
        metrics.record_job_created(namespace)
        metrics.operator_watch_events.labels(event_type='create').inc()

        # Validate the spec
        _validate_training_job_spec(spec, name, namespace)

        # Initialize status
        initial_status = {
            'state': 'Pending',
            'conditions': [
                {
                    'type': 'Initialized',
                    'status': 'True',
                    'reason': 'TrainingJobCreated',
                    'message': 'TrainingJob has been created',
                    'lastTransitionTime': datetime.utcnow().isoformat() + 'Z',
                    'lastUpdateTime': datetime.utcnow().isoformat() + 'Z',
                }
            ],
            'progress': '0%',
            'currentEpoch': 0,
            'totalEpochs': spec.get('hyperparameters', {}).get('epochs', 0),
            'workers': {
                'active': 0,
                'succeeded': 0,
                'failed': 0,
                'pending': spec.get('numWorkers', 1),
            },
            'startTime': datetime.utcnow().isoformat() + 'Z',
        }

        # Update metrics
        metrics.update_training_job_count(namespace, 'Pending', 1)

        duration = time.time() - start_time
        metrics.record_reconciliation(namespace, name, duration, 'success')

        logger.info(f"TrainingJob {namespace}/{name} created successfully")

        return initial_status

    except Exception as e:
        logger.error(f"Failed to create TrainingJob {namespace}/{name}: {e}", exc_info=True)
        duration = time.time() - start_time
        metrics.record_reconciliation(namespace, name, duration, 'error')
        metrics.record_reconciliation_error(namespace, name, type(e).__name__)
        raise kopf.PermanentError(f"Failed to create TrainingJob: {e}")


@kopf.on.update(GROUP, VERSION, PLURAL)
async def update_handler(
    spec: Dict[str, Any],
    meta: Dict[str, Any],
    status: Dict[str, Any],
    namespace: str,
    name: str,
    old: Dict[str, Any],
    new: Dict[str, Any],
    diff: kopf.Diff,
    logger: kopf.Logger,
    **kwargs
) -> Dict[str, Any]:
    """
    Handler for TrainingJob update events.

    Args:
        spec: Updated TrainingJob spec
        meta: Resource metadata
        status: Current status
        namespace: Namespace
        name: Resource name
        old: Old resource state
        new: New resource state
        diff: Diff between old and new
        logger: Kopf logger

    Returns:
        Status update dictionary
    """
    start_time = time.time()

    try:
        logger.info(f"Updating TrainingJob {namespace}/{name}")
        metrics.operator_watch_events.labels(event_type='update').inc()

        # Check if spec has changed
        spec_changes = [d for d in diff if d[1][0] == 'spec']

        if spec_changes:
            logger.info(f"Spec changed for {namespace}/{name}: {spec_changes}")

            # Handle spec updates based on current state
            current_state = status.get('state', 'Unknown')

            if current_state in ['Running', 'Initializing']:
                logger.warning(f"Cannot update running TrainingJob {namespace}/{name}")
                return {
                    'state': current_state,
                    'conditions': status.get('conditions', []) + [
                        {
                            'type': 'UpdateRejected',
                            'status': 'True',
                            'reason': 'JobRunning',
                            'message': 'Cannot update spec while job is running',
                            'lastTransitionTime': datetime.utcnow().isoformat() + 'Z',
                        }
                    ],
                }

        duration = time.time() - start_time
        metrics.record_reconciliation(namespace, name, duration, 'success')

        return status

    except Exception as e:
        logger.error(f"Failed to update TrainingJob {namespace}/{name}: {e}", exc_info=True)
        duration = time.time() - start_time
        metrics.record_reconciliation(namespace, name, duration, 'error')
        metrics.record_reconciliation_error(namespace, name, type(e).__name__)
        raise


@kopf.on.delete(GROUP, VERSION, PLURAL)
async def delete_handler(
    spec: Dict[str, Any],
    meta: Dict[str, Any],
    status: Dict[str, Any],
    namespace: str,
    name: str,
    logger: kopf.Logger,
    **kwargs
):
    """
    Handler for TrainingJob deletion events.

    Args:
        spec: TrainingJob spec
        meta: Resource metadata
        status: Current status
        namespace: Namespace
        name: Resource name
        logger: Kopf logger
    """
    start_time = time.time()

    try:
        logger.info(f"Deleting TrainingJob {namespace}/{name}")
        metrics.operator_watch_events.labels(event_type='delete').inc()

        # Delete associated Kubernetes resources
        await job_controller.delete_training_resources(name, namespace, logger)

        # Update metrics
        current_state = status.get('state', 'Unknown')
        metrics.update_training_job_count(namespace, current_state, -1)

        duration = time.time() - start_time
        metrics.record_reconciliation(namespace, name, duration, 'success')

        logger.info(f"TrainingJob {namespace}/{name} deleted successfully")

    except Exception as e:
        logger.error(f"Failed to delete TrainingJob {namespace}/{name}: {e}", exc_info=True)
        duration = time.time() - start_time
        metrics.record_reconciliation(namespace, name, duration, 'error')
        metrics.record_reconciliation_error(namespace, name, type(e).__name__)
        raise


@kopf.timer(GROUP, VERSION, PLURAL, interval=30.0)
async def reconcile_handler(
    spec: Dict[str, Any],
    meta: Dict[str, Any],
    status: Dict[str, Any],
    namespace: str,
    name: str,
    logger: kopf.Logger,
    **kwargs
) -> Dict[str, Any]:
    """
    Periodic reconciliation handler.

    This handler runs every 30 seconds to ensure the actual state
    matches the desired state.

    Args:
        spec: TrainingJob spec
        meta: Resource metadata
        status: Current status
        namespace: Namespace
        name: Resource name
        logger: Kopf logger

    Returns:
        Status update dictionary
    """
    start_time = time.time()

    try:
        current_state = status.get('state', 'Pending')

        logger.debug(f"Reconciling TrainingJob {namespace}/{name} (state: {current_state})")

        # Handle different states
        if current_state == 'Pending':
            # Create training resources
            new_status = await job_controller.create_training_resources(
                name, namespace, spec, status, logger
            )
            return new_status

        elif current_state == 'Initializing':
            # Check if resources are ready
            new_status = await job_controller.check_resources_ready(
                name, namespace, spec, status, logger
            )
            return new_status

        elif current_state == 'Running':
            # Monitor training progress
            new_status = await status_controller.update_training_status(
                name, namespace, spec, status, logger
            )

            # Handle checkpoints
            if spec.get('checkpoint', {}).get('enabled', True):
                await checkpoint_controller.manage_checkpoints(
                    name, namespace, spec, new_status, logger
                )

            # Check for completion
            if _is_training_complete(new_status):
                new_status['state'] = 'Completed'
                new_status['completionTime'] = datetime.utcnow().isoformat() + 'Z'
                new_status['conditions'].append({
                    'type': 'Completed',
                    'status': 'True',
                    'reason': 'TrainingCompleted',
                    'message': 'Training has completed successfully',
                    'lastTransitionTime': datetime.utcnow().isoformat() + 'Z',
                })
                metrics.record_job_completed(namespace, 'success')

            return new_status

        elif current_state == 'Failed':
            # Handle failure and potential retry
            backoff_limit = spec.get('failurePolicy', {}).get('backoffLimit', 3)
            restart_count = status.get('restartCount', 0)

            if restart_count < backoff_limit:
                logger.info(f"Restarting failed TrainingJob {namespace}/{name}")
                new_status = status.copy()
                new_status['state'] = 'Pending'
                new_status['restartCount'] = restart_count + 1
                metrics.job_restarted.labels(namespace=namespace, training_job=name).inc()
                return new_status

        elif current_state in ['Completed', 'Suspended']:
            # No action needed
            pass

        duration = time.time() - start_time
        metrics.record_reconciliation(namespace, name, duration, 'success')

        return status

    except Exception as e:
        logger.error(f"Reconciliation failed for {namespace}/{name}: {e}", exc_info=True)
        duration = time.time() - start_time
        metrics.record_reconciliation(namespace, name, duration, 'error')
        metrics.record_reconciliation_error(namespace, name, type(e).__name__)

        # Update status to Failed
        return {
            **status,
            'state': 'Failed',
            'failureReason': type(e).__name__,
            'failureMessage': str(e),
            'conditions': status.get('conditions', []) + [
                {
                    'type': 'Failed',
                    'status': 'True',
                    'reason': type(e).__name__,
                    'message': str(e),
                    'lastTransitionTime': datetime.utcnow().isoformat() + 'Z',
                }
            ],
        }


def _validate_training_job_spec(spec: Dict[str, Any], name: str, namespace: str) -> None:
    """
    Validate TrainingJob spec.

    Args:
        spec: TrainingJob spec
        name: Resource name
        namespace: Namespace

    Raises:
        ValueError: If validation fails
    """
    required_fields = ['model', 'dataset', 'numWorkers']

    for field in required_fields:
        if field not in spec:
            raise ValueError(f"Missing required field: {field}")

    num_workers = spec.get('numWorkers', 0)
    if num_workers < 1:
        raise ValueError(f"numWorkers must be >= 1, got {num_workers}")

    gpus_per_worker = spec.get('gpusPerWorker', 1)
    if gpus_per_worker < 0:
        raise ValueError(f"gpusPerWorker must be >= 0, got {gpus_per_worker}")

    logger.info(f"TrainingJob {namespace}/{name} spec validated successfully")


def _is_training_complete(status: Dict[str, Any]) -> bool:
    """
    Check if training is complete.

    Args:
        status: Current status

    Returns:
        True if training is complete
    """
    current_epoch = status.get('currentEpoch', 0)
    total_epochs = status.get('totalEpochs', 0)

    if total_epochs > 0 and current_epoch >= total_epochs:
        return True

    # Check success policy
    metrics = status.get('metrics', {})
    # This would check against successPolicy in spec if available

    return False


def main():
    """
    Main entry point for the operator.
    """
    logger.info("Starting TrainingJob Operator")

    # Run the operator
    kopf.run(
        clusterwide=False,
        namespace=OPERATOR_NAMESPACE,
        liveness_endpoint='http://0.0.0.0:8080/healthz',
        standalone=True,
    )


if __name__ == '__main__':
    main()
