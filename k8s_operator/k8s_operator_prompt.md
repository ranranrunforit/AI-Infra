# Reimplementation Prompt: TrainingJob Kubernetes Operator (k8s_operator)

## Goal

Build a **production-ready Kubernetes Operator** for managing ML training jobs. The operator extends Kubernetes with a custom `TrainingJob` resource and automates the entire lifecycle of distributed ML training workloads.

---

## Tech Stack

- **Language**: Python 3.11+
- **Operator Framework**: [Kopf](https://kopf.readthedocs.io/) (`kopf`)
- **Kubernetes Client**: [kubernetes](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/multi_region/src/failover/failover_controller.py#145-210) Python SDK
- **Metrics**: `prometheus_client`
- **Containerization**: Docker
- **Kubernetes Manifests**: YAML (CRD, RBAC, Deployment, Service)

**[requirements.txt](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/requirements.txt)**:
```
kopf>=1.36.0
kubernetes>=28.1.0
prometheus-client>=0.19.0
pyyaml>=6.0
```

---

## Project Structure

```
k8s_operator/
├── Dockerfile
├── Makefile
├── requirements.txt
├── README.md
├── OPERATIONS_GUIDE.md
├── api/
│   └── v1/                        # CRD Python type hints / models
├── cmd/                           # Go-style entry point placeholder
├── src/
│   ├── operator/
│   │   └── main.py                # Kopf operator entry point
│   ├── controllers/
│   │   ├── job_controller.py      # Manages K8s Job lifecycle
│   │   ├── status_controller.py   # Monitors & updates training status
│   │   └── checkpoint_controller.py # Checkpoint lifecycle management
│   ├── resources/
│   │   └── job_builder.py         # Builds K8s resource specs
│   ├── crd/
│   │   ├── trainingjob_crd.py     # CRD Python model definitions
│   │   ├── validation.py          # Spec validation logic
│   │   └── defaults.py            # Default values for spec fields
│   └── utils/
│       ├── __init__.py            # Exports: get_logger, setup_logging, metrics, get_k8s_client
│       ├── k8s_client.py          # Kubernetes client wrapper
│       ├── logger.py              # Structured JSON logging
│       └── metrics.py             # Prometheus metrics definitions
├── kubernetes/
│   ├── base/
│   │   ├── trainingjob-crd.yaml   # CRD definition
│   │   ├── rbac.yaml              # ClusterRole + ServiceAccount + Binding
│   │   ├── deployment.yaml        # Operator deployment
│   │   └── service.yaml           # Operator metrics service
│   └── monitoring/
│       ├── servicemonitor.yaml    # Prometheus ServiceMonitor
│       └── grafana-dashboard.yaml # Grafana dashboard
└── examples/
    ├── trainingjob-simple.yaml
    ├── trainingjob-distributed.yaml
    └── trainingjob-checkpoint.yaml
```

---

## Custom Resource: TrainingJob

### API Group
- **Group**: `ml.example.com`
- **Version**: `v1`
- **Plural**: `trainingjobs`
- **Kind**: `TrainingJob`
- **Namespace**: `ml-training`

### TrainingJob Spec Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| [model](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/multi_region/src/replication/model_replicator.py#541-625) | string | ✅ | — | Model architecture name |
| `dataset` | string | ✅ | — | Dataset name |
| `numWorkers` | integer | ✅ | — | Number of worker pods (1–100) |
| `gpusPerWorker` | integer | ❌ | 1 | GPUs per worker pod |
| `framework` | string | ❌ | `pytorch` | `pytorch` / `tensorflow` / `jax` |
| `image` | string | ❌ | `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime` | Container image |
| `command` | []string | ❌ | auto | Container entrypoint |
| `args` | []string | ❌ | auto | Container args |
| [env](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/resources/job_builder.py#177-255) | []object | ❌ | — | Extra env vars `[{name, value}]` |
| [resources](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/resources/job_builder.py#256-299) | object | ❌ | — | `{requests: {cpu, memory, nvidia.com/gpu}, limits: {...}}` |
| `hyperparameters` | object | ❌ | — | See below |
| `checkpoint` | object | ❌ | — | See below |
| `scheduling` | object | ❌ | — | See below |
| [monitoring](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/multi_region/src/failover/failover_controller.py#472-508) | object | ❌ | — | See below |
| `networking` | object | ❌ | — | See below |
| `failurePolicy` | object | ❌ | — | See below |
| `successPolicy` | object | ❌ | — | See below |

**Hyperparameters sub-object**:
```yaml
hyperparameters:
  learningRate: 0.001
  batchSize: 64
  epochs: 10
  optimizer: adam          # adam/sgd/adamw/rmsprop
  warmupSteps: 100
  gradientAccumulationSteps: 1
  mixedPrecision: false
  additionalParams:        # map[string]string for custom params
    weight_decay: "0.01"
```

**Checkpoint sub-object**:
```yaml
checkpoint:
  enabled: true
  frequency: 1             # Every N epochs
  storage:
    type: pvc              # pvc / s3 / gcs / nfs
    pvcName: my-pvc        # for type=pvc
    s3Bucket: my-bucket    # for type=s3
    nfsServer: 10.0.0.1   # for type=nfs
    nfsPath: /checkpoints
  resumeFrom: "s3://bucket/checkpoint-epoch-10"
  retention: 5             # How many checkpoints to keep
```

**Scheduling sub-object**:
```yaml
scheduling:
  priority: high-priority       # PriorityClass name
  nodeSelector:
    nvidia.com/gpu.product: A100-SXM4-80GB
  affinity: {}                  # raw K8s affinity spec
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
```

**Networking sub-object**:
```yaml
networking:
  backend: nccl               # nccl / gloo / mpi
  masterPort: 29500
  rdmaEnabled: false
```

**FailurePolicy sub-object**:
```yaml
failurePolicy:
  restartPolicy: OnFailure
  backoffLimit: 3
  activeDeadlineSeconds: 86400
```

**SuccessPolicy sub-object**:
```yaml
successPolicy:
  targetAccuracy: 0.95
  earlyStoppingPatience: 5
```

### TrainingJob Status Fields

```yaml
status:
  state: Running            # Pending / Initializing / Running / Completed / Failed / Suspended
  conditions: []            # [{type, status, reason, message, lastTransitionTime}]
  progress: "45%"
  currentEpoch: 2
  totalEpochs: 10
  metrics:
    loss: 0.35
    accuracy: 0.89
  workers:
    active: 4
    succeeded: 0
    failed: 0
    pending: 0
  checkpoint:
    lastCheckpointEpoch: 1
    checkpointPath: "..."
  resources:
    allocatedGPUs: 8
    allocatedNodes: 4
  startTime: "..."
  completionTime: "..."
  duration: "..."
  failureReason: ""
  restartCount: 0
```

---

## State Machine

```
Created → Pending → Initializing → Running → Completed
                                     │
                                     └──────→ Failed → Pending (retry if restartCount < backoffLimit)
```

---

## src/operator/main.py — Kopf Handlers

Use Kopf decorators. Constants at the top:
```python
GROUP = 'ml.example.com'
VERSION = 'v1'
PLURAL = 'trainingjobs'
OPERATOR_NAMESPACE = os.getenv('OPERATOR_NAMESPACE', 'ml-training')
```

### `@kopf.on.startup()`
- Configure `settings.posting.level = logging.WARNING`
- `settings.watching.connect_timeout = 60`
- `settings.watching.server_timeout = 600`
- `settings.persistence.finalizer = f'{GROUP}/trainingjob-finalizer'`
- Call `metrics.set_operator_up(True)`

### `@kopf.on.create(GROUP, VERSION, PLURAL)` → [create_handler](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/operator/main.py#70-145)
1. Call [_validate_training_job_spec(spec, name, namespace)](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/operator/main.py#389-416) — raises `kopf.PermanentError` on failure
2. Return `initial_status` dict:
   - `state: 'Pending'`
   - `conditions`: one entry with `type: 'Initialized'`
   - `progress: '0%'`, `currentEpoch: 0`, `totalEpochs: spec.hyperparameters.epochs`
   - `workers: {active:0, succeeded:0, failed:0, pending: numWorkers}`
   - `startTime: utcnow().isoformat() + 'Z'`
3. Record `metrics.record_job_created(namespace)` and `metrics.operator_watch_events.labels(event_type='create').inc()`
4. Track reconciliation duration; call `metrics.record_reconciliation(...)` on both success and error

### `@kopf.on.update(GROUP, VERSION, PLURAL)` → [update_handler](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/operator/main.py#147-218)
- If `spec_changes` detected **and** `current_state in ['Running', 'Initializing']`, log warning and return status with an `UpdateRejected` condition appended
- Otherwise return current status unchanged

### `@kopf.on.delete(GROUP, VERSION, PLURAL)` → [delete_handler](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/operator/main.py#220-265)
- Call `await job_controller.delete_training_resources(name, namespace, logger)`
- Update `metrics.update_training_job_count(namespace, current_state, -1)`

### `@kopf.timer(GROUP, VERSION, PLURAL, interval=30.0)` → [reconcile_handler](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/operator/main.py#267-387)
This is the **core reconciliation loop**. Dispatch on `current_state`:

| State | Action |
|-------|--------|
| `Pending` | `await job_controller.create_training_resources(...)` → sets state to `Initializing` |
| `Initializing` | `await job_controller.check_resources_ready(...)` → sets state to `Running` when all workers active |
| `Running` | `await status_controller.update_training_status(...)` + `await checkpoint_controller.manage_checkpoints(...)` + check [_is_training_complete()](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/operator/main.py#418-439) |
| `Failed` | If `restartCount < backoffLimit`: set `state='Pending'`, increment `restartCount` |
| `Completed` / `Suspended` | No-op |

On **exception** in reconcile: return status with `state: 'Failed'` and new `Failed` condition.

### Helper functions
```python
def _validate_training_job_spec(spec, name, namespace):
    # Required: model, dataset, numWorkers
    # numWorkers >= 1
    # gpusPerWorker >= 0

def _is_training_complete(status) -> bool:
    # True if currentEpoch >= totalEpochs (and totalEpochs > 0)
```

---

## src/controllers/job_controller.py — JobController

```python
class JobController:
    def __init__(self):
        self.k8s_client = get_k8s_client()
        self.job_builder = JobBuilder()
```

### `async create_training_resources(name, namespace, spec, status, klogger) → dict`
1. Build K8s Job: `self.job_builder.build_job(name, namespace, spec)`
2. Create Job: `self.k8s_client.create_job(namespace, k8s_job)`
3. If `numWorkers > 1`: build and create headless Service (`{name}-headless`)
4. Build and create ConfigMap (`{name}-config`) with training config
5. Record metrics: `k8s_jobs_created`, `allocated_workers`, `allocated_gpus`
6. Return new status with `state: 'Initializing'` and `ResourcesCreated` condition

### `async check_resources_ready(name, namespace, spec, status, klogger) → dict`
1. Get job `{name}-training` from K8s
2. Read `job_status.active`, `succeeded`, `failed`
3. If `active >= numWorkers` → set `state: 'Running'`, append `Running` condition
4. If `failed >= backoffLimit` → set `state: 'Failed'`, append `Failed` condition
5. Update `workers` counts in status

### `async delete_training_resources(name, namespace, klogger)`
- Delete Job `{name}-training`
- Delete Service `{name}-headless`
- Delete ConfigMap `{name}-config`

### [_build_headless_service(name, namespace, spec) → V1Service](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/controllers/job_controller.py#240-288)
- `clusterIP: None` (headless)
- Port: `spec.networking.masterPort` (default 29500)
- Selector: `{app: training-job, training-job: name}`

### [_build_config_map(name, namespace, spec) → V1ConfigMap](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/controllers/job_controller.py#289-346)
- Data: [model](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/multi_region/src/replication/model_replicator.py#541-625), `dataset`, `num_workers`, `gpus_per_worker`, `framework`
- Add `hyperparameters` as JSON string
- Add `backend`, `master_port` from networking spec

---

## src/resources/job_builder.py — JobBuilder

### [build_job(name, namespace, spec) → V1Job](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/resources/job_builder.py#17-70)
- Job name: `{name}-training`
- Labels: `{app: training-job, training-job: name, component: worker}`
- `V1JobSpec(parallelism=numWorkers, completions=numWorkers, backoff_limit=..., active_deadline_seconds=..., template=pod_template)`

### [_build_pod_template(name, namespace, spec) → V1PodTemplateSpec](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/resources/job_builder.py#71-117)
- `restart_policy`: from `failurePolicy.restartPolicy` (default `OnFailure`)
- `node_selector`, `priority_class_name`, [affinity](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/resources/job_builder.py#300-322), [tolerations](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/resources/job_builder.py#323-345) from `scheduling`
- [volumes](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/resources/job_builder.py#346-396): config volume + checkpoint volume if enabled
- `containers`: single container from [_build_container](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/resources/job_builder.py#118-176)

### [_build_container(name, spec) → V1Container](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/resources/job_builder.py#118-176)
- Default image: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime`
- Default PyTorch command: `['python', '-m', 'torch.distributed.run']`
- Default args: `--nproc_per_node=<gpus> --nnodes=$(NUM_WORKERS) --node_rank=$(WORKER_RANK) --master_addr=$(MASTER_ADDR) --master_port=$(MASTER_PORT) train.py`
- [resources](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/resources/job_builder.py#256-299): from [_build_resources](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/resources/job_builder.py#256-299) (defaults: cpu=4/8, memory=16Gi/32Gi, gpu auto-added)
- [volume_mounts](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/resources/job_builder.py#397-430): config at `/etc/training-config`, checkpoints at `/checkpoints`

### [_build_env_vars(name, spec) → List[V1EnvVar]](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/resources/job_builder.py#177-255)
Standard env vars to inject:
```
NUM_WORKERS         = spec.numWorkers
GPUS_PER_WORKER     = spec.gpusPerWorker
MASTER_ADDR         = {name}-headless
MASTER_PORT         = spec.networking.masterPort (default 29500)
NCCL_BACKEND        = spec.networking.backend.upper() (default NCCL)
POD_NAME            = fieldRef: metadata.name
POD_NAMESPACE       = fieldRef: metadata.namespace
MLFLOW_TRACKING_URI = spec.monitoring.mlflowTrackingUri (if set)
WANDB_PROJECT       = spec.monitoring.wandbProject (if set)
```
Plus any user-defined `spec.env` items.

### [_build_resources(spec) → V1ResourceRequirements](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/resources/job_builder.py#256-299)
- GPU: auto-add `nvidia.com/gpu: gpusPerWorker` to both requests and limits if `gpusPerWorker > 0`
- Defaults: `cpu: 4 / 8`, `memory: 16Gi / 32Gi`

### [_build_volumes(name, spec) → List[V1Volume]](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/resources/job_builder.py#346-396)
- Always add ConfigMap volume (`{name}-config` → `/etc/training-config`)
- If `checkpoint.enabled`: add PVC volume (`{name}-checkpoints`) or NFS volume

---

## src/controllers/status_controller.py — StatusController

```python
class StatusController:
    async def update_training_status(name, namespace, spec, status, klogger) -> dict:
        # 1. Get K8s job status
        # 2. Update workers counts (active, succeeded, failed)
        # 3. Simulate or read progress: currentEpoch, totalEpochs, progress %
        # 4. Read/simulate metrics: {loss, accuracy}
        # 5. Update Prometheus gauges (training_loss, training_accuracy, progress)
        # 6. If workers.failed > 0 and failed >= backoffLimit → set state=Failed
        # 7. If workers.succeeded >= numWorkers → mark ready for completion
```

---

## src/controllers/checkpoint_controller.py — CheckpointController

```python
class CheckpointController:
    async def manage_checkpoints(name, namespace, spec, status, klogger):
        # 1. Read checkpoint spec (frequency, storage, retention)
        # 2. Check if current epoch % frequency == 0
        # 3. If yes: create checkpoint annotation/record in status
        # 4. Apply retention policy: prune old checkpoints beyond `retention` count
        # 5. Record checkpoint path in status.checkpoint
```

---

## src/utils/metrics.py — Prometheus Metrics

Define the following metrics using `prometheus_client`:

```python
# Counters
operator_watch_events = Counter('operator_watch_events_total', '...', ['event_type'])
jobs_created_total = Counter('trainingjob_created_total', '...', ['namespace'])
jobs_completed_total = Counter('trainingjob_completed_total', '...', ['namespace', 'result'])
jobs_failed_total = Counter('trainingjob_failed_total', '...', ['namespace', 'reason'])
job_restarted = Counter('trainingjob_restarted_total', '...', ['namespace', 'training_job'])
k8s_jobs_created = Counter('k8s_jobs_created_total', '...', ['namespace'])
reconciliation_errors_total = Counter('reconciliation_errors_total', '...', ['namespace', 'name', 'error_type'])

# Gauges
operator_up = Gauge('trainingjob_operator_up', 'Operator health')
training_jobs_total = Gauge('trainingjob_count', '...', ['namespace', 'state'])
allocated_workers = Gauge('trainingjob_allocated_workers', '...', ['namespace', 'training_job'])
allocated_gpus = Gauge('trainingjob_allocated_gpus', '...', ['namespace', 'training_job'])
training_progress = Gauge('trainingjob_progress_percent', '...', ['namespace', 'training_job'])
training_loss = Gauge('trainingjob_loss', '...', ['namespace', 'training_job'])
training_accuracy = Gauge('trainingjob_accuracy', '...', ['namespace', 'training_job'])
gpu_utilization = Gauge('trainingjob_gpu_utilization_percent', '...', ['namespace', 'training_job'])

# Histograms
reconciliation_duration = Histogram('trainingjob_reconciliation_duration_seconds', '...', ['namespace', 'name', 'result'])

# Helper methods on a MetricsCollector class:
# set_operator_up(up: bool)
# record_job_created(namespace)
# record_job_completed(namespace, result)
# update_training_job_count(namespace, state, delta)
# record_reconciliation(namespace, name, duration, result)
# record_reconciliation_error(namespace, name, error_type)
```

---

## Kubernetes Manifests

### `kubernetes/base/trainingjob-crd.yaml`

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: trainingjobs.ml.example.com
spec:
  group: ml.example.com
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              required: [model, dataset, numWorkers]
              properties:
                model: {type: string}
                dataset: {type: string}
                numWorkers:
                  type: integer
                  minimum: 1
                  maximum: 100
                gpusPerWorker:
                  type: integer
                  minimum: 0
                  default: 1
                # ... all other fields
            status:
              type: object
              x-kubernetes-preserve-unknown-fields: true
  scope: Namespaced
  names:
    plural: trainingjobs
    singular: trainingjob
    kind: TrainingJob
    shortNames: [tj]
  additionalPrinterColumns:
    - name: State
      type: string
      jsonPath: .status.state
    - name: Progress
      type: string
      jsonPath: .status.progress
    - name: Epoch
      type: integer
      jsonPath: .status.currentEpoch
    - name: Age
      type: date
      jsonPath: .metadata.creationTimestamp
```

### `kubernetes/base/rbac.yaml`

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: trainingjob-operator
  namespace: ml-training
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: trainingjob-operator
rules:
  - apiGroups: [ml.example.com]
    resources: [trainingjobs, trainingjobs/status]
    verbs: [get, list, watch, create, update, patch, delete]
  - apiGroups: [batch]
    resources: [jobs]
    verbs: [get, list, watch, create, update, patch, delete]
  - apiGroups: [""]
    resources: [pods, services, configmaps, events, persistentvolumeclaims]
    verbs: [get, list, watch, create, update, patch, delete]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: trainingjob-operator
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: trainingjob-operator
subjects:
  - kind: ServiceAccount
    name: trainingjob-operator
    namespace: ml-training
```

### `kubernetes/base/deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trainingjob-operator
  namespace: ml-training
spec:
  replicas: 1
  selector:
    matchLabels:
      app: trainingjob-operator
  template:
    spec:
      serviceAccountName: trainingjob-operator
      containers:
        - name: operator
          image: trainingjob-operator:latest
          command: ["python", "-m", "src.operator.main"]
          env:
            - name: OPERATOR_NAMESPACE
              value: ml-training
            - name: LOG_LEVEL
              value: INFO
          ports:
            - containerPort: 8080    # liveness + metrics
          livenessProbe:
            httpGet: {path: /healthz, port: 8080}
          resources:
            requests: {cpu: "100m", memory: "256Mi"}
            limits:   {cpu: "500m", memory: "512Mi"}
```

---

## Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
CMD ["python", "-m", "src.operator.main"]
```

---

## Example TrainingJob Resources

### Simple (examples/trainingjob-simple.yaml)
```yaml
apiVersion: ml.example.com/v1
kind: TrainingJob
metadata:
  name: resnet-simple
  namespace: ml-training
spec:
  model: resnet50
  dataset: imagenet
  numWorkers: 1
  gpusPerWorker: 1
  hyperparameters:
    learningRate: 0.001
    batchSize: 32
    epochs: 5
```

### Distributed with Checkpoints (examples/trainingjob-distributed.yaml)
```yaml
apiVersion: ml.example.com/v1
kind: TrainingJob
metadata:
  name: bert-distributed
  namespace: ml-training
spec:
  model: bert-base-uncased
  dataset: squad
  numWorkers: 4
  gpusPerWorker: 2
  framework: pytorch
  image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
  hyperparameters:
    learningRate: 0.00005
    batchSize: 16
    epochs: 3
  checkpoint:
    enabled: true
    frequency: 1
    storage:
      type: pvc
      pvcName: bert-checkpoints
  networking:
    backend: nccl
    masterPort: 29500
  failurePolicy:
    backoffLimit: 3
```

---

## Key Design Patterns to Follow

1. **All controllers are async** — use `asyncio` patterns throughout
2. **Status is returned from handlers** — Kopf automatically patches the CR status
3. **Idempotency** — all resource creation calls should handle `AlreadyExists` gracefully
4. **Finalizers** — Kopf handles cleanup via finalizer on delete handler
5. **Structured logging** — use `logger.info(f"...")` with context (namespace/name)
6. **Metrics on all paths** — record both success and error metrics in every handler
7. **The timer drives the main reconciliation** — create/update handlers just initialize state
