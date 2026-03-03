# TrainingJob Kubernetes Operator

**Project 204: Kubernetes Operator for ML Training Jobs**

A production-ready Kubernetes operator that automates the lifecycle management of machine learning training jobs. Built using Python and the Kopf framework, this operator demonstrates advanced Kubernetes controller patterns, distributed training orchestration, and production-grade observability.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Development](#development)
- [Testing](#testing)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Production Considerations](#production-considerations)
- [Contributing](#contributing)
- [License](#license)

## Overview

The TrainingJob operator extends Kubernetes with custom resources for managing ML training workloads. It handles:

- **Automated Resource Management**: Creates and manages Kubernetes Jobs, Services, and ConfigMaps
- **Distributed Training**: Coordinates multi-node, multi-GPU training with proper networking
- **Checkpoint Management**: Automated checkpoint creation, rotation, and recovery
- **Status Tracking**: Real-time progress monitoring and status updates
- **Fault Tolerance**: Automatic retry with configurable backoff policies
- **GPU Scheduling**: Intelligent GPU allocation and resource optimization
- **Observability**: Comprehensive Prometheus metrics and structured logging

### Why Use This Operator?

Training ML models at scale requires:
- Coordinating multiple workers across nodes
- Managing GPU resources efficiently
- Handling failures gracefully with checkpointing
- Monitoring training progress in real-time
- Automating repetitive deployment tasks

This operator encapsulates these complexities into a declarative API, allowing data scientists to focus on model development while the operator handles infrastructure concerns.

## Features

### Core Capabilities

- **Custom Resource Definition (CRD)**: Declarative API for defining training jobs
- **Multi-Framework Support**: PyTorch, TensorFlow, JAX
- **Distributed Training**: Built-in support for data-parallel training
- **GPU Management**: Automatic GPU allocation and scheduling
- **Checkpoint Lifecycle**: Automated creation, rotation, and resume from checkpoint
- **Fault Tolerance**: Configurable retry policies and graceful failure handling
- **Resource Optimization**: Node affinity, tolerations, and resource quotas
- **Networking**: Headless services for worker coordination
- **Monitoring Integration**: MLflow and Weights & Biases support

### Advanced Features

- **Multi-Storage Backends**: PVC, S3, GCS, NFS for checkpoints
- **Priority Scheduling**: Priority classes and preemption support
- **Success Policies**: Early stopping based on metrics thresholds
- **Progress Tracking**: Real-time epoch and metrics monitoring
- **Event Recording**: Kubernetes events for training lifecycle
- **Prometheus Metrics**: Comprehensive operator and training metrics

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Kubernetes API Server                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ Watch TrainingJob Resources
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    TrainingJob Operator                          │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          Reconciliation Loop (Kopf)                       │  │
│  │  • Create Handler                                         │  │
│  │  • Update Handler                                         │  │
│  │  • Delete Handler                                         │  │
│  │  • Timer Handler (every 30s)                             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Controllers:                                                    │
│  ┌────────────────┐ ┌──────────────┐ ┌───────────────────┐   │
│  │ JobController  │ │StatusCtrl   │ │CheckpointCtrl    │   │
│  │- Create Jobs   │ │- Monitor    │ │- Rotation        │   │
│  │- Manage Svcs   │ │- Metrics    │ │- Validation      │   │
│  └────────────────┘ └──────────────┘ └───────────────────┘   │
│                                                                   │
│  Resource Builders:                                              │
│  ┌────────────────┐ ┌──────────────┐ ┌───────────────────┐   │
│  │  JobBuilder    │ │ServiceBuilder│ │ConfigMapBuilder  │   │
│  └────────────────┘ └──────────────┘ └───────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ Create/Update/Delete
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                  Kubernetes Resources                            │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Job (parallelism=numWorkers)                             │  │
│  │  ├─ Pod (worker-0) [GPU: 2, CPU: 8, Mem: 32Gi]         │  │
│  │  ├─ Pod (worker-1) [GPU: 2, CPU: 8, Mem: 32Gi]         │  │
│  │  └─ Pod (worker-N) [GPU: 2, CPU: 8, Mem: 32Gi]         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Headless Service (for distributed training)              │  │
│  │  • Enables worker-to-worker communication                │  │
│  │  • Provides stable DNS names                             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ ConfigMap (training configuration)                        │  │
│  │  • Hyperparameters                                        │  │
│  │  • Networking settings                                    │  │
│  │  • Framework configuration                                │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

### Component Overview

#### 1. TrainingJob CRD

The custom resource definition that users interact with:

```yaml
apiVersion: ml.example.com/v1
kind: TrainingJob
metadata:
  name: bert-training
spec:
  model: bert-base-uncased
  dataset: squad
  numWorkers: 4
  gpusPerWorker: 2
  hyperparameters:
    learningRate: 0.00005
    batchSize: 16
    epochs: 3
  checkpoint:
    enabled: true
    frequency: 1
status:
  state: Running
  progress: "45%"
  currentEpoch: 2
  metrics:
    loss: 0.45
    accuracy: 0.89
```

#### 2. Operator Core

Built with **Kopf** (Kubernetes Operator Pythonic Framework):

- **Handlers**: React to create, update, delete events
- **Timer**: Periodic reconciliation every 30 seconds
- **Finalizers**: Ensure clean resource cleanup
- **Status Updates**: Keep TrainingJob status synchronized with actual state

#### 3. Controllers

- **JobController**: Manages Kubernetes Job lifecycle
- **StatusController**: Monitors and updates training progress
- **CheckpointController**: Handles checkpoint lifecycle

#### 4. Resource Builders

- **JobBuilder**: Constructs Kubernetes Job specs with proper GPU allocation, networking, and volumes
- Creates pod templates with:
  - Container specifications
  - Environment variables for distributed training
  - Volume mounts for checkpoints and configuration
  - Resource requests and limits

### State Machine

```
 Created
    │
    ▼
 Pending ──────► Initializing ──────► Running ──────► Completed
    │                                     │
    │                                     │
    └──────► Failed ◄─────────────────────┘
                │
                ▼
           Restarted (if backoffLimit not exceeded)
```

**State Descriptions:**

- **Pending**: TrainingJob created, resources being allocated
- **Initializing**: Kubernetes resources created, waiting for all workers to start
- **Running**: All workers active, training in progress
- **Completed**: Training finished successfully
- **Failed**: Training failed, retry may occur based on policy
- **Suspended**: Manually suspended by user

## Prerequisites

### Required Software

- **Kubernetes Cluster**: v1.24+ with GPU support
- **kubectl**: v1.24+
- **Docker**: 20.10+ (for building operator image)
- **Python**: 3.11+ (for development)
- **GPU Nodes**: NVIDIA GPUs with NVIDIA GPU Operator installed

### Kubernetes Requirements

```yaml
# GPU Operator (required for GPU scheduling)
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/gpu-operator/master/deployments/gpu-operator.yaml

# Prometheus Operator (optional, for metrics)
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml
```

### Cluster Configuration

1. **GPU Nodes**: At least one node with NVIDIA GPU
2. **Storage**: StorageClass for PersistentVolumes (checkpoints)
3. **Namespace**: Create `ml-training` namespace

```bash
kubectl create namespace ml-training
```

## Quick Start

### 1. Install the Operator

```bash
# Apply CRD
kubectl apply -f kubernetes/base/trainingjob-crd.yaml

# Create RBAC
kubectl apply -f kubernetes/base/rbac.yaml

# Deploy operator
kubectl apply -f kubernetes/base/deployment.yaml
kubectl apply -f kubernetes/base/service.yaml
```

### 2. Verify Installation

```bash
# Check operator is running
kubectl get pods -n ml-training

# Check CRD is registered
kubectl get crd trainingjobs.ml.example.com
```

### 3. Create Your First TrainingJob

```bash
# Apply simple example
kubectl apply -f examples/trainingjob-simple.yaml

# Watch status
kubectl get trainingjob resnet-simple -w
```

### 4. Monitor Progress

```bash
# Get detailed status
kubectl describe trainingjob resnet-simple

# View logs from worker pods
kubectl logs -l training-job=resnet-simple --tail=100
```

## Installation

### From Source

#### 1. Clone the Repository

```bash
git clone https://github.com/ai-infra-curriculum/ai-infra-senior-engineer-solutions.git
cd projects/project-204-k8s-operator
```

#### 2. Build the Operator Image

```bash
docker build -t trainingjob-operator:latest .

# Tag and push to your registry
docker tag trainingjob-operator:latest your-registry/trainingjob-operator:v1.0.0
docker push your-registry/trainingjob-operator:v1.0.0
```

#### 3. Update Deployment

```bash
# Edit kubernetes/base/deployment.yaml
# Change image to your registry URL

# Apply manifests
kubectl apply -f kubernetes/base/
```

### Using Kustomize

```bash
# Base installation
kubectl apply -k kubernetes/base

# With monitoring
kubectl apply -k kubernetes/overlays/with-monitoring
```

### Helm Chart (Future)

```bash
helm repo add trainingjob https://charts.example.com
helm install my-operator trainingjob/trainingjob-operator
```

## Usage

### Basic Training Job

```yaml
apiVersion: ml.example.com/v1
kind: TrainingJob
metadata:
  name: mnist-training
  namespace: ml-training
spec:
  model: cnn
  dataset: mnist
  numWorkers: 2
  gpusPerWorker: 1
  hyperparameters:
    learningRate: 0.001
    batchSize: 64
    epochs: 10
```

### Distributed Training with 4 GPUs

```yaml
apiVersion: ml.example.com/v1
kind: TrainingJob
metadata:
  name: bert-distributed
spec:
  model: bert-base
  dataset: wiki text
  numWorkers: 2
  gpusPerWorker: 2
  framework: pytorch
  image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
  resources:
    requests:
      memory: 32Gi
      cpu: "8"
    limits:
      memory: 64Gi
      cpu: "16"
  networking:
    backend: nccl
    masterPort: 29500
```

### Resume from Checkpoint

```yaml
apiVersion: ml.example.com/v1
kind: TrainingJob
metadata:
  name: gpt-resume
spec:
  model: gpt2
  dataset: wikitext
  numWorkers: 4
  gpusPerWorker: 2
  checkpoint:
    enabled: true
    frequency: 2
    resumeFrom: s3://checkpoints/gpt-training/checkpoint-epoch-10
    storage:
      type: s3
      s3Bucket: ml-checkpoints
```

### Advanced Configuration

```yaml
apiVersion: ml.example.com/v1
kind: TrainingJob
metadata:
  name: advanced-training
spec:
  model: resnet152
  dataset: imagenet
  numWorkers: 8
  gpusPerWorker: 4

  # Container configuration
  image: nvcr.io/nvidia/pytorch:23.10-py3
  command: ["python", "-m", "torch.distributed.run"]
  args:
    - "--nproc_per_node=4"
    - "train.py"

  # Environment variables
  env:
    - name: NCCL_DEBUG
      value: INFO
    - name: NCCL_IB_DISABLE
      value: "1"

  # Resource requirements
  resources:
    requests:
      memory: 128Gi
      cpu: "32"
      nvidia.com/gpu: "4"
    limits:
      memory: 256Gi
      cpu: "64"
      nvidia.com/gpu: "4"

  # Hyperparameters
  hyperparameters:
    learningRate: 0.0001
    batchSize: 128
    epochs: 100
    optimizer: adamw
    warmupSteps: 1000
    gradientAccumulationSteps: 4
    mixedPrecision: true
    additionalParams:
      weight_decay: "0.01"
      label_smoothing: "0.1"

  # Checkpoint configuration
  checkpoint:
    enabled: true
    frequency: 5
    storage:
      type: s3
      s3Bucket: ml-checkpoints
    retention: 10
    resumeFrom: s3://ml-checkpoints/resnet152/checkpoint-epoch-50

  # Scheduling
  scheduling:
    priority: high-priority
    nodeSelector:
      nvidia.com/gpu.product: A100-SXM4-80GB
      node-type: gpu-large
    affinity:
      podAntiAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
                - key: training-job
                  operator: In
                  values:
                    - advanced-training
            topologyKey: kubernetes.io/hostname
    tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule

  # Monitoring
  monitoring:
    enabled: true
    metricsPort: 8080
    tensorboardEnabled: true
    mlflowTrackingUri: http://mlflow.ml-infra.svc.cluster.local:5000
    wandbProject: advanced-training

  # Networking
  networking:
    backend: nccl
    masterPort: 29500
    rdmaEnabled: false

  # Failure policy
  failurePolicy:
    restartPolicy: OnFailure
    backoffLimit: 3
    activeDeadlineSeconds: 86400  # 24 hours

  # Success policy
  successPolicy:
    targetAccuracy: 0.95
    earlyStoppingPatience: 5
```

### Managing Training Jobs

#### List Training Jobs

```bash
kubectl get trainingjobs -n ml-training

# Output:
# NAME              STATE      PROGRESS   EPOCH   AGE
# bert-training     Running    45%        2       30m
# resnet-training   Completed  100%       10      2h
```

#### Get Detailed Status

```bash
kubectl describe trainingjob bert-training -n ml-training
```

#### View Logs

```bash
# All workers
kubectl logs -l training-job=bert-training --tail=100 -f

# Specific worker
kubectl logs bert-training-training-xyz12 -f
```

#### Delete Training Job

```bash
kubectl delete trainingjob bert-training
```

### Monitoring Training Progress

#### Using kubectl

```bash
# Watch status in real-time
kubectl get trainingjob bert-training -w

# Get metrics from status
kubectl get trainingjob bert-training -o jsonpath='{.status.metrics}'
```

#### Using Prometheus

Query training metrics:

```promql
# GPU utilization
trainingjob_gpu_utilization_percent{namespace="ml-training", training_job="bert-training"}

# Training loss
trainingjob_loss{namespace="ml-training", training_job="bert-training"}

# Progress
trainingjob_progress_percent{namespace="ml-training", training_job="bert-training"}
```

## API Reference

### TrainingJob Spec

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model architecture to train |
| `dataset` | string | Yes | Dataset name |
| `numWorkers` | integer | Yes | Number of worker replicas (1-100) |
| `gpusPerWorker` | integer | No | GPUs per worker (default: 1) |
| `framework` | string | No | ML framework (pytorch/tensorflow/jax) |
| `image` | string | No | Container image |
| `command` | []string | No | Container command |
| `args` | []string | No | Container arguments |
| `env` | []object | No | Environment variables |
| `resources` | object | No | Resource requests/limits |
| `hyperparameters` | object | No | Training hyperparameters |
| `checkpoint` | object | No | Checkpoint configuration |
| `scheduling` | object | No | Scheduling policies |
| `monitoring` | object | No | Monitoring configuration |
| `networking` | object | No | Networking settings |
| `failurePolicy` | object | No | Failure handling |
| `successPolicy` | object | No | Success criteria |

### TrainingJob Status

| Field | Type | Description |
|-------|------|-------------|
| `state` | string | Current state (Pending/Initializing/Running/Completed/Failed) |
| `conditions` | []object | Status conditions |
| `progress` | string | Training progress percentage |
| `currentEpoch` | integer | Current epoch number |
| `totalEpochs` | integer | Total epochs |
| `metrics` | object | Current training metrics |
| `workers` | object | Worker status (active/succeeded/failed) |
| `checkpoint` | object | Checkpoint information |
| `resources` | object | Allocated resources |
| `startTime` | string | Training start time |
| `completionTime` | string | Training completion time |
| `duration` | string | Total duration |
| `failureReason` | string | Failure reason if failed |
| `restartCount` | integer | Number of restarts |

### Hyperparameters Object

| Field | Type | Description |
|-------|------|-------------|
| `learningRate` | number | Learning rate |
| `batchSize` | integer | Batch size |
| `epochs` | integer | Number of epochs |
| `optimizer` | string | Optimizer (adam/sgd/adamw/rmsprop) |
| `warmupSteps` | integer | Warmup steps |
| `gradientAccumulationSteps` | integer | Gradient accumulation steps |
| `mixedPrecision` | boolean | Enable mixed precision |
| `additionalParams` | map[string]string | Additional parameters |

### Checkpoint Object

| Field | Type | Description |
|-------|------|-------------|
| `enabled` | boolean | Enable checkpointing |
| `frequency` | integer | Checkpoint every N epochs |
| `storage` | object | Storage configuration |
| `resumeFrom` | string | Checkpoint path to resume from |
| `retention` | integer | Number of checkpoints to retain |

## Development

### Local Development Setup

#### 1. Clone and Setup

```bash
git clone <repository>
cd project-204-k8s-operator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

#### 2. Run Locally (Outside Cluster)

```bash
# Ensure you have kubeconfig pointing to a cluster
export KUBECONFIG=~/.kube/config

# Run the operator
python -m src.operator.main
```

#### 3. Run in Cluster

```bash
# Build image
docker build -t trainingjob-operator:dev .

# Load to kind cluster (for local testing)
kind load docker-image trainingjob-operator:dev

# Deploy
kubectl apply -f kubernetes/base/
```

### Code Structure

```
src/
├── operator/
│   └── main.py           # Main operator entry point with Kopf handlers
├── controllers/
│   ├── job_controller.py       # Manages Kubernetes Job lifecycle
│   ├── status_controller.py    # Updates training status
│   └── checkpoint_controller.py # Manages checkpoints
├── resources/
│   └── job_builder.py    # Builds Kubernetes resource specs
├── crd/
│   ├── trainingjob_crd.py     # CRD Python model
│   ├── validation.py    # Spec validation logic
│   └── defaults.py      # Default values
└── utils/
    ├── k8s_client.py    # Kubernetes client wrapper
    ├── logger.py        # Structured logging
    └── metrics.py       # Prometheus metrics
```

### Adding a New Feature

Example: Adding support for model versioning

1. **Update CRD** (`kubernetes/base/trainingjob-crd.yaml`):
```yaml
spec:
  properties:
    modelVersion:
      type: string
      description: Model version to train
```

2. **Update Controller** (`src/controllers/job_controller.py`):
```python
def create_training_resources(...):
    model_version = spec.get('modelVersion', 'latest')
    # Use model_version in resource creation
```

3. **Update Builder** (`src/resources/job_builder.py`):
```python
def _build_env_vars(...):
    env_vars.append(client.V1EnvVar(
        name='MODEL_VERSION',
        value=spec.get('modelVersion', 'latest')
    ))
```

4. **Add Tests** (`tests/test_versioning.py`):
```python
def test_model_version_env_var():
    spec = {'modelVersion': 'v2.0'}
    builder = JobBuilder()
    env_vars = builder._build_env_vars('test', spec)
    assert any(e.name == 'MODEL_VERSION' and e.value == 'v2.0' for e in env_vars)
```

### Running Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests (requires cluster)
pytest tests/integration/

# All tests with coverage
pytest --cov=src tests/
```

### Debugging

#### Enable Debug Logging

```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Or in deployment
kubectl set env deployment/trainingjob-operator LOG_LEVEL=DEBUG -n ml-training
```

#### View Operator Logs

```bash
kubectl logs -f deployment/trainingjob-operator -n ml-training
```

#### Use Kopf CLI

```bash
# Install kopf CLI
pip install kopf

# Run with verbose logging
kopf run --verbose src/operator/main.py
```

## Testing

### Unit Tests

Test individual components in isolation:

```bash
pytest tests/unit/test_job_builder.py
pytest tests/unit/test_validators.py
pytest tests/unit/test_status_controller.py
```

### Integration Tests

Test the operator with a real Kubernetes cluster:

```bash
# Requires a test cluster (kind, minikube, etc.)
pytest tests/integration/test_trainingjob_lifecycle.py
```

### End-to-End Tests

Full workflow testing:

```bash
# Create training job
kubectl apply -f tests/e2e/fixtures/simple-training.yaml

# Run validation script
./tests/e2e/validate-training.sh
```

## Monitoring

### Prometheus Metrics

The operator exposes metrics on port 8080:

```bash
# Access metrics
kubectl port-forward svc/trainingjob-operator 8080:8080 -n ml-training
curl http://localhost:8080/metrics
```

**Available Metrics:**

```
# Reconciliation
trainingjob_reconciliation_total{namespace, training_job, result}
trainingjob_reconciliation_duration_seconds{namespace, training_job}
trainingjob_reconciliation_errors_total{namespace, training_job, error_type}

# Training Jobs
trainingjob_total{namespace, state}
trainingjob_progress_percent{namespace, training_job}
trainingjob_current_epoch{namespace, training_job}
trainingjob_duration_seconds{namespace, training_job, state}

# Resources
trainingjob_allocated_gpus{namespace, training_job}
trainingjob_allocated_workers{namespace, training_job}

# Training Metrics
trainingjob_loss{namespace, training_job}
trainingjob_accuracy{namespace, training_job}
trainingjob_gpu_utilization_percent{namespace, training_job}

# Lifecycle
trainingjob_created_total{namespace}
trainingjob_completed_total{namespace, result}
trainingjob_failed_total{namespace, reason}
```

### Grafana Dashboard

Import the provided Grafana dashboard:

```bash
# Dashboard JSON
kubectl create configmap grafana-dashboard \
  --from-file=monitoring/grafana/dashboard.json \
  -n monitoring
```

Key panels:
- Active Training Jobs by State
- Training Progress Over Time
- GPU Utilization
- Loss and Accuracy Trends
- Operator Performance (reconciliation time, errors)

### Alerts

Configure Prometheus alerts:

```yaml
groups:
  - name: trainingjob
    rules:
      - alert: TrainingJobFailed
        expr: trainingjob_failed_total > 0
        annotations:
          summary: Training job failed

      - alert: HighGPUIdleTime
        expr: trainingjob_gpu_utilization_percent < 50
        for: 10m
        annotations:
          summary: GPU utilization below 50% for 10 minutes
```

## Troubleshooting

### Common Issues

#### 1. Operator Not Starting

**Symptoms**: Operator pod in CrashLoopBackOff

**Solutions**:
```bash
# Check logs
kubectl logs deployment/trainingjob-operator -n ml-training

# Common causes:
# - Missing RBAC permissions
# - Invalid kubeconfig
# - CRD not installed

# Fix RBAC
kubectl apply -f kubernetes/base/rbac.yaml

# Ensure CRD exists
kubectl get crd trainingjobs.ml.example.com
```

#### 2. TrainingJob Stuck in Pending

**Symptoms**: Job stays in Pending state

**Solutions**:
```bash
# Check events
kubectl describe trainingjob <name>

# Common causes:
# - Insufficient GPU resources
# - Node selector not matching any nodes
# - Image pull errors

# Check node capacity
kubectl describe nodes | grep -A5 "Capacity:"

# Check pod events
kubectl get pods -l training-job=<name>
kubectl describe pod <pod-name>
```

#### 3. Workers Not Communicating

**Symptoms**: Distributed training fails, NCCL errors

**Solutions**:
```bash
# Check headless service
kubectl get svc <name>-headless

# Check networking
kubectl exec <pod-name> -- nslookup <name>-headless

# Common causes:
# - Headless service not created
# - NetworkPolicy blocking traffic
# - NCCL misconfiguration

# Check NCCL debug logs
kubectl logs <pod-name> | grep NCCL
```

#### 4. High Memory Usage

**Symptoms**: Pods being OOM killed

**Solutions**:
```yaml
# Increase memory limits
spec:
  resources:
    limits:
      memory: 64Gi  # Increase this
    requests:
      memory: 32Gi
```

#### 5. Checkpoints Not Saving

**Symptoms**: Checkpoints not appearing in storage

**Solutions**:
```bash
# Check PVC exists
kubectl get pvc

# Check volume mounts
kubectl describe pod <pod-name> | grep -A10 "Mounts:"

# Check S3 credentials (if using S3)
kubectl get secret aws-credentials
```

### Debug Mode

Enable comprehensive debugging:

```yaml
# deployment.yaml
env:
  - name: LOG_LEVEL
    value: "DEBUG"
  - name: NCCL_DEBUG
    value: "INFO"
  - name: TORCH_DISTRIBUTED_DEBUG
    value: "DETAIL"
```

### Collecting Diagnostic Information

```bash
# Operator logs
kubectl logs deployment/trainingjob-operator -n ml-training > operator-logs.txt

# Training job description
kubectl describe trainingjob <name> > trainingjob.yaml

# Worker pod logs
kubectl logs -l training-job=<name> --all-containers > worker-logs.txt

# Events
kubectl get events -n ml-training --sort-by='.lastTimestamp' > events.txt
```

## Production Considerations

### High Availability

Run multiple operator replicas:

```yaml
spec:
  replicas: 3  # Instead of 1
  strategy:
    type: RollingUpdate
```

**Note**: Kopf handles leader election automatically.

### Resource Limits

Set appropriate operator resource limits:

```yaml
resources:
  requests:
    cpu: 500m
    memory: 512Mi
  limits:
    cpu: 2000m
    memory: 2Gi
```

### Security

#### 1. RBAC Principle of Least Privilege

Review and minimize permissions in `kubernetes/base/rbac.yaml`.

#### 2. Pod Security

```yaml
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 1000
  containers:
    - securityContext:
        allowPrivilegeEscalation: false
        capabilities:
          drop: ["ALL"]
```

#### 3. Network Policies

Restrict operator network access:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: trainingjob-operator
spec:
  podSelector:
    matchLabels:
      app: trainingjob-operator
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: prometheus
      ports:
        - port: 8080
  egress:
    - to:
        - namespaceSelector: {}
      ports:
        - port: 443  # Kubernetes API
```

### Performance Tuning

#### 1. Reconciliation Interval

Adjust timer interval based on workload:

```python
@kopf.timer(GROUP, VERSION, PLURAL, interval=60.0)  # 60 seconds instead of 30
```

#### 2. Batch Processing

Process multiple events efficiently:

```python
# Configure kopf batch settings
settings.batching.worker_limit = 10
settings.batching.idle_timeout = 5.0
settings.batching.batch_window = 1.0
```

#### 3. Metrics Cardinality

Limit metrics labels to avoid high cardinality:
- Use `training_job` label judiciously
- Consider aggregating per-namespace instead of per-job

### Scaling

The operator can handle:
- **100+ concurrent training jobs**
- **1000+ pods** across jobs
- **Multiple namespaces**

For larger scale:
- Use horizontal pod autoscaling
- Consider sharding by namespace
- Implement caching for K8s API calls

### Backup and Recovery

#### 1. Backup CRDs

```bash
# Export all training jobs
kubectl get trainingjobs -A -o yaml > trainingjobs-backup.yaml
```

#### 2. Checkpoint Storage

Ensure checkpoints are stored in durable external storage (S3, GCS, NFS).

#### 3. Disaster Recovery

Document procedures for:
- Recreating the operator
- Recovering training jobs
- Restoring from checkpoints

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run linting and tests
5. Submit a pull request

### Code Standards

- Follow PEP 8 style guide
- Add type hints
- Write docstrings
- Include unit tests
- Update documentation

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Kopf](https://kopf.readthedocs.io/)
- Inspired by Kubeflow Training Operator
- Uses NVIDIA GPU Operator for GPU management

## Contact

For questions or support:
- **Issues**: [GitHub Issues](https://github.com/ai-infra-curriculum/issues)
- **Email**: ai-infra-curriculum@joshua-ferguson.com
- **Documentation**: [Full Docs](https://docs.ai-infra-curriculum.com)

## Additional Resources

- [Kubernetes Operators](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/)
- [Custom Resource Definitions](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)
- [Distributed Training with PyTorch](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [Kopf Framework](https://kopf.readthedocs.io/)

---

**Version**: 1.0.0
**Last Updated**: October 2025
**Maintainers**: AI Infrastructure Curriculum Team
