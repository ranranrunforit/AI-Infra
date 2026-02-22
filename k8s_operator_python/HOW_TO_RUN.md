# How to Run: Kubernetes Operator

This guide provides instructions on how to set up, run, and test the Kubernetes Operator for ML Training Jobs.

## Prerequisites

- **Python 3.9+** installed
- **Docker** installed (optional, for containerized run)
- **Kubernetes Cluster** (Kind, Minikube, or cloud provider)
- **kubectl** configured to point to your cluster

## 1. Local Setup

### Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:
```bash
pip install kopf kubernetes prometheus-client
```

## 2. Running the Operator Locally

### Option A: Using Kopf CLI (Recommended for Development)

Run the operator source code directly against your configured Kubernetes cluster.

```bash
# Set log level (optional)
export LOG_LEVEL=INFO

# Run the operator
kopf run src/operator/main.py --verbose
```
*Note: Ensure your `kubectl` context is set correctly before running.*

### Option B: Running as a Python Module

```bash
python -m src.operator.main
```

## 3. Deployment to Kubernetes

### Build Docker Image

```bash
docker build -t my-k8s-operator:latest .
```

### Deploy to Cluster

Apply the manifests in the `kubernetes` directory.

```bash
# 1. Install CRD
kubectl apply -f kubernetes/base/trainingjob-crd.yaml

# 2. Deploy Operator
kubectl apply -f kubernetes/base/deployment.yaml
kubectl apply -f kubernetes/base/rbac.yaml
```

## 4. Usage

### Create a TrainingJob

Apply an example TrainingJob manifest to trigger the operator.

```yaml
# example-job.yaml
apiVersion: ml.example.com/v1
kind: TrainingJob
metadata:
  name: example-model-training
spec:
  model: "resnet50"
  dataset: "cifar10"
  numWorkers: 2
  gpusPerWorker: 0
  hyperparameters:
    batchSize: 32
    learningRate: 0.001
    epochs: 10
```

```bash
kubectl apply -f example-job.yaml
```

### Verify Status

Check the status of the created job:

```bash
kubectl get trainingjob example-model-training -o yaml
```

## 5. Testing

Run unit tests to verify logic.

```bash
pytest tests/
```

## 6. Accessing Metrics

The operator exposes Prometheus metrics on port `9090`.

```bash
# Forward port if running in cluster
kubectl port-forward svc/operator-service 9090:9090

# Access metrics
curl http://localhost:9090/metrics
```
