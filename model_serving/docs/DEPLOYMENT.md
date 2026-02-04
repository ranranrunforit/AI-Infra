# Deployment Guide - High-Performance Model Serving

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Deployment](#local-deployment)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Cloud Provider Specific](#cloud-provider-specific)
- [GPU Node Setup](#gpu-node-setup)
- [Storage Configuration](#storage-configuration)
- [Network Configuration](#network-configuration)
- [TLS/SSL Setup](#tlsssl-setup)
- [CI/CD Pipeline](#cicd-pipeline)
- [Deployment Strategies](#deployment-strategies)
- [Disaster Recovery](#disaster-recovery)
- [Monitoring Setup](#monitoring-setup)

---

## Prerequisites

### System Requirements

**Hardware**:
- NVIDIA GPU (V100, A100, H100 recommended)
- CUDA 12.1 or later
- Minimum 16GB GPU memory
- Minimum 32GB system RAM
- 100GB+ free disk space

**Software**:
- Ubuntu 20.04+ or equivalent Linux distribution
- Docker 20.10+
- Kubernetes 1.24+ (for production)
- kubectl CLI
- Helm 3.x (optional)
- NVIDIA drivers 525+
- NVIDIA Container Toolkit

### Access Requirements

- Docker registry access (Docker Hub or private registry)
- Kubernetes cluster admin credentials
- Cloud provider credentials (AWS/GCP/Azure)
- Model storage access (S3, GCS, or NFS)

---

## Local Deployment

### Step 1: Clone and Setup

```bash
# Clone repository
git clone https://github.com/ai-infra-curriculum/ai-infra-senior-engineer-solutions.git
cd ai-infra-senior-engineer-solutions/projects/project-202-model-serving

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

Example `.env`:
```bash
# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
LOG_LEVEL=INFO

# GPU
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_FRACTION=0.9

# Models
MODEL_CACHE_DIR=/tmp/model_cache
DEFAULT_MODEL=resnet50-fp16

# Monitoring
METRICS_ENABLED=true
PROMETHEUS_PORT=9090

# Tracing
JAEGER_ENABLED=true
JAEGER_HOST=localhost
JAEGER_PORT=6831
```

### Step 3: Prepare Models

```bash
# Create model directory
mkdir -p models

# Convert PyTorch model to TensorRT
python scripts/convert_model.py \
    --model resnet50 \
    --precision fp16 \
    --batch-size 32 \
    --output models/resnet50-fp16.trt

# Verify model
ls -lh models/
```

### Step 4: Start Server

```bash
# Start server
python -m uvicorn src.serving.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1

# In another terminal, test
curl http://localhost:8000/health
```

**Expected Output**:
```json
{
  "status": "healthy",
  "models_loaded": ["resnet50-fp16"],
  "gpu_available": true,
  "uptime_seconds": 5.2,
  "version": "1.0.0"
}
```

---

## Docker Deployment

### Step 1: Build Docker Image

```bash
# Build image
docker build -t model-serving:latest -f docker/Dockerfile .

# Verify image
docker images | grep model-serving

# Tag for registry
docker tag model-serving:latest your-registry.com/model-serving:v1.0.0
```

### Step 2: Run with Docker

```bash
# Run single container
docker run --rm -d \
    --name model-serving \
    --gpus all \
    -p 8000:8000 \
    -p 9090:9090 \
    -v $(pwd)/models:/models \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e MODEL_CACHE_DIR=/models \
    model-serving:latest

# View logs
docker logs -f model-serving

# Test
curl http://localhost:8000/health
```

### Step 3: Docker Compose Stack

```bash
# Start full stack (includes Jaeger, Prometheus, Grafana)
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Access services
# - Model API: http://localhost:8000
# - Jaeger UI: http://localhost:16686
# - Prometheus: http://localhost:9091
# - Grafana: http://localhost:3000 (admin/admin)

# Stop stack
docker-compose -f docker/docker-compose.yml down
```

### Step 4: Push to Registry

```bash
# Login to registry
docker login your-registry.com

# Push image
docker push your-registry.com/model-serving:v1.0.0

# Verify push
docker pull your-registry.com/model-serving:v1.0.0
```

---

## Kubernetes Deployment

### Development Environment

```bash
# Apply base configuration
kubectl apply -k kubernetes/base/

# Verify deployment
kubectl get pods -n model-serving
kubectl get svc -n model-serving

# View logs
kubectl logs -n model-serving -l app=model-serving -f

# Port forward for testing
kubectl port-forward -n model-serving svc/model-serving 8000:80
curl http://localhost:8000/health
```

### Staging Environment

```bash
# Apply staging overlay
kubectl apply -k kubernetes/overlays/staging/

# Verify
kubectl get pods -n model-serving-staging

# Run smoke tests
./scripts/smoke-test.sh staging
```

### Production Environment

```bash
# Review production configuration
cat kubernetes/overlays/prod/kustomization.yaml

# Apply with dry-run first
kubectl apply -k kubernetes/overlays/prod/ --dry-run=client

# Apply production deployment
kubectl apply -k kubernetes/overlays/prod/

# Verify deployment status
kubectl rollout status deployment/prod-model-serving -n model-serving

# Check pod status
kubectl get pods -n model-serving -l app=model-serving

# Check HPA
kubectl get hpa -n model-serving

# Check service
kubectl get svc -n model-serving
```

### Verification Checklist

```bash
# 1. Pods are running
kubectl get pods -n model-serving | grep Running

# 2. Health checks passing
kubectl exec -n model-serving <pod-name> -- curl localhost:8000/health

# 3. Metrics available
kubectl exec -n model-serving <pod-name> -- curl localhost:9090/metrics

# 4. GPU accessible
kubectl exec -n model-serving <pod-name> -- nvidia-smi

# 5. Service accessible
export SVC_IP=$(kubectl get svc -n model-serving model-serving -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl http://$SVC_IP/health

# 6. Auto-scaling configured
kubectl describe hpa -n model-serving

# 7. Resource limits set
kubectl describe pod -n model-serving <pod-name> | grep -A 5 Limits
```

---

## Cloud Provider Specific

### AWS EKS

#### Setup EKS Cluster

```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create cluster with GPU nodes
eksctl create cluster \
    --name model-serving-cluster \
    --region us-west-2 \
    --nodegroup-name gpu-nodes \
    --node-type p3.2xlarge \
    --nodes 2 \
    --nodes-min 2 \
    --nodes-max 10 \
    --managed

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verify GPU nodes
kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"
```

#### Configure EBS for Model Storage

```bash
# Install EBS CSI driver
eksctl create iamserviceaccount \
    --name ebs-csi-controller-sa \
    --namespace kube-system \
    --cluster model-serving-cluster \
    --attach-policy-arn arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy \
    --approve

kubectl apply -k "github.com/kubernetes-sigs/aws-ebs-csi-driver/deploy/kubernetes/overlays/stable/?ref=master"

# Create storage class
kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
  encrypted: "true"
volumeBindingMode: WaitForFirstConsumer
EOF
```

#### Configure Load Balancer

```bash
# Install AWS Load Balancer Controller
helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
    -n kube-system \
    --set clusterName=model-serving-cluster

# Update service to use NLB
kubectl patch svc model-serving -n model-serving -p '
{
  "metadata": {
    "annotations": {
      "service.beta.kubernetes.io/aws-load-balancer-type": "nlb",
      "service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled": "true"
    }
  }
}'
```

### GCP GKE

#### Setup GKE Cluster

```bash
# Create GKE cluster with GPU nodes
gcloud container clusters create model-serving-cluster \
    --zone us-central1-a \
    --machine-type n1-standard-8 \
    --accelerator type=nvidia-tesla-v100,count=1 \
    --num-nodes 2 \
    --enable-autoscaling \
    --min-nodes 2 \
    --max-nodes 10 \
    --addons GcePersistentDiskCsiDriver

# Get credentials
gcloud container clusters get-credentials model-serving-cluster

# Install NVIDIA GPU device plugin
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

#### Configure GCS for Model Storage

```bash
# Create GCS bucket
gsutil mb gs://model-serving-models

# Create service account
gcloud iam service-accounts create model-serving-sa

# Grant access
gsutil iam ch serviceAccount:model-serving-sa@PROJECT_ID.iam.gserviceaccount.com:objectViewer gs://model-serving-models

# Create Kubernetes secret
kubectl create secret generic gcs-key \
    --from-file=key.json=~/gcs-key.json \
    -n model-serving
```

### Azure AKS

#### Setup AKS Cluster

```bash
# Create resource group
az group create --name ModelServingRG --location eastus

# Create AKS cluster with GPU
az aks create \
    --resource-group ModelServingRG \
    --name model-serving-cluster \
    --node-count 2 \
    --node-vm-size Standard_NC6s_v3 \
    --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group ModelServingRG --name model-serving-cluster

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
```

---

## GPU Node Setup

### Install NVIDIA Drivers

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install NVIDIA drivers
sudo apt-get install -y nvidia-driver-525

# Reboot
sudo reboot

# Verify installation
nvidia-smi
```

### Install CUDA Toolkit

```bash
# Download CUDA installer
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run

# Install
sudo sh cuda_12.1.0_530.30.02_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

### Install NVIDIA Container Toolkit

```bash
# Add repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu20.04 nvidia-smi
```

### Kubernetes GPU Operator (Alternative)

```bash
# Add NVIDIA Helm repo
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# Install GPU Operator
helm install --wait --generate-name \
    -n gpu-operator --create-namespace \
    nvidia/gpu-operator

# Verify
kubectl get pods -n gpu-operator
kubectl describe node <node-name> | grep nvidia.com/gpu
```

---

## Storage Configuration

### Local Storage (Development)

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-pv-local
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: local-storage
  local:
    path: /mnt/models
  nodeAffinity:
    required:
      nodeSelectorTerms:
      - matchExpressions:
        - key: kubernetes.io/hostname
          operator: In
          values:
          - gpu-node-1
```

### NFS Storage (Multi-node)

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-pv-nfs
spec:
  capacity:
    storage: 500Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: nfs
  nfs:
    server: nfs-server.example.com
    path: /exports/models
```

### Cloud Storage (S3/GCS)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: model-serving
spec:
  initContainers:
  - name: model-downloader
    image: amazon/aws-cli
    command:
    - sh
    - -c
    - |
      aws s3 sync s3://my-models-bucket/ /models/
    volumeMounts:
    - name: models
      mountPath: /models
    env:
    - name: AWS_ACCESS_KEY_ID
      valueFrom:
        secretKeyRef:
          name: aws-credentials
          key: access-key-id
    - name: AWS_SECRET_ACCESS_KEY
      valueFrom:
        secretKeyRef:
          name: aws-credentials
          key: secret-access-key
  containers:
  - name: model-serving
    image: model-serving:latest
    volumeMounts:
    - name: models
      mountPath: /models
  volumes:
  - name: models
    emptyDir: {}
```

---

## Network Configuration

### Ingress Controller

```bash
# Install NGINX Ingress Controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx \
    --namespace ingress-nginx \
    --create-namespace

# Create Ingress resource
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: model-serving-ingress
  namespace: model-serving
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.model-serving.example.com
    secretName: tls-secret
  rules:
  - host: api.model-serving.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: model-serving
            port:
              number: 80
EOF
```

### Service Mesh (Istio)

```bash
# Install Istio
istioctl install --set profile=default -y

# Enable sidecar injection
kubectl label namespace model-serving istio-injection=enabled

# Create VirtualService
kubectl apply -f - <<EOF
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: model-serving
  namespace: model-serving
spec:
  hosts:
  - model-serving.example.com
  gateways:
  - model-serving-gateway
  http:
  - match:
    - uri:
        prefix: /v1/
    route:
    - destination:
        host: model-serving
        port:
          number: 80
    timeout: 30s
    retries:
      attempts: 3
      perTryTimeout: 10s
EOF
```

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: model-serving-policy
  namespace: model-serving
spec:
  podSelector:
    matchLabels:
      app: model-serving
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 9090
```

---

## TLS/SSL Setup

### Generate Self-Signed Certificate (Development)

```bash
# Generate certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout tls.key -out tls.crt \
    -subj "/CN=api.model-serving.local/O=model-serving"

# Create Kubernetes secret
kubectl create secret tls tls-secret \
    --cert=tls.crt \
    --key=tls.key \
    -n model-serving
```

### Use Let's Encrypt (Production)

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF

# Update Ingress to use cert-manager
kubectl annotate ingress model-serving-ingress \
    -n model-serving \
    cert-manager.io/cluster-issuer=letsencrypt-prod
```

---

## CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy Model Serving

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: pytest tests/ -v --cov=src

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Build Docker image
      run: docker build -t model-serving:${{ github.sha }} -f docker/Dockerfile .
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker tag model-serving:${{ github.sha }} ${{ secrets.DOCKER_REGISTRY }}/model-serving:${{ github.sha }}
        docker push ${{ secrets.DOCKER_REGISTRY }}/model-serving:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3
    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBECONFIG }}
    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/model-serving \
          model-serving=${{ secrets.DOCKER_REGISTRY }}/model-serving:${{ github.sha }} \
          -n model-serving
        kubectl rollout status deployment/model-serving -n model-serving
```

### GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

test:
  stage: test
  image: python:3.10
  script:
    - pip install -r requirements.txt
    - pytest tests/ -v --cov=src

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA -f docker/Dockerfile .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

deploy:production:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl set image deployment/model-serving model-serving=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA -n model-serving
    - kubectl rollout status deployment/model-serving -n model-serving
  only:
    - main
```

---

## Deployment Strategies

### Rolling Update

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-serving
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # Max 1 extra pod during update
      maxUnavailable: 1  # Max 1 pod down during update
```

**Process**:
1. Create 1 new pod with new version
2. Wait for it to be ready
3. Terminate 1 old pod
4. Repeat until all pods updated

### Blue-Green Deployment

```bash
# Deploy green version
kubectl apply -f kubernetes/green-deployment.yaml

# Wait for green to be ready
kubectl rollout status deployment/model-serving-green

# Switch traffic
kubectl patch service model-serving -p '{"spec":{"selector":{"version":"green"}}}'

# Verify
curl http://api.example.com/health

# Rollback if needed
kubectl patch service model-serving -p '{"spec":{"selector":{"version":"blue"}}}'

# Clean up old version
kubectl delete deployment model-serving-blue
```

### Canary Deployment

See [RUNBOOK.md](RUNBOOK.md#canary-deployments) for detailed canary procedures.

---

## Disaster Recovery

### Backup Procedures

```bash
# Backup Kubernetes resources
kubectl get all -n model-serving -o yaml > backup-$(date +%Y%m%d).yaml

# Backup models
aws s3 sync /models/ s3://model-backup-bucket/models/

# Backup configuration
kubectl get configmap -n model-serving -o yaml > configmap-backup.yaml
kubectl get secret -n model-serving -o yaml > secret-backup.yaml
```

### Recovery Procedures

```bash
# Restore Kubernetes resources
kubectl apply -f backup-20240115.yaml

# Restore models
aws s3 sync s3://model-backup-bucket/models/ /models/

# Verify
kubectl get pods -n model-serving
curl http://api.example.com/health
```

### Multi-Region Setup

```bash
# Primary region (us-west-2)
kubectl config use-context us-west-2

# Deploy primary
kubectl apply -k kubernetes/overlays/prod/

# Secondary region (us-east-1)
kubectl config use-context us-east-1

# Deploy secondary
kubectl apply -k kubernetes/overlays/prod/

# Configure DNS failover (Route53 example)
aws route53 create-health-check ...
```

---

## Monitoring Setup

See [Step-by-Step Guide](STEP_BY_STEP.md#step-8-monitoring-and-observability) for detailed monitoring setup.

**Quick Setup**:

```bash
# Install Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace

# Install Jaeger
kubectl create namespace monitoring
kubectl apply -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/master/deploy/crds/jaegertracing.io_jaegers_crd.yaml
kubectl apply -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/master/deploy/operator.yaml -n monitoring

# Access UIs
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
kubectl port-forward -n monitoring svc/jaeger-query 16686:16686
```

---

## Post-Deployment Validation

```bash
# Run validation script
./scripts/validate-deployment.sh

# Expected checks:
# ✓ All pods running
# ✓ Health checks passing
# ✓ Metrics endpoint accessible
# ✓ GPU accessible from pods
# ✓ Models loaded
# ✓ API responds to requests
# ✓ Auto-scaling configured
# ✓ Monitoring dashboards available
```

---

## References

- [API Reference](API.md)
- [Architecture Documentation](ARCHITECTURE.md)
- [Step-by-Step Implementation Guide](STEP_BY_STEP.md)
- [Operations Runbook](RUNBOOK.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)

---

