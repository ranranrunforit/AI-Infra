# Production ML System - Deployment Guide

This guide will walk you through deploying the complete production ML system from scratch.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start (Local Development)](#quick-start-local-development)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [CI/CD Setup](#cicd-setup)
5. [Production Deployment](#production-deployment)
6. [Monitoring Setup](#monitoring-setup)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

Install these tools before starting:

```bash
# Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Minikube (for local testing)
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
```

### Required Accounts

- GitHub account (for CI/CD)
- Container registry access (Docker Hub, GHCR, or cloud provider)
- Kubernetes cluster (Minikube for local, GKE/EKS/AKS for production)

---

## Quick Start (Local Development)

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd project-05-production-ml-capstone

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Set Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
nano .env
```

Example `.env` file:
```bash
MLFLOW_TRACKING_URI=http://localhost:5000
MODEL_NAME=image-classifier
MODEL_VERSION=latest
API_KEYS=dev-test-key-123
LOG_LEVEL=DEBUG
ENVIRONMENT=development
```

### Step 3: Run Locally

```bash
# Option 1: Run with Flask (development)
python src/main.py

# Option 2: Run with Gunicorn (production-like)
gunicorn --bind 0.0.0.0:5000 --workers 2 src.main:app

# Test the API
curl http://localhost:5000/health
```

### Step 4: Test with Docker

```bash
# Build Docker image
docker build -t ml-api:local .

# Run container
docker run -d -p 5000:5000 \
  -e MODEL_NAME=image-classifier \
  -e API_KEYS=test-key \
  --name ml-api \
  ml-api:local

# Test
curl http://localhost:5000/health

# View logs
docker logs ml-api

# Stop and remove
docker stop ml-api && docker rm ml-api
```

---

## Kubernetes Deployment

### Step 1: Start Minikube

```bash
# Start Minikube with sufficient resources
minikube start --cpus=4 --memory=8192 --disk-size=20g

# Enable addons
minikube addons enable ingress
minikube addons enable metrics-server

# Verify
kubectl cluster-info
kubectl get nodes
```

### Step 2: Create Namespace

```bash
kubectl create namespace ml-system-dev
kubectl config set-context --current --namespace=ml-system-dev
```

### Step 3: Create Kubernetes Resources

Create `k8s/deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api
  namespace: ml-system-dev
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: ml-api
        image: ml-api:local
        imagePullPolicy: Never
        ports:
        - containerPort: 5000
        env:
        - name: MODEL_NAME
          value: "image-classifier"
        - name: API_KEYS
          value: "dev-test-key"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            cpu: 250m
            memory: 512Mi
          limits:
            cpu: 1000m
            memory: 1Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ml-api
  namespace: ml-system-dev
spec:
  selector:
    app: ml-api
  ports:
  - port: 80
    targetPort: 5000
  type: ClusterIP
```

### Step 4: Deploy to Minikube

```bash
# Load Docker image into Minikube
minikube image load ml-api:local

# Apply deployment
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods
kubectl get svc

# Port forward to access
kubectl port-forward svc/ml-api 8080:80

# Test
curl http://localhost:8080/health
```

---

## CI/CD Setup

### Step 1: GitHub Repository Setup

```bash
# Initialize git if not done
git init
git add .
git commit -m "Initial commit"

# Create GitHub repository and push
gh repo create production-ml-system --public --source=. --remote=origin --push
```

### Step 2: Configure GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Add these secrets:

```
KUBE_CONFIG_STAGING: <base64-encoded kubeconfig>
KUBE_CONFIG_PRODUCTION: <base64-encoded kubeconfig>
STAGING_API_KEY: <staging-api-key>
PRODUCTION_API_KEY: <production-api-key>
SLACK_WEBHOOK: <slack-webhook-url> (optional)
```

To get base64-encoded kubeconfig:
```bash
cat ~/.kube/config | base64 -w 0
```

### Step 3: Set Up GitHub Actions Workflows

The workflows are already in `.github/workflows/`:
- `ci.yml` - Runs on every push/PR
- `cd.yml` - Deploys to staging/production

### Step 4: Test CI Pipeline

```bash
# Make a change and push
git add .
git commit -m "Test CI pipeline"
git push origin main

# Check GitHub Actions tab in your repository
```

---

## Production Deployment

### Step 1: Choose Cloud Provider

#### Option A: Google Kubernetes Engine (GKE)

```bash
# Create GKE cluster
gcloud container clusters create ml-system-prod \
  --num-nodes=3 \
  --machine-type=n1-standard-4 \
  --zone=us-central1-a

# Get credentials
gcloud container clusters get-credentials ml-system-prod \
  --zone=us-central1-a
```

#### Option B: Amazon EKS

```bash
# Create EKS cluster (requires eksctl)
eksctl create cluster \
  --name ml-system-prod \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.xlarge \
  --nodes 3
```

#### Option C: Azure AKS

```bash
# Create AKS cluster
az aks create \
  --resource-group ml-system-rg \
  --name ml-system-prod \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-addons monitoring
```

### Step 2: Install NGINX Ingress Controller

```bash
# Add Helm repo
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

# Install
helm install nginx-ingress ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.service.type=LoadBalancer
```

### Step 3: Install cert-manager (for TLS)

```bash
# Add Helm repo
helm repo add jetstack https://charts.jetstack.io
helm repo update

# Install
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set installCRDs=true

# Apply ClusterIssuer
kubectl apply -f security/cert-manager.yaml
```

### Step 4: Deploy Application

```bash
# Create production namespace
kubectl create namespace ml-system-production

# Deploy using Helm (create Helm chart first)
helm upgrade --install ml-system ./helm/ml-system \
  --namespace ml-system-production \
  --values ./helm/ml-system/values-production.yaml \
  --set api.image.tag=v1.0.0 \
  --wait
```

### Step 5: Set Up DNS

```bash
# Get LoadBalancer IP
kubectl get svc -n ingress-nginx

# Add A record in your DNS provider:
# api.yourdomain.com -> <LoadBalancer-IP>
```

### Step 6: Verify Deployment

```bash
# Check pods
kubectl get pods -n ml-system-production

# Check service
kubectl get svc -n ml-system-production

# Test API
curl https://api.yourdomain.com/health
```

---

## Monitoring Setup

### Step 1: Install Prometheus

```bash
# Add Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace
```

### Step 2: Install Grafana

Grafana is included in kube-prometheus-stack. Access it:

```bash
# Get Grafana password
kubectl get secret -n monitoring prometheus-grafana \
  -o jsonpath="{.data.admin-password}" | base64 --decode

# Port forward
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Access at http://localhost:3000
# Username: admin
# Password: <from above>
```

### Step 3: Import Dashboards

1. Go to Grafana
2. Import dashboard ID: 315 (Kubernetes cluster monitoring)
3. Import dashboard ID: 1860 (Node Exporter)
4. Create custom dashboard for ML metrics

---

## Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n <namespace>

# Check logs
kubectl logs <pod-name> -n <namespace>

# Check events
kubectl get events -n <namespace> --sort-by='.lastTimestamp'
```

### Image Pull Errors

```bash
# Check if image exists
docker pull <image-name>

# Create image pull secret
kubectl create secret docker-registry regcred \
  --docker-server=<your-registry> \
  --docker-username=<username> \
  --docker-password=<password> \
  --docker-email=<email> \
  -n <namespace>
```

### Service Not Accessible

```bash
# Check service
kubectl get svc -n <namespace>

# Check endpoints
kubectl get endpoints -n <namespace>

# Test from within cluster
kubectl run -it --rm debug --image=alpine --restart=Never -- sh
apk add curl
curl http://ml-api.ml-system-dev/health
```

### Certificate Issues

```bash
# Check certificate
kubectl get certificate -n <namespace>
kubectl describe certificate <cert-name> -n <namespace>

# Check cert-manager logs
kubectl logs -n cert-manager deploy/cert-manager -f
```


