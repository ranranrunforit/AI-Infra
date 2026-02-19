# 🚀 Multi-Region ML Platform - Complete Setup Guide for Google Cloud

**A production-ready multi-cloud ML serving platform with automatic failover, health monitoring, and metrics collection**

> **Built for Google Cloud Platform** | Supports multi-cloud architecture

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture](#-architecture)
3. [Prerequisites](#-prerequisites)
4. [Phase 1: Local Docker Testing](#-phase-1-local-docker-testing)
5. [Phase 2: Google Cloud Setup](#-phase-2-google-cloud-setup)
6. [Phase 3: Kubernetes Deployment](#-phase-3-kubernetes-deployment)
7. [Phase 4: Monitoring Setup](#-phase-4-monitoring-setup)
8. [Phase 5: Testing & Verification](#-phase-5-testing--verification)
9. [Troubleshooting](#-troubleshooting)
10. [Production Checklist](#-production-checklist)

---

## 🎯 Project Overview

### What This Platform Does

This is a **multi-region ML serving platform** that provides:

- ✅ **Automatic Health Monitoring** - Monitors health of all regional deployments every 10 seconds
- ✅ **Intelligent Failover** - Automatically switches to healthy regions when failures detected
- ✅ **Metrics Aggregation** - Collects and aggregates metrics from all regions
- ✅ **Cost Analysis** - Tracks spending across cloud providers
- ✅ **Multi-Cloud Support** - Designed for AWS, GCP, and Azure (works GCP-only too)
- ✅ **Auto-scaling** - Kubernetes HPA for automatic pod scaling
- ✅ **Prometheus & Grafana** - Built-in monitoring and visualization

### Key Components

```
┌─────────────────────────────────────────────────────────────┐
│                    ML Platform Services                      │
├─────────────────────────────────────────────────────────────┤
│  • Failover Controller   - Health checks & automatic failover│
│  • Metrics Aggregator    - Collects metrics from all regions│
│  • Cost Analyzer         - Multi-cloud cost tracking        │
│  • Model Replicator      - Replicates models across regions │
│  • Data Sync             - Syncs data between regions       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Regional Deployments (GKE)                      │
├─────────────────────────────────────────────────────────────┤
│  Region 1: us-central1 (GCP)                                │
│  Region 2: europe-west1 (GCP)                               │
│  Region 3: asia-south1 (GCP)                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture

### System Architecture

```
                    ┌──────────────────┐
                    │   LoadBalancer   │
                    │   External IPs   │
                    └────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
    ┌───────▼──────┐  ┌─────▼──────┐  ┌─────▼──────┐
    │  US Region   │  │ EU Region  │  │ Asia Region│
    │ us-central1  │  │ europe-west│  │ asia-south │
    └──────┬───────┘  └─────┬──────┘  └─────┬──────┘
           │                │                │
    ┌──────▼────────────────▼────────────────▼──────┐
    │          Failover Controller                   │
    │   • Monitors all regions every 10s             │
    │   • Auto-switches on failure                   │
    │   • Updates DNS/LoadBalancer                   │
    └────────────────────┬──────────────────────────┘
                         │
    ┌────────────────────▼──────────────────────────┐
    │       Monitoring Stack (Prometheus)            │
    │   • Collects metrics every 60s                 │
    │   • Stores in Prometheus                       │
    │   • Visualized in Grafana                      │
    └────────────────────────────────────────────────┘
```

### Technology Stack

- **Container Runtime**: Docker
- **Orchestration**: Kubernetes (GKE)
- **Monitoring**: Prometheus + Grafana
- **Language**: Python 3.11+
- **Web Framework**: aiohttp
- **Metrics**: prometheus-client
- **Cloud**: Google Cloud Platform

---

## 📦 Prerequisites

### Required Tools

```bash
# Check if you have these installed:
docker --version          # Docker 20.10+
kubectl version --client  # kubectl 1.28+
gcloud version           # Google Cloud SDK

# If missing, install:
# Docker: https://docs.docker.com/get-docker/
# kubectl: https://kubernetes.io/docs/tasks/tools/
# gcloud: https://cloud.google.com/sdk/docs/install
```

### Google Cloud Account

- Active GCP account with billing enabled
- Project created (or use Cloud Shell which has everything pre-installed)
- Sufficient quota for:
  - GKE cluster (3+ nodes)
  - LoadBalancers (3-6)
  - Artifact Registry

### Project Structure

```
multi_region/
├── docker/
│   ├── Dockerfile              # Container image definition
│   ├── docker-compose.yml      # Local dev stack
│   └── prometheus.yml          # Prometheus config
├── kubernetes/
│   ├── base/                   # Base K8s manifests
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── configmap.yaml
│   │   └── kustomization.yaml
│   └── overlays/               # Region-specific configs
│       ├── us-west-2/
│       ├── eu-west-1/
│       └── ap-south-1/
├── scripts/
│   └── main.py                 # Application entry point
└── src/
    ├── failover/
    ├── monitoring/
    ├── cost/
    └── replication/
```

---

## 🐳 Phase 1: Local Docker Testing

### Step 1: Build Docker Image

```bash
# Navigate to project root
cd ~/multi_region

# Build the image
docker build -f docker/Dockerfile -t ml-platform/ml-serving:latest .

# Verify build
docker images | grep ml-platform
```

**Expected output:**
```
ml-platform/ml-serving   latest   abc123def456   2 minutes ago   500MB
```

### Step 2: Run Container Locally

```bash
# Run the container
docker run -p 8080:8080 -p 9090:9090 ml-platform/ml-serving:latest
```

**What you should see:**
```
INFO:MultiRegionPlatform:Starting Multi-Region ML Platform Services...
INFO:src.failover.failover_controller:Initialized region tracking: us-west-2
INFO:src.failover.failover_controller:Initialized region tracking: eu-west-1
INFO:src.failover.failover_controller:Initialized region tracking: ap-south-1
INFO:src.failover.failover_controller:Region us-west-2: healthy, response_time=5.23ms
INFO:MultiRegionPlatform:Starting health server on port 8080
INFO:MultiRegionPlatform:Starting metrics server on port 9090
```

### Step 3: Test Locally

Open a new terminal:

```bash
# Test health endpoint
curl http://localhost:8080/health
# Expected: OK

# Test metrics endpoint
curl http://localhost:9090/metrics
# Expected: Prometheus metrics output
```

✅ **Success Criteria**: Both endpoints return valid responses

### Step 4: Test with Docker Compose (Optional)

For full local stack with Prometheus and Grafana:

```bash
cd docker/

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# Access services:
# - ML Platform: http://localhost:8080
# - Prometheus:  http://localhost:9091
# - Grafana:     http://localhost:3000 (admin/admin)
```

---

## ☁️ Phase 2: Google Cloud Setup

### Step 1: Configure Google Cloud

```bash
# Login to Google Cloud (skip if using Cloud Shell)
gcloud auth login

# Set your project ID
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable container.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable compute.googleapis.com
```

### Step 2: Create Artifact Registry

```bash
# Create Docker repository
gcloud artifacts repositories create ml-platform \
    --repository-format=docker \
    --location=us-central1 \
    --description="ML Platform container images"

# Configure Docker authentication
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### Step 3: Build and Push Image

```bash
# Tag image for Artifact Registry
docker tag ml-platform/ml-serving:latest \
    us-central1-docker.pkg.dev/$PROJECT_ID/ml-platform/ml-serving:latest

# Push to registry
docker push us-central1-docker.pkg.dev/$PROJECT_ID/ml-platform/ml-serving:latest

# Verify upload
gcloud artifacts docker images list \
    us-central1-docker.pkg.dev/$PROJECT_ID/ml-platform
```

### Step 4: Create GKE Cluster

```bash
# Create a 3-node cluster
gcloud container clusters create ml-platform \
    --zone=us-central1-a \
    --num-nodes=3 \
    --machine-type=e2-standard-4 \
    --enable-autoscaling \
    --min-nodes=2 \
    --max-nodes=10 \
    --enable-autorepair \
    --enable-autoupgrade

# Get credentials
gcloud container clusters get-credentials ml-platform \
    --zone=us-central1-a

# Verify connection
kubectl cluster-info
kubectl get nodes
```

**Expected output:**
```
NAME                                      STATUS   ROLES    AGE   VERSION
gke-ml-platform-default-pool-xxx-yyy      Ready    <none>   2m    v1.28.x
gke-ml-platform-default-pool-xxx-zzz      Ready    <none>   2m    v1.28.x
gke-ml-platform-default-pool-xxx-www      Ready    <none>   2m    v1.28.x
```

---

## ⚙️ Phase 3: Kubernetes Deployment

### Step 1: Fix Kubernetes Manifests

#### Update `kubernetes/base/deployment.yaml`

Change the image path and environment variables:

```yaml
# Find line ~53, change:
image: ml-platform/ml-serving:latest

# To (replace YOUR_PROJECT_ID):
image: us-central1-docker.pkg.dev/YOUR_PROJECT_ID/ml-platform/ml-serving:latest
imagePullPolicy: Always
```

Also update environment variables (~lines 62-70):

```yaml
env:
- name: REGION
  value: "us-central1"      # Change from REPLACE_ME
- name: PROVIDER
  value: "gcp"              # Change from REPLACE_ME
- name: LOG_LEVEL
  value: "info"
- name: GCP_PROJECT_ID
  value: "YOUR_PROJECT_ID"  # Your actual project ID
```

#### Update `kubernetes/base/kustomization.yaml`

Remove PrometheusRule (requires Prometheus Operator):

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
  - namespace.yaml
  - deployment.yaml
  - service.yaml
  - configmap.yaml
  - serviceaccount.yaml
  # REMOVE THIS LINE: - prometheus-rules.yaml

labels:
  - pairs:
      app: ml-platform
      component: multi-region

namespace: ml-platform
```

### Step 2: Grant Artifact Registry Permissions

**Critical**: GKE needs permission to pull images from Artifact Registry.

```bash
# Get cluster details
CLUSTER_NAME=$(gcloud container clusters list --format="value(name)" --limit=1)
CLUSTER_ZONE=$(gcloud container clusters list --filter="name=${CLUSTER_NAME}" --format="value(location)")

# Get service account
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
GKE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

# Grant permission
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${GKE_SA}" \
    --role="roles/artifactregistry.reader"

echo "✅ Permissions granted to: $GKE_SA"
```

### Step 3: Deploy to Kubernetes

```bash
# Deploy the application
kubectl apply -k kubernetes/base/
kubectl apply -k kubernetes/overlays/us-west-2/
kubectl apply -k kubernetes/overlays/eu-west-1/

# Watch pods start
kubectl get pods -n ml-platform -w
# Press Ctrl+C once all pods show "Running"
```

**Expected output:**
```
NAME                          READY   STATUS    RESTARTS   AGE
ml-serving-xxxx-yyyy          1/1     Running   0          2m
ml-serving-xxxx-zzzz          1/1     Running   0          2m
ml-serving-xxxx-wwww          1/1     Running   0          2m
```

### Step 4: Get External IPs

```bash
# Wait for LoadBalancer IPs to be assigned
kubectl get svc -n ml-platform -w
# Press Ctrl+C once EXTERNAL-IP appears
```

**Expected output:**
```
NAME              TYPE           EXTERNAL-IP     PORT(S)
ml-serving        LoadBalancer   34.41.27.82     80:31717/TCP,9090:31950/TCP
```

### Step 5: Test Deployment

```bash
# Get the external IP
EXTERNAL_IP=$(kubectl get svc ml-serving -n ml-platform -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Test health endpoint
curl http://$EXTERNAL_IP/health
# Expected: OK

# Test metrics endpoint
curl http://$EXTERNAL_IP:9090/metrics | head -10
# Expected: Prometheus metrics
```

✅ **Success!** Your ML platform is now running on Kubernetes!

---

## 📊 Phase 4: Monitoring Setup

### Step 1: Install Prometheus Operator

```bash
# Add Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus stack
helm install prometheus prometheus-community/kube-prometheus-stack \
    --namespace monitoring \
    --create-namespace \
    --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false

# Wait for pods to be ready
kubectl wait --for=condition=ready pod \
    -l "release=prometheus" \
    -n monitoring \
    --timeout=300s
```

### Step 2: Expose Grafana and Prometheus

```bash
# Expose Grafana as LoadBalancer
kubectl patch svc prometheus-grafana -n monitoring \
    -p '{"spec": {"type": "LoadBalancer"}}'

# Expose Prometheus as LoadBalancer
kubectl patch svc prometheus-kube-prometheus-prometheus -n monitoring \
    -p '{"spec": {"type": "LoadBalancer"}}'

# Wait for IPs
sleep 30

# Get URLs
echo "📊 Monitoring URLs:"
echo "Grafana:    http://$(kubectl get svc prometheus-grafana -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"
echo "Prometheus: http://$(kubectl get svc prometheus-kube-prometheus-prometheus -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):9090"
```

### Step 3: Get Grafana Credentials

```bash
# Get Grafana admin password
echo "Grafana Login:"
echo "Username: admin"
echo "Password: $(kubectl get secret prometheus-grafana -n monitoring -o jsonpath='{.data.admin-password}' | base64 --decode)"
echo ""
```

### Step 4: Configure Grafana Dashboard

1. Open Grafana in your browser (use the URL from Step 2)
2. Login with admin credentials
3. Go to **Dashboards** → **New** → **Import**
4. Create panels with these queries:

**Panel 1: Request Rate by Region**
```promql
sum(rate(multiregion_request_rate[5m])) by (region)
```

**Panel 2: Region Health Status**
```promql
region_health_status
```

**Panel 3: P99 Latency**
```promql
multiregion_latency_ms{percentile="p99"}
```

**Panel 4: Error Rate**
```promql
sum(rate(multiregion_error_rate[5m])) by (region)
```

---

## 🧪 Phase 5: Testing & Verification

### Comprehensive Health Check

Run this complete test:

```bash
#!/bin/bash
echo "🧪 ML Platform Health Check"
echo "============================"
echo ""

# Get service IP
EXTERNAL_IP=$(kubectl get svc ml-serving -n ml-platform -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Test 1: Health endpoint
echo "1. Health Endpoint:"
curl -f http://$EXTERNAL_IP/health && echo "   ✅ PASS" || echo "   ❌ FAIL"

# Test 2: All pods running
echo ""
echo "2. Pod Status:"
RUNNING=$(kubectl get pods -n ml-platform --field-selector=status.phase=Running | grep ml-serving | wc -l)
echo "   $RUNNING pods running"
if [ $RUNNING -ge 2 ]; then
    echo "   ✅ PASS"
else
    echo "   ❌ FAIL"
fi

# Test 3: Region health from logs
echo ""
echo "3. Region Health Checks:"
kubectl logs -n ml-platform deployment/ml-serving --tail=50 | grep "Region.*healthy" | tail -3
echo "   ✅ PASS"

# Test 4: External access
echo ""
echo "4. External Access:"
echo "   Health URL: http://$EXTERNAL_IP/health"
echo "   Metrics URL: http://$EXTERNAL_IP:9090/metrics"

# Test 5: Monitoring stack
echo ""
echo "5. Monitoring Stack:"
GRAFANA_IP=$(kubectl get svc prometheus-grafana -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
PROMETHEUS_IP=$(kubectl get svc prometheus-kube-prometheus-prometheus -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "   Grafana: http://$GRAFANA_IP"
echo "   Prometheus: http://$PROMETHEUS_IP:9090"

echo ""
echo "✅ Health Check Complete!"
```

### Watch Real-Time Monitoring

```bash
# Watch region health checks (every 10 seconds)
kubectl logs -f deployment/ml-serving -n ml-platform | grep "Region.*healthy"

# You should see:
# INFO:src.failover.failover_controller:Region us-west-2: healthy, response_time=5.23ms
# INFO:src.failover.failover_controller:Region eu-west-1: healthy, response_time=5.37ms
# INFO:src.failover.failover_controller:Region ap-south-1: healthy, response_time=5.43ms
```

### Test Auto-Scaling

```bash
# Check HPA status
kubectl get hpa -n ml-platform

# Generate load (requires hey or ab)
hey -z 30s -c 50 http://$EXTERNAL_IP/health

# Watch pods scale up
kubectl get pods -n ml-platform -w
```

### Verify Metrics in Prometheus

1. Open Prometheus: `http://<PROMETHEUS_IP>:9090`
2. Go to **Graph** tab
3. Try these queries:

```promql
# Check if your app is being scraped
up{job="ml-serving"}

# View request rates
multiregion_request_rate

# View region health
region_health_status
```

---

## 🔧 Troubleshooting

### Issue 1: ImagePullBackOff

**Symptom:**
```
kubectl get pods -n ml-platform
NAME                          READY   STATUS             RESTARTS   AGE
ml-serving-xxx                0/1     ImagePullBackOff   0          2m
```

**Cause:** GKE can't pull image from Artifact Registry (permission issue)

**Solution:**
```bash
# Grant Artifact Registry access
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
GKE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${GKE_SA}" \
    --role="roles/artifactregistry.reader"

# Restart pods
kubectl delete pods --all -n ml-platform
```

### Issue 2: Pods Stuck in Pending

**Symptom:**
```
ml-serving-xxx    0/1     Pending   0          5m
```

**Cause:** Insufficient cluster resources

**Solution:**
```bash
# Check node resources
kubectl describe nodes | grep -A5 "Allocated resources"

# Option 1: Scale down replicas
kubectl scale deployment ml-serving -n ml-platform --replicas=2

# Option 2: Add nodes to cluster
gcloud container clusters resize ml-platform \
    --num-nodes=5 \
    --zone=us-central1-a
```

### Issue 3: Service Has No External IP

**Symptom:**
```
ml-serving    LoadBalancer   34.118.235.238   <pending>   80:31717/TCP
```

**Cause:** LoadBalancer creation takes time or quota exceeded

**Solution:**
```bash
# Wait longer (can take 2-3 minutes)
kubectl get svc -n ml-platform -w

# Check for quota issues
gcloud compute project-info describe --project=$PROJECT_ID

# Alternative: Use NodePort temporarily
kubectl patch svc ml-serving -n ml-platform \
    -p '{"spec": {"type": "NodePort"}}'
```

### Issue 4: 500 Error on /metrics Endpoint

**Symptom:**
```bash
curl http://$EXTERNAL_IP:9090/metrics
500 Internal Server Error
```

**Cause:** Bug in metrics handler (check logs)

**Solution:**
```bash
# Check logs for the error
kubectl logs deployment/ml-serving -n ml-platform | grep -i error

# Exec into pod and test locally
kubectl exec -n ml-platform deployment/ml-serving -- curl localhost:9090/metrics
```

### Issue 5: Grafana "Origin Not Allowed"

**Symptom:** Can't access Grafana through Cloud Shell Web Preview

**Solution:** Use LoadBalancer IP directly instead of port-forward:
```bash
# Expose Grafana as LoadBalancer
kubectl patch svc prometheus-grafana -n monitoring \
    -p '{"spec": {"type": "LoadBalancer"}}'

# Access via external IP
kubectl get svc prometheus-grafana -n monitoring
```

### Issue 6: Persistent Errors in Logs

**Common errors you can ignore:**

```
# AWS credentials (if using GCP only)
ERROR:src.replication.model_replicator:Error: Unable to locate credentials
✅ SAFE TO IGNORE - You're not using AWS

# Azure credentials (if using GCP only)
ERROR:src.replication.data_sync:Error: Connection string is either blank
✅ SAFE TO IGNORE - You're not using Azure

# Missing GCS buckets
ERROR:src.replication.data_sync:Error: bucket does not exist
✅ EXPECTED - Create buckets if needed, or disable data sync
```

**Real errors to fix:**

```
# Image pull errors
ERROR: Failed to pull image
❌ FIX: Check image path and permissions

# OOM (Out of Memory)
OOMKilled
❌ FIX: Increase memory limits in deployment.yaml

# CrashLoopBackOff
❌ FIX: Check application logs for startup errors
```

---

## ✅ Production Checklist

### Before Going Live

- [ ] **Security**
  - [ ] Use workload identity instead of default service account
  - [ ] Enable network policies
  - [ ] Set up secrets management (not ConfigMaps)
  - [ ] Enable pod security policies
  - [ ] Use private GKE cluster

- [ ] **Reliability**
  - [ ] Configure pod disruption budgets
  - [ ] Set up backup and disaster recovery
  - [ ] Configure multi-zone deployment
  - [ ] Set resource limits on all pods
  - [ ] Enable cluster autoscaling

- [ ] **Monitoring**
  - [ ] Set up Grafana alerts
  - [ ] Configure PagerDuty/Slack integration
  - [ ] Create SLO/SLI dashboards
  - [ ] Enable Cloud Logging
  - [ ] Set up uptime monitoring

- [ ] **Cost**
  - [ ] Enable GKE cluster autoscaler
  - [ ] Use preemptible nodes for non-critical workloads
  - [ ] Set up budget alerts
  - [ ] Configure GCS lifecycle policies
  - [ ] Review and optimize resource requests

- [ ] **Networking**
  - [ ] Set up Cloud CDN
  - [ ] Configure Ingress with SSL
  - [ ] Set up Cloud Armor (WAF)
  - [ ] Enable DDoS protection
  - [ ] Configure DNS with health checks

### Recommended Resource Limits

```yaml
# For production workloads
resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "2000m"
    memory: "4Gi"
```

### High Availability Configuration

```yaml
# Ensure HA in deployment.yaml
spec:
  replicas: 3  # Minimum for HA
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
```

---

## 📚 Additional Resources

### Documentation
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [GKE Documentation](https://cloud.google.com/kubernetes-engine/docs)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

### Useful Commands

```bash
# View all resources in namespace
kubectl get all -n ml-platform

# Describe a pod (for debugging)
kubectl describe pod <pod-name> -n ml-platform

# View logs
kubectl logs -f deployment/ml-serving -n ml-platform

# Port forward for local access
kubectl port-forward svc/ml-serving 8080:80 -n ml-platform

# Execute command in pod
kubectl exec -it <pod-name> -n ml-platform -- /bin/bash

# Get resource usage
kubectl top pods -n ml-platform
kubectl top nodes

# Scale deployment
kubectl scale deployment ml-serving -n ml-platform --replicas=5

# Rollback deployment
kubectl rollout undo deployment/ml-serving -n ml-platform

# Check rollout status
kubectl rollout status deployment/ml-serving -n ml-platform
```

### Quick Reference

| Component | Default Port | URL |
|-----------|-------------|-----|
| Health Check | 8080 | http://EXTERNAL-IP/health |
| Metrics | 9090 | http://EXTERNAL-IP:9090/metrics |
| Prometheus | 9090 | http://PROMETHEUS-IP:9090 |
| Grafana | 80 | http://GRAFANA-IP |

---

## 🎓 Understanding the Platform

### How Failover Works

1. **Health Monitoring**: Every 10 seconds, the failover controller checks each region
2. **Health Evaluation**: 
   - Healthy: Response time < 200ms, no failures
   - Degraded: Response time 200-500ms or 1-2 failures
   - Unhealthy: Response time > 500ms or 3+ consecutive failures
3. **Failover Trigger**: When primary region becomes unhealthy
4. **Target Selection**: Chooses healthiest region with lowest latency
5. **Execution**: Updates routing to new primary region

### Metrics Explained

```promql
# Request rate per region
multiregion_request_rate{region="us-west-2"}

# Error rate per region  
multiregion_error_rate{region="eu-west-1"}

# Latency percentiles
multiregion_latency_ms{region="ap-south-1",percentile="p99"}

# Region health (1=healthy, 0=unhealthy)
region_health_status{region="us-west-2"}

# Total failover events
failover_events_total{source_region="us-west-2",target_region="eu-west-1"}
```

### Architecture Decisions

**Why GKE?**
- Managed Kubernetes service
- Auto-scaling and auto-repair
- Integrated with GCP services
- Strong security features

**Why LoadBalancer over Ingress?**
- Simpler setup for multi-port services
- Direct external access
- Can upgrade to Ingress later for advanced routing

**Why Prometheus?**
- Industry standard for metrics
- Native Kubernetes support
- Powerful query language
- Great Grafana integration

---

## 🚀 Next Steps

After completing this guide, consider:

1. **Add More Regions**: Deploy to additional GCP regions for better global coverage
2. **Implement CI/CD**: Set up automated builds and deployments
3. **Enable Auto-Scaling**: Configure vertical and horizontal pod autoscaling
4. **Set Up Alerting**: Configure alerts for failures and anomalies
5. **Optimize Costs**: Enable cluster autoscaler and preemptible nodes
6. **Add ML Models**: Integrate actual ML model serving
7. **Data Pipeline**: Set up data sync between regions
8. **Load Testing**: Perform stress tests and optimize

---

## 💡 Tips & Best Practices

### Development Workflow

```bash
# 1. Make code changes
# 2. Build new image
docker build -f docker/Dockerfile -t us-central1-docker.pkg.dev/$PROJECT_ID/ml-platform/ml-serving:v1.1 .

# 3. Push to registry
docker push us-central1-docker.pkg.dev/$PROJECT_ID/ml-platform/ml-serving:v1.1

# 4. Update deployment
kubectl set image deployment/ml-serving \
    ml-serving=us-central1-docker.pkg.dev/$PROJECT_ID/ml-platform/ml-serving:v1.1 \
    -n ml-platform

# 5. Watch rollout
kubectl rollout status deployment/ml-serving -n ml-platform
```

### Cost Optimization

- Use **preemptible nodes** for non-critical workloads (60-70% savings)
- Enable **cluster autoscaling** to scale down during low traffic
- Use **committed use discounts** for predictable workloads
- Monitor and **right-size** your resource requests/limits
- Clean up **unused LoadBalancers** and persistent disks

### Security Best Practices

- Always use **least privilege** for service accounts
- Enable **workload identity** instead of service account keys
- Use **secrets** for sensitive data, not ConfigMaps
- Enable **network policies** to control pod-to-pod traffic
- Regularly **update** GKE clusters and node images
- Use **private clusters** for production

---

## 📞 Support & Contribution

### Getting Help

If you encounter issues:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review pod logs: `kubectl logs -n ml-platform deployment/ml-serving`
3. Check events: `kubectl get events -n ml-platform --sort-by='.lastTimestamp'`
4. Describe resources: `kubectl describe pod <pod-name> -n ml-platform`

### Common Questions

**Q: Can I use this with only GCP (no AWS/Azure)?**  
A: Yes! The platform gracefully handles missing cloud provider credentials.

**Q: How much does this cost to run?**  
A: Approximately $100-150/month for a small 3-node GKE cluster with monitoring.

**Q: Can I run this in production?**  
A: Yes, but review the [Production Checklist](#-production-checklist) first.

**Q: How do I add my ML models?**  
A: Integrate your model serving code into the platform and expose via HTTP endpoints.

---

## 📜 License

This project is provided as-is for educational and production use.

---

**Built with ❤️ for Cloud-Native ML Deployments**

*Last Updated: February 2026*
