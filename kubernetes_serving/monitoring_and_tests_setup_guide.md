# Monitoring & Testing Setup Guide

Complete guide to set up monitoring and run tests for your Kubernetes model serving project.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Monitoring Setup](#monitoring-setup)
3. [Running Tests](#running-tests)
4. [Load Testing](#load-testing)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

```powershell
# 1. Install Minikube (if not installed)
winget install Kubernetes.minikube

# 2. Install kubectl
winget install Kubernetes.kubectl

# 3. Install Helm
winget install Helm.Helm

# 4. Install K6 (for load testing)
winget install k6

# 5. Python packages
pip install kubernetes pytest requests
```

### Start Minikube

```powershell
# Start Minikube with sufficient resources
minikube start --cpus=4 --memory=8192 --driver=docker

# Enable required addons
minikube addons enable metrics-server
minikube addons enable ingress

# Verify
kubectl get nodes
kubectl get pods -A

# Build Docker image
docker build -f docker/Dockerfile -t model-api:v1.0 .

# Load Docker image into Minikube
minikube image load model-api:v1.0

```

---

## Monitoring Setup

### Step 1: Install Prometheus & Grafana using Helm

```powershell
# Add Prometheus community Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Create monitoring namespace
kubectl create namespace monitoring

# Install Prometheus & Grafana (this takes 2-3 minutes)
helm install prometheus prometheus-community/kube-prometheus-stack `
  --namespace monitoring `
  --create-namespace

# Wait for pods to be ready
kubectl wait --for=condition=ready pod --all -n monitoring --timeout=300s

# Verify installation
kubectl get pods -n monitoring
```

Expected output:
```
NAME                                                     READY   STATUS    RESTARTS   AGE
alertmanager-prometheus-kube-prometheus-alertmanager-0   2/2     Running   0          2m
prometheus-grafana-xxxx                                  3/3     Running   0          2m
prometheus-kube-prometheus-operator-xxxx                 1/1     Running   0          2m
prometheus-kube-state-metrics-xxxx                       1/1     Running   0          2m
prometheus-prometheus-kube-prometheus-prometheus-0       2/2     Running   0          2m
```

### Step 2: Deploy Your Application

```powershell
# Create ml-serving namespace
kubectl create namespace ml-serving

# Apply your Kubernetes manifests
kubectl apply -f kubernetes/configmap.yaml -n ml-serving
kubectl apply -f kubernetes/deployment.yaml -n ml-serving
kubectl apply -f kubernetes/service.yaml -n ml-serving
kubectl apply -f kubernetes/hpa.yaml -n ml-serving

# Wait for pods
kubectl get pods -n ml-serving -w
# Press Ctrl+C when all pods are Running (1/1 READY)
```

### Step 3: Apply ServiceMonitor

```powershell
# Apply the ServiceMonitor to tell Prometheus where to scrape metrics
kubectl apply -f monitoring/servicemonitor.yaml -n ml-serving

# Verify ServiceMonitor was created
kubectl get servicemonitor -n ml-serving

# Check if Prometheus is picking it up (wait 30 seconds)
kubectl logs -n monitoring prometheus-prometheus-kube-prometheus-prometheus-0 -c prometheus | grep "model-api"
```

### Step 4: Access Prometheus

```powershell
# Port-forward Prometheus (in a separate terminal)
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
```

Open browser: http://localhost:9090

**Verify targets:**
1. Go to Status → Targets
2. Look for `ml-serving/model-api-monitor`
3. State should be **UP** (green)

**Query metrics:**
```promql
# In the Graph tab, try these queries:
model_api_requests_total
model_api_request_duration_seconds
rate(model_api_requests_total[5m])
```

### Step 5: Access Grafana

```powershell
# Get Grafana admin password
$GRAFANA_PASSWORD = kubectl get secret --namespace monitoring prometheus-grafana -o jsonpath="{.data.admin-password}" | ForEach-Object { [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($_)) }
Write-Host "Grafana Password: $GRAFANA_PASSWORD"

# Port-forward Grafana (in a separate terminal)
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
```

Open browser: http://localhost:3000
- Username: **admin**
- Password: (from above command)

### Step 6: Import Dashboard

1. In Grafana, click **+ → Import** (left sidebar)
2. Click **Upload JSON file**
3. Select `monitoring/grafana-dashboard.json`
4. Select **Prometheus** as the data source
5. Click **Import**

You should now see your dashboard with panels showing:
- Request Rate
- Running Pods
- Latency Percentiles
- Memory/CPU Usage
- Error Rate
- Inference Time

---

## Running Tests

### Kubernetes Integration Tests

```powershell
# Make sure your service is accessible
kubectl port-forward -n ml-serving svc/model-api-service 8080:80

# In another terminal, run tests
cd tests
pytest test_k8s.py -v

# Run specific test class
pytest test_k8s.py::TestDeployment -v

# Run specific test
pytest test_k8s.py::TestDeployment::test_deployment_exists -v

# Skip slow tests
pytest test_k8s.py -v -m "not slow"

# Run only slow tests (scaling, rolling update)
pytest test_k8s.py -v -m slow
```

Expected output:
```
tests/test_k8s.py::TestDeployment::test_deployment_exists PASSED
tests/test_k8s.py::TestDeployment::test_deployment_replicas PASSED
tests/test_k8s.py::TestPods::test_all_pods_running PASSED
tests/test_k8s.py::TestService::test_service_exists PASSED
...
=================== 20 passed, 5 skipped in 15.23s ===================
```

---

## Load Testing

### Option 1: K6 Load Test (Recommended)

```powershell
# Make sure service is accessible
kubectl port-forward -n ml-serving svc/model-api-service 8080:80

# In another terminal, run load test
cd tests
k6 run load-test.js

# Custom duration
k6 run --duration 2m --vus 20 load-test.js

# Set custom API URL
$env:API_URL="http://localhost:8080"
k6 run load-test.js
```

Expected output:
```
          /\      |‾‾| /‾‾/   /‾‾/   
     /\  /  \     |  |/  /   /  /    
    /  \/    \    |     (   /   ‾‾\  
   /          \   |  |\  \ |  (‾)  | 
  / __________ \  |__| \__\ \_____/ .io

  execution: local
     script: load-test.js
     output: -

  scenarios: (100.00%) 1 scenario, 100 max VUs, 5m30s max duration

     ✓ health check status is 200
     ✓ prediction status is 200
     ✓ prediction has predictions

     checks.........................: 100.00% ✓ 15234      ✗ 0     
     data_received..................: 45 MB   150 kB/s
     data_sent......................: 23 MB   76 kB/s
     http_req_duration..............: avg=125ms min=45ms  p(95)=380ms
     successful_predictions.........: 5078    16.925/s
     vus............................: 0       min=0      max=100
```

### Option 2: Python Load Test

```powershell
# Simple concurrent requests test
python -c "
import requests
import concurrent.futures

def make_request():
    r = requests.get('http://localhost:8080/health')
    return r.status_code == 200

with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    futures = [executor.submit(make_request) for _ in range(1000)]
    results = [f.result() for f in futures]
    print(f'Success rate: {sum(results)/len(results)*100}%')
"
```

### Monitoring During Load Test

While load test is running:

1. **Watch HPA scaling:**
```powershell
kubectl get hpa -n ml-serving -w
```

2. **Watch pods scaling:**
```powershell
kubectl get pods -n ml-serving -w
```

3. **Watch Grafana dashboard:**
   - Open http://localhost:3000
   - Watch Request Rate, CPU, Memory panels
   - Should see HPA scale from 3 → 10 pods

---

## Troubleshooting

### Prometheus Not Scraping

**Check ServiceMonitor:**
```powershell
kubectl describe servicemonitor model-api-monitor -n ml-serving
```

**Check Prometheus targets:**
1. http://localhost:9090/targets
2. Look for errors in "Last Scrape" column

**Common fix:**
```powershell
# Restart Prometheus
kubectl delete pod -n monitoring prometheus-prometheus-kube-prometheus-prometheus-0
```

### Grafana Dashboard Shows "No Data"

**Check Prometheus data source:**
1. Grafana → Configuration → Data Sources
2. Click Prometheus
3. Click "Test" button
4. Should say "Data source is working"

**Check metrics exist:**
```powershell
# Port-forward Prometheus
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090

# Visit http://localhost:9090
# Try query: model_api_requests_total
```

### Tests Failing

**Service not accessible:**
```powershell
# Check if port-forward is running
netstat -ano | findstr :8080

# Restart port-forward
kubectl port-forward -n ml-serving svc/model-api-service 8080:80
```

**Pods not ready:**
```powershell
kubectl get pods -n ml-serving
kubectl describe pod <pod-name> -n ml-serving
kubectl logs <pod-name> -n ml-serving
```

### K6 Load Test Fails

**Install K6:**
```powershell
winget install k6
# Or download from: https://k6.io/docs/getting-started/installation/
```

**Check API is accessible:**
```powershell
curl http://localhost:8080/health
```

---

## Quick Reference

### Port Forwards (Run in separate terminals)

```powershell
# Prometheus
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090

# Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Your API
kubectl port-forward -n ml-serving svc/model-api-service 8080:80
```

### Useful Commands

```powershell
# Check everything is running
kubectl get all -n ml-serving
kubectl get all -n monitoring

# View logs
kubectl logs -n ml-serving deployment/model-api --tail=50 -f

# Describe resources
kubectl describe deployment model-api -n ml-serving
kubectl describe hpa model-api-hpa -n ml-serving

# Execute into pod
kubectl exec -it <pod-name> -n ml-serving -- /bin/sh

# Delete and redeploy
kubectl delete -f kubernetes/ -n ml-serving
kubectl apply -f kubernetes/ -n ml-serving
```
