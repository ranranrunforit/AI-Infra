# ⚡ Quick Reference - Multi-Region ML Platform

**Essential commands for day-to-day operations**

---

## 🚀 Quick Start Commands

```bash
# Set your project
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Build and push
docker build -f docker/Dockerfile -t us-central1-docker.pkg.dev/$PROJECT_ID/ml-platform/ml-serving:latest .
docker push us-central1-docker.pkg.dev/$PROJECT_ID/ml-platform/ml-serving:latest

# Deploy
kubectl apply -k kubernetes/base/

# Check status
kubectl get pods -n ml-platform
kubectl get svc -n ml-platform
```

---

## 📊 Monitoring Commands

```bash
# Get all URLs
echo "App:        http://$(kubectl get svc ml-serving -n ml-platform -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"
echo "Grafana:    http://$(kubectl get svc prometheus-grafana -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"
echo "Prometheus: http://$(kubectl get svc prometheus-kube-prometheus-prometheus -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):9090"

# Get Grafana password
kubectl get secret prometheus-grafana -n monitoring -o jsonpath="{.data.admin-password}" | base64 --decode
```

---

## 🔍 Debugging Commands

```bash
# View logs
kubectl logs -f deployment/ml-serving -n ml-platform

# Get pod details
kubectl describe pod <pod-name> -n ml-platform

# Check events
kubectl get events -n ml-platform --sort-by='.lastTimestamp'

# Exec into pod
kubectl exec -it <pod-name> -n ml-platform -- /bin/bash

# Check resource usage
kubectl top pods -n ml-platform
kubectl top nodes
```

---

## 🎯 Testing Commands

```bash
# Test health
EXTERNAL_IP=$(kubectl get svc ml-serving -n ml-platform -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl http://$EXTERNAL_IP/health

# Watch region health
kubectl logs -f deployment/ml-serving -n ml-platform | grep "Region.*healthy"

# Run full health check
kubectl get pods -n ml-platform && \
kubectl get svc -n ml-platform && \
curl -s http://$EXTERNAL_IP/health
```

---

## 🔧 Management Commands

```bash
# Scale deployment
kubectl scale deployment ml-serving -n ml-platform --replicas=5

# Update image
kubectl set image deployment/ml-serving \
    ml-serving=us-central1-docker.pkg.dev/$PROJECT_ID/ml-platform/ml-serving:v2.0 \
    -n ml-platform

# Restart deployment
kubectl rollout restart deployment/ml-serving -n ml-platform

# Check rollout status
kubectl rollout status deployment/ml-serving -n ml-platform

# Rollback
kubectl rollout undo deployment/ml-serving -n ml-platform
```

---

## 🆘 Emergency Commands

```bash
# Delete all pods (they'll restart)
kubectl delete pods --all -n ml-platform

# Get comprehensive status
kubectl get all -n ml-platform

# Check cluster health
kubectl get componentstatuses

# Force delete stuck pod
kubectl delete pod <pod-name> -n ml-platform --force --grace-period=0
```

---

## 🧹 Cleanup Commands

```bash
# Delete application
kubectl delete -k kubernetes/base/

# Delete monitoring
helm uninstall prometheus -n monitoring
kubectl delete namespace monitoring

# Delete cluster
gcloud container clusters delete ml-platform --zone=us-central1-a

# Delete artifact registry
gcloud artifacts repositories delete ml-platform --location=us-central1
```

---

## 📈 Prometheus Queries

```promql
# App is being scraped
up{job="ml-serving"}

# Request rate
rate(multiregion_request_rate[5m])

# Error rate
rate(multiregion_error_rate[5m])

# Region health
region_health_status

# Latency
multiregion_latency_ms{percentile="p99"}
```

---

## 🎯 Common Tasks

### Deploy New Version
```bash
docker build -f docker/Dockerfile -t us-central1-docker.pkg.dev/$PROJECT_ID/ml-platform/ml-serving:v1.1 .
docker push us-central1-docker.pkg.dev/$PROJECT_ID/ml-platform/ml-serving:v1.1
kubectl set image deployment/ml-serving ml-serving=us-central1-docker.pkg.dev/$PROJECT_ID/ml-platform/ml-serving:v1.1 -n ml-platform
kubectl rollout status deployment/ml-serving -n ml-platform
```

### Check if Everything is Working
```bash
kubectl get pods -n ml-platform | grep Running
EXTERNAL_IP=$(kubectl get svc ml-serving -n ml-platform -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl http://$EXTERNAL_IP/health
kubectl logs deployment/ml-serving -n ml-platform --tail=20 | grep healthy
```

### Debug 500 Error
```bash
kubectl logs deployment/ml-serving -n ml-platform | grep -i error
kubectl exec -n ml-platform deployment/ml-serving -- curl -v localhost:9090/metrics
```

### Scale Based on Load
```bash
# Manual scale
kubectl scale deployment ml-serving -n ml-platform --replicas=10

# Check HPA
kubectl get hpa -n ml-platform

# Describe HPA
kubectl describe hpa ml-serving -n ml-platform
```

---

## 📦 File Locations

```
kubernetes/base/deployment.yaml    - Main deployment config
kubernetes/base/service.yaml       - LoadBalancer service
kubernetes/base/configmap.yaml     - Application config
scripts/main.py                    - Application entry point
docker/Dockerfile                  - Container image definition
```

---

## 🔐 Important Notes

- Always use full image paths: `us-central1-docker.pkg.dev/PROJECT/REPO/IMAGE:TAG`
- Default credentials: Grafana (admin/check-secret), Prometheus (no auth)
- Health endpoint: `/health`, Metrics endpoint: `/metrics`
- Logs show region health every 10 seconds
- Metrics collected every 60 seconds

---

**Keep this handy for quick reference!** 📌
