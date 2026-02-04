# Operations Runbook - High-Performance Model Serving

## Overview

This runbook provides step-by-step procedures for common operational tasks. Designed for on-call engineers and operators managing the model serving platform.

**Document Audience**: DevOps Engineers, SREs, On-Call Personnel

---

## Table of Contents

- [Quick Reference](#quick-reference)
- [Starting and Stopping Services](#starting-and-stopping-services)
- [Model Deployment](#model-deployment)
- [Scaling Operations](#scaling-operations)
- [Backup and Restore](#backup-and-restore)
- [Log Analysis](#log-analysis)
- [Performance Tuning](#performance-tuning)
- [Incident Response](#incident-response)
- [Maintenance Windows](#maintenance-windows)
- [Common Operational Tasks](#common-operational-tasks)

---

## Quick Reference

### Essential Commands

```bash
# Check service health
kubectl get pods -n model-serving
curl http://api.example.com/health

# View logs
kubectl logs -n model-serving -l app=model-serving --tail=100

# Check metrics
curl http://api.example.com/metrics | grep model_serving

# Check GPU usage
kubectl exec -n model-serving <pod-name> -- nvidia-smi

# Restart deployment
kubectl rollout restart deployment/model-serving -n model-serving

# Scale deployment
kubectl scale deployment/model-serving --replicas=5 -n model-serving
```

### Service URLs (Production)

- **API**: https://api.model-serving.example.com
- **Prometheus**: https://prometheus.example.com
- **Grafana**: https://grafana.example.com
- **Jaeger**: https://jaeger.example.com
- **Kubernetes Dashboard**: https://k8s-dashboard.example.com

### On-Call Contacts

- **Primary On-Call**: pagerduty-ml-infra@example.com
- **Secondary**: ml-platform-team@example.com
- **Escalation**: engineering-leads@example.com

---

## Starting and Stopping Services

### Start Service (Kubernetes)

```bash
# Check current status
kubectl get deployment -n model-serving

# If deployment doesn't exist, create it
kubectl apply -k kubernetes/overlays/prod/

# Verify pods are starting
kubectl get pods -n model-serving -w

# Check logs
kubectl logs -n model-serving -l app=model-serving -f

# Verify health
kubectl exec -n model-serving <pod-name> -- curl localhost:8000/health
```

**Expected Timeline**: 2-5 minutes for pods to be ready

### Stop Service

```bash
# Scale to zero (keeps deployment)
kubectl scale deployment/model-serving --replicas=0 -n model-serving

# Or delete deployment (removes everything)
kubectl delete deployment model-serving -n model-serving

# Verify
kubectl get pods -n model-serving
```

### Restart Service

```bash
# Rolling restart (zero downtime)
kubectl rollout restart deployment/model-serving -n model-serving

# Watch restart progress
kubectl rollout status deployment/model-serving -n model-serving

# Verify all pods restarted
kubectl get pods -n model-serving -o wide
```

### Start Docker Compose Stack (Development)

```bash
cd /path/to/project
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f model-serving

# Check status
docker-compose -f docker/docker-compose.yml ps
```

---

## Model Deployment

### Deploy New Model

**Procedure**:

1. **Upload Model to Storage**

```bash
# Upload to S3
aws s3 cp resnet50-fp16.trt s3://model-storage/models/

# Or copy to NFS
cp resnet50-fp16.trt /mnt/nfs/models/
```

2. **Load Model in Pod**

```bash
# Using API
curl -X POST "https://api.example.com/v1/models/resnet50-fp16/load?model_format=tensorrt" \
  -H "Authorization: Bearer YOUR_API_KEY"

# Expected response
# {"model":"resnet50-fp16","format":"tensorrt","status":"loaded","load_time_seconds":2.3}
```

3. **Verify Model Loaded**

```bash
# Check health endpoint
curl https://api.example.com/health

# Should list new model in models_loaded array
```

4. **Test Inference**

```bash
# Send test request
curl -X POST https://api.example.com/v1/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d @test-request.json
```

### Update Existing Model

**Procedure**:

1. **Upload New Version**

```bash
# Upload with version suffix
aws s3 cp resnet50-fp16-v2.trt s3://model-storage/models/resnet50-fp16-v2.trt
```

2. **Unload Old Version**

```bash
curl -X POST "https://api.example.com/v1/models/resnet50-fp16/unload" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

3. **Load New Version**

```bash
curl -X POST "https://api.example.com/v1/models/resnet50-fp16-v2/load?model_format=tensorrt" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

4. **Verify and Test**

```bash
# Check health
curl https://api.example.com/health

# Run inference test
./scripts/test-inference.sh resnet50-fp16-v2
```

---

## Scaling Operations

### Manual Scaling

**Scale Up**:

```bash
# Increase replicas
kubectl scale deployment/model-serving --replicas=10 -n model-serving

# Verify scaling
kubectl get pods -n model-serving -w

# Check HPA status
kubectl get hpa -n model-serving
```

**Scale Down**:

```bash
# Decrease replicas
kubectl scale deployment/model-serving --replicas=3 -n model-serving

# Wait for pods to terminate gracefully
kubectl get pods -n model-serving -w
```

### Auto-Scaling Configuration

**View Current HPA**:

```bash
kubectl get hpa -n model-serving -o yaml
```

**Modify HPA**:

```bash
# Edit HPA
kubectl edit hpa model-serving-hpa -n model-serving

# Or patch specific values
kubectl patch hpa model-serving-hpa -n model-serving --patch '
spec:
  minReplicas: 5
  maxReplicas: 30
'
```

**Monitor Auto-Scaling**:

```bash
# Watch HPA decisions
kubectl get hpa -n model-serving -w

# View HPA events
kubectl describe hpa model-serving-hpa -n model-serving

# Check metrics
kubectl top pods -n model-serving
```

### Vertical Scaling (Increase Resources)

```bash
# Edit deployment to increase resource limits
kubectl edit deployment model-serving -n model-serving

# Update resources section:
resources:
  requests:
    memory: "16Gi"
    cpu: "4"
    nvidia.com/gpu: "1"
  limits:
    memory: "32Gi"
    cpu: "8"
    nvidia.com/gpu: "1"

# Restart to apply changes
kubectl rollout restart deployment/model-serving -n model-serving
```

---

## Backup and Restore

### Backup Procedures

**Daily Backup** (Automated):

```bash
#!/bin/bash
# /opt/scripts/backup-model-serving.sh

DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups/model-serving/$DATE"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup Kubernetes resources
kubectl get all -n model-serving -o yaml > $BACKUP_DIR/resources.yaml
kubectl get configmap -n model-serving -o yaml > $BACKUP_DIR/configmaps.yaml
kubectl get secret -n model-serving -o yaml > $BACKUP_DIR/secrets.yaml

# Backup models
aws s3 sync /models/ s3://backup-bucket/models/$DATE/

# Backup Prometheus data
tar -czf $BACKUP_DIR/prometheus-data.tar.gz /var/lib/prometheus/

echo "Backup completed: $BACKUP_DIR"
```

**On-Demand Backup**:

```bash
# Run backup script
sudo /opt/scripts/backup-model-serving.sh

# Verify backup
ls -lh /backups/model-serving/$(date +%Y%m%d)/
```

### Restore Procedures

**Restore from Backup**:

```bash
#!/bin/bash
# /opt/scripts/restore-model-serving.sh

BACKUP_DATE=$1

if [ -z "$BACKUP_DATE" ]; then
  echo "Usage: $0 YYYYMMDD"
  exit 1
fi

BACKUP_DIR="/backups/model-serving/$BACKUP_DATE"

# Restore Kubernetes resources
kubectl apply -f $BACKUP_DIR/resources.yaml
kubectl apply -f $BACKUP_DIR/configmaps.yaml
kubectl apply -f $BACKUP_DIR/secrets.yaml

# Restore models
aws s3 sync s3://backup-bucket/models/$BACKUP_DATE/ /models/

# Verify
kubectl get pods -n model-serving
```

**Execute Restore**:

```bash
# Restore from specific date
sudo /opt/scripts/restore-model-serving.sh 20240115

# Verify service
kubectl get pods -n model-serving
curl https://api.example.com/health
```

---

## Log Analysis

### View Real-Time Logs

```bash
# All pods
kubectl logs -n model-serving -l app=model-serving -f

# Specific pod
kubectl logs -n model-serving <pod-name> -f

# Previous instance (if crashed)
kubectl logs -n model-serving <pod-name> --previous
```

### Search Logs

```bash
# Search for errors
kubectl logs -n model-serving -l app=model-serving --tail=1000 | grep ERROR

# Search for specific request
kubectl logs -n model-serving -l app=model-serving | grep "trace_id=abc123"

# Count errors by type
kubectl logs -n model-serving -l app=model-serving --tail=10000 | grep ERROR | sort | uniq -c | sort -rn
```

### Log Aggregation (ELK/Splunk)

```bash
# Kibana query
{
  "query": {
    "bool": {
      "must": [
        {"match": {"kubernetes.namespace": "model-serving"}},
        {"match": {"level": "ERROR"}},
        {"range": {"@timestamp": {"gte": "now-1h"}}}
      ]
    }
  }
}

# View in Kibana
https://kibana.example.com/app/discover
```

### Log Analysis Scripts

```bash
# Analyze latency
kubectl logs -n model-serving -l app=model-serving --tail=10000 | \
  grep "latency_ms" | \
  awk '{print $NF}' | \
  sort -n | \
  awk 'BEGIN {sum=0; count=0} {sum+=$1; count++; vals[count]=$1} END {
    print "Count:", count
    print "Mean:", sum/count
    print "P50:", vals[int(count*0.5)]
    print "P95:", vals[int(count*0.95)]
    print "P99:", vals[int(count*0.99)]
  }'
```

---

## Performance Tuning

### GPU Utilization

**Check GPU Usage**:

```bash
# From host
nvidia-smi

# From pod
kubectl exec -n model-serving <pod-name> -- nvidia-smi

# Continuous monitoring
kubectl exec -n model-serving <pod-name> -- watch -n 1 nvidia-smi
```

**Optimize GPU Memory**:

```bash
# Adjust GPU memory fraction
kubectl set env deployment/model-serving GPU_MEMORY_FRACTION=0.95 -n model-serving

# Adjust vLLM memory utilization
kubectl set env deployment/model-serving VLLM_GPU_MEMORY_UTILIZATION=0.90 -n model-serving
```

### Batch Size Tuning

**Adjust Dynamic Batching**:

```bash
# Increase batch size for higher throughput
kubectl set env deployment/model-serving MAX_BATCH_SIZE=64 -n model-serving

# Decrease timeout for lower latency
kubectl set env deployment/model-serving BATCH_TIMEOUT_MS=5 -n model-serving

# Restart to apply
kubectl rollout restart deployment/model-serving -n model-serving
```

**Monitor Batch Performance**:

```bash
# Check batch size distribution
curl http://api.example.com/metrics | grep model_serving_batch_size
```

### TensorRT Optimization

**Re-optimize Engine**:

```bash
# Re-build with higher optimization level
python scripts/convert_model.py \
    --model resnet50 \
    --precision fp16 \
    --optimization-level 5 \
    --output models/resnet50-fp16-optimized.trt

# Deploy optimized engine
aws s3 cp models/resnet50-fp16-optimized.trt s3://model-storage/models/
```

---

## Incident Response

### High Error Rate Alert

**Procedure**:

1. **Identify Scope**

```bash
# Check error rate
curl http://api.example.com/metrics | grep model_serving_requests_total

# View recent errors
kubectl logs -n model-serving -l app=model-serving --tail=500 | grep ERROR
```

2. **Check Dependencies**

```bash
# Check GPU availability
kubectl exec -n model-serving <pod-name> -- nvidia-smi

# Check model availability
curl https://api.example.com/health
```

3. **Mitigate**

```bash
# If specific pod is problematic, restart it
kubectl delete pod <pod-name> -n model-serving

# If widespread, rollback deployment
kubectl rollout undo deployment/model-serving -n model-serving
```

4. **Verify Resolution**

```bash
# Monitor error rate
watch -n 5 'curl -s http://api.example.com/metrics | grep model_serving_requests_total'
```

### High Latency Alert

**Procedure**:

1. **Identify Bottleneck**

```bash
# Check P99 latency
curl http://api.example.com/metrics | grep model_serving_request_duration_seconds

# View trace
Open Jaeger UI and search for slow traces
```

2. **Check Resources**

```bash
# CPU/Memory
kubectl top pods -n model-serving

# GPU
kubectl exec -n model-serving <pod-name> -- nvidia-smi
```

3. **Scale if Needed**

```bash
# Increase replicas
kubectl scale deployment/model-serving --replicas=10 -n model-serving
```

### Service Down Alert

**Procedure**:

1. **Check Pod Status**

```bash
kubectl get pods -n model-serving
kubectl describe pod <pod-name> -n model-serving
```

2. **View Crash Logs**

```bash
kubectl logs -n model-serving <pod-name> --previous
```

3. **Common Fixes**

```bash
# If OOMKilled, increase memory
kubectl edit deployment model-serving -n model-serving
# Update memory limits

# If ImagePullBackOff, check registry
kubectl describe pod <pod-name> -n model-serving

# If CrashLoopBackOff, check logs and configuration
kubectl logs -n model-serving <pod-name>
```

---

## Maintenance Windows

### Pre-Maintenance Checklist

```bash
# 1. Announce maintenance
# Post to #ml-platform Slack channel

# 2. Backup current state
sudo /opt/scripts/backup-model-serving.sh

# 3. Document current configuration
kubectl get deployment model-serving -n model-serving -o yaml > pre-maintenance-config.yaml

# 4. Verify rollback plan
cat rollback-plan.md
```

### During Maintenance

```bash
# 1. Set maintenance mode (optional)
kubectl annotate service model-serving maintenance=true -n model-serving

# 2. Perform updates
kubectl apply -k kubernetes/overlays/prod/

# 3. Monitor rollout
kubectl rollout status deployment/model-serving -n model-serving

# 4. Run smoke tests
./scripts/smoke-test.sh production
```

### Post-Maintenance Checklist

```bash
# 1. Verify all pods healthy
kubectl get pods -n model-serving

# 2. Test critical paths
./scripts/integration-test.sh

# 3. Monitor metrics for 15 minutes
watch -n 10 'curl -s http://api.example.com/metrics | grep model_serving_requests_total'

# 4. Remove maintenance annotation
kubectl annotate service model-serving maintenance- -n model-serving

# 5. Announce completion
# Post to #ml-platform Slack channel
```

---

## Common Operational Tasks

### Add New GPU Node

```bash
# 1. Provision node with GPU
# (Cloud provider specific)

# 2. Install NVIDIA drivers
ssh node-new
sudo apt-get install -y nvidia-driver-525
sudo reboot

# 3. Join Kubernetes cluster
# (Cluster-specific join command)

# 4. Verify GPU accessible
kubectl describe node node-new | grep nvidia.com/gpu

# 5. Label node
kubectl label node node-new accelerator=nvidia-gpu

# 6. Pods will automatically schedule on new node
kubectl get pods -n model-serving -o wide
```

### Update API Key

```bash
# 1. Generate new API key
export NEW_API_KEY=$(openssl rand -hex 32)

# 2. Update secret
kubectl create secret generic api-keys \
    --from-literal=key1=$NEW_API_KEY \
    -n model-serving \
    --dry-run=client -o yaml | kubectl apply -f -

# 3. Restart pods to pick up new secret
kubectl rollout restart deployment/model-serving -n model-serving

# 4. Update clients with new key
# (Coordinate with API consumers)

# 5. Revoke old key after grace period
# (Remove from API_KEYS environment variable)
```

### Update TLS Certificate

```bash
# 1. Obtain new certificate
# (Let's Encrypt or internal CA)

# 2. Update secret
kubectl create secret tls tls-secret \
    --cert=new-cert.crt \
    --key=new-key.key \
    -n model-serving \
    --dry-run=client -o yaml | kubectl apply -f -

# 3. Verify ingress picks up new cert
kubectl describe ingress model-serving-ingress -n model-serving

# 4. Test HTTPS connection
curl -v https://api.example.com/health
```

### Drain Node for Maintenance

```bash
# 1. Cordon node (prevent new pods)
kubectl cordon node-1

# 2. Drain node (evict pods gracefully)
kubectl drain node-1 --ignore-daemonsets --delete-emptydir-data

# 3. Perform maintenance
ssh node-1
sudo apt-get update && sudo apt-get upgrade -y
sudo reboot

# 4. Uncordon node
kubectl uncordon node-1

# 5. Verify pods return
kubectl get pods -n model-serving -o wide
```

---

## Monitoring Dashboards

### Key Metrics to Watch

**Service Health**:
- Pod count and status
- Container restarts
- Health check success rate

**Performance**:
- Request rate (QPS)
- P50/P95/P99 latency
- Error rate

**Resources**:
- GPU utilization
- GPU memory usage
- CPU/Memory usage

**Business**:
- Models loaded
- Requests per model
- Active users

### Grafana Dashboards

- **Model Serving Overview**: http://grafana.example.com/d/model-serving-overview
- **GPU Metrics**: http://grafana.example.com/d/gpu-metrics
- **Request Tracing**: http://grafana.example.com/d/request-tracing


---

## References

- [API Reference](API.md)
- [Architecture Documentation](ARCHITECTURE.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)


---

