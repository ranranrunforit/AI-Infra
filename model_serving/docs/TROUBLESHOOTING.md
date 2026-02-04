# Troubleshooting Guide - High-Performance Model Serving

## Overview

Comprehensive troubleshooting guide organized by symptom. Each issue includes symptoms, root causes, solutions, and prevention tips.

**Quick Navigation**: Use Ctrl+F to search for your error message or symptom.

---

## Table of Contents

- [Model Loading Issues](#model-loading-issues)
- [GPU and CUDA Errors](#gpu-and-cuda-errors)
- [Performance Issues](#performance-issues)
- [Memory Problems](#memory-problems)
- [Network and Connectivity](#network-and-connectivity)
- [Kubernetes Deployment Problems](#kubernetes-deployment-problems)
- [Debugging Tools and Commands](#debugging-tools-and-commands)
- [Log Interpretation](#log-interpretation)

---

## Model Loading Issues

### Issue: Model File Not Found

**Symptoms**:
```
ERROR: Model file not found: /models/resnet50-fp16.trt
FileNotFoundError: [Errno 2] No such file or directory: '/models/resnet50-fp16.trt'
```

**Root Causes**:
1. Model not uploaded to storage
2. Incorrect model path in configuration
3. PersistentVolume not mounted
4. Permissions issue

**Solutions**:

```bash
# 1. Verify model exists in storage
aws s3 ls s3://model-storage/models/resnet50-fp16.trt
# Or for local/NFS
ls -lh /mnt/models/resnet50-fp16.trt

# 2. Check PVC is mounted
kubectl describe pod <pod-name> -n model-serving | grep -A 5 Volumes
kubectl exec -n model-serving <pod-name> -- ls -la /models/

# 3. Check file permissions
kubectl exec -n model-serving <pod-name> -- ls -lh /models/

# 4. Copy model to correct location
kubectl cp resnet50-fp16.trt model-serving/<pod-name>:/models/

# 5. Update model path in config
kubectl set env deployment/model-serving MODEL_CACHE_DIR=/models -n model-serving
```

**Prevention**:
- Use init containers to download models before app starts
- Implement health checks that verify model presence
- Document model deployment procedure

---

### Issue: TensorRT Engine Version Mismatch

**Symptoms**:
```
ERROR: Engine deserialization failed - version mismatch or corrupted file
RuntimeError: TensorRT version mismatch: engine built with 8.5.1, runtime is 8.6.1
```

**Root Causes**:
1. Engine built with different TensorRT version
2. CUDA version incompatibility
3. Corrupted engine file

**Solutions**:

```bash
# 1. Check TensorRT version
kubectl exec -n model-serving <pod-name> -- python -c "import tensorrt; print(tensorrt.__version__)"

# 2. Rebuild engine with correct version
python scripts/convert_model.py \
    --model resnet50 \
    --precision fp16 \
    --output models/resnet50-fp16-new.trt

# 3. Verify CUDA version matches
kubectl exec -n model-serving <pod-name> -- nvcc --version
kubectl exec -n model-serving <pod-name> -- nvidia-smi

# 4. Use version-specific engine naming
# resnet50-fp16-trt86-cuda121.trt
```

**Prevention**:
- Include TensorRT/CUDA versions in engine filename
- Document engine build environment
- Use Docker images with pinned TensorRT versions

---

### Issue: Model Loading Timeout

**Symptoms**:
```
ERROR: Model loading timeout after 30 seconds
TimeoutError: Model resnet50-fp16 failed to load within timeout
```

**Root Causes**:
1. Large model taking too long to load
2. Slow storage (network latency)
3. Insufficient memory
4. Resource contention

**Solutions**:

```bash
# 1. Increase timeout
kubectl set env deployment/model-serving MODEL_LOAD_TIMEOUT=120 -n model-serving

# 2. Use init container for pre-loading
# Add to deployment spec:
initContainers:
- name: model-loader
  image: model-serving:latest
  command: ['python', 'scripts/preload-models.sh']
  volumeMounts:
  - name: models
    mountPath: /models

# 3. Check storage performance
kubectl exec -n model-serving <pod-name> -- dd if=/models/resnet50-fp16.trt of=/dev/null bs=1M
# Should be >100 MB/s for good performance

# 4. Use faster storage class
# Update PVC to use SSD storage class
storageClassName: fast-ssd
```

**Prevention**:
- Pre-load critical models during pod initialization
- Use local SSD or high-performance storage
- Implement lazy loading for non-critical models

---

## GPU and CUDA Errors

### Issue: CUDA Out of Memory

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 15.90 GiB total capacity; 
13.50 GiB already allocated; 1.80 GiB free; 13.60 GiB reserved in total by PyTorch)
```

**Root Causes**:
1. Batch size too large
2. Multiple models loaded simultaneously
3. Memory fragmentation
4. Memory leak

**Solutions**:

```bash
# 1. Check current GPU memory usage
kubectl exec -n model-serving <pod-name> -- nvidia-smi

# 2. Reduce batch size
kubectl set env deployment/model-serving MAX_BATCH_SIZE=16 -n model-serving

# 3. Reduce GPU memory fraction
kubectl set env deployment/model-serving GPU_MEMORY_FRACTION=0.85 -n model-serving

# 4. For vLLM, reduce memory utilization
kubectl set env deployment/model-serving VLLM_GPU_MEMORY_UTILIZATION=0.85 -n model-serving

# 5. Clear GPU cache (temporary fix)
kubectl exec -n model-serving <pod-name> -- python -c "
import torch
torch.cuda.empty_cache()
print('Cache cleared')
"

# 6. Unload unused models
curl -X POST "http://api.example.com/v1/models/unused-model/unload" \
  -H "Authorization: Bearer YOUR_API_KEY"

# 7. Restart pod (last resort)
kubectl delete pod <pod-name> -n model-serving
```

**Prevention**:
- Monitor GPU memory usage proactively
- Implement automatic model unloading for least-recently-used models
- Use INT8 quantization to reduce memory footprint
- Implement circuit breaker to reject requests when memory is high

---

### Issue: CUDA Driver Version Mismatch

**Symptoms**:
```
ERROR: CUDA driver version is insufficient for CUDA runtime version
RuntimeError: Detected that PyTorch and torchvision were compiled with different CUDA versions
```

**Root Causes**:
1. Host NVIDIA drivers outdated
2. Container CUDA version incompatible with host
3. Multiple CUDA versions installed

**Solutions**:

```bash
# 1. Check host CUDA driver version
nvidia-smi
# Look for "Driver Version: 525.xx"

# 2. Check container CUDA version
kubectl exec -n model-serving <pod-name> -- nvcc --version

# 3. Update NVIDIA drivers on host
sudo apt-get update
sudo apt-get install -y nvidia-driver-525
sudo reboot

# 4. Use compatible base image
# Update Dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu20.04
# Ensure CUDA version matches driver capabilities

# 5. Verify compatibility matrix
# CUDA 12.1 requires driver >= 525.60.13
# CUDA 11.8 requires driver >= 520.61.05
```

**Prevention**:
- Document required driver versions
- Use GPU operator for automated driver management
- Pin CUDA versions in Dockerfile

---

### Issue: No GPU Detected

**Symptoms**:
```
WARNING: GPU not available, falling back to CPU
torch.cuda.is_available() returns False
nvidia-smi: command not found (in container)
```

**Root Causes**:
1. NVIDIA Container Toolkit not installed
2. GPU not requested in pod spec
3. GPU node selector not configured
4. No GPU nodes available

**Solutions**:

```bash
# 1. Verify host can see GPU
nvidia-smi

# 2. Check NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu20.04 nvidia-smi
# If fails, reinstall toolkit:
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 3. Verify pod requests GPU
kubectl get pod <pod-name> -n model-serving -o yaml | grep -A 2 limits
# Should show: nvidia.com/gpu: "1"

# 4. Check GPU device plugin running
kubectl get pods -n kube-system | grep nvidia-device-plugin

# 5. Verify GPU nodes available
kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"

# 6. Update deployment to request GPU
kubectl patch deployment model-serving -n model-serving --patch '
spec:
  template:
    spec:
      containers:
      - name: model-serving
        resources:
          limits:
            nvidia.com/gpu: "1"
'

# 7. Add node selector
kubectl patch deployment model-serving -n model-serving --patch '
spec:
  template:
    spec:
      nodeSelector:
        accelerator: nvidia-gpu
'
```

**Prevention**:
- Always specify GPU resources in pod spec
- Use node selectors to ensure scheduling on GPU nodes
- Monitor GPU node availability
- Set up alerts for GPU node failures

---

### Issue: GPU Thermal Throttling

**Symptoms**:
```
WARNING: GPU temperature at 85°C, performance degraded
Inference latency increased from 2ms to 10ms
```

**Root Causes**:
1. Insufficient cooling
2. High ambient temperature
3. GPU overutilization
4. Dust accumulation

**Solutions**:

```bash
# 1. Check GPU temperature
kubectl exec -n model-serving <pod-name> -- nvidia-smi --query-gpu=temperature.gpu --format=csv
# Safe: <80°C, Warning: 80-85°C, Critical: >85°C

# 2. Reduce GPU utilization
kubectl scale deployment/model-serving --replicas=3 -n model-serving
# Distribute load across more GPUs

# 3. Reduce clock speeds (temporary)
kubectl exec -n model-serving <pod-name> -- nvidia-smi -pl 200
# Set power limit to 200W (default may be 250W)

# 4. Check cooling system
# Physical inspection required

# 5. Monitor continuously
watch -n 1 'kubectl exec -n model-serving <pod-name> -- nvidia-smi'
```

**Prevention**:
- Monitor GPU temperature in Prometheus
- Set alerts for temperature >80°C
- Ensure proper data center cooling
- Regular hardware maintenance
- Implement load balancing to prevent GPU hotspots

---

## Performance Issues

### Issue: High Latency (P99 > 100ms)

**Symptoms**:
```
Prometheus alert: HighLatency
P99 latency: 350ms (target: <100ms)
Customer complaints about slow responses
```

**Root Causes**:
1. GPU overutilization
2. Large batch sizes causing queueing delay
3. Network latency
4. Insufficient replicas
5. Cold cache / model warmup needed

**Diagnostic Steps**:

```bash
# 1. Check current latency distribution
curl http://api.example.com/metrics | grep model_serving_request_duration_seconds

# 2. View Jaeger traces for slow requests
# Open Jaeger UI, filter by duration > 100ms

# 3. Check GPU utilization
kubectl exec -n model-serving <pod-name> -- nvidia-smi --query-gpu=utilization.gpu --format=csv

# 4. Check replica count vs load
kubectl get hpa -n model-serving
kubectl top pods -n model-serving

# 5. Analyze batch processing
kubectl logs -n model-serving -l app=model-serving | grep batch_size
```

**Solutions**:

```bash
# Solution 1: Reduce batch timeout (lower latency, lower throughput)
kubectl set env deployment/model-serving BATCH_TIMEOUT_MS=5 -n model-serving

# Solution 2: Scale up
kubectl scale deployment/model-serving --replicas=10 -n model-serving

# Solution 3: Implement model warmup
# Add warmup to startup:
kubectl patch deployment model-serving -n model-serving --patch '
spec:
  template:
    spec:
      containers:
      - name: model-serving
        lifecycle:
          postStart:
            exec:
              command: ["python", "scripts/warmup-models.py"]
'

# Solution 4: Use faster precision
# Convert to INT8 for lower latency
python scripts/convert_model.py --model resnet50 --precision int8

# Solution 5: Enable request timeout
kubectl set env deployment/model-serving REQUEST_TIMEOUT_SEC=10 -n model-serving
```

**Prevention**:
- Set up latency SLO alerts
- Implement auto-scaling based on P99 latency
- Regular performance testing
- Capacity planning based on traffic patterns

---

### Issue: Low Throughput (QPS)

**Symptoms**:
```
Expected: 1000 QPS per GPU
Actual: 300 QPS per GPU
Load tests show capacity constraints
```

**Root Causes**:
1. Batch size too small
2. Synchronous processing bottleneck
3. CPU bottleneck in pre/post-processing
4. Network bandwidth limitation

**Diagnostic Steps**:

```bash
# 1. Check actual QPS
curl http://api.example.com/metrics | grep rate

# 2. Check batch size distribution
curl http://api.example.com/metrics | grep model_serving_batch_size

# 3. Check CPU utilization
kubectl top pods -n model-serving

# 4. Run load test
python benchmarks/load_test.py --url http://api.example.com --concurrent 100 --duration 60
```

**Solutions**:

```bash
# Solution 1: Increase batch size
kubectl set env deployment/model-serving MAX_BATCH_SIZE=64 -n model-serving

# Solution 2: Increase batch timeout (more batching)
kubectl set env deployment/model-serving BATCH_TIMEOUT_MS=20 -n model-serving

# Solution 3: Increase workers (if CPU-bound)
kubectl set env deployment/model-serving UVICORN_WORKERS=4 -n model-serving

# Solution 4: Optimize preprocessing
# Review and optimize code in _preprocess_inputs()

# Solution 5: Increase CPU allocation
kubectl patch deployment model-serving -n model-serving --patch '
spec:
  template:
    spec:
      containers:
      - name: model-serving
        resources:
          requests:
            cpu: "8"
          limits:
            cpu: "16"
'
```

**Prevention**:
- Regular load testing
- Monitor batch size metrics
- Profile CPU usage patterns
- Benchmark different configurations

---

### Issue: Inconsistent Latency (High P99/P50 Ratio)

**Symptoms**:
```
P50 latency: 5ms
P99 latency: 200ms (40x worse)
Some requests are very slow
```

**Root Causes**:
1. Cold model loading
2. Garbage collection pauses
3. Batch processing variations
4. Network retries
5. GPU context switching

**Solutions**:

```bash
# 1. Enable detailed tracing
kubectl set env deployment/model-serving TRACE_SAMPLE_RATE=1.0 -n model-serving

# 2. Analyze slow traces in Jaeger
# Look for outliers and identify bottlenecks

# 3. Implement model warmup
python scripts/warmup-models.py

# 4. Tune garbage collection
kubectl set env deployment/model-serving PYTHONHASHSEED=0 -n model-serving
kubectl set env deployment/model-serving MALLOC_TRIM_THRESHOLD_=100000 -n model-serving

# 5. Use consistent batch sizes
kubectl set env deployment/model-serving ENFORCE_BATCH_SIZE=true -n model-serving
```

---

## Memory Problems

### Issue: OOMKilled (Out of Memory)

**Symptoms**:
```
kubectl get pods shows: OOMKilled
Pod restarts frequently
Logs end abruptly without error message
```

**Root Causes**:
1. Memory limit too low
2. Memory leak
3. Too many models loaded
4. Large input data

**Solutions**:

```bash
# 1. Check current memory usage
kubectl top pod <pod-name> -n model-serving

# 2. Check pod events
kubectl describe pod <pod-name> -n model-serving | grep -A 10 Events

# 3. Increase memory limits
kubectl patch deployment model-serving -n model-serving --patch '
spec:
  template:
    spec:
      containers:
      - name: model-serving
        resources:
          requests:
            memory: "16Gi"
          limits:
            memory: "32Gi"
'

# 4. Check for memory leaks
# Monitor memory over time
kubectl exec -n model-serving <pod-name> -- ps aux | grep python
# Look for increasing RSS values

# 5. Implement memory limits per request
kubectl set env deployment/model-serving MAX_INPUT_SIZE_MB=10 -n model-serving

# 6. Unload unused models
curl -X POST http://api.example.com/v1/models/unused-model/unload \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Prevention**:
- Set appropriate memory limits based on load testing
- Monitor memory usage trends
- Implement automated model unloading
- Regular memory profiling

---

### Issue: Memory Leak

**Symptoms**:
```
Memory usage continuously increasing
Pod memory grows from 4GB to 30GB over days
Eventually leads to OOMKilled
```

**Diagnostic Steps**:

```bash
# 1. Monitor memory over time
watch -n 60 'kubectl top pod -n model-serving | grep model-serving'

# 2. Use memory profiler
kubectl exec -n model-serving <pod-name> -- python -m memory_profiler src/serving/server.py

# 3. Check Python object count
kubectl exec -n model-serving <pod-name> -- python -c "
import gc
gc.collect()
print(f'Objects: {len(gc.get_objects())}')
"

# 4. Use heapy for memory analysis
kubectl exec -it -n model-serving <pod-name> -- python
>>> from guppy import hpy
>>> h = hpy()
>>> print(h.heap())
```

**Solutions**:

```bash
# 1. Restart pods regularly (temporary workaround)
kubectl rollout restart deployment/model-serving -n model-serving

# 2. Implement automatic restarts based on memory threshold
# Add to deployment:
livenessProbe:
  exec:
    command:
    - sh
    - -c
    - |
      MEMORY=$(ps aux | grep python | awk '{sum+=$6} END {print sum}')
      [ $MEMORY -lt 30000000 ]  # Restart if >30GB
  initialDelaySeconds: 300
  periodSeconds: 60

# 3. Use memory limit per worker
# If using multiple workers, each has separate memory space

# 4. Profile and fix code
# Common leaks:
# - Unclosed file handles
# - Circular references
# - Large caches not cleared
# - Event listeners not removed
```

**Prevention**:
- Regular memory profiling in development
- Code reviews focusing on resource management
- Automated memory tests in CI/CD
- Monitor memory trends in production

---

## Network and Connectivity

### Issue: Connection Timeout

**Symptoms**:
```
curl: (28) Connection timed out after 30000 milliseconds
Client errors: "Request timeout"
```

**Root Causes**:
1. Service not accessible
2. Firewall blocking connections
3. Ingress misconfigured
4. Slow inference causing timeout

**Solutions**:

```bash
# 1. Check service is running
kubectl get svc -n model-serving

# 2. Test from within cluster
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl http://model-serving.model-serving.svc.cluster.local/health

# 3. Check ingress configuration
kubectl get ingress -n model-serving
kubectl describe ingress model-serving-ingress -n model-serving

# 4. Check network policies
kubectl get networkpolicy -n model-serving

# 5. Increase timeout
# On ingress:
kubectl annotate ingress model-serving-ingress \
  nginx.ingress.kubernetes.io/proxy-read-timeout=300 \
  -n model-serving

# 6. Test connectivity from different locations
curl -v -m 60 http://api.example.com/health
```

**Prevention**:
- Set appropriate timeouts based on P99 latency
- Implement health checks
- Monitor connection errors
- Document network requirements

---

### Issue: DNS Resolution Failure

**Symptoms**:
```
Could not resolve host: model-serving.model-serving.svc.cluster.local
getaddrinfo: Name or service not known
```

**Solutions**:

```bash
# 1. Check CoreDNS is running
kubectl get pods -n kube-system | grep coredns

# 2. Test DNS from pod
kubectl run -it --rm debug --image=busybox --restart=Never -- \
  nslookup model-serving.model-serving.svc.cluster.local

# 3. Check service exists
kubectl get svc -n model-serving

# 4. Verify service endpoints
kubectl get endpoints model-serving -n model-serving

# 5. Check DNS configuration
kubectl get configmap -n kube-system coredns -o yaml
```

---

## Kubernetes Deployment Problems

### Issue: ImagePullBackOff

**Symptoms**:
```
kubectl get pods shows: ImagePullBackOff or ErrImagePull
Pod stays in Pending state
```

**Root Causes**:
1. Image doesn't exist
2. Registry authentication failure
3. Network connectivity to registry
4. Image tag incorrect

**Solutions**:

```bash
# 1. Check pod events
kubectl describe pod <pod-name> -n model-serving

# 2. Verify image exists
docker pull your-registry.com/model-serving:v1.0.0

# 3. Check image pull secret
kubectl get secret -n model-serving | grep docker
kubectl describe secret <image-pull-secret> -n model-serving

# 4. Create image pull secret if missing
kubectl create secret docker-registry regcred \
  --docker-server=your-registry.com \
  --docker-username=<username> \
  --docker-password=<password> \
  --docker-email=<email> \
  -n model-serving

# 5. Add secret to service account
kubectl patch serviceaccount default -n model-serving -p '
{
  "imagePullSecrets": [{"name": "regcred"}]
}'

# 6. Update deployment to use correct image
kubectl set image deployment/model-serving \
  model-serving=your-registry.com/model-serving:v1.0.0 \
  -n model-serving
```

---

### Issue: CrashLoopBackOff

**Symptoms**:
```
kubectl get pods shows: CrashLoopBackOff
Pod continuously restarting
```

**Diagnostic Steps**:

```bash
# 1. Check pod status
kubectl describe pod <pod-name> -n model-serving

# 2. View current logs
kubectl logs -n model-serving <pod-name>

# 3. View previous container logs
kubectl logs -n model-serving <pod-name> --previous

# 4. Check liveness probe
kubectl describe pod <pod-name> -n model-serving | grep -A 10 Liveness
```

**Common Causes and Solutions**:

**Cause: Application error on startup**
```bash
# View logs to identify error
kubectl logs -n model-serving <pod-name> --previous

# Common fixes:
# - Environment variable missing
kubectl set env deployment/model-serving REQUIRED_VAR=value -n model-serving

# - Config file missing
kubectl create configmap app-config --from-file=config.yaml -n model-serving

# - Permissions issue
kubectl exec -n model-serving <pod-name> -- chmod +x /app/startup.sh
```

**Cause: Health check failing**
```bash
# Disable health check temporarily to see app logs
kubectl patch deployment model-serving -n model-serving --patch '
spec:
  template:
    spec:
      containers:
      - name: model-serving
        livenessProbe: null
        readinessProbe: null
'

# Fix underlying issue, then re-enable health checks
```

**Cause: Port conflict**
```bash
# Check what port app is trying to bind
kubectl logs -n model-serving <pod-name> | grep -i port

# Update service/deployment to match
kubectl patch svc model-serving -n model-serving --patch '
spec:
  ports:
  - port: 80
    targetPort: 8000
'
```

---

### Issue: Insufficient Resources

**Symptoms**:
```
Pod stuck in Pending state
Events: "0/5 nodes are available: 2 Insufficient nvidia.com/gpu, 3 node(s) didn't match node selector"
```

**Solutions**:

```bash
# 1. Check pod resource requests
kubectl describe pod <pod-name> -n model-serving | grep -A 5 Requests

# 2. Check node resources
kubectl describe nodes | grep -A 10 Allocatable

# 3. Check GPU availability
kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"

# 4. Solutions:
# Option A: Add more GPU nodes
# (Cloud provider specific)

# Option B: Reduce resource requests
kubectl patch deployment model-serving -n model-serving --patch '
spec:
  template:
    spec:
      containers:
      - name: model-serving
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
'

# Option C: Remove node selector
kubectl patch deployment model-serving -n model-serving --patch '
spec:
  template:
    spec:
      nodeSelector: null
'
```

---

## Debugging Tools and Commands

### Essential Debug Commands

```bash
# Get pod status
kubectl get pods -n model-serving -o wide

# Describe pod (see events)
kubectl describe pod <pod-name> -n model-serving

# View logs
kubectl logs -n model-serving <pod-name> -f
kubectl logs -n model-serving <pod-name> --previous

# Execute command in pod
kubectl exec -it -n model-serving <pod-name> -- bash

# Port forward for local testing
kubectl port-forward -n model-serving <pod-name> 8000:8000

# Check resource usage
kubectl top pod -n model-serving
kubectl top node

# Get events
kubectl get events -n model-serving --sort-by='.lastTimestamp'
```

### Debugging with Debug Container

```bash
# Add ephemeral debug container (K8s 1.23+)
kubectl debug -it <pod-name> -n model-serving --image=busybox --target=model-serving

# Or use traditional sidecar
kubectl patch deployment model-serving -n model-serving --patch '
spec:
  template:
    spec:
      containers:
      - name: debug
        image: nicolaka/netshoot
        command: ["sleep", "infinity"]
'
```

### Network Debugging

```bash
# Test DNS
kubectl run -it --rm debug --image=busybox --restart=Never -- nslookup model-serving.model-serving.svc.cluster.local

# Test connectivity
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- curl http://model-serving.model-serving.svc.cluster.local/health

# Trace route
kubectl run -it --rm debug --image=nicolaka/netshoot --restart=Never -- traceroute model-serving.model-serving.svc.cluster.local

# Check network policies
kubectl describe networkpolicy -n model-serving
```

### GPU Debugging

```bash
# Check GPU from pod
kubectl exec -n model-serving <pod-name> -- nvidia-smi
kubectl exec -n model-serving <pod-name> -- nvidia-smi -L

# Continuous monitoring
kubectl exec -n model-serving <pod-name> -- watch -n 1 nvidia-smi

# Check CUDA availability
kubectl exec -n model-serving <pod-name> -- python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
"
```

---

## Log Interpretation

### Understanding Log Levels

```
DEBUG - Detailed information for diagnosing problems
INFO - General informational messages
WARNING - Warning messages (non-critical)
ERROR - Error messages (operation failed)
CRITICAL - Critical errors (service impaired)
```

### Common Log Patterns

**Successful Request**:
```json
{
  "timestamp": "2024-01-15T10:30:00.123Z",
  "level": "INFO",
  "message": "Request completed",
  "trace_id": "abc123",
  "model": "resnet50-fp16",
  "latency_ms": 1.2,
  "status": "success"
}
```

**Failed Request**:
```json
{
  "timestamp": "2024-01-15T10:30:05.456Z",
  "level": "ERROR",
  "message": "Prediction failed: CUDA out of memory",
  "trace_id": "def456",
  "model": "resnet50-fp16",
  "exception": "RuntimeError: CUDA out of memory...",
  "status": "error"
}
```

**Startup Logs**:
```
INFO:__main__:Starting Model Serving Server
INFO:model_loader:ModelLoader initialized with cache_dir=/tmp/model_cache
INFO:batch_processor:DynamicBatchProcessor initialized: max_batch_size=32, timeout_ms=10
INFO:tracer:Tracing initialized: service=model-serving
INFO:__main__:Server startup complete
INFO:uvicorn:Started server process [1]
INFO:uvicorn:Waiting for application startup.
INFO:uvicorn:Application startup complete.
INFO:uvicorn:Uvicorn running on http://0.0.0.0:8000
```

---

## Getting Help

### Internal Resources
- **Slack**: #ml-platform
- **Wiki**: https://wiki.example.com/model-serving
- **Runbook**: [RUNBOOK.md](RUNBOOK.md)

### External Resources
- **TensorRT Documentation**: https://docs.nvidia.com/deeplearning/tensorrt/
- **vLLM GitHub**: https://github.com/vllm-project/vllm
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Kubernetes Documentation**: https://kubernetes.io/docs/

### Creating a Support Ticket

Include the following information:

```
Subject: [Model Serving] Brief description of issue

Environment:
- Cluster: production/staging/dev
- Namespace: model-serving
- Pod: <pod-name>
- Deployment version: v1.2.3

Symptoms:
- What is happening
- When did it start
- How often does it occur

Steps to Reproduce:
1. ...
2. ...
3. ...

Logs:
<attach relevant logs>

What I've tried:
- ...
- ...

Impact:
- Users affected: X
- Severity: Critical/High/Medium/Low
```

---

## References

- [API Reference](API.md)
- [Architecture Documentation](ARCHITECTURE.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Operations Runbook](RUNBOOK.md)
- [Step-by-Step Implementation Guide](STEP_BY_STEP.md)
