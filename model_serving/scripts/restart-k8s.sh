#!/bin/bash
# =============================================================================
# Restart Kubernetes Deployment (Clean Slate)
# =============================================================================

set -e

echo "=== 1. Deleting existing resources ==="
kubectl delete -k kubernetes/observability/ --ignore-not-found
kubectl delete -k kubernetes/base/ --ignore-not-found
echo "✓ Resources deleted"
echo ""

echo "=== 2. Deploying Base Resources ==="
kubectl apply -k kubernetes/base/
echo "✓ Base resources applied"
echo ""

echo "=== 3. Deploying Observability Stack (Jaeger, Prometheus, Grafana) ==="
kubectl apply -k kubernetes/observability/
echo "✓ Observability resources applied"
echo ""

echo "=== 4. Waiting for Pods to be Ready ==="
echo "Waiting for model-serving..."
kubectl wait --for=condition=Ready pod -l app=model-serving --timeout=300s

echo "Waiting for observability pods..."
kubectl wait --for=condition=Ready pod -l app=jaeger --timeout=300s
kubectl wait --for=condition=Ready pod -l app=prometheus --timeout=300s
kubectl wait --for=condition=Ready pod -l app=grafana --timeout=300s

echo ""
echo "=== ✅ Deployment Complete! ==="
echo "Run 'bash scripts/access-services.sh' to access the UIs."
