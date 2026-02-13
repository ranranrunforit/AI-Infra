#!/bin/bash
# =============================================================================
# Teardown Kubernetes Deployment
# =============================================================================

echo "=== Deleting Resources ==="
kubectl delete -k kubernetes/observability/ --ignore-not-found
kubectl delete -k kubernetes/base/ --ignore-not-found

echo ""
echo "=== Cleaning up Port Forwards ==="
pkill -f "kubectl port-forward" || true

echo ""
echo "=== Done! ==="
