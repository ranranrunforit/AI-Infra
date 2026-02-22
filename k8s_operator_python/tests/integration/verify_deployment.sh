#!/bin/bash
set -e

echo "Starting End-to-End Verification..."

# 1. Check CRD
echo "Checking CRD..."
kubectl get crd trainingjobs.ml.example.com || (echo "CRD not found" && exit 1)

# 2. Check Operator Deployment
echo "Checking Operator Pod..."
kubectl wait --for=condition=available --timeout=60s deployment/trainingjob-operator -n ml-training || (echo "Operator deployment failed" && exit 1)

# 3. Check Service Monitor
echo "Checking Service Monitor..."
kubectl get servicemonitor trainingjob-operator-monitor -n ml-training || (echo "ServiceMonitor not found" && exit 1)

# 4. Check Metrics Endpoint
echo "Verifying Metrics Endpoint..."
# Port forward in background
kubectl port-forward svc/operator-service 9090:9090 -n ml-training > /dev/null 2>&1 &
PF_PID=$!
sleep 5

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:9090/metrics)
kill $PF_PID

if [ "$HTTP_CODE" -eq "200" ]; then
    echo "Metrics endpoint is UP (200 OK)"
else
    echo "Metrics endpoint failed with code $HTTP_CODE"
    exit 1
fi

echo "Verification Successful!"
