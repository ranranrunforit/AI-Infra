#!/bin/bash
# =============================================================================
# Access Services via Port Forwarding
# =============================================================================

echo "=== Port Forwarding Services ==="
echo "Keep this terminal open!"
echo ""
echo "  - Model Serving API:  http://localhost:8000"
echo "  - Jaeger UI:          http://localhost:16686"
echo "  - Prometheus UI:      http://localhost:9090"
echo "  - Grafana UI:         http://localhost:3000 (admin/admin)"
echo ""

# Kill existing port-forwards
pkill -f "kubectl port-forward" || true

# Forward ports in background
kubectl port-forward svc/model-serving 8000:8000 &
kubectl port-forward svc/jaeger 16686:16686 &
kubectl port-forward svc/prometheus 9090:9090 &
kubectl port-forward svc/grafana 3000:3000 &

# Wait
wait
