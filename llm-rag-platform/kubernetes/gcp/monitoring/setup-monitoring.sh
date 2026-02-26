#!/bin/bash
# =============================================================================
# GKE Monitoring Stack Setup Script
# Project 303: Enterprise LLM Platform with RAG
#
# Run this from the root of the reference-implementation/ directory:
#   bash kubernetes/gcp/monitoring/setup-monitoring.sh
#
# Pre-requisite: kubectl must be connected to your GKE cluster
#   gcloud container clusters get-credentials rag-platform-cluster --region=us-central1
# =============================================================================
set -euo pipefail

GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin123}"   # Override via env var

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Project 303 – GKE Monitoring Stack Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Create monitoring namespace (idempotent) ───────────────────────────────
echo ""
echo "▶ Step 1/5: Creating monitoring namespace..."
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

# ── 2. Load LLM alerting rules from the existing llm-rules.yaml ──────────────
echo ""
echo "▶ Step 2/5: Applying LLM Prometheus rules (from monitoring/prometheus/llm-rules.yaml)..."
kubectl create configmap llm-prometheus-rules \
    --from-file=llm-rules.yaml=monitoring/prometheus/llm-rules.yaml \
    -n monitoring \
    --dry-run=client -o yaml | kubectl apply -f -

# ── 3. Create Grafana admin password secret (idempotent) ─────────────────────
echo ""
echo "▶ Step 3/5: Creating Grafana admin secret..."
kubectl create secret generic grafana-secrets \
    --from-literal=admin-password="${GRAFANA_PASSWORD}" \
    -n monitoring \
    --dry-run=client -o yaml | kubectl apply -f -

echo "  Grafana admin password: ${GRAFANA_PASSWORD}"
echo "  (change via: GRAFANA_PASSWORD=mypassword bash setup-monitoring.sh)"

# ── 4. Deploy Prometheus and Grafana ─────────────────────────────────────────
echo ""
echo "▶ Step 4/5: Deploying Prometheus..."
kubectl apply -f kubernetes/gcp/monitoring/prometheus.yaml

echo ""
echo "▶ Step 5/5: Deploying Grafana..."
kubectl apply -f kubernetes/gcp/monitoring/grafana.yaml

# ── 5. Wait for pods to be ready ─────────────────────────────────────────────
echo ""
echo "⏳ Waiting for Prometheus to be ready..."
kubectl rollout status deployment/prometheus -n monitoring --timeout=120s

echo ""
echo "⏳ Waiting for Grafana to be ready..."
kubectl rollout status deployment/grafana -n monitoring --timeout=120s

# ── 6. Print access info ──────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Monitoring stack deployed!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Get Grafana LoadBalancer IP (may take 1-2 minutes to provision)
GRAFANA_IP=$(kubectl get svc grafana-service -n monitoring \
    -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")

if [ "$GRAFANA_IP" != "pending" ] && [ -n "$GRAFANA_IP" ]; then
    echo ""
    echo "  📈 Grafana:    http://${GRAFANA_IP}"
    echo "     Login:      admin / ${GRAFANA_PASSWORD}"
    echo "     Dashboard:  LLM RAG Platform (auto-loaded)"
else
    echo ""
    echo "  📈 Grafana IP is still provisioning. Check with:"
    echo "     kubectl get svc grafana-service -n monitoring"
    echo ""
    echo "  OR access via port-forward (no public IP needed):"
    echo "     kubectl port-forward svc/grafana-service 3000:80 -n monitoring"
    echo "     Open: http://localhost:3000  (admin / ${GRAFANA_PASSWORD})"
fi

echo ""
echo "  📊 Prometheus (internal):  http://prometheus-service.monitoring:9090"
echo "     Port-forward:  kubectl port-forward svc/prometheus-service 9090:9090 -n monitoring"
echo ""
echo "  🔥 Hot-reload Prometheus config (after editing prometheus.yaml):"
echo "     kubectl rollout restart deployment/prometheus -n monitoring"
echo ""
echo "  📋 Check all resources:"
echo "     kubectl get all -n monitoring"
