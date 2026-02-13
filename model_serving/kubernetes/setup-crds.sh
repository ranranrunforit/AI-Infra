#!/bin/bash
# =============================================================================
# Install CRDs required for full Kubernetes deployment
#
# This installs the Custom Resource Definitions (CRDs) for:
#   - Prometheus Operator (ServiceMonitor, PrometheusRule)
#   - Vertical Pod Autoscaler (VPA)
#
# After running this script, you can deploy all resources without CRD errors.
#
# Usage:
#   bash kubernetes/setup-crds.sh
#
# Or from Windows PowerShell (via WSL2/minikube):
#   wsl bash kubernetes/setup-crds.sh
# =============================================================================

set -e

echo "=== Installing Kubernetes CRDs ==="
echo ""

# --------------------------------------------------------------------------
# 1. Prometheus Operator CRDs (ServiceMonitor, PrometheusRule, etc.)
# --------------------------------------------------------------------------
echo "[1/2] Installing Prometheus Operator CRDs..."

PROM_VERSION="v0.72.0"
PROM_CRD_BASE="https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/${PROM_VERSION}/example/prometheus-operator-crd"

for crd in \
    monitoring.coreos.com_servicemonitors.yaml \
    monitoring.coreos.com_prometheusrules.yaml \
    monitoring.coreos.com_podmonitors.yaml \
    monitoring.coreos.com_probes.yaml; do
    echo "  Installing ${crd}..."
    kubectl apply --server-side -f "${PROM_CRD_BASE}/${crd}" 2>/dev/null || \
    kubectl apply -f "${PROM_CRD_BASE}/${crd}"
done

echo "  ✓ Prometheus Operator CRDs installed"
echo ""

# --------------------------------------------------------------------------
# 2. Vertical Pod Autoscaler (VPA) CRDs
# --------------------------------------------------------------------------
echo "[2/2] Installing VPA CRDs..."

VPA_VERSION="1.1.2"
VPA_CRD_BASE="https://raw.githubusercontent.com/kubernetes/autoscaler/vertical-pod-autoscaler-${VPA_VERSION}/vertical-pod-autoscaler/deploy"

for crd in \
    vpa-v1-crd-gen.yaml; do
    echo "  Installing ${crd}..."
    kubectl apply -f "${VPA_CRD_BASE}/${crd}"
done

echo "  ✓ VPA CRDs installed"
echo ""

# --------------------------------------------------------------------------
# Verify
# --------------------------------------------------------------------------
echo "=== Verifying CRDs ==="
echo ""

echo "Prometheus Operator CRDs:"
kubectl get crd servicemonitors.monitoring.coreos.com 2>/dev/null && echo "  ✓ ServiceMonitor" || echo "  ✗ ServiceMonitor (missing)"
kubectl get crd prometheusrules.monitoring.coreos.com 2>/dev/null && echo "  ✓ PrometheusRule" || echo "  ✗ PrometheusRule (missing)"

echo ""
echo "VPA CRDs:"
kubectl get crd verticalpodautoscalers.autoscaling.k8s.io 2>/dev/null && echo "  ✓ VerticalPodAutoscaler" || echo "  ✗ VerticalPodAutoscaler (missing)"

echo ""
echo "=== Done! You can now deploy: kubectl apply -k kubernetes/base/ ==="
