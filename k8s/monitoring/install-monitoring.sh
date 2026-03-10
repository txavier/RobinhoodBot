#!/bin/bash
# ============================================================================
# Install Prometheus + Grafana monitoring stack for Ray metrics
# ============================================================================
# Usage:
#   ./k8s/monitoring/install-monitoring.sh
#
# Prerequisites:
#   - kubectl configured for your cluster
#   - Helm 3 installed (https://helm.sh/docs/intro/install/)
#
# This script:
#   1. Installs kube-prometheus-stack (Prometheus + Grafana) via Helm
#   2. Creates PodMonitors to scrape Ray metrics from optimizer pods
#   3. Downloads and imports Ray's Grafana dashboards
# ============================================================================

set -e

NAMESPACE="prometheus-system"
HELM_RELEASE="prometheus"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Step 1: Add Helm repos ==="
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

echo ""
echo "=== Step 2: Create monitoring namespace ==="
kubectl create namespace "$NAMESPACE" 2>/dev/null || echo "Namespace $NAMESPACE already exists"

echo ""
echo "=== Step 3: Install kube-prometheus-stack ==="
helm upgrade --install "$HELM_RELEASE" prometheus-community/kube-prometheus-stack \
  --namespace "$NAMESPACE" \
  --values "$SCRIPT_DIR/prometheus-values.yaml" \
  --wait

echo ""
echo "=== Step 4: Apply PodMonitors for Ray metrics ==="
kubectl apply -f "$SCRIPT_DIR/ray-pod-monitor.yaml"

echo ""
echo "=== Step 5: Import Ray Grafana dashboards ==="
DASHBOARD_URL="https://raw.githubusercontent.com/ray-project/kuberay/master/config/grafana/default_grafana_dashboard.json"
TEMP_FILE=$(mktemp)
echo "Downloading Ray Grafana dashboard from KubeRay repo..."
if curl -sfL "$DASHBOARD_URL" -o "$TEMP_FILE" && [ -s "$TEMP_FILE" ]; then
  kubectl create configmap ray-grafana-dashboards \
    --namespace "$NAMESPACE" \
    --from-file="default-dashboard.json=$TEMP_FILE" \
    --dry-run=client -o yaml | \
    kubectl apply -f -
  kubectl label configmap ray-grafana-dashboards \
    --namespace "$NAMESPACE" \
    grafana_dashboard=1 \
    --overwrite
  kubectl annotate configmap ray-grafana-dashboards \
    --namespace "$NAMESPACE" \
    grafana_folder=Ray \
    --overwrite
  echo "Ray Grafana dashboard imported successfully."
else
  echo "WARNING: Failed to download dashboard. Import manually from:"
  echo "  $DASHBOARD_URL"
fi
rm -f "$TEMP_FILE"

echo ""
echo "=== Installation complete! ==="
echo ""
echo "To access Grafana:"
echo "  kubectl port-forward -n $NAMESPACE svc/prometheus-grafana 3005:http-web"
echo "  Open http://localhost:3005"
echo "  Login: admin / prom-operator"
echo ""
echo "To access Prometheus:"
echo "  kubectl port-forward -n $NAMESPACE svc/prometheus-kube-prometheus-prometheus 9090:http-web"
echo "  Open http://localhost:9090"
echo ""
echo "To check Ray metrics are being scraped:"
echo "  1. Open Prometheus at http://localhost:9090/targets"
echo "  2. Look for 'podMonitor/prometheus-system/ray-optimizer-monitor'"
echo "  3. Run query: ray_node_cpu_utilization"
