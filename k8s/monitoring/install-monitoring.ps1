# ============================================================================
# Install Prometheus + Grafana monitoring stack for Ray metrics
# ============================================================================
# Usage:
#   .\k8s\monitoring\install-monitoring.ps1
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

$ErrorActionPreference = "Stop"

$Namespace = "prometheus-system"
$HelmRelease = "prometheus"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=== Step 1: Add Helm repos ==="
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

Write-Host ""
Write-Host "=== Step 2: Create monitoring namespace ==="
kubectl create namespace $Namespace 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "Namespace $Namespace already exists" }

Write-Host ""
Write-Host "=== Step 3: Install kube-prometheus-stack ==="
helm upgrade --install $HelmRelease prometheus-community/kube-prometheus-stack `
  --namespace $Namespace `
  --values "$ScriptDir\prometheus-values.yaml" `
  --wait

Write-Host ""
Write-Host "=== Step 4: Apply PodMonitors for Ray metrics ==="
kubectl apply -f "$ScriptDir\ray-pod-monitor.yaml"

Write-Host ""
Write-Host "=== Step 5: Import Ray Grafana dashboards ==="
$DashboardUrl = "https://raw.githubusercontent.com/ray-project/kuberay/master/config/grafana/default_grafana_dashboard.json"
$TempFile = New-TemporaryFile
Write-Host "Downloading Ray Grafana dashboard from KubeRay repo..."
try {
    Invoke-WebRequest -Uri $DashboardUrl -OutFile $TempFile -UseBasicParsing
    if ((Get-Item $TempFile).Length -gt 0) {
        kubectl create configmap ray-grafana-dashboards `
          --namespace $Namespace `
          --from-file="default-dashboard.json=$TempFile" `
          --dry-run=client -o yaml | kubectl apply -f -
        kubectl label configmap ray-grafana-dashboards `
          --namespace $Namespace `
          grafana_dashboard=1 `
          --overwrite
        kubectl annotate configmap ray-grafana-dashboards `
          --namespace $Namespace `
          grafana_folder=Ray `
          --overwrite
        Write-Host "Ray Grafana dashboard imported successfully."
    }
} catch {
    Write-Host "WARNING: Failed to download dashboard. Import manually from:"
    Write-Host "  $DashboardUrl"
} finally {
    Remove-Item $TempFile -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "=== Installation complete! ==="
Write-Host ""
Write-Host "To access Grafana:"
Write-Host "  kubectl port-forward -n $Namespace svc/prometheus-grafana 3005:http-web"
Write-Host "  Open http://localhost:3005"
Write-Host "  Get password: kubectl get secret -n $Namespace prometheus-grafana -o jsonpath=`"{.data.admin-password}`" | ForEach-Object { [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String(`$_)) }"
Write-Host ""
Write-Host "To access Prometheus:"
Write-Host "  kubectl port-forward -n $Namespace svc/prometheus-kube-prometheus-prometheus 9090:http-web"
Write-Host "  Open http://localhost:9090"
