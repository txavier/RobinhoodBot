# Genetic Optimizer — Kubernetes Quick Start

Deploy **only** the genetic optimizer on Kubernetes (Rancher Desktop, minikube, or any K8s cluster) — no trading bot.

## Prerequisites

- Kubernetes cluster running (e.g., [Rancher Desktop](https://rancherdesktop.io/))
- `kubectl` configured and connected to your cluster
- A `robinhoodbot/config.py` file (copy from `config.py.sample` and edit)

Verify your cluster is reachable:

```bash
kubectl get nodes
```

## Quick Start

### 1. Build the Docker image

```bash
docker build -t robinhoodbot:latest .
```

For **Rancher Desktop** (K3s with containerd), the image is automatically available to the cluster.

For **minikube**, load the image:

```bash
minikube image load robinhoodbot:latest
```

For a **remote cluster**, push to your registry:

```bash
docker tag robinhoodbot:latest your-registry.com/robinhoodbot:latest
docker push your-registry.com/robinhoodbot:latest
```

### 2. Create the namespace

```bash
kubectl apply -f k8s/namespace.yaml
```

### 3. Create the config secret

```bash
kubectl create secret generic robinhoodbot-config \
  --from-file=config.py=robinhoodbot/config.py \
  -n robinhoodbot
```

### 4. Create persistent volume claims

```bash
kubectl apply -f k8s/pvc.yaml -f k8s/optimizer-cache-pvc.yaml
```

### 5. Launch the optimizer job

```bash
kubectl apply -f k8s/optimizer-job.yaml
```

### 6. Follow the logs

```bash
kubectl logs -f -n robinhoodbot job/robinhoodbot-optimizer
```

## Monitoring & Management

```bash
# Check pod status
kubectl get pods -n robinhoodbot

# Check job status
kubectl get jobs -n robinhoodbot

# View recent logs (last 50 lines)
kubectl logs -n robinhoodbot job/robinhoodbot-optimizer --tail=50

# Interactive shell into the running pod
kubectl exec -it -n robinhoodbot job/robinhoodbot-optimizer -- bash
```

## Re-running the Optimizer

Jobs are immutable — delete the old one before creating a new run:

```bash
kubectl delete job robinhoodbot-optimizer -n robinhoodbot
kubectl apply -f k8s/optimizer-job.yaml
```

To run with different parameters, edit the `args` in `k8s/optimizer-job.yaml` before applying, or create an inline job:

```bash
kubectl create job optimizer-quick -n robinhoodbot \
  --image=robinhoodbot:latest \
  -- python3 genetic_optimizer_intraday.py \
     --num-stocks=50 --generations=20 --population=30 --real-data
```

## Updating the Config

```bash
kubectl create secret generic robinhoodbot-config \
  --from-file=config.py=robinhoodbot/config.py \
  -n robinhoodbot \
  --dry-run=client -o yaml | kubectl apply -f -
```

## Cleanup

```bash
# Delete just the optimizer job
kubectl delete job robinhoodbot-optimizer -n robinhoodbot

# Delete everything (namespace and all resources)
kubectl delete namespace robinhoodbot
```

## Dashboard

### Install monitoring (one-time setup)

**Linux / macOS / WSL:**
```bash
./k8s/monitoring/install-monitoring.sh
```

**Windows PowerShell:**
```powershell
.\k8s\monitoring\install-monitoring.ps1
```

### View the dashboard

```bash
# Port-forward Grafana
kubectl port-forward -n prometheus-system svc/prometheus-grafana 3005:http-web
# Open http://localhost:3005
# Get password:
kubectl get secret -n prometheus-system prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 -d
# Login: admin / <password from above>
# Navigate to Dashboards → Ray → Default Dashboard
```

## Quick Reference

```bash
docker build -t robinhoodbot:latest .
kubectl delete job robinhoodbot-optimizer -n robinhoodbot
kubectl apply -f k8s/optimizer-job.yaml
kubectl logs -f -n robinhoodbot job/robinhoodbot-optimizer
kubectl port-forward -n robinhoodbot job/robinhoodbot-optimizer 8265:8265
kubectl cp robinhoodbot/POD_NAME:/app/data/genetic_optimization_intraday_result.json ./robinhoodbot/genetic_optimization_intraday_result.json
```
