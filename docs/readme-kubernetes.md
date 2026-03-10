# RobinhoodBot — Kubernetes Setup

## Prerequisites

- A Kubernetes cluster (local: [minikube](https://minikube.sigs.k8s.io/), [kind](https://kind.sigs.k8s.io/), [Docker Desktop](https://docs.docker.com/desktop/kubernetes/); cloud: EKS, GKE, AKS)
- [kubectl](https://kubernetes.io/docs/tasks/tools/) configured to talk to your cluster
- The Docker image built and accessible to your cluster (local or pushed to a registry)

## Build & Push the Image

```bash
# Build locally
docker build -t robinhoodbot:latest .

# For a remote cluster, tag and push to your registry
docker tag robinhoodbot:latest your-registry.com/robinhoodbot:latest
docker push your-registry.com/robinhoodbot:latest
```

If using **minikube**, load the image directly:
```bash
minikube image load robinhoodbot:latest
```

## Kubernetes Manifests

### 1. Namespace (optional)

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: robinhoodbot
```

### 2. Secret — Credentials (config.py)

Store your `config.py` as a Kubernetes Secret (it contains passwords and tokens):

```bash
kubectl create secret generic robinhoodbot-config \
  --from-file=config.py=robinhoodbot/config.py \
  -n robinhoodbot
```

### 3. ConfigMap — Log / Trade History Seed Files

For first-time setup, seed empty JSON files so the bot can start fresh:

```bash
# Create empty seed files (only needed if you don't have existing data)
echo '{}' > /tmp/tradehistory-real.json
echo '{}' > /tmp/tradehistory.json
echo '[]' > /tmp/log.json
echo '[]' > /tmp/console_log.json
echo '{}' > /tmp/buy_reasons.json

# Or, to migrate existing data:
kubectl create configmap robinhoodbot-data \
  --from-file=tradehistory-real.json=robinhoodbot/tradehistory-real.json \
  --from-file=tradehistory.json=robinhoodbot/tradehistory.json \
  --from-file=log.json=robinhoodbot/log.json \
  --from-file=console_log.json=robinhoodbot/console_log.json \
  --from-file=buy_reasons.json=robinhoodbot/buy_reasons.json \
  -n robinhoodbot
```

> **Note:** ConfigMaps have a 1 MiB limit. If your log files exceed this, use a PersistentVolumeClaim instead (see section below).

### 4. PersistentVolumeClaim — For Large Log Files

When log files grow beyond ConfigMap limits, use persistent storage:

```yaml
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: robinhoodbot-data
  namespace: robinhoodbot
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 500Mi
```

### 5. Deployment — Trading Bot

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: robinhoodbot
  namespace: robinhoodbot
  labels:
    app: robinhoodbot
spec:
  replicas: 1  # IMPORTANT: Never run more than 1 replica — duplicate trades!
  strategy:
    type: Recreate  # Don't run two pods simultaneously during updates
  selector:
    matchLabels:
      app: robinhoodbot
  template:
    metadata:
      labels:
        app: robinhoodbot
    spec:
      containers:
        - name: bot
          image: robinhoodbot:latest
          imagePullPolicy: IfNotPresent
          env:
            - name: TZ
              value: "America/New_York"
          volumeMounts:
            - name: config
              mountPath: /app/config.py
              subPath: config.py
              readOnly: true
            - name: data
              mountPath: /app/data
          livenessProbe:
            exec:
              command: ["pgrep", "-f", "main.py"]
            initialDelaySeconds: 30
            periodSeconds: 60
            failureThreshold: 3
          resources:
            requests:
              memory: "256Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
              cpu: "500m"
      volumes:
        - name: config
          secret:
            secretName: robinhoodbot-config
        - name: data
          persistentVolumeClaim:
            claimName: robinhoodbot-data
```

> **Warning:** `replicas` must be `1`. Running multiple replicas will cause duplicate buy/sell orders.

### 6. Job — Genetic Optimizer (on-demand)

```yaml
# k8s/optimizer-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: robinhoodbot-optimizer
  namespace: robinhoodbot
spec:
  backoffLimit: 1
  ttlSecondsAfterFinished: 86400  # Auto-cleanup after 24h
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: optimizer
          image: robinhoodbot:latest
          imagePullPolicy: IfNotPresent
          command: ["python3", "genetic_optimizer_intraday.py"]
          args:
            - "--num-stocks=125"
            - "--max-positions=10"
            - "--generations=30"
            - "--population=40"
            - "--real-data"
            - "--resume"
            - "--validate-real"
            - "--days=86"
            - "--train-test-split=0.7"
            - "--optimize-filters"
          env:
            - name: TZ
              value: "America/New_York"
            - name: PYTHONUNBUFFERED
              value: "1"
          volumeMounts:
            - name: config
              mountPath: /app/config.py
              subPath: config.py
              readOnly: true
            - name: data
              mountPath: /app/data
            - name: optimizer-cache
              mountPath: /app/.yfinance_cache
          resources:
            requests:
              memory: "1Gi"
              cpu: "500m"
            limits:
              memory: "4Gi"
              cpu: "2000m"
      volumes:
        - name: config
          secret:
            secretName: robinhoodbot-config
        - name: data
          persistentVolumeClaim:
            claimName: robinhoodbot-data
        - name: optimizer-cache
          persistentVolumeClaim:
            claimName: optimizer-cache
```

Optimizer cache PVC:

```yaml
# k8s/optimizer-cache-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: optimizer-cache
  namespace: robinhoodbot
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 2Gi
```

## Deploying

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Create the secret (config.py)
kubectl create secret generic robinhoodbot-config \
  --from-file=config.py=robinhoodbot/config.py \
  -n robinhoodbot

# Create PVCs
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/optimizer-cache-pvc.yaml

# Deploy the bot
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods -n robinhoodbot
```

## Common Commands

### Bot Management

```bash
# View bot logs (live)
kubectl logs -f deployment/robinhoodbot -n robinhoodbot

# Check pod status
kubectl get pods -n robinhoodbot

# Interactive shell
kubectl exec -it deployment/robinhoodbot -n robinhoodbot -- bash

# Restart the bot (after config changes)
kubectl rollout restart deployment/robinhoodbot -n robinhoodbot

# Stop the bot
kubectl scale deployment robinhoodbot --replicas=0 -n robinhoodbot

# Start the bot
kubectl scale deployment robinhoodbot --replicas=1 -n robinhoodbot
```

### Run the Optimizer

```bash
# Start an optimizer job
kubectl apply -f k8s/optimizer-job.yaml

# Watch optimizer progress
kubectl logs -f job/robinhoodbot-optimizer -n robinhoodbot

# Delete the job when done (or let ttlSecondsAfterFinished handle it)
kubectl delete job robinhoodbot-optimizer -n robinhoodbot
```

To run with different parameters, edit the `args` in `optimizer-job.yaml` or create the job inline:

```bash
kubectl create job optimizer-quick -n robinhoodbot \
  --image=robinhoodbot:latest \
  -- python3 genetic_optimizer_intraday.py \
     --num-stocks=50 --generations=20 --population=30 --real-data
```

### Update Config

```bash
# Update the secret after editing config.py
kubectl create secret generic robinhoodbot-config \
  --from-file=config.py=robinhoodbot/config.py \
  -n robinhoodbot \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart bot to pick up changes
kubectl rollout restart deployment/robinhoodbot -n robinhoodbot
```

### Update the Image

```bash
# Rebuild and load
docker build -t robinhoodbot:latest .

# For minikube:
minikube image load robinhoodbot:latest

# For remote registry:
docker push your-registry.com/robinhoodbot:latest

# Roll out the new image
kubectl rollout restart deployment/robinhoodbot -n robinhoodbot
```

## Persisted Data

| Volume | Contents | Mount Path |
|--------|----------|------------|
| `robinhoodbot-data` PVC | Trade history, logs, buy reasons | `/app/data/` |
| `robinhoodbot-config` Secret | `config.py` (credentials) | `/app/config.py` |
| `optimizer-cache` PVC | Yahoo Finance cached data | `/app/.yfinance_cache` |

## Key Differences from Docker Compose

| Aspect | Docker Compose | Kubernetes |
|--------|---------------|------------|
| Config storage | Bind-mounted file | Secret (encrypted at rest) |
| Data persistence | Bind-mounted files | PersistentVolumeClaim |
| Restart policy | `unless-stopped` | Deployment controller |
| Optimizer | `docker compose run --rm` | Job (one-shot) |
| Health check | Docker healthcheck | liveness/readiness probes |
| Scaling guard | Single container | `replicas: 1` + `Recreate` strategy |

## Monitoring — Prometheus & Grafana for Ray Metrics

The optimizer runs [Ray](https://docs.ray.io/) internally and exports Prometheus metrics on port 8080.
A monitoring stack (Prometheus + Grafana) can be installed to visualize CPU, memory, task, and actor metrics during optimization runs.

### Prerequisites

- [Helm 3](https://helm.sh/docs/intro/install/) installed
- `kubectl` configured for your cluster

### Install

```bash
./k8s/monitoring/install-monitoring.sh
```

This script:
1. Adds the `prometheus-community` Helm repo
2. Creates a `prometheus-system` namespace
3. Installs [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack) (Prometheus + Grafana)
4. Applies a `PodMonitor` to scrape Ray metrics from optimizer pods
5. Downloads and imports the [Ray Grafana dashboard](https://github.com/ray-project/kuberay/tree/master/config/grafana) from the KubeRay repo

### Access Grafana

```bash
kubectl port-forward -n prometheus-system svc/prometheus-grafana 3000:http-web
```
Open http://localhost:3000 — login: `admin` / `prom-operator`

Navigate to **Dashboards → Ray → Default Dashboard** to see:
- Cluster utilization (CPU, memory, disk, network)
- Task and actor counts by state
- Logical CPU/GPU usage
- Object store memory
- Per-component and per-node hardware breakdowns

### Access Prometheus

```bash
kubectl port-forward -n prometheus-system svc/prometheus-kube-prometheus-prometheus 9090:http-web
```
Open http://localhost:9090 — try queries like `ray_node_cpu_utilization` or `ray_tasks`.

### Verify Metrics Are Scraped

1. Open Prometheus at http://localhost:9090/targets
2. Look for a target named `podMonitor/prometheus-system/ray-optimizer-monitor`
3. Status should be **UP** while the optimizer job is running

### Files

| File | Purpose |
|------|---------|
| `k8s/monitoring/install-monitoring.sh` | One-command install script |
| `k8s/monitoring/prometheus-values.yaml` | Helm chart values (Grafana config, Prometheus storage/retention) |
| `k8s/monitoring/ray-pod-monitor.yaml` | PodMonitor CRD — tells Prometheus to scrape Ray metrics |

### Uninstall

```bash
helm uninstall prometheus -n prometheus-system
kubectl delete namespace prometheus-system
```

## Troubleshooting

**Pod in CrashLoopBackOff:**
```bash
kubectl logs deployment/robinhoodbot -n robinhoodbot --previous
kubectl describe pod -l app=robinhoodbot -n robinhoodbot
```

**Config secret not mounting:**
```bash
kubectl get secret robinhoodbot-config -n robinhoodbot -o yaml
```

**PVC stuck in Pending:**
```bash
kubectl describe pvc robinhoodbot-data -n robinhoodbot
# Verify your cluster has a default StorageClass:
kubectl get storageclass
```

**Optimizer job stuck:**
```bash
kubectl describe job robinhoodbot-optimizer -n robinhoodbot
kubectl logs job/robinhoodbot-optimizer -n robinhoodbot
```

**Remove everything:**
```bash
kubectl delete namespace robinhoodbot
```
