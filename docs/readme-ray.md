# Ray Cluster — Distributed Genetic Optimizer

The genetic optimizer runs across multiple Kubernetes nodes using [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/index.html) to distribute workloads. This guide covers the architecture, setup, and operation.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Kubernetes Cluster                                              │
│                                                                  │
│  ┌────────────────────┐   ┌────────────────────────────────────┐ │
│  │  Optimizer Job      │   │  Ray Head (num-cpus=0)             │ │
│  │  (driver process)   │──▶│  Coordination, GCS, Dashboard      │ │
│  │  RAY_ADDRESS=ray:// │   │  Port 6379 (GCS)                   │ │
│  │  ...head-svc:10001  │   │  Port 8265 (Dashboard)             │ │
│  └────────────────────┘   │  Port 10001 (Client)               │ │
│                            └────────────────────────────────────┘ │
│                                          │                        │
│          ┌───────────────┬───────────────┼───────────────┐        │
│          ▼               ▼               ▼               ▼        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────┐ │
│  │ Ray Worker 1 │ │ Ray Worker 2 │ │ Ray Worker 3 │ │ Worker 4 │ │
│  │ 6 CPUs, 4Gi  │ │ 6 CPUs, 4Gi  │ │ 6 CPUs, 4Gi  │ │ 6C, 4Gi │ │
│  │ Node A       │ │ Node B       │ │ Node C       │ │ Node D   │ │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────┘ │
│                                                                  │
│  Total: 24 CPUs across 4 worker nodes                            │
└──────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- The Ray **head** has `num-cpus: 0` — it only handles coordination (GCS, dashboard, scheduling), no gene evaluations.
- The **optimizer job** is a separate K8s Job that connects to the Ray cluster as a client via `RAY_ADDRESS`. It runs the driver process (downloads data, manages generations) but offloads all `evaluate_candidate` calls to workers via `ray.remote`.
- Workers use **pod anti-affinity** to spread across physical nodes.
- `RAY_DISABLE_MEMORY_MONITOR=1` is set on all pods to prevent OOM-kills from Ray's internal memory monitor.

## Prerequisites

1. **Kubernetes cluster** with multiple worker nodes
2. **Local Docker registry** running on the control plane (see [readme-kubernetes.md](readme-kubernetes.md))
3. **KubeRay operator** installed (see [Installation](#kuberay-operator-installation) below)
4. **StorageClass** available (e.g., Rancher local-path-provisioner)

## KubeRay Operator Installation

Install the KubeRay operator via Helm (one-time setup):

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

helm install kuberay-operator kuberay/kuberay-operator \
  --namespace kuberay-system \
  --create-namespace \
  --version 1.3.0
```

Verify:

```bash
kubectl get pods -n kuberay-system
# Should show kuberay-operator pod Running
```

## Manifests

| File | Purpose |
|------|---------|
| `k8s/ray-cluster.yaml` | RayCluster CR — 1 head + 4 workers |
| `k8s/optimizer-job.yaml` | K8s Job — optimizer driver process |
| `k8s/optimizer-data-pvc.yaml` | PVC for results & checkpoints (1Gi) |
| `k8s/optimizer-cache-pvc.yaml` | PVC for yfinance cache (2Gi) |

## Usage

### Deploy the optimizer (recommended)

```bash
./deploy.sh optimizer
```

This command:
1. Creates PVCs (`optimizer-data`, `optimizer-cache`)
2. Deploys the RayCluster (head + 4 workers)
3. Waits for the head pod to be ready
4. Deletes any previous optimizer job
5. Launches the optimizer job

### Monitor progress

```bash
# Tail optimizer logs
./deploy.sh logs-optimizer

# Ray cluster status (CPU/memory usage, connected nodes)
./deploy.sh ray-status

# General cluster status
./deploy.sh status
```

### Stop the optimizer

```bash
./deploy.sh optimizer-stop
```

This deletes both the optimizer job **and** the Ray cluster to free resources.

### Access the Ray Dashboard

The Ray head exposes a dashboard on port 8265. Port-forward to access it:

```bash
# Find the head pod
kubectl get pods -n robinhoodbot -l ray-role=head

# Port-forward
kubectl port-forward <head-pod-name> 8265:8265 -n robinhoodbot
```

Then open http://localhost:8265 in your browser.

## Resource Allocation

| Component | CPU (req/limit) | Memory (req/limit) | Count |
|-----------|----------------|---------------------|-------|
| Ray Head | 1 / 2 | 1Gi / 2Gi | 1 |
| Ray Worker | 6 / 7 | 2Gi / 4Gi | 4 |
| Optimizer Job | 500m / 1000m | 1Gi / 2Gi | 1 |
| **Total** | **26.5 / 32** | **11.5Gi / 20Gi** | **6 pods** |

The head advertises `num-cpus: 0` to Ray so no tasks are scheduled on it. Each worker advertises `num-cpus: 6` (slightly below its 7 CPU limit) to leave headroom.

### Adjusting worker count or CPUs

Edit `k8s/ray-cluster.yaml`:

```yaml
workerGroupSpecs:
  - groupName: workers
    replicas: 4          # ← Change number of workers
    rayStartParams:
      num-cpus: "6"      # ← CPUs per worker (advertised to Ray)
    template:
      spec:
        containers:
          - resources:
              requests:
                cpu: "6"   # ← Should match num-cpus
              limits:
                cpu: "7"   # ← num-cpus + 1 for overhead
```

## How It Works

1. **KubeRay operator** watches for `RayCluster` CRDs and creates the head + worker pods.
2. **Ray head** starts the GCS (Global Control Store) and registers workers as they connect.
3. **Optimizer job** starts and connects to the head via `RAY_ADDRESS=ray://optimizer-ray-head-svc:10001`.
4. The optimizer downloads stock data locally (in the driver), then distributes gene evaluations to workers using `@ray.remote` decorated functions.
5. Each generation's candidates are evaluated in parallel across all 24 CPUs.
6. Results and checkpoints are written to the `optimizer-data` PVC (mounted on the head pod and optimizer job).

### Code changes for cluster mode

The optimizer (`genetic_optimizer_intraday.py`) detects the `RAY_ADDRESS` environment variable:

```python
ray_address = os.environ.get("RAY_ADDRESS", "")
connecting_to_cluster = bool(ray_address)

init_kwargs = {"logging_level": logging.WARNING}
if not connecting_to_cluster:
    # Local mode — set system config and metrics port
    init_kwargs["_system_config"] = {"worker_register_timeout_seconds": 60}
    init_kwargs["_metrics_export_port"] = 8080
else:
    log(f"Connecting to existing Ray cluster at: {ray_address}")

ray.init(**init_kwargs)
```

When `RAY_ADDRESS` is set, `ray.init()` connects as a client instead of starting a local cluster. The `_system_config` and `_metrics_export_port` parameters are skipped because they're only valid for cluster starters, not clients.

## Troubleshooting

### Workers stuck in `Init:0/1`

Workers are pulling the Docker image. Check pull progress:

```bash
kubectl describe pod <worker-pod> -n robinhoodbot
```

If image pull fails, verify the local registry is accessible from worker nodes:

```bash
# From a worker node
curl http://192.168.87.35:5000/v2/_catalog
```

### Optimizer can't connect to Ray head

Check the head pod is running and the service exists:

```bash
kubectl get svc -n robinhoodbot | grep ray
kubectl logs <head-pod> -n robinhoodbot | head -20
```

### Ray reports fewer CPUs than expected

Check which workers have connected:

```bash
./deploy.sh ray-status
# or
kubectl exec <head-pod> -n robinhoodbot -- ray status
```

Workers may still be starting. Wait for all worker pods to be `Running` and `Ready`.

### OOM kills

If workers are killed with `OOMKilled`:
1. Increase worker memory limits in `ray-cluster.yaml`
2. Reduce `--num-stocks` or `--population` in `optimizer-job.yaml`
3. The `/dev/shm` `emptyDir` (used for Ray object store) has a `sizeLimit` of 1Gi per worker — increase if needed

### Resume after failure

The optimizer supports `--resume` and writes checkpoints to the `optimizer-data` PVC. To restart from the last checkpoint:

```bash
./deploy.sh optimizer-stop
./deploy.sh optimizer
```

The new job will pick up from the latest checkpoint file.
