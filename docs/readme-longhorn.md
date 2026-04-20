# RobinhoodBot — Longhorn Distributed Storage

Longhorn provides replicated block storage across the bare-metal K8s cluster. All persistent volumes (bot data, optimizer data, Garage S3, dev sandboxes, monitoring) use Longhorn RWO (ReadWriteOnce) block storage.

## Cluster Context

| Node | IP | Disk | RAM | Role |
|---|---|---|---|---|
| control-plane | 192.168.x.35 | 227G | 8GB | Control plane (excluded from Longhorn storage) |
| k8s-node-1 | 192.168.x.70 | 44G | 16GB | Worker |
| k8s-node-2 | 192.168.x.71 | 44G | 8GB | Worker |
| k8s-node-3 | 192.168.x.69 | 44G | 8GB | Worker |
| k8s-node-4 | 192.168.x.68 | 44G | 8GB | Worker |

WiFi-connected cluster — timeouts and resource limits are tuned accordingly.

## Installation

### 1. Prerequisites — iSCSI on All Workers

Longhorn requires `open-iscsi` on every node. Apply the official prerequisite DaemonSet:

```bash
kubectl create namespace longhorn-system

kubectl apply -f https://raw.githubusercontent.com/longhorn/longhorn/v1.8.1/deploy/prerequisite/longhorn-iscsi-installation.yaml
```

Wait for 4/4 pods on each DaemonSet (one per worker).

### 2. Helm Values

Values file: `k8s/longhorn-values.yaml` — tuned for WiFi + low-resource nodes:

| Setting | Value | Reason |
|---|---|---|
| `defaultReplicaCount` | 2 | Balance between redundancy and 44G disk space |
| `storageMinimalAvailablePercentage` | 25 | Reserve disk for OS |
| `concurrentReplicaRebuildPerNodeLimit` | 1 | Avoid saturating WiFi |
| `nodeDownPodDeletionPolicy` | delete-both-statefulset-and-deployment-pod | Auto-recover after node failures |
| `autoDeletePodWhenVolumeDetachedUnexpectedly` | true | Restart pods on volume detach |
| `rwxVolumeFastFailover` | true | Lease-based health monitoring for RWX recovery |
| `persistence.defaultClass` | false | Longhorn is opt-in (not default StorageClass) |
| `persistence.reclaimPolicy` | Retain | Don't delete data on PVC removal |
| Longhorn UI | NodePort 30081 | Access at `http://<any-node-ip>:30081` |
| Manager resources | 128Mi–512Mi | Lightweight for 8GB nodes |
| Driver resources | 64Mi–256Mi | Lightweight for 8GB nodes |

### 3. Install via Helm

```bash
# Sync values to control plane
scp k8s/longhorn-values.yaml ${REMOTE_USER}@${REMOTE_HOST}:~/dev/RobinhoodBot/k8s/

# Install (run on control plane or via SSH)
helm repo add longhorn https://charts.longhorn.io
helm repo update
helm install longhorn longhorn/longhorn \
  --namespace longhorn-system \
  --version 1.8.1 \
  --values k8s/longhorn-values.yaml
```

Image pulls over WiFi may take a few minutes. Some pods may show `ImagePullBackOff` temporarily — they self-resolve on retry. Expect 38 pods total when fully running.

### 4. Exclude Control Plane from Data Storage

```bash
kubectl label node <control-plane-node> node.longhorn.io/create-default-disk=false
```

### 5. Verify

```bash
# All 38 pods running
kubectl -n longhorn-system get pods --no-headers | awk '{print $3}' | sort | uniq -c

# Storage classes (longhorn should be present, not default)
kubectl get sc
```

Expected storage classes:

| Class | Provisioner | Reclaim | Default |
|---|---|---|---|
| `local-path` | rancher.io/local-path | Delete | Yes |
| `longhorn` | driver.longhorn.io | Retain | No |
| `longhorn-static` | driver.longhorn.io | Delete | No |

## Smoke Test

```bash
# Create test PVC
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: longhorn-test
spec:
  accessModes: [ReadWriteOnce]
  storageClassName: longhorn
  resources:
    requests:
      storage: 128Mi
---
apiVersion: v1
kind: Pod
metadata:
  name: longhorn-test-pod
spec:
  containers:
  - name: test
    image: busybox
    command: ["sh", "-c", "echo 'Longhorn works!' > /data/test.txt && cat /data/test.txt && sleep 3600"]
    volumeMounts:
    - name: data
      mountPath: /data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: longhorn-test
EOF

# Verify
kubectl logs longhorn-test-pod

# Cleanup
kubectl delete pod longhorn-test-pod
kubectl delete pvc longhorn-test
```
