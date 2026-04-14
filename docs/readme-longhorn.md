# RobinhoodBot — Longhorn Distributed Storage

Longhorn provides replicated block storage across the bare-metal K8s cluster. It complements the existing NFS storage class by offering data replication across worker nodes.

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

### 1. Prerequisites — iSCSI & NFS on All Workers

Longhorn requires `open-iscsi` on every node. Apply the official prerequisite DaemonSets:

```bash
kubectl create namespace longhorn-system

kubectl apply -f https://raw.githubusercontent.com/longhorn/longhorn/v1.8.1/deploy/prerequisite/longhorn-iscsi-installation.yaml
kubectl apply -f https://raw.githubusercontent.com/longhorn/longhorn/v1.8.1/deploy/prerequisite/longhorn-nfs-installation.yaml
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
| `autoDeletePodWhenVolumeDetachedUnexpectedly` | true | Restart pods on volume detach — clears stale NFS inode caches |
| `rwxVolumeFastFailover` | true | Lease-based health monitoring for RWX share-manager recovery |
| `persistence.defaultClass` | false | NFS remains the default StorageClass |
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
| `nfs` | nfs-subdir-external-provisioner | Retain | No |
| `longhorn` | driver.longhorn.io | Retain | No |
| `longhorn-static` | driver.longhorn.io | Delete | No |

## Mounting Longhorn on the Dev Machine

To access a Longhorn volume from outside the cluster (e.g., your dev laptop), create a ReadWriteMany PVC and expose the NFS share-manager via NodePort.

### 1. Create a Longhorn RWX PVC

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: longhorn-shared
  namespace: longhorn-system
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: longhorn
  resources:
    requests:
      storage: 10Gi
```

```bash
kubectl apply -f <above>
```

### 2. Trigger the Share-Manager

Longhorn only starts the NFS share-manager pod when a consumer exists. Create a lightweight trigger pod:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: longhorn-mount-trigger
  namespace: longhorn-system
spec:
  containers:
  - name: pause
    image: registry.k8s.io/pause:3.9
    volumeMounts:
    - name: data
      mountPath: /data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: longhorn-shared
```

Wait for the share-manager pod to appear:

```bash
kubectl -n longhorn-system get pods | grep share-manager
```

### 3. Expose via NodePort

Longhorn creates a ClusterIP service for the share-manager. Create a stable NodePort service to expose it:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: longhorn-shared-nfs
  namespace: longhorn-system
spec:
  type: NodePort
  selector:
    longhorn.io/share-manager: <PV-NAME>
  ports:
  - port: 2049
    targetPort: 2049
    nodePort: 30049
    protocol: TCP
    name: nfs
```

Get the PV name from:

```bash
kubectl -n longhorn-system get pvc longhorn-shared -o jsonpath='{.spec.volumeName}'
```

> **Note:** If the Longhorn-managed ClusterIP service already has NodePort 30049 allocated, revert it to ClusterIP first:
> ```bash
> kubectl -n longhorn-system patch svc <PV-NAME> -p '{"spec":{"type":"ClusterIP"}}' --type=merge
> ```

### 4. Mount on the Dev Machine

```bash
sudo mkdir -p /mnt/longhorn
sudo mount -t nfs4 -o port=30049 <CONTROL_PLANE_IP>:/ /mnt/longhorn
```

Any node IP works since NodePort routes cluster-wide. Using the control plane IP is convenient.

### 5. Persist Across Reboots (fstab)

Add to `/etc/fstab`:

```
<CONTROL_PLANE_IP>:/ /mnt/longhorn nfs4 port=30049,_netdev,x-systemd.automount 0 0
```

### Current State (April 8, 2026)

- **PVC:** `longhorn-shared` — 10Gi RWX, bound
- **Share-manager:** Running on a worker node
- **NodePort service:** `longhorn-shared-nfs` on port 30049
- **Trigger pod:** `longhorn-mount-trigger` (pause container)
- **Mount point:** `/mnt/longhorn` on dev machine

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
