# Longhorn Upgrade Benefits: v1.8.1 → v1.11.1

Upgrade path: `v1.8.1 → v1.8.2 → v1.9.2 → v1.10.2 → v1.11.1`  
Completed: April 2026

---

## Reliability & Data Recovery

| Benefit | Introduced | What It Means for Your Cluster |
|---|---|---|
| **Offline Replica Rebuilding** | v1.9 | Degraded volumes auto-rebuild replicas even while detached — no manual intervention needed |
| **Delta Replica Rebuilding** | v1.10 | Rebuilds only transfer changed blocks instead of full data — dramatically faster recovery after a node comes back |
| **Fixed: Memory leak in instance-manager** | v1.11.1 | Critical fix — proxy connection leak caused high RAM use; directly relevant given your tight-memory nodes |
| **Fixed: Failed replicas accumulate during engine upgrade** | v1.11.1 | Prevents replica pile-up after rolling upgrades like the ones we just ran |
| **Fixed: Block disk never schedulable after re-adding** | v1.11.1 | Fixes a real operational annoyance when disks are removed/re-added |
| **Fixed: Storage double-counting causing scheduling failures** | v1.11.1 | Incorrect space math no longer causes healthy nodes to reject new replicas |
| **Backoff retry for instance-manager pod re-creation** | v1.10 | Under resource pressure, Longhorn backs off gracefully instead of hammering the API |

---

## Backup & Restore

| Benefit | Introduced | What It Means for Your Cluster |
|---|---|---|
| **Fixed: S3/GCS backup fails at 95%** | v1.11.1 | Resolves backup incompatibility with S3-compatible stores (relevant to your Garage S3 setup) |
| **Recurring System Backups** | v1.9 | Can now schedule automated full Longhorn system backups (settings, volumes, recurring jobs) |
| **Configurable Backup Block Size** | v1.10 | Tune backup efficiency per volume based on workload I/O patterns |
| **Snapshot auto-delete after backup** | v1.9 | Option to clean up the source snapshot automatically when a backup completes |

---

## Scheduling & Storage Management

| Benefit | Introduced | What It Means for Your Cluster |
|---|---|---|
| **CSIStorageCapacity support** | v1.10 | Kubernetes checks actual node disk space before scheduling pods — reduces `pending` PVCs from scheduling to full nodes |
| **CSI topology-aware PV nodeAffinity** | v1.11.1 | Better replica placement control aligned with K8s topology labels |
| **`best-effort` locality: schedule ≥1 replica locally** | v1.10 | Improves data locality — at least one replica will be on the workload's node when possible |
| **Orphaned Instance Deletion** | v1.9 | Auto-cleanup of leftover engines/replicas from failed operations — reduces wasted disk space |
| **Improved disk space un-schedulable messages** | v1.10 | Clearer UI/logs when a volume can't schedule due to disk space |

---

## Observability & Operations

| Benefit | Introduced | What It Means for Your Cluster |
|---|---|---|
| **New Prometheus metrics: Replica, Engine, Rebuild** | v1.9 | Your Prometheus/Grafana stack now gets per-replica and rebuild-in-progress metrics |
| **Replica Rebuild QoS** | v1.10 | Set bandwidth limits on rebuilds to prevent saturating node storage throughput |
| **Volume Attachment Summary in UI** | v1.10 | Each volume page shows all attachment tickets — easier to diagnose stuck volumes |
| **Nanosecond-precision log timestamps** | v1.10 | Better ordering when debugging rapid sequences of events |
| **Instance manager log collection improvements** | v1.10 | Support bundle and log collection now captures IM logs properly |
| **longhornctl `node-selector` option** | v1.10 | Run preflight checks / ops against specific nodes only |

---

## API & Stability

| Benefit | Introduced | What It Means for Your Cluster |
|---|---|---|
| **IPv6 support (V1)** | v1.10 | Your cluster's IPv6 nodes can now host Longhorn volumes correctly |
| **CRD fully migrated to `v1beta2`** | v1.9/v1.10 | `v1beta1` API removed — cleaner, forward-compatible resource definitions |
| **Standardized image registry override** | v1.10 | Easier to point Longhorn at your local registry (192.168.87.35:5000) for air-gapped pulls |
| **Consolidated V1/V2 settings format** | v1.10 | Single settings UI covers both engines; per-engine JSON overrides supported |
| **Validation moved to webhook** | v1.10 | Invalid operations caught immediately at admission rather than silently failing later |

---

## Key Highlights for This Cluster

The most immediately impactful for this setup:

- **Memory leak fix** (v1.11.1) — prevents gradual RAM exhaustion on instance-manager pods
- **S3 backup fix for Garage** (v1.11.1) — large backups were silently failing at 95% completion before this
- **Offline replica rebuilding** (v1.9) — volumes self-heal during node flaps without manual attach/detach cycles
- **Delta rebuilds** (v1.10) — next time a node like `k8s-node-394cdd` flaps, replicas rebuild far faster with far less network I/O

---

## Cluster State at Upgrade Completion

| Node | Status | Notes |
|---|---|---|
| `theo-thinkpad-l430` | Control plane | Healthy |
| `k8s-node-d1792a` | Ready, kured reboot pending | Reboot scheduled since 2026-04-02 |
| `k8s-node-d7e31b` | Ready, kured reboot pending | Reboot scheduled since 2026-04-02 |
| `k8s-node-12dc7f` | Ready | 14G root disk — needs expansion or lower image-gc threshold |
| `k8s-node-394cdd` | Ready (unstable) | 29 NotReady events in 2 days — physical investigation warranted |
| `k8s-node-352ec6` | Permanently cordoned | Offline |
