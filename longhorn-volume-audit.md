# Longhorn Volume Audit — Cluster-Wide Analysis

**Date:** April 19, 2026  
**Total Longhorn PVs:** 33  
**Reclaim Policy:** All set to `Retain`

---

## `robinhoodbot` namespace (6 PVs)

| PV | PVC | Size | Status | Pod Using It | Data Synopsis | Action |
|---|---|---|---|---|---|---|
| `pvc-3c40651e…` | `robinhoodbot-bot-data` | 2Gi | **Bound** | `robinhoodbot-765c8df98c-hhvfw` | Bot runtime: optimizer results JSON, robinhood auth tokens, test symbols (35MB used) | **KEEP** — active bot storage |
| `pvc-6667abaf…` | `robinhoodbot-optimizer-data` | 3Gi | **Bound** | *None* (Job not running) | Optimizer job working dir: yfinance cache, checkpoints, ray data. Synced to Garage S3. | **KEEP** — used when optimizer job runs |
| `pvc-b9e3b1d4…` | `garage-data` | 5Gi | **Bound** | `garage-0` | Garage S3 object store data dir (`/data`) | **KEEP** — active S3 backend |
| `pvc-cd2881c3…` | `robinhoodbot-data-longhorn` | 3Gi | **Bound** | *None* | Legacy RWX shared vol from before Garage S3 migration. No pod references it in current manifests. | **DELETE** — orphaned, superseded by bot-data + Garage |
| `pvc-eef04d13…` | `garage-data` (old) | 5Gi | **Released** | *None* | Previous garage-data PV from earlier deployment | **DELETE** — released orphan |
| `pvc-baa9a65b…` | `recovery-old-data` | 3Gi | **Released** | *None* | One-time recovery volume, no longer needed | **DELETE** — released orphan |

---

## `robinhoodbot-dev` namespace (9 PVs)

| PV | PVC | Size | Status | Pod Using It | Data Synopsis | Action |
|---|---|---|---|---|---|---|
| `pvc-07728c3a…` | `dev-git-data-pvc` | 10Gi | **Bound** | `dev-sandbox-…rmbk7` | Git workspace mount at `/workspace` (3.7MB used — freshly cloned) | **KEEP** — current dev sandbox |
| `pvc-8e3e0788…` | `dev-home-data-pvc` | 5Gi | **Bound** | `dev-sandbox-…rmbk7` | Agent home dir at `/home/agent` (1.2GB — tools, cache, extensions) | **KEEP** — current dev sandbox |
| `pvc-dc9b314c…` | `dev-settings-pvc` | 2Gi | **Bound** | `dev-sandbox-…rmbk7` | VS Code settings/extensions persistence | **KEEP** — current dev sandbox |
| `pvc-003d30e3…` | `dev-git-data-pvc` (old) | 10Gi | **Released** | *None* | Previous dev workspace iteration | **DELETE** — released orphan |
| `pvc-b4e0758d…` | `dev-git-data-pvc` (old) | 10Gi | **Released** | *None* | Previous dev workspace iteration | **DELETE** — released orphan |
| `pvc-0d94d425…` | `dev-home-data-pvc` (old) | 5Gi | **Released** | *None* | Previous home dir iteration | **DELETE** — released orphan |
| `pvc-879bf542…` | `dev-home-data-pvc` (old) | 5Gi | **Released** | *None* | Previous home dir iteration | **DELETE** — released orphan |
| `pvc-9d423813…` | `dev-home-data-pvc` (old) | 5Gi | **Released** | *None* | Previous home dir iteration | **DELETE** — released orphan |
| `pvc-0f307f4f…` | `dev-settings-pvc` (old) | 2Gi | **Released** | *None* | Previous settings iteration | **DELETE** — released orphan |
| `pvc-d1a9ec04…` | `dev-settings-pvc` (old) | 2Gi | **Released** | *None* | Previous settings iteration | **DELETE** — released orphan |

---

## `fingerprint-dev` namespace (14 PVs)

### Bound (8 PVs — need namespace access to verify pods)

| PV | PVC | Size | Status | Data Synopsis | Action |
|---|---|---|---|---|---|
| `pvc-00d7abb0…` | `client-claude-memory-pvc` | 1Gi | **Bound** | Claude AI memory for client dev sandbox | **REVIEW** — verify pod exists |
| `pvc-37e41b4f…` | `client-copilot-memory-pvc` | 1Gi | **Bound** | Copilot memory for client dev sandbox | **REVIEW** |
| `pvc-609633ee…` | `server-claude-memory-pvc` | 1Gi | **Bound** | Claude AI memory for server dev sandbox | **REVIEW** |
| `pvc-6f67e646…` | `server-copilot-memory-pvc` | 1Gi | **Bound** | Copilot memory for server dev sandbox | **REVIEW** |
| `pvc-793ac950…` | `client-home-data-pvc` | 5Gi | **Bound** | Home dir for client dev sandbox | **REVIEW** |
| `pvc-a8c66364…` | `server-home-data-pvc` | 5Gi | **Bound** | Home dir for server dev sandbox | **REVIEW** |
| `pvc-ae566e3c…` | `server-workspace-pvc` | 10Gi | **Bound** | Workspace for server dev sandbox | **REVIEW** |
| `pvc-b0f16ebc…` | `client-workspace-pvc` | 10Gi | **Bound** | Workspace for client dev sandbox | **REVIEW** |

### Released (6 PVs — all orphaned)

| PV | PVC | Size | Status | Data Synopsis | Action |
|---|---|---|---|---|---|
| `pvc-57b716c9…` | `git-data-pvc` (old) | 10Gi | **Released** | Old dev workspace | **DELETE** |
| `pvc-20f0fa62…` | `home-data-pvc` (old) | 5Gi | **Released** | Old home dir | **DELETE** |
| `pvc-27aacdc5…` | `home-data-pvc` (old) | 5Gi | **Released** | Old home dir | **DELETE** |
| `pvc-6ac973be…` | `copilot-memory-pvc` (old) | 1Gi | **Released** | Old copilot memory | **DELETE** |
| `pvc-cedd7e8e…` | `copilot-memory-pvc` (old) | 1Gi | **Released** | Old copilot memory | **DELETE** |
| `pvc-dbb09c30…` | `claude-memory-pvc` (old) | 1Gi | **Released** | Old claude memory | **DELETE** |
| `pvc-e69f8c34…` | `claude-memory-pvc` (old) | 1Gi | **Released** | Old claude memory | **DELETE** |

---

## `longhorn-system` namespace (1 PV)

| PV | PVC | Size | Status | Pod Using It | Data Synopsis | Action |
|---|---|---|---|---|---|---|
| `pvc-9dab4099…` | `longhorn-shared` | 10Gi | **Bound** | Longhorn share-manager | NFS share-manager backing volume (the old RWX mechanism) | **DELETE** — once all NFS/RWX longhorn use is removed |

---

## `default` namespace (1 PV)

| PV | PVC | Size | Status | Pod Using It | Data Synopsis | Action |
|---|---|---|---|---|---|---|
| `pvc-ec01f905…` | `longhorn-test` | 128Mi | **Released** | *None* | Longhorn test/validation volume | **DELETE** — test artifact |

---

## NFS-Class PVs (also backed by Longhorn share-manager)

| PV | PVC | Namespace | Size | Status | Action |
|---|---|---|---|---|---|
| `pvc-529a9db2…` | `robinhoodbot-data-nfs` | robinhoodbot | 3Gi | **Released** | **DELETE** — legacy NFS share |
| `pvc-1c2810a6…` | `copilot-memory-pvc` | fingerprint-dev | 1Gi | **Released** | **DELETE** |
| `pvc-361acaa0…` | `git-data-pvc` | fingerprint-dev | 10Gi | **Released** | **DELETE** |
| `pvc-452102a6…` | `git-data-pvc` | fingerprint-dev | 10Gi | **Released** | **DELETE** |
| `pvc-47bfb470…` | `git-data-pvc` | fingerprint-dev | 10Gi | **Released** | **DELETE** |
| `pvc-497934cf…` | `claude-memory-pvc` | fingerprint-dev | 1Gi | **Released** | **DELETE** |
| `pvc-ad166a50…` | `git-data-pvc` | fingerprint-dev | 10Gi | **Released** | **DELETE** |
| `pvc-c0738529…` | `home-data-pvc` | fingerprint-dev | 5Gi | **Released** | **DELETE** |
| `pvc-cef0fea3…` | `git-data-pvc` | fingerprint-dev | 10Gi | **Released** | **DELETE** |
| `pvc-22cde9fc…` | `bitnet-dev-pvc` | bitnet-dev | 20Gi | **Bound** | **REVIEW** — active? |
| `pvc-38a8fb08…` | `kube-prometheus-stack-grafana` | monitoring | 1Gi | **Bound** | **KEEP** — Grafana data |
| `pvc-fe793992…` | `prometheus-…-db-…` | monitoring | 5Gi | **Bound** | **KEEP** — Prometheus TSDB |

---

## Summary

| Category | Count | Total Size | Action |
|---|---|---|---|
| **Active Longhorn (keep)** | 11 | ~55Gi | Keep — in use by running pods/jobs |
| **Released/Orphaned Longhorn** | 15 | ~78Gi | **Delete** — all are old iterations with Retain policy |
| **Bound but unused (`robinhoodbot-data-longhorn`)** | 1 | 3Gi | **Delete** — legacy RWX vol, superseded by Garage S3 |
| **Longhorn NFS share-manager (`longhorn-shared`)** | 1 | 10Gi | **Delete** — after confirming no remaining NFS dependents |
| **fingerprint-dev Bound (need access to verify)** | 8 | ~34Gi | **Review** — can't verify pod bindings without namespace access |
| **NFS Released PVs** | 9 | ~60Gi | **Delete** — all orphaned NFS shares |
| **NFS Bound PVs (monitoring/bitnet)** | 3 | ~26Gi | **Review/Keep** |

### Cleanup Totals

- **~26 PVs safe to delete immediately** = **~151Gi** of Longhorn capacity reclaimed
- **8 PVs need fingerprint-dev namespace access** to verify before deleting
- All Released PVs have `Retain` reclaim policy — deleting the PV object alone won't remove the Longhorn volume
- Must also delete from the **Longhorn UI** or `volumes.longhorn.io` CRD after removing the PV/PVC

### K8s Manifests to Remove

- [`k8s/robinhoodbot-data-longhorn-pvc.yaml`](k8s/robinhoodbot-data-longhorn-pvc.yaml) — legacy RWX PVC, no longer referenced by any deployment
- [`k8s/longhorn-storageclass.yaml`](k8s/longhorn-storageclass.yaml) — review if still needed after migration
- [`k8s/longhorn-values.yaml`](k8s/longhorn-values.yaml) — review if Longhorn itself is being removed
