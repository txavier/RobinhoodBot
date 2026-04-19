# Optimizer ↔ Bot File Sharing via Garage S3

The bot and optimizer run as separate Kubernetes workloads — the bot is a long-running Deployment while the optimizer is a batch Job backed by a Ray cluster. They each need a small set of files from the other: the bot produces trade history and symbol lists that the optimizer uses for fitness evaluation and validation, and the optimizer produces optimized parameter results that the bot loads on startup. Previously these files were shared through an NFS volume backed by Longhorn, but the RWX (ReadWriteMany) access mode required Longhorn to run an NFS share-manager pod, which introduced significant I/O latency that slowed down both workloads. To eliminate that overhead, NFS was replaced with Garage — a lightweight S3-compatible object store — backed by Longhorn RWO (ReadWriteOnce) block storage. Each pod now gets its own dedicated PVC with fast block-level performance, and the handful of small JSON files that need to cross pod boundaries are exchanged through the S3 bucket asynchronously: a sidecar in the bot pod uploads outputs weekly, and init containers in each workload download what they need at startup.

```mermaid
flowchart TB
    subgraph S3["☁️ Garage S3 Bucket (robinhoodbot)"]
        direction TB
        shared_tests["shared/genetic_optimizer_test_symbols.json"]
        shared_trade["shared/tradehistory-real.json"]
        shared_buy["shared/buy_reasons.json"]
        opt_result["optimizer/genetic_optimization_intraday_result.json"]
        opt_ckpt["optimizer/...checkpoint.json"]
        ray_data[".ray_shared_data/*.pkl"]
    end

    subgraph Bot["🤖 Bot Pod"]
        direction TB
        bot_sidecar["S3 Sidecar\n(weekly upload)"]
        bot_init["S3 Init Container\n(download on start)"]
        bot_main["main.py"]
        bot_pvc[("bot-data PVC\n(Longhorn RWO)")]
    end

    subgraph Optimizer["⚙️ Optimizer Job"]
        direction TB
        opt_startup["S3 Download\n(on startup)"]
        opt_main["genetic_optimizer_intraday.py"]
        opt_save["S3 Upload\n(checkpoint + final)"]
        opt_pvc[("optimizer-data PVC\n(Longhorn RWO)")]
    end

    subgraph Ray["🔄 Ray Workers"]
        ray_worker["_load_shared_data()\nS3 fallback download"]
        ray_scratch[("emptyDir scratch")]
    end

    %% Bot writes → S3
    bot_main --> bot_pvc
    bot_sidecar -- "upload" --> shared_tests
    bot_sidecar -- "upload" --> shared_trade
    bot_sidecar -- "upload" --> shared_buy

    %% Bot reads from S3 on init
    opt_result -- "download" --> bot_init
    bot_init --> bot_pvc

    %% Optimizer reads from S3 on startup
    shared_tests -- "download" --> opt_startup
    shared_trade -- "download" --> opt_startup
    shared_buy -- "download" --> opt_startup
    opt_ckpt -- "download\n(--resume)" --> opt_startup
    opt_startup --> opt_pvc

    %% Optimizer writes to S3
    opt_main --> opt_pvc
    opt_save -- "upload" --> opt_result
    opt_save -- "upload" --> opt_ckpt

    %% Ray shared data
    opt_main -- "upload pickle" --> ray_data
    ray_data -- "download on\ncache miss" --> ray_worker
    ray_worker --> ray_scratch
```

## Flow Summary

| Direction | Files | Trigger |
|-----------|-------|---------|
| **Bot → S3** | `tradehistory-real.json`, `buy_reasons.json`, `genetic_optimizer_test_symbols.json` | Sidecar uploads weekly |
| **S3 → Optimizer** | Same 3 files + checkpoint (for `--resume`) | Init download at job start |
| **Optimizer → S3** | `genetic_optimization_intraday_result.json`, checkpoint | After each generation + final save |
| **S3 → Bot** | `genetic_optimization_intraday_result.json` | Init container on pod start |
| **Optimizer → S3 → Ray Workers** | `.ray_shared_data/*.pkl` (CV folds, market data) | Upload after generation; workers download on cache miss |
