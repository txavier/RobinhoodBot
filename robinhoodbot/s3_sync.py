"""
S3 Sync Utility for RobinhoodBot

Provides file upload/download to a Garage (S3-compatible) bucket for sharing
data between the bot and optimizer pods without NFS.

Configuration via environment variables:
    S3_ENDPOINT        - Garage S3 endpoint (e.g. http://garage.garage.svc.cluster.local:3900)
    S3_ACCESS_KEY_ID   - Garage access key
    S3_SECRET_ACCESS_KEY - Garage secret key
    S3_BUCKET          - Bucket name (default: robinhoodbot)
    S3_REGION          - Region (default: garage)

Usage as CLI:
    python3 s3_sync.py upload <local_path> <s3_key>
    python3 s3_sync.py download <s3_key> <local_path>
    python3 s3_sync.py bot-sidecar          # Loop: upload bot outputs every 7 days (default)
    python3 s3_sync.py download-bot-inputs  # One-shot: download shared files for bot
    python3 s3_sync.py download-optimizer-inputs  # One-shot: download files for optimizer

Usage as library:
    from s3_sync import upload_file, download_file, s3_enabled
"""

import os
import sys
import time
import logging

log = logging.getLogger(__name__)

# ── S3 client (lazy singleton) ──────────────────────────────────────────────

_s3_client = None

def _get_client():
    global _s3_client
    if _s3_client is None:
        import boto3
        from botocore.config import Config
        _s3_client = boto3.client(
            's3',
            endpoint_url=os.environ['S3_ENDPOINT'],
            aws_access_key_id=os.environ['S3_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['S3_SECRET_ACCESS_KEY'],
            region_name=os.environ.get('S3_REGION', 'garage'),
            config=Config(
                signature_version='s3v4',
                retries={'max_attempts': 3, 'mode': 'standard'},
                connect_timeout=10,
                read_timeout=30,
            ),
        )
    return _s3_client

def _bucket():
    return os.environ.get('S3_BUCKET', 'robinhoodbot')

def s3_enabled():
    """Return True if S3 sync is configured."""
    return bool(os.environ.get('S3_ENDPOINT'))


# ── Core upload / download ──────────────────────────────────────────────────

def upload_file(local_path, s3_key):
    """Upload a local file to S3. Silently skips if file doesn't exist."""
    if not os.path.isfile(local_path):
        return False
    try:
        _get_client().upload_file(local_path, _bucket(), s3_key)
        log.info("S3 upload: %s -> s3://%s/%s", local_path, _bucket(), s3_key)
        return True
    except Exception as e:
        log.warning("S3 upload failed for %s: %s", local_path, e)
        return False


def download_file(s3_key, local_path):
    """Download from S3 to a local path. Returns True on success, False if key doesn't exist."""
    try:
        os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
        _get_client().download_file(_bucket(), s3_key, local_path)
        log.info("S3 download: s3://%s/%s -> %s", _bucket(), s3_key, local_path)
        return True
    except _get_client().exceptions.NoSuchKey:
        log.debug("S3 key not found: %s (skipping)", s3_key)
        return False
    except Exception as e:
        # ClientError with 404 status also means key doesn't exist
        if hasattr(e, 'response') and e.response.get('Error', {}).get('Code') == '404':
            log.debug("S3 key not found: %s (skipping)", s3_key)
            return False
        log.warning("S3 download failed for %s: %s", s3_key, e)
        return False


def upload_directory(local_dir, s3_prefix):
    """Upload all files in a local directory to S3 under the given prefix."""
    if not os.path.isdir(local_dir):
        return
    for filename in os.listdir(local_dir):
        filepath = os.path.join(local_dir, filename)
        if os.path.isfile(filepath):
            upload_file(filepath, f"{s3_prefix}/{filename}")


# ── High-level sync operations ──────────────────────────────────────────────

# Files the bot writes that the optimizer needs
BOT_OUTPUT_FILES = [
    # (local_path_relative_to_data_dir, s3_key)
    ("genetic_optimizer_test_symbols.json", "shared/genetic_optimizer_test_symbols.json"),
]

# Files under the bot's log directory
BOT_LOG_FILES = [
    ("tradehistory-real.json", "shared/tradehistory-real.json"),
    ("buy_reasons.json", "shared/buy_reasons.json"),
]

# Files the optimizer writes that the user/bot might want
OPTIMIZER_OUTPUT_FILES = [
    ("genetic_optimization_intraday_result.json", "optimizer/genetic_optimization_intraday_result.json"),
    ("genetic_optimization_intraday_result.checkpoint.json", "optimizer/genetic_optimization_intraday_result.checkpoint.json"),
]


def sync_bot_outputs_to_s3():
    """Upload bot output files to S3 (called periodically by sidecar)."""
    data_dir = os.environ.get('DATA_DIR', '/app/data')
    log_dir = os.environ.get('LOG_DIR', '/app/logs')

    for local_rel, s3_key in BOT_OUTPUT_FILES:
        upload_file(os.path.join(data_dir, local_rel), s3_key)
    for local_rel, s3_key in BOT_LOG_FILES:
        upload_file(os.path.join(log_dir, local_rel), s3_key)


def download_bot_inputs_from_s3():
    """Download files the bot needs from S3 (init container)."""
    data_dir = os.environ.get('DATA_DIR', '/app/data')
    # The bot might want the latest optimizer result for reference
    download_file("optimizer/genetic_optimization_intraday_result.json",
                  os.path.join(data_dir, "genetic_optimization_intraday_result.json"))


def download_optimizer_inputs_from_s3():
    """Download files the optimizer needs from S3 (init container)."""
    data_dir = os.environ.get('DATA_DIR', '/app/data')
    log_dir = os.environ.get('LOG_DIR', '/app/logs')

    for local_rel, s3_key in BOT_OUTPUT_FILES:
        download_file(s3_key, os.path.join(data_dir, local_rel))
    for local_rel, s3_key in BOT_LOG_FILES:
        download_file(s3_key, os.path.join(log_dir, local_rel))

    # Download checkpoint for --resume
    for local_rel, s3_key in OPTIMIZER_OUTPUT_FILES:
        download_file(s3_key, os.path.join(data_dir, local_rel))


def upload_optimizer_results_to_s3():
    """Upload optimizer outputs to S3 (called after optimization completes)."""
    data_dir = os.environ.get('DATA_DIR', '/app/data')
    for local_rel, s3_key in OPTIMIZER_OUTPUT_FILES:
        upload_file(os.path.join(data_dir, local_rel), s3_key)


def upload_ray_shared_data_to_s3(data_dir):
    """Upload .ray_shared_data/ pickle files to S3 for Ray workers."""
    upload_directory(data_dir, ".ray_shared_data")


def download_ray_shared_file_from_s3(s3_key, local_path):
    """Download a single .ray_shared_data file from S3 (called by Ray workers)."""
    return download_file(s3_key, local_path)


# ── Sidecar loop ────────────────────────────────────────────────────────────

def bot_sidecar_loop(interval_seconds=604800):
    """Run in a sidecar container: periodically sync bot outputs to S3."""
    log.info("Bot S3 sidecar started (interval=%ds)", interval_seconds)
    while True:
        try:
            sync_bot_outputs_to_s3()
        except Exception as e:
            log.warning("Sidecar sync error: %s", e)
        time.sleep(interval_seconds)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [s3_sync] %(message)s')

    if len(sys.argv) < 2:
        print("Usage: python3 s3_sync.py <command> [args...]")
        print("Commands: upload, download, bot-sidecar, download-bot-inputs, download-optimizer-inputs, upload-optimizer-results")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'upload' and len(sys.argv) == 4:
        upload_file(sys.argv[2], sys.argv[3])
    elif cmd == 'download' and len(sys.argv) == 4:
        download_file(sys.argv[2], sys.argv[3])
    elif cmd == 'bot-sidecar':
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 300
        bot_sidecar_loop(interval)
    elif cmd == 'download-bot-inputs':
        download_bot_inputs_from_s3()
    elif cmd == 'download-optimizer-inputs':
        download_optimizer_inputs_from_s3()
    elif cmd == 'upload-optimizer-results':
        upload_optimizer_results_to_s3()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == '__main__':
    main()
