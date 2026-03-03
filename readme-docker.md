# RobinhoodBot — Docker Setup

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (v20.10+)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2+, included with Docker Desktop)

## Quick Start

```bash
# Build the image
docker compose build

# Start the bot (runs in background)
docker compose up -d
```

The bot will loop via `run.sh`, executing `main.py` every 7 minutes — the same behavior as running it directly on the host.

## Configuration

`config.py` is **bind-mounted** into the container, not baked into the image. This means:

- Your credentials never exist in the Docker image (safe to push to a registry)
- You can edit `robinhoodbot/config.py` on the host and restart the container — no rebuild needed
- The sample config (`config.py.sample`) is included in the image as a reference

## Common Commands

### Bot Management

```bash
# Start the bot
docker compose up -d

# View live logs
docker compose logs -f bot

# Stop the bot
docker compose down

# Restart after config changes
docker compose restart bot

# Interactive shell inside the container
docker compose exec bot bash
```

### Health Check

The bot container includes a health check that verifies `main.py` is running:

```bash
docker inspect --format='{{.State.Health.Status}}' robinhoodbot
```

### Rebuild After Code Changes

Only needed when you modify Python source files (not `config.py`):

```bash
docker compose build
docker compose up -d
```

## Genetic Optimizer

The optimizer runs as a separate on-demand service. It is **not** started by `docker compose up`.

```bash
# Basic optimizer run
docker compose run --rm optimizer \
  --num-stocks 125 --generations 20 --population 30 --real-data

# With train-test split (86 days so training gets 60 days at 70/30)
docker compose run --rm optimizer \
  --num-stocks 125 --generations 20 --population 30 \
  --real-data --days 86 --train-test-split 0.7

# Resume a previous run
docker compose run --rm optimizer \
  --num-stocks 125 --generations 20 --population 30 \
  --real-data --resume

# With filter optimization and real-data validation
docker compose run --rm optimizer \
  --num-stocks 125 --generations 30 --population 40 \
  --real-data --resume --validate-real --optimize-filters
```

Yahoo Finance data is cached in a Docker named volume (`optimizer-cache`) so it persists across runs without re-downloading.

## Persisted Data

These files are bind-mounted from `robinhoodbot/` on the host and persist across container restarts:

| File | Purpose |
|------|---------|
| `config.py` | Bot configuration and credentials |
| `tradehistory-real.json` | Real trade history |
| `tradehistory.json` | Trade history |
| `log.json` | Trade log |
| `console_log.json` | Console output log |
| `buy_reasons.json` | Buy decision reasons |
| `genetic_optimization_intraday_result.json` | Optimizer best result |
| `genetic_optimization_intraday_result.checkpoint.json` | Optimizer checkpoint |

## What's NOT in the Image

The `.dockerignore` ensures these are excluded from the Docker image:

- `config.py` — credentials (mounted at runtime)
- All `.json` log/trade files — data (mounted at runtime)
- AI/optimizer artifacts — `ai_changelog.json`, `ai_prompts.md`, etc.
- `.venv`, `__pycache__`, `.git`

## Timezone

The container defaults to `America/New_York` (US Eastern) for market hours. Change it in `docker-compose.yml` if needed:

```yaml
environment:
  - TZ=America/Chicago  # Central time
```

## Troubleshooting

**Container keeps restarting:**
```bash
docker compose logs bot  # Check for errors
```

**Exit code 75 (Robinhood password required):**
The bot will stop its loop if the Robinhood session expires and the network is up. Re-authenticate:
```bash
docker compose exec bot python3 main.py
```

**Check container status:**
```bash
docker compose ps
```

**Remove everything (containers, volumes, images):**
```bash
docker compose down -v --rmi all
```
