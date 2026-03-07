# ============================================================================
# RobinhoodBot Docker Container
# ============================================================================
# Runs the trading bot (main.py) in a loop via run.sh.
# Trade history and logs are persisted via Docker volumes.
#
# Build:  docker build -t robinhoodbot .
# Run:    docker compose up -d
# Logs:   docker compose logs -f bot
# Shell:  docker compose exec bot bash
# ============================================================================

FROM python:3.11-slim

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=America/New_York

# Install minimal OS deps (ping for connectivity checks in run.sh, tzdata for timezone)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        iputils-ping \
        tzdata \
        procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY robinhoodbot/requirements.txt .
# pyotp is imported but not in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt pyotp

# Copy application code (config.py excluded via .dockerignore — mounted at runtime)
COPY robinhoodbot/ .

# Copy sample config as fallback (container won't start without a real config mount)
COPY robinhoodbot/config.py.sample /app/config.py.sample

# Make shell scripts executable
RUN chmod +x run.sh run_optimizer.sh

# The bot loops internally via run.sh (runs main.py every 7 minutes)
CMD ["bash", "run.sh"]
