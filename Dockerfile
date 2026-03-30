# ── Base image ────────────────────────────────────────────────────────────────
# python:3.11-slim is a minimal Debian image with Python pre-installed.
# "slim" keeps the image small by stripping documentation and non-essential files.
FROM python:3.11-slim

# libgomp1 is required by LightGBM (OpenMP parallelism)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer.
# If requirements.txt hasn't changed, pip install is skipped on rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY . .

# ── Default command ───────────────────────────────────────────────────────────
# Overridden per-service in docker-compose.yml.
# 0.0.0.0 makes the server reachable from outside the container.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
