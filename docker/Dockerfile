# syntax=docker/dockerfile:1.7
# Heart Disease prediction API — Flask served by gunicorn.
#
# Build:  docker build -f docker/Dockerfile -t heart-api:latest .
# Run:    docker run --rm -p 5000:5000 heart-api:latest
# Health: curl http://localhost:5000/health
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1. System packages required at runtime by numpy / scipy / scikit-learn
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Python deps — copied first so this layer is cached across code changes
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 3. Application code + the trained model artefact
COPY src/ ./src/
COPY models/heart_pipeline.joblib ./models/heart_pipeline.joblib

# 4. Runtime config
ENV MODEL_PATH=/app/models/heart_pipeline.joblib \
    HOST=0.0.0.0 \
    PORT=5000

EXPOSE 5000

# Non-root for least privilege
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:5000/health || exit 1

# Two workers is enough for the assignment workload; tune via env if needed.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", \
     "--timeout", "60", "--access-logfile", "-", "src.api.app:app"]
