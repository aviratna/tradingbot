###############################################################################
# Trading Bot — Full App Dockerfile
# Serves FastAPI + Quant Engine + OSINT Intelligence Layer
# Target: GCP e2-small / e2-medium (Linux x86-64)
###############################################################################

FROM python:3.11-slim

# ---------- System dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---------- Working directory ----------
WORKDIR /tradingbot

# ---------- Python dependencies ----------
# Copy both requirements files and install in one layer to maximise cache hits
COPY requirements.txt ./requirements.txt
COPY quant/requirements_quant.txt ./quant/requirements_quant.txt

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -r quant/requirements_quant.txt

# ---------- Application code ----------
COPY . .

# ---------- Runtime directories ----------
RUN mkdir -p quant/data/snapshots logs

# ---------- Non-root user (security best practice) ----------
RUN adduser --disabled-password --gecos "" appuser \
 && chown -R appuser:appuser /tradingbot
USER appuser

# ---------- Expose port ----------
EXPOSE 8000

# ---------- Health check ----------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ---------- Start command ----------
# Workers=1 keeps the quant engine in a single process (shared state)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
