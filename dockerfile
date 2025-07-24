# Build stage
FROM python:3.12-slim-bookworm as builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv pip install --no-cache-dir --system -r requirements.txt

# Final stage
FROM python:3.12-slim-bookworm

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/app/.config/Ultralytics

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user and set up directory
RUN useradd -m -d /app appuser
WORKDIR /app

# Create config directory and set permissions
RUN mkdir -p /app/.config/Ultralytics && \
    chown -R appuser:appuser /app/.config

# Copy only necessary files with proper permissions
COPY --from=builder --chown=appuser:appuser /usr/local/ /usr/local/
COPY --chown=appuser:appuser app.py extractor.py db_utils.py ./
COPY --chown=appuser:appuser models/ ./models/

# Switch to non-root user
USER appuser

# Set up Ultralytics config
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')  # This will create the config file" || true

# Health check (only if you have a /health endpoint)
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
