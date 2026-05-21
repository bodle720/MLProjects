FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Keep runtime-generated config/cache files out of the project tree.
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV YOLO_CONFIG_DIR=/tmp/ultralytics
ENV PYTHONPATH=/deployment

WORKDIR /deployment

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY model_runtime ./model_runtime

RUN mkdir -p app/tmp/uploads app/logs /tmp/matplotlib /tmp/ultralytics && \
    adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /deployment /tmp/matplotlib /tmp/ultralytics

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]