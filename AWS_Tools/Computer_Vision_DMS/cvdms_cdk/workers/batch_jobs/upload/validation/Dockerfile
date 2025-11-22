# Use a lightweight Python base image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed by Pillow and scientific libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libjpeg-dev \
        zlib1g-dev \
        libpng-dev \
        libtiff5-dev \
        libopenjp2-7-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY batch_jobs/upload/validation/requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY batch_jobs/upload/validation/src/ .
COPY common/ /opt/common/

ENV PYTHONPATH="/opt/common:${PYTHONPATH}"

# Set the default command to run your validation job
CMD ["python", "main.py"]

