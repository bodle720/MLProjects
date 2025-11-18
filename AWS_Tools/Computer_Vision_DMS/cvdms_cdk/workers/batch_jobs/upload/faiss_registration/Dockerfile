# Use a lightweight Python base image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY batch_jobs/upload/faiss_registration/requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY batch_jobs/upload/faiss_registration/src/ .
COPY common/ /opt/common/

ENV PYTHONPATH="/opt/common:${PYTHONPATH}"

# Set the default command to run your internal deup job
CMD ["python", "main.py"]