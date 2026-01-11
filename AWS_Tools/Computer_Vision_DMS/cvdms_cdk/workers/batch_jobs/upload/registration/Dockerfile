# Use a lightweight Python base image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY batch_jobs/upload/registration/requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY batch_jobs/upload/registration/src/ .

# Copy the python package so /opt/common/common exists
COPY common/python/ /opt/common/

ENV PYTHONPATH=/opt/common:/usr/local/lib/python3.12/site-packages:/usr/local/lib/python3/site-packages
ENV PYTHONUNBUFFERED=1

# Set the default command to run your registration job
CMD ["python", "main.py"]