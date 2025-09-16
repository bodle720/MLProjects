# Use a lightweight Python base image
FROM public.ecr.aws/lambda/python:3.12

# Set working directory inside the container
WORKDIR /var/task

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY src/ .

# Set the CMD to your handler
CMD ["app.lambda_handler"]

