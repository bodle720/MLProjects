# Use the AWS Lambda Python 3.12 base image
FROM public.ecr.aws/lambda/python:3.12

# Set working directory inside the container
WORKDIR /var/task

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ .

# Set the CMD to your handler (module.function)
CMD ["main.lambda_handler"]
