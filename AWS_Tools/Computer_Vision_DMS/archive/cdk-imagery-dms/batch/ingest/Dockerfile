FROM public.ecr.aws/lambda/python:3.12

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy code
COPY split.py .

CMD ["python", "split.py"]
