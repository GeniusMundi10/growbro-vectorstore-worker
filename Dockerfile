# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy only the files needed for the worker
COPY vectorstore_webhook_worker.py .
COPY rag_dynamic.py .
COPY rag_utils.py .
COPY supabase_client.py .

# Expose the port your Flask app runs on
EXPOSE 8001

# Run the worker
CMD ["python", "vectorstore_webhook_worker.py"]