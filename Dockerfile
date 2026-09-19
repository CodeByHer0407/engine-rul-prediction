# Base Python image
FROM python:3.11-slim

# Python configuration
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependency required by LightGBM
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy runtime dependencies
COPY requirements-api.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application source code
COPY src/ ./src/

# Create model directory
RUN mkdir -p /app/models

# Expose API port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]