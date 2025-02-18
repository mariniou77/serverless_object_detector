# Use Python 3.10 slim as a base (adjust as needed)
FROM python:3.10-slim

# Install system dependencies needed by OpenCV, ultralytics, etc.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy everything into the container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Default command (will be overridden by docker-compose)
CMD ["python", "--version"]