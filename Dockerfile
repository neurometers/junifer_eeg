# Use Python 3.12 as base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies that might be needed for scientific packages
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first for better caching
COPY requirements.txt .

# Create a modified requirements.txt without the editable junifer_eeg line
RUN grep -v "junifer_eeg" requirements.txt > requirements_docker.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy the entire project
COPY pyproject.toml setup.py ./
COPY junifer_eeg/ ./junifer_eeg/

# Install junifer_eeg in editable mode
RUN pip install -e .

# Set the entrypoint to use junifer command
ENTRYPOINT ["junifer"]

# Default command (can be overridden)
CMD ["--help"]

