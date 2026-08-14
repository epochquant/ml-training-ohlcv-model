# Use official PyTorch GPU runtime image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system utilities & Google Cloud SDK components
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gsutil

# Copy Kronos ML training source code
COPY model/ ./model/
COPY training/ ./training/
COPY data/ ./data/
COPY configs/ ./configs/
COPY tests/ ./tests/
COPY run_training_pipeline.py .

# Set default execution entrypoint
ENTRYPOINT ["python", "run_training_pipeline.py"]
CMD ["--non-interactive"]
