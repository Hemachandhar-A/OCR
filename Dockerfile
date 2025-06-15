FROM python:3.11-slim

# Install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && git lfs install \
 && apt-get purge -y --auto-remove \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy everything (must include .git and .gitattributes for LFS)
COPY . .

# Pull large files (model files) tracked by Git LFS
RUN git lfs pull

# Change directory into backend
WORKDIR /app/backend

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run app with gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT", "--timeout", "120", "--workers", "2"]
