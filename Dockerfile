FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Enable Git LFS
RUN git lfs install

# Clone your repo and pull LFS files
WORKDIR /app
RUN git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git .
RUN git lfs pull

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Set Flask environment
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

EXPOSE 8080

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
