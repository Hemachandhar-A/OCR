# Use a slim Python image
FROM python:3.11-slim

# Install system dependencies needed for OpenCV and Git LFS
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Clean up apt caches to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Install git-lfs for fetching large files (globally in the container)
RUN git lfs install

# Set the working directory to /app (where Railway clones your repo by default)
WORKDIR /app

RUN git clone https://github.com/Hemachandhar-A/OCR.git . # <--- Your repo content is placed here in /app

# Now, change the working directory *into* the backend folder.
# This is crucial so that subsequent commands (like pip install and gunicorn)
# are executed relative to where your Python app files and requirements.txt are.
WORKDIR /app/backend

RUN git lfs pull
# Install Python dependencies from the backend's requirements.txt
# This must happen AFTER requirements.txt has been copied into the container's /app/backend.
RUN pip install --no-cache-dir -r requirements.txt
# --- CRITICAL CHANGE END ---

# Environment variables (good to keep)
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# The EXPOSE instruction is for documentation; it doesn't actually publish the port.
# Railway handles port mapping based on the container's binding (0.0.0.0:$PORT).
# You can remove it or keep it, it won't impact functionality on Railway.
# EXPOSE 8080 # This is less relevant with $PORT

# Command to run the application using Gunicorn
# Use the $PORT environment variable provided by Railway
CMD sh -c 'gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2'
