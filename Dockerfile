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
RUN git clone https://github.com/Hemachandhar-A/OCR.git .


# Copy the content of your 'backend' folder from your Git repository (build context)
# into the '/app/backend' directory inside the Docker container.
COPY ./backend /app/backend

# Now, change the working directory *into* the backend folder.
# This ensures that subsequent commands are executed relative to where your Python app files are.
WORKDIR /app/backend

# *** CRITICAL ADDITION/MODIFICATION FOR LFS FILES ***
# Run git lfs pull *after* the backend directory content has been copied.
# This will download the actual large files tracked by LFS into this directory.
RUN git lfs pull

# Install Python dependencies from the backend's requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Command to run the application using Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT", "--timeout", "120", "--workers", "2"]