# -------- TensorFlow-Compatible Python Slim Image ----------
FROM python:3.11-slim

# Set working directory
WORKDIR /GITOCR

# Install system dependencies for building packages + TensorFlow support
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy backend code only
COPY pixelpantry_backend/ .

# Install Python dependencies (TensorFlow + Flask + others)
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Expose port
EXPOSE 8080

# Start the Flask app with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
