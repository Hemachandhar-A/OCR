FROM python:3.11-slim

WORKDIR /GITOCR

# Install system packages including libGL
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

ENV FLASK_APP=app.py
ENV FLASK_ENV=production

EXPOSE 8080

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
