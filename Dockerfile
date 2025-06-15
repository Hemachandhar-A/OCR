FROM python:3.11-slim

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

RUN git lfs install

# Clone the full repo
WORKDIR /app
RUN git clone https://github.com/Hemachandhar-A/OCR.git .

RUN git lfs pull

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Change working directory to backend
WORKDIR /app/backend

ENV FLASK_APP=app.py
ENV FLASK_ENV=production

EXPOSE 8080

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
