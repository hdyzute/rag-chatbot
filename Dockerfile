FROM python:3.11-slim

WORKDIR /app

# Install curl
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install packages
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt && \
    rm -rf /root/.cache

# Copy ALL application files
COPY . .

# Create chroma directory
RUN mkdir -p /app/chroma_dbs

EXPOSE 5000

CMD ["python", "app.py"]