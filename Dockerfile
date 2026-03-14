FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (needed for some Python libraries like psycopg2 or FAISS)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 8000

ENV PYTHONPATH=/app

CMD ["python3", "cli/main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
