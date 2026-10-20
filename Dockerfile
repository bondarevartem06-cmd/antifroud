FROM python:3.11-slim

# libgomp1 нужен LightGBM и XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
ENV PYTHONUNBUFFERED=1
