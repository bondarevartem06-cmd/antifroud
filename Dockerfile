# ---------- общий слой: Python, библиотеки модели, БД и код проекта ----------
FROM python:3.11-slim AS base

# libgomp1 нужен LightGBM и XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Библиотеки для PostgreSQL и Alembic - отдельным слоем, чтобы не пересобирать тяжёлые библиотеки выше
COPY requirements-db.txt .
RUN pip install --no-cache-dir -r requirements-db.txt

COPY *.py alembic.ini ./
COPY alembic ./alembic
ENV PYTHONUNBUFFERED=1


# ---------- Apache Airflow: отдельное виртуальное окружение, чтобы не менять версии библиотек модели ----------
FROM base AS airflow
ARG AIRFLOW_VERSION=2.10.5
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/airflow-venv \
    && /opt/airflow-venv/bin/pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" \
        apache-airflow-providers-postgres \
        --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.11.txt"

RUN useradd -m -u 50000 -s /bin/bash airflow \
    && mkdir -p /opt/airflow/dags /opt/airflow/logs /opt/airflow/work \
    && chown -R airflow:airflow /opt/airflow \
    && printf '#!/bin/sh\nexec python /app/airflow_entrypoint.py cli "$@"\n' > /usr/local/bin/af \
    && chmod +x /usr/local/bin/af

ENV AIRFLOW_HOME=/opt/airflow \
    AIRFLOW__CORE__EXECUTOR=LocalExecutor \
    AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags \
    AIRFLOW__CORE__LOAD_EXAMPLES=False \
    AIRFLOW__WEBSERVER__EXPOSE_CONFIG=False
USER airflow
CMD ["python", "/app/airflow_entrypoint.py"]


# ---------- итоговый образ приложения (последняя стадия = цель по умолчанию) ----------
FROM base AS app
