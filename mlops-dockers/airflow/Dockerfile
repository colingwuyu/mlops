ARG AIRFLOW_VER=2.1.2
ARG PYTHON_VER=3.7

FROM apache/airflow:${AIRFLOW_VER}-python${PYTHON_VER}

ARG MLFLOW_VER=1.18.0

RUN pip install --no-cache-dir --upgrade "mlflow==${MLFLOW_VER}"

USER root
RUN mkdir -p /app
COPY . /app

ENV PATH "/app:${PATH}"
ENV PYTHONPATH "${PYTHONPATH}:/app"
