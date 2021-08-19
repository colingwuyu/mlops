ARG JUP=jupyter
ARG BASE_CONTAINER=$JUP/base-notebook

FROM $BASE_CONTAINER

ARG conda_env=python37
ARG py_ver=3.7

LABEL maintainer="Colin Wang <colingwuyu@gmail.com>"

USER root

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    g++ curl git-all \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER ${NB_USER}

RUN conda create --quiet --yes \ 
    -p "${CONDA_DIR}/envs/${conda_env}" \
    python=${py_ver} ipython ipykernel psycopg2 \
    && conda clean --all -f -y

RUN "${CONDA_DIR}/envs/${conda_env}/bin/python" -m ipykernel install --user --name="${conda_env}"

COPY ./requirements/mlops_req.txt "/home/${NB_USER}/"
RUN "${CONDA_DIR}/envs/${conda_env}/bin/pip3" install -r "/home/${NB_USER}/mlops_req.txt" \
    && fix-permissions "${CONDA_DIR}" \
    &&  fix-permissions "/home/${NB_USER}"

ENV PATH "${CONDA_DIR}/envs/${conda_env}/bin:${PATH}"
ENV CONDA_DEFAULT_ENV ${conda_env}

RUN rm "/home/${NB_USER}/mlops_req.txt"

EXPOSE 5000

WORKDIR "/home/${NB_USER}"
COPY mlops-dockers/mlops-base/wait-for-it.sh "/home/${NB_USER}/"

USER root
RUN chmod a+x wait-for-it.sh

RUN mkdir -p /app/pipeline_config
COPY . /app
RUN chown -R ${NB_USER} /app

USER ${NB_USER}

ENV PATH "/app:${PATH}"
ENV PYTHONPATH "${PYTHONPATH}:/app"

RUN mkdir "/home/${NB_USER}/notebooks"
RUN mkdir "/home/${NB_USER}/mlflow_artifacts"

ENTRYPOINT ["tini", "-g", "--"] 

