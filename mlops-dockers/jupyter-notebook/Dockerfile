ARG JUP=jupyter
ARG BASE_CONTAINER=$JUP/base-notebook

FROM $BASE_CONTAINER

ARG conda_env=python37
ARG py_ver=3.7

LABEL maintainer="Colin Wang <colingwuyu@gmail.com>"

RUN conda create --quiet --yes \ 
    -p "${CONDA_DIR}/envs/${conda_env}" \
    python=${py_ver} ipython ipykernel && \
    conda clean --all -f -y

RUN "${CONDA_DIR}/envs/${conda_env}/bin/python" -m ipykernel install --user --name="${conda_env}"

RUN mkdir "/home/${NB_USER}/notebooks"

RUN mamba install --quiet --yes -p "${CONDA_DIR}/envs/${conda_env}" \
    'mlflow=1.18.0' \
    'pyarrow=3.0.0' \
    'tensorflow=2.4.1' && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}" && \
    fix-permissions "/home/${NB_USER}/notebooks"

ENV PATH "${CONDA_DIR}/envs/${conda_env}/bin:${PATH}"
ENV CONDA_DEFAULT_ENV ${conda_env}

# Hadoop client
# install openJDK 1.8
USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
    openjdk-8-jdk \
    curl \
    gnupg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Setup env
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/

RUN curl -O https://dist.apache.org/repos/dist/release/hadoop/common/KEYS

RUN gpg --import KEYS

ENV HADOOP_VERSION 3.2.2
ENV HADOOP_URL https://www.apache.org/dist/hadoop/common/hadoop-$HADOOP_VERSION/hadoop-$HADOOP_VERSION.tar.gz
ENV HADOOP_USER ${NB_USER}
ENV HADOOP_PREFIX /usr/local/hadoop
ENV HADOOP_HOME /usr/local/hadoop
ENV HADOOP_COMMON_HOME /usr/local/hadoop
ENV HADOOP_HDFS_HOME /usr/local/hadoop
ENV HADOOP_CONF_DIR /opt/cluster-conf

# download hadoop
RUN set -x \
    && curl -fSL "$HADOOP_URL" -o /tmp/hadoop.tar.gz \
    && curl -fSL "$HADOOP_URL.asc" -o /tmp/hadoop.tar.gz.asc \
    && gpg --verify /tmp/hadoop.tar.gz.asc \
    && tar -xvf /tmp/hadoop.tar.gz -C /usr/local \
    && rm /tmp/hadoop.tar.gz* \
    && ln -s /usr/local/hadoop-${HADOOP_VERSION} /usr/local/hadoop \
    && groupadd -r hadoop \
    && groupadd -r $HADOOP_USER && usermod -a -g $HADOOP_USER -G hadoop $HADOOP_USER

RUN mkdir -p $HADOOP_CONF_DIR

# Setup permissions and ownership (httpfs tomcat conf for 600 permissions)
RUN chown -R $HADOOP_USER:hadoop /usr/local/hadoop-${HADOOP_VERSION} && chmod -R 775 $HADOOP_CONF_DIR

# set up hadoop user and bin path
ENV HADOOP_USER_NAME $HADOOP_USER
ENV PATH="${HADOOP_PREFIX}/bin:${PATH}"

ADD ./entrypoint.sh /entrypoint.sh

RUN chmod a+x /entrypoint.sh

USER ${NB_USER}

ENTRYPOINT ["tini", "-g", "--", "/entrypoint.sh"]
CMD ["start-notebook.sh"]
