DOCKER_NETWORK = mlops-bridge
ENV_FILE = hadoop.env
HADOOP_VERSION = 3.2.1
current_branch = 2.0.0-hadoop${HADOOP_VERSION}-java8
build:
	docker build -t hadoop-base:$(current_branch) ./base
	docker build -t hadoop-namenode:$(current_branch) ./namenode
	docker build -t hadoop-datanode:$(current_branch) ./datanode
	docker build -t hadoop-resourcemanager:$(current_branch) ./resourcemanager
	docker build -t hadoop-nodemanager:$(current_branch) ./nodemanager
	docker build -t hadoop-historyserver:$(current_branch) ./historyserver
	docker build -t hadoop-submit:$(current_branch) ./submit

wordcount:
	docker build -t hadoop-wordcount ./submit
	docker run --network ${DOCKER_NETWORK} --env-file ${ENV_FILE} hadoop-base:$(current_branch) hdfs dfs -mkdir -p /input/
	docker run --network ${DOCKER_NETWORK} --env-file ${ENV_FILE} hadoop-base:$(current_branch) hdfs dfs -copyFromLocal -f /opt/hadoop-${HADOOP_VERSION}/README.txt /input/
	docker run --network ${DOCKER_NETWORK} --env-file ${ENV_FILE} hadoop-wordcount
	docker run --network ${DOCKER_NETWORK} --env-file ${ENV_FILE} hadoop-base:$(current_branch) hdfs dfs -cat /output/*
	docker run --network ${DOCKER_NETWORK} --env-file ${ENV_FILE} hadoop-base:$(current_branch) hdfs dfs -rm -r /output
	docker run --network ${DOCKER_NETWORK} --env-file ${ENV_FILE} hadoop-base:$(current_branch) hdfs dfs -rm -r /input

up:
	docker-compose up

down:
	docker-compose down