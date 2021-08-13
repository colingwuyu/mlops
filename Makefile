.PHONY: all clean

build:
	docker build . -f ./mlops-dockers/mlops-base/Dockerfile --build-arg conda_env=python37 --build-arg py_ver="3.7" -t mlops-base
	docker build . -f ./mlops-dockers/postgres/Dockerfile -t mlops-postgres
	
up-prod:
	docker-compose up -d
	
up-experiment:
	docker-compose up -d iris_serving notebook mlflow_server postgres

down:
	docker-compose down