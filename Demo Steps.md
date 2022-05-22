# Demo Steps

## Servers

- mlflow: <http://localhost:5000/>
- airflow: <http://localhost:8080/> (user-airflow; password-airflow)
- notebook: <http://localhost:8888/> (password-notebook)
- serving: <http://localhost:3000/>

## Preq

1. no registered model in mlflow
2. run `./clean.sh` in `examples/iris/serving/` folder

## Development Env Demo

1. `make up-experiment`: boost experiment dev environment
2. open `jupyter lab` (find token in notebook docker)
3. walk through `sandbox_experiment` notebook
4. after done, `make down` to shutdown environment

## Prod Serving Demo

1. `make up-prod`: boost prod demo environment
2. open `Mlflow` to check no registered model
3. open `iris monitoring` process to see the log
4. open `airflow webserver`: walk through iris dags
5. check training done, open `iris_serving` model_ver page to show loaded model version
6. open `jupyter lab` (find token in notebook docker)
7. open `config_prod.yaml` to show data_monitoring and performance_monitoring settings
8. use `prod_demo` notebook to hack production model
