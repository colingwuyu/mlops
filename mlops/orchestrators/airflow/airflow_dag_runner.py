from typing import Any, Dict, Optional, Text, Union
import yaml

from airflow import models
from airflow.operators.python import PythonOperator

from mlops.orchestrators import pipeline as pipeline_module
from mlops.orchestrators.airflow import airflow_component, airflow_comm
from mlops.orchestrators.airflow.airflow_component import (
    AIRFLOW__CELERY__BROKER_URL,
    DOCKER_COMPONENT_REDIS_DB,
)


class AirflowPipelineConfig:
    def __init__(self, airflow_dag_config: Optional[Dict[Text, Any]] = None):
        self.airflow_dag_config = airflow_dag_config or {}


class AirflowDagRunner:
    """Airflow DAG runner."""

    def __init__(
        self, config: Optional[Union[Dict[Text, Any], AirflowPipelineConfig]] = None
    ):
        """Initializes AirflowDagRunner."""
        if config and not isinstance(config, AirflowPipelineConfig):
            config = AirflowPipelineConfig(airflow_dag_config=config)
        self._config = config

    def run(self, pipeline: pipeline_module.Pipeline) -> None:
        """Runs given logical pipeline locally.

        Args:
            pipeline (pipeline_py.Pipeline): Logical pipeline containing pipeline components.
        """
        airflow_dag = models.DAG(
            dag_id=pipeline.name.replace(" ", "_") + "_DAG",
            default_args=self._config.airflow_dag_config["default_args"],
        )
        start_pipeline_task = PythonOperator(
            task_id="transit_airflow_comm_status_training",
            python_callable=_transit_status_training,
            dag=airflow_dag,
            default_args=self._config.airflow_dag_config["default_args"],
        )
        init_mlflow_task = PythonOperator(
            task_id="init_mlflow",
            python_callable=_exec_init_run,
            op_args=[pipeline],
            dag=airflow_dag,
            default_args=self._config.airflow_dag_config["default_args"],
        )
        init_mlflow_task.set_upstream(start_pipeline_task)
        end_mlflow_task = PythonOperator(
            task_id="end_mlflow",
            python_callable=_exec_end_run,
            dag=airflow_dag,
            default_args=self._config.airflow_dag_config["default_args"],
        )
        end_pipeline_task = PythonOperator(
            task_id="transit_airflow_comm_status_done",
            python_callable=_transit_status_done,
            default_args=self._config.airflow_dag_config["default_args"],
        )
        end_pipeline_task.set_upstream(end_mlflow_task)

        component_impl_map = {}
        for component_name, component_spec in pipeline.operators.items():
            if component_name in component_impl_map:
                continue

            current_airflow_component = airflow_component.MlopsComponentDockerOperator(
                mlops_component_spec=component_spec,
                dag=airflow_dag,
                default_args=self._config.airflow_dag_config["default_args"],
            )
            component_impl_map[component_name] = current_airflow_component

            for upstream_node in component_spec.upstreams:
                if upstream_node not in component_impl_map:
                    upstream_airflow_component = (
                        airflow_component.MlopsComponentDockerOperator(
                            mlops_component_spec=pipeline.operators[upstream_node],
                            dag=airflow_dag,
                            default_args=self._config.airflow_dag_config[
                                "default_args"
                            ],
                        )
                    )
                    component_impl_map[upstream_node] = upstream_airflow_component
                current_airflow_component.set_upstream(
                    component_impl_map[upstream_node]
                )

            if component_spec.pipeline_init:
                current_airflow_component.set_upstream(init_mlflow_task)

            if component_spec.pipeline_end:
                current_airflow_component.set_downstream(end_mlflow_task)
        return airflow_dag


def _transit_status_training():
    airflow_comm.AirflowCommStatus.TRAINING.save()


def _transit_status_done():
    cur_status = airflow_comm.AirflowCommStatus.status()
    assert cur_status == airflow_comm.AirflowCommStatus.TRAINING
    print(cur_status.transit())


def _exec_init_run(pipeline: pipeline_module.Pipeline):
    import redis

    r = redis.Redis.from_url(
        AIRFLOW__CELERY__BROKER_URL[:-1], db=DOCKER_COMPONENT_REDIS_DB
    )
    print(pipeline)
    pipeline.mlflow_info.init_mlflow_run()

    r.hset("mlflow_runids", "pipeline_mlflow_runid", pipeline.run_id)
    r.set("pipeline", yaml.dump(pipeline._serialize()))


def _exec_end_run():
    import redis

    r = redis.Redis.from_url(
        AIRFLOW__CELERY__BROKER_URL[:-1], db=DOCKER_COMPONENT_REDIS_DB
    )

    pipeline = pipeline_module.Pipeline.load(
        yaml.load(r.get("pipeline").decode("utf-8"))
    )

    run_ids = r.hgetall("mlflow_runids")
    for run_name, run_id in run_ids.items():
        run_name = run_name.decode("utf-8")
        run_id = run_id.decode("utf-8")
        if run_name == "pipeline_mlflow_runid":
            pipeline.mlflow_info.mlflow_run_id = run_id
        else:
            pipeline.operators[run_name].run_id = run_id

    print(pipeline)
    pipeline.log_mlflow()
    r.delete("mlflow_runids")
    r.delete("pipeline")
