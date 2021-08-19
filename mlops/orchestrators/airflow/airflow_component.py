import json
import os

from airflow.operators.docker_operator import DockerOperator
from docker.types import Mount

from mlops.orchestrators.datatype import ComponentSpec
from mlops.orchestrators.airflow.airflow_util import create_entry_point

AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME")
AIRFLOW__CELERY__BROKER_URL = os.environ.get("AIRFLOW__CELERY__BROKER_URL")
AIRFLOW_COMPONENT_SUBDIR = "dags/pipeline_config"
DOCKER_COMPONENT_DIR = "/app/pipeline_config"
# COMPONENT_MAIN = "main.py"
REDIS_URL = "REDIS_URL"
REDIS_DB = "REDIS_DB"
DOCKER_COMPONENT_REDIS_DB = "1"


# a wrapper class
class MlopsComponentDockerOperator(DockerOperator):
    def __init__(self, mlops_component_spec: ComponentSpec, *args, **kwargs):
        entry_point_module_dir = os.path.join(AIRFLOW_HOME, AIRFLOW_COMPONENT_SUBDIR)
        entry_point_module_name = create_entry_point(
            mlops_component_spec, entry_point_module_dir
        )
        entry_point_module = os.path.join(DOCKER_COMPONENT_DIR, entry_point_module_name)
        task_id = mlops_component_spec.name
        super().__init__(
            image="mlops-base:latest",
            task_id=task_id,
            container_name=f"iris_train_dag_task__{task_id}",
            command=f"python {entry_point_module}",
            api_version="auto",
            auto_remove=True,
            docker_url="unix://var/run/docker.sock",
            environment={
                REDIS_URL: AIRFLOW__CELERY__BROKER_URL[:-1],
                REDIS_DB: DOCKER_COMPONENT_REDIS_DB,
            },
            working_dir="/app",
            mounts=[
                Mount(
                    target="/home/jovyan/mlflow_artifacts",
                    source="mlops_mlflow_artifacts",
                ),
                Mount(target=DOCKER_COMPONENT_DIR, source="mlops_pipeline_config"),
            ],
            network_mode="mlops-bridge",
            *args,
            **kwargs,
        )

    # TODO: add arguments by set downstream and upstream
    # def set_downstream(
    #     self,
    #     task_or_task_list,
    #     edge_modifier,
    # ) -> None:
    #     """
    #     Set a task or a task list to be directly downstream from the current
    #     task. Required by TaskMixin.
    #     """
    #
    # super().set_downstream(task_or_task_list, edge_modifier=edge_modifier)

    # def execute(self, context):
    #     # get the last log line from docker stdout
    #     docker_log = super().execute(context)

    #     # push XComs from the json
    #     if docker_log:
    #         try:
    #             result = json.loads(docker_log.decode().replace("'", '"'))
    #             for key in result.keys():
    #                 context["ti"].xcom_push(key=key, value=result[key])
    #         except:
    #             pass

    #     return docker_log
