import json

from airflow.operators.docker_operator import DockerOperator
from docker.types import Mount


# a wrapper class
class MlopsComponentDockerOperator(DockerOperator):
    def __init__(self, task_id, upstreams=[], *args, **kwargs):
        pipeline_mlflow_runid = (
            "{{ ti.xcom_pull(task_ids='data_gen', key='pipeline_mlflow_runid')  }}"
        )
        component_mlflow_runid = (
            "{{{{ ti.xcom_pull(task_ids={}, key='component_mlflow_runid') }}}}"
        )
        if upstreams:
            command_args = "mlflow_ids=pipeline:{}".format(
                pipeline_mlflow_runid)
            for upstream in upstreams:
                str_upstream = "'" + upstream + "'"
                command_args += ";{}:{}".format(
                    str_upstream,
                    component_mlflow_runid.format(str_upstream),
                )
        else:
            command_args = ""
        super().__init__(
            image="mlops-base:latest",
            task_id=task_id,
            container_name=f"iris_train_dag_task__{task_id}",
            command=f"python -Wignore -m examples.iris.{task_id}.main " +
            command_args,
            api_version="auto",
            auto_remove=True,
            docker_url="unix://var/run/docker.sock",
            working_dir="/app",
            mounts=[
                Mount(
                    target="/home/jovyan/mlflow_artifacts",
                    source="mlops_mlflow_artifacts",
                ),
                Mount(target="/app/pipeline_config", source="pipeline_config"),
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

    def execute(self, context):
        # get the last log line from docker stdout
        docker_log = super().execute(context)

        # push XComs from the json
        if docker_log:
            try:
                result = json.loads(docker_log.decode().replace("'", '"'))
                for key in result.keys():
                    context["ti"].xcom_push(key=key, value=result[key])
            except:
                pass

        return docker_log
