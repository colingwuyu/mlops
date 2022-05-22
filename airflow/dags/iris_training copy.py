import os
from datetime import datetime

from mlops.config import Config as conf
from mlops.orchestrators import pipeline as pipeline_module
from mlops.orchestrators.airflow.airflow_dag_runner import AirflowPipelineConfig, AirflowDagRunner

default_args = {
    "owner": "airflow",
    "description": "IRIS Training Pipeline",
    "depend_on_past": False,
    "start_date": datetime(2021, 1, 1),
}

_airflow_config = {
    "default_args": default_args,
    # "schedule_interval": "0 0 1 * *",
    "catchup": False,
}

conf.load(
    config_from=os.path.join(
        os.environ.get(
            "AIRFLOW_HOME"), "dags/pipeline_config/prod_pipeline.yaml"
    )
)

pipeline = pipeline_module.Pipeline.load(conf.settings)

DAG = AirflowDagRunner(AirflowPipelineConfig(_airflow_config)).run(pipeline)
