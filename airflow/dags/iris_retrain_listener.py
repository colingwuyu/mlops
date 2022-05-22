import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.filesystem import FileSensor
from airflow.exceptions import AirflowSensorTimeout

from mlops.config import Config as conf
from mlops.orchestrators import pipeline as pipeline_module


default_args = {
    "owner": "airflow",
    "description": "IRIS Re-Training Pipeline Trigger",
    "depend_on_past": False,
    "start_date": datetime(2021, 1, 1),
}


def _failure_callback(context):
    if isinstance(context["exception"], AirflowSensorTimeout):
        print(context)
        print("Sensor timed out")


# def _transit_status_training():
#     cur_status = AirflowCommStatus.status()
#     assert cur_status == AirflowCommStatus.RETRAIN
#     print(cur_status.transit())

conf.load(
    config_from=os.path.join(
        os.environ.get(
            "AIRFLOW_HOME"), "dags/pipeline_config/prod_pipeline.yaml"
    )
)

pipeline = pipeline_module.Pipeline.load(conf.settings)
pipeline_dag_name = pipeline.name.replace(" ", "_") + "_DAG"

with DAG(
    "iris_retrain_listener_dag",
    schedule_interval=timedelta(seconds=60),
    default_args=default_args,
    catchup=False,
) as dag:

    watch_retrain_file_task = FileSensor(
        task_id="watch_retrain_file_task",
        filepath="/opt/airflow/comm/retrain",
        fs_conn_id="airflow_comm",
        poke_interval=5,
        timeout=30,
        on_failure_callback=_failure_callback,
        mode="poke",
    )

    retrain_trigger_task = TriggerDagRunOperator(
        task_id="retrain_trigger", trigger_dag_id=pipeline_dag_name
    )

    watch_retrain_file_task >> retrain_trigger_task

    # transit_status_task = PythonOperator(
    #     task_id="transit_airflow_comm_training",
    #     python_callable=_transit_status_training,
    # )

    # retrain_trigger_task >> transit_status_task
