from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.exceptions import AirflowSensorTimeout
from datetime import datetime

default_args = {"start_date": datetime(2021, 1, 1)}


def _inference():
    pass


def _sns():
    pass


def _failure_callback(context):
    if isinstance(context["exception"], AirflowSensorTimeout):
        print(context)
        print("Sensor timed out")


with DAG(
    "iris_batch_inference_dag",
    schedule_interval="@daily",
    default_args=default_args,
    catchup=False,
) as dag:
    wait_for_input_data_sensor = FileSensor(
        task_id=f"wait_for_input_data",
        poke_interval=60,
        timeout=60 * 30,
        mode="reschedule",
        on_failure_callback=_failure_callback,
        filepath=f"inbound/data.csv",
        fs_conn_id=f"airflow_comm",
    )
    inference_task = PythonOperator(
        task_id="batch_inference",
        python_callable=_inference,
    )
    sns_task = PythonOperator(
        task_id="sns_email",
        python_callable=_sns,
    )
    wait_for_input_data_sensor >> inference_task >> sns_task
