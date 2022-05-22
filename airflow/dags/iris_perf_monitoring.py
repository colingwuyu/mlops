from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.sensors.filesystem import FileSensor

from datetime import datetime

default_args = {"start_date": datetime(2021, 1, 1)}


def _eval_sampling():
    pass


def _perf_eval():
    pass


with DAG(
    "iris_performance_monitoring_dag",
    schedule_interval="@daily",
    default_args=default_args,
    catchup=False,
) as dag:
    eval_sampling_task = PythonOperator(
        task_id="performance_evaluation_sampling",
        python_callable=_eval_sampling,
    )

    wait_for_labels_task = FileSensor(
        task_id="wait_for_labels",
        filepath="inbound/label_data.csv",
        fs_conn_id="airflow_comm",
        poke_interval=5,
        timeout=30,
        mode="poke",
    )
    perf_eval_task = PythonOperator(
        task_id="performance_evaluation",
        python_callable=_perf_eval,
    )
    retrain_trigger_task = TriggerDagRunOperator(
        task_id="iris_training_dag_trigger",
        trigger_dag_id="iris_retrain",
        trigger_rule=TriggerRule.ALL_FAILED,
    )
    eval_sampling_task >> wait_for_labels_task >> perf_eval_task >> retrain_trigger_task
