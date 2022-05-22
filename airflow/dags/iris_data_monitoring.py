from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime

default_args = {"start_date": datetime(2021, 1, 1)}


def _new_data_count():
    pass


def _data_validate():
    pass


def _sns():
    pass


with DAG(
    "iris_data_monitoring_dag",
    schedule_interval="@daily",
    default_args=default_args,
    catchup=False,
) as dag:
    data_validation_task = PythonOperator(
        task_id="data_validation",
        python_callable=_data_validate,
    )
    perf_eval_trigger_task = TriggerDagRunOperator(
        task_id="performance_monitoring_trigger",
        trigger_dag_id="iris_performance_monitoring",
        trigger_rule=TriggerRule.ALL_FAILED,
    )
    data_validation_task >> perf_eval_trigger_task
