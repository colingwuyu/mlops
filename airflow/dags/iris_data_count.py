from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime

default_args = {"start_date": datetime(2021, 1, 1)}


def _new_data_count():
    pass


def _perf_eval_threshold():
    pass


def _data_val_threshold():
    pass


with DAG(
    "iris_data_count_dag",
    schedule_interval="@daily",
    default_args=default_args,
    catchup=False,
) as dag:
    data_count_task = PythonOperator(
        task_id="data_count",
        python_callable=_new_data_count,
    )
    perf_count_threshold_task = PythonOperator(
        task_id="perf_eval_threshould", python_callable=_perf_eval_threshold
    )
    perf_eval_trigger_task = TriggerDagRunOperator(
        task_id="performance_monitoring_trigger",
        trigger_dag_id="iris_performance_monitoring_dag",
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )
    data_val_threshold_task = PythonOperator(
        task_id="data_val_threshould", python_callable=_data_val_threshold
    )
    data_val_trigger_task = TriggerDagRunOperator(
        task_id="data_monitoring_trigger",
        trigger_dag_id="iris_data_monitoring_dag",
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    data_count_task >> [perf_count_threshold_task, data_val_threshold_task]
    perf_count_threshold_task >> perf_eval_trigger_task
    data_val_threshold_task >> data_val_trigger_task
