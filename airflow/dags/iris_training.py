from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator

from airflow_comm import AirflowCommStatus
from operators.mlops import MlopsComponentDockerOperator


def _transit_status_training():
    AirflowCommStatus.TRAINING.save()


def _transit_status_done():
    cur_status = AirflowCommStatus.status()
    assert cur_status == AirflowCommStatus.TRAINING
    print(cur_status.transit())


default_args = {
    "owner": "airflow",
    "description": "IRIS Training Pipeline",
    "depend_on_past": False,
    "start_date": datetime(2021, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
with DAG(
    "iris_training_dag",
    default_args=default_args,
    schedule_interval="0 0 1 * *",
    catchup=False,
) as dag:
    start_train_task = PythonOperator(
        task_id="transit_airflow_comm_status_training",
        python_callable=_transit_status_training,
    )
    data_gen_task = MlopsComponentDockerOperator(task_id="data_gen")
    start_train_task >> data_gen_task

    data_validation_task = MlopsComponentDockerOperator(
        task_id="data_validation",
        upstreams=["data_gen"],
    )
    data_gen_task >> data_validation_task

    feature_transform_task = MlopsComponentDockerOperator(
        task_id="feature_transform",
        upstreams=["data_gen"],
    )
    data_validation_task >> feature_transform_task

    lda_task = MlopsComponentDockerOperator(
        task_id="linear_discriminant_analysis",
        upstreams=["data_gen", "data_validation", "feature_transform"],
    )
    [data_gen_task, data_validation_task, feature_transform_task] >> lda_task

    lr_task = MlopsComponentDockerOperator(
        task_id="logistic_regression",
        upstreams=["data_gen", "data_validation", "feature_transform"],
    )
    [data_gen_task, data_validation_task, feature_transform_task] >> lr_task

    model_eval_task = MlopsComponentDockerOperator(
        task_id="model_eval",
        upstreams=[
            "data_gen",
            "linear_discriminant_analysis",
            "logistic_regression",
        ],
    )
    [data_gen_task, lda_task, lr_task] >> model_eval_task

    model_pub_task = MlopsComponentDockerOperator(
        task_id="model_pub", upstreams=["data_gen", "model_eval"]
    )
    [data_gen_task, model_eval_task] >> model_pub_task

    transit_status_task = PythonOperator(
        task_id="transit_airflow_comm_status_done", python_callable=_transit_status_done
    )

    model_pub_task >> transit_status_task
