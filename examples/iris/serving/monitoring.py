import time
import os
import json
import requests
from pathlib import Path
from enum import Enum

import pandas as pd

from mlops.config import Config as conf
import mlops.model_analysis as mlops_ma
import mlops.data_validation as mlops_dv
from mlops.utils.otherutils import FileLineCounter
from mlops.utils.mlflowutils import MlflowUtils
import mlops.serving.model as mlops_model
from examples.iris.serving.helper import load_model

conf.load("config_{}.yaml".format(os.environ.get("IRIS_ENV", "local")))

LABEL_DATA = os.path.join(
    conf.settings.data_path.data_store, conf.settings.data_path.label_data
)
LABEL_DATA_FILE = os.path.join(LABEL_DATA, "data.csv")

UPLOAD_DATA = os.path.join(
    conf.settings.data_path.data_store, conf.settings.data_path.upload_data
)
UPLOAD_DATA_FILE = os.path.join(UPLOAD_DATA, "data.csv")
DATA_REPORT_PATH = "templates/data_reports"


DATA_COUNTER_FILE = os.path.join(
    UPLOAD_DATA, conf.settings.data_monitoring.data_counter_file
)
PERF_DATA_COUNTER_FILE = os.path.join(
    UPLOAD_DATA, conf.settings.performance_monitoring.perf_counter_file
)
PERF_COUNTER_FILE = os.path.join(
    LABEL_DATA, conf.settings.performance_monitoring.perf_counter_file
)
EVAL_RESULT_PATH = "static/performance_reports"
EVAL_REPORT_PATH = "templates/performance_reports"
IRIS_SERVING_URL = (
    "http://iris_serving:3000"
    if os.environ.get("IRIS_ENV")
    else "http://localhost:3000"
)


AIRFLOW_COMM_PATH = "static/data/airflow_comm"


MlflowUtils.init_mlflow_client(
    conf.settings.mlflow.tracking_uri, conf.settings.mlflow.registry_uri
)
print(conf.settings.serving.model_name)
if not MlflowUtils.mlflow_client.search_registered_models(
    f"name='{conf.settings.serving.model_name}'"
):
    model = None
else:
    model = mlops_model.load_model(
        f"models:/{conf.settings.serving.model_name}/Production"
    )


class AirflowCommStatus(Enum):
    IDLE = "empty"
    RETRAIN = "retrain"
    TRAINING = "training"
    DONE = "done"

    @classmethod
    def status(cls):
        status_files = os.listdir(AIRFLOW_COMM_PATH)
        assert len(status_files) <= 1
        if not status_files:
            cur_status = cls("empty")
            Path(cur_status._status_file).touch()
            return cur_status
        return cls(status_files[0])

    @property
    def _status_file(self):
        return f"{AIRFLOW_COMM_PATH}/{self.value}"

    def transit(self):
        if self == AirflowCommStatus.IDLE:
            os.remove(self._status_file)
            new_status = AirflowCommStatus.RETRAIN
            Path(new_status._status_file).touch()
        elif self == AirflowCommStatus.RETRAIN:
            os.remove(self._status_file)
            new_status = AirflowCommStatus.TRAINING
            Path(new_status._status_file).touch()
        elif self == AirflowCommStatus.TRAINING:
            os.remove(self._status_file)
            new_status = AirflowCommStatus.DONE
            Path(new_status._status_file).touch()
        elif self == AirflowCommStatus.DONE:
            os.remove(self._status_file)
            new_status = AirflowCommStatus.IDLE
            Path(new_status._status_file).touch()
        return new_status


def performance_eval(start_row, end_row):
    global model
    print(f"Monitoring: performance evaluation examples {start_row} to {end_row}")
    eval_data = pd.read_csv(
        LABEL_DATA_FILE,
        skiprows=list(range(1, start_row))
        + list(range(end_row + 1, perf_counter.file_lines + 1)),
    )
    eval_path = os.path.join(
        EVAL_RESULT_PATH, f"Evaluation examples {start_row}-{end_row}"
    )
    prev_eval_path = os.path.join(
        EVAL_RESULT_PATH,
        f"Evaluation examples {start_row-perf_counter.inc_interval}-{end_row-perf_counter.inc_interval}",
    )
    prev_eval_data_stats = None
    if os.path.exists(prev_eval_path):
        prev_eval_data_stats = mlops_ma.get_eval_data_stats(prev_eval_path)

    validation_result = mlops_ma.validate_model(
        model,
        eval_data,
        prev_data_stat=prev_eval_data_stats,
        output_path=eval_path,
    )
    mlops_ma.view_report(
        eval_path,
        report_save_path=eval_path.replace(EVAL_RESULT_PATH, EVAL_REPORT_PATH),
    )
    print(
        "Monitoring: performance report is ready in %s..."
        % eval_path.replace(EVAL_RESULT_PATH, EVAL_REPORT_PATH)
    )
    return validation_result


if __name__ == "__main__":
    print("start monitoring...")
    upload_data_counter = FileLineCounter(
        data_file=UPLOAD_DATA_FILE,
        inc_interval=conf.settings.data_monitoring.count_frequency,
        counter_file=DATA_COUNTER_FILE,
        initial_point=0,
    )
    perf_counter = FileLineCounter(
        data_file=LABEL_DATA_FILE,
        inc_interval=conf.settings.performance_monitoring.count_frequency,
        counter_file=PERF_COUNTER_FILE,
        initial_point=150,
    )

    if model is None:
        print("Monitoring: load_model None...")
    else:
        response = requests.get(f"{IRIS_SERVING_URL}/reload_model")
        print(response.text)
    while True:
        airflow_comm_status = AirflowCommStatus.status()
        if airflow_comm_status == AirflowCommStatus.DONE:
            airflow_comm_status = airflow_comm_status.transit()
        if airflow_comm_status != AirflowCommStatus.IDLE:
            print("Monitoring: waiting for model ready...")
            time.sleep(30)
            continue
        if not model:
            airflow_comm_status.transit()
            print("Monitoring: start training...")
            while AirflowCommStatus.status() != AirflowCommStatus.DONE:
                print("Monitoring: wait for training finish...")
                time.sleep(30)
            AirflowCommStatus.status().transit()
            print("Monitoring: training done...")
            model = load_model("Production")
            response = requests.get(f"{IRIS_SERVING_URL}/reload_model")
            print(response.text)

        require_retrain = False

        for start_row, end_row in upload_data_counter.move():
            print(f"Monitoring: data validation examples {start_row} to {end_row}")
            upload_data = pd.read_csv(
                UPLOAD_DATA_FILE,
                skiprows=list(range(1, start_row))
                + list(range(end_row + 1, upload_data_counter.file_lines + 1)),
            )
            prev_start_row = start_row - upload_data_counter.inc_interval
            prev_end_row = end_row - upload_data_counter.inc_interval

            prev_upload_data = None
            if prev_start_row >= 0:
                prev_upload_data = pd.read_csv(
                    UPLOAD_DATA_FILE,
                    skiprows=list(range(1, prev_start_row))
                    + list(range(prev_end_row + 1, upload_data_counter.file_lines + 1)),
                )
            data_report_path = os.path.join(
                DATA_REPORT_PATH, f"Evaluation examples {start_row}-{end_row}"
            )
            _, pass_skew_test = mlops_dv.view_report(
                model=model,
                serving_data=upload_data,
                prev_data=prev_upload_data,
                report_save_path=data_report_path,
            )
            print(
                "Monitoring: data validation report is ready in %s..."
                % data_report_path
            )

            if not pass_skew_test:
                # Fail data skew validation
                # Triger performance evaluation
                print("Monitoring: data skew detected...")
                while (
                    perf_counter.count_lines() < perf_counter._initial_point + end_row
                ):
                    print("Monitoring: wait for label data uploading...")
                    # waiting for label data
                    time.sleep(10)
                validation_result = performance_eval(
                    start_row + perf_counter._initial_point,
                    end_row + perf_counter._initial_point,
                )
                require_retrain |= validation_result is mlops_ma.VALIDATION_FAIL
                if require_retrain:
                    print(
                        "Monitoring: performance evaluation fails, require model retrain..."
                    )
                    perf_counter.move_to(end_row + perf_counter._initial_point)

        for start_row, end_row in perf_counter.move():
            validation_result = performance_eval(start_row, end_row)
            require_retrain |= validation_result is mlops_ma.VALIDATION_FAIL
            if require_retrain:
                print(
                    "Monitoring: performance evaluation fails, require model retrain..."
                )

        if require_retrain:
            airflow_comm_status.transit()
            print("Monitoring: start training...")
            while AirflowCommStatus.status() != AirflowCommStatus.DONE:
                print("Monitoring: wait for training finish...")
                time.sleep(30)
            response = requests.get(f"{IRIS_SERVING_URL}/reload_model")
            print("Monitoring: " + response.text)
            AirflowCommStatus.status().transit()
            print("Monitoring: training done...")
            model = load_model("Production")
        time.sleep(10)
