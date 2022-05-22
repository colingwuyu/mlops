import os
from enum import Enum
from pathlib import Path

AIRFLOW_COMM_PATH = "/opt/airflow/comm"


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

    def save(self):
        for f in os.listdir(f"{AIRFLOW_COMM_PATH}/"):
            os.remove(f"{AIRFLOW_COMM_PATH}/{f}")
        Path(self._status_file).touch()

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
