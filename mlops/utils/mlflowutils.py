import os
import json

import mlflow
from mlflow.tracking import MlflowClient

from mlops.utils.sysutils import is_windows


class MlflowUtils:
    mlflow_client: MlflowClient = None

    @classmethod
    def init_mlflow_client(cls, tracking_uri: str, registry_uri: str):
        if (not cls.mlflow_client) or (
            cls.mlflow_client
            and (
                cls.mlflow_client._tracking_client.tracking_uri != tracking_uri
                or cls.mlflow_client._registry_uri != registry_uri
            )
        ):
            # reset the client for tracking and registry servers
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_registry_uri(registry_uri)
            cls.mlflow_client = MlflowClient(
                tracking_uri=tracking_uri, registry_uri=registry_uri
            )

    @staticmethod
    def print_experiment_info(experiment):
        print("Name: {}".format(experiment.name))
        print("Experiment Id: {}".format(experiment.experiment_id))
        print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

    @classmethod
    def _get_run(cls, run_id: str):
        return cls.mlflow_client.get_run(run_id)

    @classmethod
    def get_parameters(cls, run_id: str):
        return cls._get_run(run_id).data.params

    @classmethod
    def get_parameter(cls, run_id: str, param_name: str, default=None):
        return cls.get_parameters(run_id).get(param_name, default)

    @classmethod
    def get_tags(cls, run_id: str):
        return cls._get_run(run_id).data.tags

    @classmethod
    def get_tag(cls, run_id: str, tag_name: str, default=None):
        return cls.get_tags(run_id).get(tag_name, default)

    @classmethod
    def get_metrics(cls, run_id: str):
        return cls._get_run(run_id).data.metrics

    @classmethod
    def get_metric(cls, run_id: str, metric_name: str, default=None):
        return cls.get_metrics(run_id).get(metric_name, default)

    @staticmethod
    def _path(file_path):
        if is_windows():
            return file_path[8:]  # remove appendix 'file:///' appearing in path
        return file_path

    @classmethod
    def get_artifact_path(cls, run_id: str, artifact_file: str):
        run = cls._get_run(run_id)
        artifact_path = cls._path(os.path.join(run.info.artifact_uri, artifact_file))
        return artifact_path

    class ArtifactFileObj(object):
        """Context manager for artifact file object"""

        def __init__(self, run_id: str, artifact_file: str) -> None:
            artifact_path = MlflowUtils.get_artifact_path(run_id, artifact_file)
            self.artifact_file_obj = open(artifact_path)

        def __enter__(self):
            return self.artifact_file_obj

        def __exit__(self, type, value, traceback):
            self.artifact_file_obj.close()

    @classmethod
    def load_dict(cls, run_id: str, artifact_file: str):
        assert (
            artifact_file.split(".")[-1] == "json"
        ), "MlflowUtils.load_dict expects artifact_file in json format with file extention '.json'."
        with cls.ArtifactFileObj(run_id, artifact_file) as dict_f:
            data = json.load(dict_f)
        return data

    @classmethod
    def add_run_note(cls, run_id: str, note: str):
        cls.mlflow_client.set_tag(run_id, "mlflow.note.content", note)

    @classmethod
    def get_exp_id(cls, exp_name: str):
        return cls.mlflow_client.get_experiment_by_name(exp_name).experiment_id

    @classmethod
    def get_run_name(cls, run_id: str):
        run = MlflowUtils.mlflow_client.get_run(run_id=run_id)
        return run.data.tags["mlflow.runName"]

    @classmethod
    def close_active_runs(cls):
        while mlflow.active_run():
            mlflow.end_run()

    @classmethod
    def download_artifact(
        cls,
        run_id: str,
        artifact_name: str,
        output_path=None,
    ):
        artifact_uri = cls.get_artifact_path(run_id, artifact_name)
        return mlflow.tracking.artifact_utils._download_artifact_from_uri(
            artifact_uri, output_path=output_path
        )

    @classmethod
    def get_latest_versions(cls, name, stages):
        return cls.mlflow_client.get_latest_versions(name, stages)