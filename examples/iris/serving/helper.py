import os

from mlops.config import Config as conf
import mlops.serving.model as mlops_model
from mlops.utils.mlflowutils import MlflowUtils

conf.load("config_{}.yaml".format(os.environ.get("IRIS_ENV", "local")))


def retrain():
    from mlops.orchestrators.pipeline import Pipeline
    from mlops.orchestrators.local.local_dag_runner import LocalDagRunner

    pipeline = Pipeline(
        name=conf.settings.model.name,
        components=conf.settings.model.ct_pipeline,
        mlflow_conf=conf.settings.mlflow,
    )
    LocalDagRunner().run(pipeline)


def load_model(stage: str):
    MlflowUtils.init_mlflow_client(
        conf.settings.mlflow.tracking_uri, conf.settings.mlflow.registry_uri
    )

    if not MlflowUtils.mlflow_client.search_registered_models(
        f"name='{conf.settings.serving.model_name}'"
    ):
        return None
    return mlops_model.load_model(f"models:/{conf.settings.serving.model_name}/{stage}")
