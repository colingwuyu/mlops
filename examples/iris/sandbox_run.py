import os

import mlflow

from mlops.config import Config as conf
from mlops.utils.mlflowutils import MlflowUtils


def retrain():
    from mlops.orchestrators.pipeline import Pipeline
    from mlops.orchestrators.local.local_dag_runner import LocalDagRunner

    pipeline = Pipeline.load(conf.settings)
    LocalDagRunner().run(pipeline)


def load_model(stage: str):
    MlflowUtils.init_mlflow_client(
        conf.settings.mlflow.tracking_uri, conf.settings.mlflkow.registry_uri
    )
    return mlflow.pyfunc.load_model(
        model_uri=f"models:/{conf.settings.serving.model_name}/{stage}"
    )


if __name__ == "__main__":
    conf.load(config_from="examples/iris/sandbox_config_{}.yaml".format(
        os.environ.get("IRIS_ENV", "local")))
    retrain()
