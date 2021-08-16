import os

import mlflow

from mlops.config import Config as conf
from mlops.utils.mlflowutils import MlflowUtils


def retrain():
    from mlops.orchetrators.pipeline import Pipeline
    from mlops.orchetrators.local_dag_runner import LocalDagRunner

    pipeline = Pipeline(
        pipeline_name=conf.settings.model.name,
        pipeline_def=conf.settings.model.ct_pipeline,
        mlflow_conf=conf.settings.mlflow,
    )
    LocalDagRunner().run(pipeline)


def load_model(stage: str):
    MlflowUtils.init_mlflow_client(
        conf.settings.mlflow.tracking_uri, conf.settings.mlflkow.registry_uri
    )
    return mlflow.pyfunc.load_model(
        model_uri=f"models:/{conf.settings.serving.model_name}/{stage}"
    )


if __name__ == "__main__":
    retrain()
