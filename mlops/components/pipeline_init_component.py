import mlflow

from mlops.components.base_component import base_component
from mlops.orchetrators.datatype import PipelineInfo
from mlops.utils.mlflowutils import MlflowUtils


def pipeline_init_component(name: str = None, note: str = None):
    def inner_init_component(run_func):
        def inner_mlflow_wrapper(*args, **kwargs):
            assert (
                "pipeline_info" in kwargs.keys(),
                "ERROR: argument 'pipeline' is required for a pipeline_init_component.",
            )
            pipeline_info: PipelineInfo = kwargs["pipeline_info"]
            MlflowUtils.init_mlflow_client(
                pipeline_info.mlflow_tracking_uri, pipeline_info.model_registry_uri
            )
            pipelinne_mlflow_run = mlflow.start_run(
                experiment_id=MlflowUtils.get_exp_id(pipeline_info.mlflow_exp_id),
                run_name=pipeline_info.name,
            )
            pipeline_info.mlflow_run_id = pipelinne_mlflow_run.info.run_id
            base_component_ops = base_component(name, note)(run_func)
            return base_component_ops(*args, **kwargs)

        return inner_mlflow_wrapper

    return inner_init_component
