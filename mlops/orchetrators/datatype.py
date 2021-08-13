"""common data types for orchestration."""
from typing import List
from copy import deepcopy

from mlops.utils.strutils import pretty_dict, pad_tab


class ComponentSpec(object):
    """ComponentSpec contains specifications for executing component

    Attributes:
        run_id: mlflow run id for finished component (None if not executed)
        args: arguments for component
        component_module: module for the component
    """

    def __init__(
        self,
        args: dict = {},
        component_module: str = None,
        run_id: str = None,
        upstreams: List[str] = None,
    ):
        self.args = args
        self.component_module = component_module
        self.run_id = run_id
        self.upstreams = upstreams

    def __repr__(self) -> str:
        short_args = deepcopy(self.args)
        short_args.pop("pipeline_info")
        return f"ComponentSpec(\n\tcomponent_module: {self.component_module}\n\trun_id: {self.run_id}\n\targs:\n{pad_tab(pretty_dict(short_args))}\n\tupstreams: {self.upstreams}\n)"


class PipelineInfo(object):
    """PipelineInfo contains information of pipeline

    Attributes:
        name: pipeline name
        mlflow_tracking_uri: mlflow tracking uri for the pipeline execution
        mlflow_registry_uri: mlflow registry uri for model registry
        mlflow_exp_id: mlflow experiment name for the pipeline execution
    """

    def __init__(
        self,
        name: str,
        mlflow_tracking_uri: str,
        mlflow_registry_uri: str,
        mlflow_exp_id: str,
        mlflow_run_id: str = None,
    ) -> None:
        self.name = name
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.model_registry_uri = mlflow_registry_uri
        self.mlflow_exp_id = mlflow_exp_id
        self.mlflow_run_id = mlflow_run_id

    def __repr__(self) -> str:
        return "Pipeline(\n\tname: %s, \n\tmlflow_uri: %s, \n\tmlflow_experiment_name: %s, \n\tmlflow_runid: %s)" % (
            self.name,
            self.mlflow_tracking_uri,
            self.mlflow_exp_id,
            self.mlflow_run_id,
        )
