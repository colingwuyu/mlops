"""common data types for orchestration."""
import os
import sys
from typing import Dict, List
from copy import deepcopy
import contextlib

import mlflow

from mlops.utils.strutils import pretty_dict, pad_tab
from mlops.utils.mlflowutils import MlflowUtils


class ComponentSpec(object):
    """ComponentSpec contains specifications for executing component

    Attributes:
        run_id: mlflow run id for finished component (None if not executed)
        args: arguments for component
        component_module: module for the component
    """

    def __init__(
        self,
        name: str,
        args: dict = {},
        module_dir: str = None,
        module_file: str = None,
        module: str = None,
        run_id: str = None,
        upstreams: List[str] = None,
        pipeline_init: bool = False,
        pipeline_end: bool = False,
    ):
        self.name = name
        self.args = args
        self.module_dir = module_dir
        self.module_file = module_file
        self.module = module
        self.run_id = run_id
        self.upstreams = upstreams
        self.pipeline_init = pipeline_init
        self.pipeline_end = pipeline_end

    @property
    def module_full_path(self):
        if self.module_file:
            return os.path.join(self.module_dir, self.module_file)
        else:
            return None

    @contextlib.contextmanager
    def include_module(self):
        if self.module_dir and (self.module_dir not in sys.path):
            sys.path.insert(0, self.module_dir)
        yield
        if self.module_dir and (self.module_dir in sys.path):
            sys.path.pop(sys.path.index(self.module_dir))

    def __repr__(self) -> str:
        short_args = deepcopy(self.args)
        if "mlflow_info" in short_args:
            short_args.pop("mlflow_info")
        if "upstream_ids" in short_args:
            short_args.pop("upstream_ids")
        if "mlflow_run" in short_args:
            short_args.pop("mlflow_run")
        if self.module:
            module_repr = f"module: {self.module}\n\t"
        elif self.module_file:
            module_repr = f"module_file: {self.module_file}\n\t"
        return (
            f"ComponentSpec(\n\tname: {self.name}\n\t"
            + module_repr
            + f"run_id: {self.run_id}\n\targs:\n{pad_tab(pretty_dict(short_args))}\n\tupstreams: {self.upstreams}\n)"
        )

    def _serialize(self) -> Dict:
        component_dict = dict()
        short_args = deepcopy(self.args)
        if "mlflow_info" in short_args:
            short_args.pop("mlflow_info")
        if "upstream_ids" in short_args:
            short_args.pop("upstream_ids")
        if "mlflow_run" in short_args:
            short_args.pop("mlflow_run")

        if self.pipeline_init:
            component_dict["pipeline_init"] = True
        if self.pipeline_end:
            component_dict["pipeline_end"] = True
        if self.module_file:
            component_dict["module_file"] = self.module_file
        if self.module:
            component_dict["module"] = self.module
        if short_args:
            component_dict["args"] = dict(short_args)
        if self.upstreams:
            component_dict["upstreams"] = self.upstreams
        if self.run_id:
            component_dict["run_id"] = self.run_id
        return component_dict


class MLFlowInfo(object):
    """PipelineInfo contains information of pipeline

    Attributes:
        name: pipeline name
        mlflow_tracking_uri: mlflow tracking uri for the pipeline execution
        mlflow_registry_uri: mlflow registry uri for model registry
        mlflow_exp_id: mlflow experiment name for the pipeline execution
    """

    def __init__(
        self,
        mlflow_tracking_uri: str,
        mlflow_registry_uri: str,
        mlflow_exp_id: str,
        name: str = None,
        mlflow_run_id: str = None,
        mlflow_tags: dict = None,
    ) -> None:
        self.name = name
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_registry_uri = mlflow_registry_uri
        self.mlflow_exp_id = mlflow_exp_id
        self.mlflow_run_id = mlflow_run_id
        MlflowUtils.init_mlflow_client(
            self.mlflow_tracking_uri, self.mlflow_registry_uri
        )
        if mlflow_run_id is not None:
            self.name = MlflowUtils.get_run_name(mlflow_run_id)
        self.mlflow_tags = mlflow_tags

    def __repr__(self) -> str:
        return "Pipeline(\n\tname: %s, \n\tmlflow_uri: %s, \n\tmlflow_experiment_name: %s, \n\tmlflow_runid: %s)" % (
            self.name,
            self.mlflow_tracking_uri,
            self.mlflow_exp_id,
            self.mlflow_run_id,
        )

    # def init_mlflow_run(self):
    #     mlflow_run = mlflow.start_run(
    #         experiment_id=MlflowUtils.get_exp_id(self.mlflow_exp_id), run_name=self.name
    #     )
    #     self.mlflow_run_id = mlflow_run.info.run_id
    #     mlflow.end_run()

    def _serialize(self):
        mlflow_dict = dict()
        mlflow_dict["exp_id"] = self.mlflow_exp_id
        mlflow_dict["tracking_uri"] = self.mlflow_tracking_uri
        mlflow_dict["registry_uri"] = self.mlflow_registry_uri
        return mlflow_dict
