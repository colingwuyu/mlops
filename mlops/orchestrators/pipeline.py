from collections import namedtuple
import copy
from typing import Union, Dict, Text

import mlflow
from mlflow.tracking.context.git_context import _get_git_commit
from mlflow.projects.utils import (
    _fetch_project,
    _get_git_repo_url,
    _is_valid_branch_name,
    _is_local_uri,
)
from mlflow.utils.mlflow_tags import (
    MLFLOW_GIT_REPO_URL,
    MLFLOW_GIT_COMMIT,
    MLFLOW_GIT_BRANCH,
)

from mlops import consts
from mlops.orchestrators import datatype
from mlops.utils.collectionutils import convert_to_namedtuple
from mlops.utils.strutils import pad_tab
from mlops.utils.mlflowutils import MlflowUtils


ARTIFACT_PIPELINE = "pipeline.yml"


class Pipeline:
    """
    Pipeline defined in dictionary or namedtuple

    Example (in yaml format):

    name: "IRIS Training Pipeline"
    uri: https://github.com/colingwuyu/mlops.git
    version: c833597f1676099c008dad3ca6aa311e9681f9ff
    components:
        data_gen:
            module_file: examples/iris/data_gen/__init__.py
            args:
                url: "http://iris_serving:3000/data"
                x_headers:
                - sepal-length
                - sepal-width
                - petal-length
                - petal-width
                y_header: species
                test_size: 0.2
        data_validation:
            module_file: examples/iris/data_validation/__init__.py
            upstreams:
                - data_gen
        feature_transform:
            module_file: examples/iris/feature_transform/__init__.py
            args:
                transform_approach: min-max
            upstreams:
                - data_gen
        logistic_regression:
            module_file: examples/iris/logistic_regression/__init__.py
            upstreams:
                - data_gen
                - data_validation
                - feature_transform
        linear_discriminant_analysis:
            module_file: linear_discriminant_analysis/__init__.py
            upstreams:
                - data_gen
                - data_validation
                - feature_transform
        model_eval:
            module_file: examples/iris/model_eval/__init__.py
            upstreams:
                - data_gen
                - logistic_regression
                - linear_discriminant_analysis
    mlflow:
        exp_id: "Sandbox"
        tracking_uri: http://mlflow_server:5000
        registry_uri: http://mlflow_server:5000
    """

    def __init__(
        self,
        name: str,
        uri: str,
        components: Union[dict, namedtuple],
        version: str = None,
        mlflow_conf: Union[dict, namedtuple] = None,
        run_id: str = None,
    ):
        if isinstance(components, dict):
            components = convert_to_namedtuple(components)
        if isinstance(mlflow_conf, dict):
            mlflow_conf = convert_to_namedtuple(mlflow_conf)

        self.name = name
        self.operators: Dict[Text, datatype.ComponentSpec] = {}
        self.uri = uri
        self.version = version

        # fetch codes
        module_dir = _fetch_project(uri, version)
        mlflow_tags = {}
        if not _is_local_uri(uri):
            source_version = _get_git_commit(module_dir)
            if source_version is not None:
                mlflow_tags[MLFLOW_GIT_COMMIT] = source_version
            repo_url = _get_git_repo_url(module_dir)
            if repo_url is not None:
                mlflow_tags[MLFLOW_GIT_REPO_URL] = repo_url
            if _is_valid_branch_name(module_dir, version):
                mlflow_tags[MLFLOW_GIT_BRANCH] = version

        self.mlflow_info = datatype.MLFlowInfo(
            name=self.name,
            mlflow_tracking_uri=mlflow_conf.tracking_uri,
            mlflow_registry_uri=mlflow_conf.registry_uri,
            mlflow_exp_id=mlflow_conf.exp_id,
            mlflow_tags=mlflow_tags,
        )
        if run_id:
            self.mlflow_info.mlflow_run_id = run_id
        # build component operators
        for component_name, component_val in components._asdict().items():
            component_val = component_val._asdict()
            component_args = component_val.get("args")
            cur_component_args = (
                copy.deepcopy(component_args._asdict()
                              ) if component_args else {}
            )
            cur_component_args["mlflow_info"] = self.mlflow_info
            cur_component_spec = datatype.ComponentSpec(
                name=component_name,
                args=cur_component_args,
                module_dir=module_dir,
                module_file=component_val.get("module_file"),
                module=component_val.get("module"),
                run_id=component_val.get("run_id"),
                upstreams=component_val.get("upstreams", []),
                pipeline_init=component_val.get("pipeline_init", False),
                pipeline_end=component_val.get("pipeline_end", False),
            )
            self.operators.update({component_name: cur_component_spec})

    def get_pipeline_init_component_spec(self):
        for component_spec in self.operators.values():
            if component_spec.pipeline_init:
                return component_spec

    @classmethod
    def load(cls, pipeline: Union[dict, namedtuple], run_id: str = None):
        if not isinstance(pipeline, dict):
            pipeline = pipeline._asdict()
        return cls(
            name=pipeline.get("name"),
            uri=pipeline.get("uri"),
            components=pipeline.get("components"),
            version=pipeline.get("version"),
            mlflow_conf=pipeline.get("mlflow"),
            run_id=run_id,
        )

    def __repr__(self) -> str:
        str_operators = "(\n"
        for ops_name, ops in self.operators.items():
            str_operators += "\t" + ops_name + \
                " :\n" + pad_tab(str(ops), 2) + "\n"
        str_operators += ")"
        return f"Pipeline(\n\tname: {self.name}\n\trun_id: {self.mlflow_info.mlflow_run_id}\n\toperators:\n{pad_tab(str_operators)})"

    @property
    def run_ids(self):
        return {k: {"run_id": v.run_id} for k, v in self.operators.items()}

    @property
    def run_id(self):
        return self.mlflow_info.mlflow_run_id

    def _serialize(self):
        pipeline_yml = dict()
        pipeline_yml["name"] = self.name
        pipeline_yml["uri"] = self.uri
        if self.version:
            pipeline_yml["version"] = self.version
        components = dict()
        for component_name, component_spec in self.operators.items():
            component_val = component_spec._serialize()
            components[component_name] = component_val
        pipeline_yml["components"] = components
        pipeline_yml["mlflow"] = self.mlflow_info._serialize()
        return pipeline_yml

    def init_mlflow_pipeline(self):
        exp_id = MlflowUtils.get_exp_id(self.mlflow_info.mlflow_exp_id)
        pipeline_run = mlflow.start_run(
            experiment_id=exp_id, run_name=self.name)
        self.mlflow_info.mlflow_run_id = pipeline_run.info.run_id
        for component_name, component_spec in self.operators.items():
            if component_spec.run_id is None:
                component_run = mlflow.start_run(
                    nested=True, experiment_id=exp_id, run_name=component_name)
                component_spec.run_id = component_spec.args[consts.ARG_MLFLOW_RUN] = component_run.info.run_id

                component_dict = component_spec._serialize()
                for component_attr_name, component_attr_val in component_dict.items():
                    if component_attr_name != "args":
                        mlflow.set_tag(
                            f"component.{component_attr_name}", component_attr_val)
                    else:
                        for arg_name, arg_val in component_attr_val.items():
                            mlflow.log_param(arg_name, arg_val)
                mlflow.end_run('SCHEDULED')
        mlflow.log_dict(
            self._serialize(), ARTIFACT_PIPELINE
        )
        mlflow.end_run('SCHEDULED')

    def end_mlflow_pipeline(self):
        mlflow.start_run(run_id=self.run_id)
        mlflow.end_run()

    # def log_mlflow(self):
    #     MlflowUtils.mlflow_client.log_dict(
    #         self.run_id, self._serialize(), ARTIFACT_PIPELINE
    #     )

    @classmethod
    def view(cls, mlflow_info: datatype.MLFlowInfo) -> "Pipeline":
        pipeline_dict = MlflowUtils.load_dict(
            mlflow_info.mlflow_run_id, ARTIFACT_PIPELINE
        )
        return cls.load(pipeline_dict, run_id=mlflow_info.mlflow_run_id)
