from collections import namedtuple
import copy
import os
import tempfile
from typing import Union

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt

from mlops.orchestrators import datatype
from mlops.utils.collectionutils import convert_to_namedtuple
from mlops.utils.strutils import pad_tab


def _dag_level(graph, node):
    ancestors = nx.ancestors(graph, node)
    max_ancestor_level = -1
    for a in ancestors:
        max_ancestor_level = max(max_ancestor_level, _dag_level(graph, a))
    return max_ancestor_level + 1


class Pipeline:
    """
    Pipeline defined in dictionary or namedtuple

    Example (in yaml format):

    ct_pipeline:
        name: iris
        components:
            data_extraction:
                url: "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
            dataset_split:
                x_headers:
                - sepal-length
                - sepal-width
                - petal-length
                - petal-width
                y_header:
                - class
                test_size: 0.2
            data_validation:
            feature_transform:
            train_model:
            model_eval:
        dag: # component: downstream components separated by ','
            data_extraction:
                - dataset_split
                - data_validation
                - feature_transform
                - train_model
                - model_eval
            dataset_split:
                - data_validation
                - feature_transform
                - train_model
                - model_eval
            feature_transform:
                - train_model
                - model_eval
            train_model:
                - model_eval
    """

    def __init__(
        self,
        pipeline_name: str,
        pipeline_module_root: str,
        pipeline_def: Union[dict, namedtuple],
        mlflow_conf: Union[dict, namedtuple] = None,
    ):
        self.pipeline_name = pipeline_name
        self.oprators = {}
        self.mlflow_info = datatype.MLFlowInfo(
            name=self.pipeline_name,
            mlflow_tracking_uri=mlflow_conf.tracking_uri,
            mlflow_registry_uri=mlflow_conf.registry_uri,
            mlflow_exp_id=mlflow_conf.exp_id,
        )
        if isinstance(pipeline_def, dict):
            pipeline_def = convert_to_namedtuple(pipeline_def)
        if isinstance(mlflow_conf, dict):
            pipeline_def = convert_to_namedtuple(mlflow_conf)
        # build component operators
        for component_name, component_val in pipeline_def.components._asdict().items():
            module_file = os.path.join(
                pipeline_module_root, component_val.get("module_file")
            )
            component_args = component_val.get("args")
            cur_component_args = (
                copy.deepcopy(component_args._asdict()) if component_args else {}
            )
            cur_component_args["mlflow_info"] = self.mlflow_info
            cur_component_spec = datatype.ComponentSpec(
                name=component_name,
                args=cur_component_args,
                module_file=module_file,
                run_id=component_val.get("run_id"),
            )
            self.oprators.update({component_name: cur_component_spec})

        # build DAG
        self.dag = nx.DiGraph()
        for component_name, downstream_names in pipeline_def.dag._asdict().items():
            self.dag.add_node(component_name)
            if downstream_names:
                self.dag.add_edges_from(
                    [(component_name, ds_name) for ds_name in downstream_names]
                )
        for component_name, component_spec in self.oprators.items():
            component_spec.upstreams = list(self.dag.pred[component_name])

    def __repr__(self) -> str:
        str_operators = "(\n"
        for ops_name, ops in self.oprators.items():
            str_operators += "\t" + ops_name + " :\n" + pad_tab(str(ops), 2) + "\n"
        str_operators += ")"
        return f"Pipeline(\n\tname: {self.pipeline_name}\n\trun_id: {self.mlflow_info.mlflow_run_id}\n\toperators:\n{pad_tab(str_operators)})"

    def plot_dag(self):
        pos = {}
        levels = {}
        nodes = self.oprators.keys()
        for node in nodes:
            levels[node] = _dag_level(self.dag, node)
        levels = pd.Series(levels)
        heights = {}
        labels = {}
        for l in levels.unique():
            heights[l] = 0
        for node in nodes:
            l = levels[node]
            h = heights[l]
            heights[l] += 1
            pos[node] = (l * 5, h)
            labels[node] = ".".join(node.split(".")[1:])
        nx.draw_networkx_nodes(self.dag, pos)
        nx.draw_networkx_edges(self.dag, pos, connectionstyle="arc3,rad=0.3")
        nx.draw_networkx_labels(self.dag, pos, labels=labels)
        tmpd = tempfile.mkdtemp()
        dag_img = os.path.join(tmpd, "dag.png")
        plt.savefig(dag_img, format="PNG")
        return tmpd, dag_img

    def run_ids(self):
        return {k: {"run_id": v.run_id} for k, v in self.oprators.items()}
