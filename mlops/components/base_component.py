import os

import mlflow

from mlops import consts
from mlops.orchestrators.datatype import MLFlowInfo
from mlops.utils.mlflowutils import MlflowUtils
from mlops.utils.functionutils import lazy_property


def log_component_desc(component_filename):
    component_desc_file = os.path.join(os.path.dirname(
        os.path.abspath(component_filename)), "README.md")

    if os.path.exists(component_desc_file):
        with open(component_desc_file, 'r') as component_desc_io:
            component_desc = component_desc_io.read()
        mlflow.set_tag("mlflow.note.content", component_desc)
        if "Task type" in component_desc:
            # add task type tag
            note_lines = component_desc.split("\n")
            n = 0
            task_type = []
            while True:
                if "Task type" in note_lines[n]:
                    n += 1
                    while "##" not in note_lines[n]:
                        task_type.append(
                            "-".join(
                                note_lines[n].split("-")[1:]
                            ).strip()
                        )
                        n += 1
                    break
                n += 1
            mlflow.set_tags({"task.type": ",".join(task_type)})


def base_component(run_func):
    def inner_mlflow_wrapper(*args, **kwargs):
        import inspect

        frm = inspect.stack()[1]
        mlflow_info: MLFlowInfo = kwargs["mlflow_info"]
        mlflow_run_id = kwargs[consts.ARG_MLFLOW_RUN]
        MlflowUtils.init_mlflow_client(
            mlflow_info.mlflow_tracking_uri, mlflow_info.mlflow_registry_uri
        )
        # if not mlflow.active_run():
        #     mlflow.start_run(run_id=mlflow_info.mlflow_run_id)
        # else:
        #     assert mlflow.active_run().info.run_id == mlflow_info.mlflow_run_id
        with mlflow.start_run(run_id=mlflow_run_id):
            log_component_desc(frm.filename)
            run_func(*args, **kwargs)
            return mlflow_info.mlflow_run_id, mlflow_run_id

    return inner_mlflow_wrapper


class Artifacts(dict):
    ...


class BaseComponent:
    def __init__(self) -> None:
        ...

    @lazy_property
    def layzy_prop(self):
        return 100
