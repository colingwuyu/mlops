from os import pipe

import mlflow

from mlops.orchetrators.datatype import PipelineInfo
from mlops.utils.mlflowutils import MlflowUtils

ARG_MLFLOW_RUN_ID = "mlflow_run"


def base_component(name: str = None, note: str = None):
    if not name:
        import inspect

        frm = inspect.stack()[1]
        mod = inspect.getmodule(frm[0])
        if mod:
            name = mod.__name__
        else:
            name = frm.filename + ":" + frm.function

    def inner_base_component(run_func):
        def inner_mlflow_wrapper(*args, **kwargs):
            assert (
                "pipeline_info" in kwargs.keys(),
                "ERROR: argument 'pipeline' is required for a base_component.",
            )
            pipeline_info: PipelineInfo = kwargs["pipeline_info"]
            MlflowUtils.init_mlflow_client(
                pipeline_info.mlflow_tracking_uri, pipeline_info.model_registry_uri
            )
            if not mlflow.active_run():
                mlflow.start_run(run_id=pipeline_info.mlflow_run_id)
            with mlflow.start_run(
                experiment_id=MlflowUtils.get_exp_id(pipeline_info.mlflow_exp_id),
                run_name=name,
                nested=True,
            ) as active_run:
                active_run_id = active_run.info.run_id
                if note:
                    MlflowUtils.add_run_note(active_run_id, note)
                    if "Task type" in note:
                        # add task type tag
                        note_lines = note.split("\n")
                        n = 0
                        task_type = []
                        while True:
                            if "Task type" in note_lines[n]:
                                n += 1
                                while "##" not in note_lines[n]:
                                    task_type.append(
                                        "-".join(note_lines[n].split("-")[1:]).strip()
                                    )
                                    n += 1
                                break
                            n += 1
                        mlflow.set_tags({"task.type": ",".join(task_type)})
                for arg_name, arg_value in kwargs.items():
                    if arg_name == "upstream_ids":
                        mlflow.log_params(kwargs["upstream_ids"])
                    elif arg_name != "pipeline_info":
                        mlflow.log_param(arg_name, arg_value)
                kwargs.update({ARG_MLFLOW_RUN_ID: active_run})
                run_func(*args, **kwargs)
            return pipeline_info.mlflow_run_id, active_run_id

        return inner_mlflow_wrapper

    return inner_base_component
