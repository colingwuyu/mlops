import mlflow

from mlops.orchestrators.datatype import MLFlowInfo
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
                "mlflow_info" in kwargs.keys(),
                "ERROR: argument 'pipeline' is required for a base_component.",
            )
            mlflow_info: MLFlowInfo = kwargs["mlflow_info"]
            MlflowUtils.init_mlflow_client(
                mlflow_info.mlflow_tracking_uri, mlflow_info.mlflow_registry_uri
            )
            if not mlflow.active_run():
                mlflow.start_run(run_id=mlflow_info.mlflow_run_id)
            else:
                assert mlflow.active_run().info.run_id == mlflow_info.mlflow_run_id
            with mlflow.start_run(
                experiment_id=MlflowUtils.get_exp_id(mlflow_info.mlflow_exp_id),
                run_name=name,
                nested=True,
                tags=mlflow_info.mlflow_tags,
            ) as active_run:
                component_mlflow_run_id = active_run.info.run_id
                if note:
                    MlflowUtils.add_run_note(component_mlflow_run_id, note)
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
                    elif arg_name != "mlflow_info":
                        mlflow.log_param(arg_name, arg_value)
                kwargs.update({ARG_MLFLOW_RUN_ID: active_run})
                run_func(*args, **kwargs)
            return mlflow_info.mlflow_run_id, component_mlflow_run_id

        return inner_mlflow_wrapper

    return inner_base_component
