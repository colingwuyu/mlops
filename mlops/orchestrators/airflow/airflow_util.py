from typing import Text
import os

from mlops.utils.strutils import var_to_str
from mlops.orchestrators.datatype import ComponentSpec


def create_entry_point(
    mlops_component_spec: ComponentSpec, entry_point_module_dir: Text
):
    set_args = "args=dict()\n"
    for arg_name, arg_value in mlops_component_spec.args.items():
        if arg_name == "mlflow_info":
            set_args += f"""
args[{var_to_str(arg_name)}] = MLFlowInfo(
    name={var_to_str(arg_value.name)},
    mlflow_tracking_uri={var_to_str(arg_value.mlflow_tracking_uri)},
    mlflow_registry_uri={var_to_str(arg_value.mlflow_registry_uri)},
    mlflow_exp_id={var_to_str(arg_value.mlflow_exp_id)},
    mlflow_run_id={var_to_str(arg_value.mlflow_run_id)},
    mlflow_tags={var_to_str(arg_value.mlflow_tags)}
)\n
            """
        elif isinstance(arg_value, str):
            set_args += f"""args[{var_to_str(arg_name)}]={var_to_str(arg_value)}\n"""
        else:
            set_args += f"""args[{var_to_str(arg_name)}]={var_to_str(arg_value)}\n"""
    entry_point_script = f"""
import importlib.util
import sys
import os

import redis

from mlops.orchestrators.datatype import ComponentSpec, MLFlowInfo

r = redis.Redis.from_url(os.environ.get("REDIS_URL"), db=os.environ.get("REDIS_DB"))

{set_args}

upstreams = {var_to_str(mlops_component_spec.upstreams)}
upstream_run_ids = dict()

pipeline_mlflow_runid = r.hget("mlflow_runids", "pipeline_mlflow_runid") 
if pipeline_mlflow_runid is not None:
    args["mlflow_info"].mlflow_run_id = pipeline_mlflow_runid.decode("utf-8")


if upstreams:
    upstream_redis_runids = r.hmget("mlflow_runids", upstreams)
    for i, upstream_runid in enumerate(upstream_redis_runids):
        if upstream_runid is None:
            print("ERROR: Upstream - " + upstreams[i] + " mlflow run id is not available.")
            sys.exist(1)
        upstream_run_ids[upstreams[i]] = upstream_runid.decode("utf-8")

args["upstream_ids"] = upstream_run_ids


cur_component_spec = ComponentSpec(
                name={var_to_str(mlops_component_spec.name)},
                args=args,
                module_dir={var_to_str(mlops_component_spec.module_dir)},
                module_file={var_to_str(mlops_component_spec.module_file)},
                module={var_to_str(mlops_component_spec.module)},
                run_id={var_to_str(mlops_component_spec.run_id)},
                upstreams=upstreams,
                pipeline_init={var_to_str(mlops_component_spec.pipeline_init)},
                pipeline_end={var_to_str(mlops_component_spec.pipeline_end)},
            )

with cur_component_spec.include_module():
    if {var_to_str(mlops_component_spec.module_file)} is not None:
        spec = importlib.util.spec_from_file_location("mlops_component", {var_to_str(mlops_component_spec.module_full_path)})
        ops_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ops_module)
    elif {var_to_str(mlops_component_spec.module)} is not None:
        ops_module = importlib.import_module({var_to_str(mlops_component_spec.module)})

    pipeline_mlflow_runid, component_mlfow_runid = ops_module.run_func(**args)
    r.hset("mlflow_runids", {var_to_str(mlops_component_spec.name)}, component_mlfow_runid)

"""
    with open(
        os.path.join(entry_point_module_dir, f"{mlops_component_spec.name}_main.py"),
        "w",
    ) as entry_point_module_file:
        entry_point_module_file.write(entry_point_script)
    return f"{mlops_component_spec.name}_main.py"
