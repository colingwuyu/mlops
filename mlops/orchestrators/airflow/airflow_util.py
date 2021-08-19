from typing import Text
import os

from mlops.orchestrators.datatype import ComponentSpec


def create_entry_point(
    mlops_component_spec: ComponentSpec, entry_point_module_dir: Text
):
    args = mlops_component_spec.args
    upstreams = mlops_component_spec.upstreams
    module_file = mlops_component_spec.module_file
    component_name = mlops_component_spec.name

    set_args = "args=dict()\n"
    for arg_name, arg_value in args.items():
        if arg_name == "mlflow_info":
            if arg_value.mlflow_run_id is None:
                mlflow_run_id = "None"
            else:
                mlflow_run_id = f"'{arg_value.mlflow_run_id}'"
            set_args += f"""
args['{arg_name}'] = MLFlowInfo(
    name='{arg_value.name}',
    mlflow_tracking_uri='{arg_value.mlflow_tracking_uri}',
    mlflow_registry_uri='{arg_value.mlflow_registry_uri}',
    mlflow_exp_id='{arg_value.mlflow_exp_id}',
    mlflow_run_id={mlflow_run_id},
    mlflow_tags={arg_value.mlflow_tags}
)\n
            """
        elif isinstance(arg_value, str):
            set_args += f"""args['{arg_name}']='{arg_value}'\n"""
        else:
            set_args += f"""args['{arg_name}']={arg_value}\n"""

    entry_point_script = f"""
import importlib.util
import sys
import os

import redis

from mlops.orchestrators.datatype import ComponentSpec, MLFlowInfo

r = redis.Redis.from_url(os.environ.get("REDIS_URL"), db=os.environ.get("REDIS_DB"))

{set_args}

upstreams = {upstreams}
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

spec = importlib.util.spec_from_file_location("mlops_component", '{module_file}')
ops_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops_module)
pipeline_mlflow_runid, component_mlfow_runid = ops_module.run_func(**args)
r.hset("mlflow_runids", '{component_name}', component_mlfow_runid)

"""
    with open(
        os.path.join(entry_point_module_dir, f"{component_name}_main.py"), "w"
    ) as entry_point_module_file:
        entry_point_module_file.write(entry_point_script)
    return f"{component_name}_main.py"
