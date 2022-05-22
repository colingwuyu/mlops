import redis
import os
import sys
import importlib.util
import yaml
import time

from mlops.orchestrators.datatype import ComponentSpec
from mlops.orchestrators import pipeline as pipeline_module
from mlops.utils.mlflowutils import MlflowUtils
from mlops.consts import ARG_MLFLOW_RUN


component_name = sys.argv[1]
print("Execute component " + component_name)

r = redis.Redis.from_url(os.environ.get("REDIS_URL"),
                         db=os.environ.get("REDIS_DB"))

pipeline = pipeline_module.Pipeline.load(
    yaml.load(r.get("pipeline").decode("utf-8")),
    r.get("pipeline_run_id").decode("utf-8")
)
component_spec: ComponentSpec = pipeline.operators[component_name]

upstream_run_ids = dict()
for component_name in component_spec.upstreams:
    upstream_run_ids[component_name] = pipeline.operators[component_name].run_id
component_spec.args["upstream_ids"] = upstream_run_ids
component_spec.args[ARG_MLFLOW_RUN] = component_spec.run_id


with component_spec.include_module():
    if component_spec.module_file:
        spec = importlib.util.spec_from_file_location(
            "mlops_component", component_spec.module_full_path)
        component_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(component_module)
    elif component_spec.module:
        component_module = importlib.import_module(component_spec.module)

    component_mlflow_run = MlflowUtils.mlflow_client.get_run(
        component_spec.run_id)
    start = time.time()
    print("\nRunning operation %s..." % (component_spec.name))
    if component_mlflow_run.info.status == 'FINISHED':
        print("\nOperation %s has been executed already..." %
              component_spec.name)
        sys.exit()
    _, component_mlflow_run_id = component_module.run_func(
        **component_spec.args)
    end = time.time()
    component_spec.run_id = component_mlflow_run_id
    print(
        "\nExecution of operator %s took %s seconds" % (
            component_spec.name, end - start)
    )
