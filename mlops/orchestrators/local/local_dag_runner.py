import time
import importlib
import importlib.util

from mlops.orchestrators import pipeline as pipeline_module, datatype
from mlops.utils.mlflowutils import MlflowUtils


class LocalDagRunner:
    """Local DAG runner."""

    def __init__(self):
        """Initializes LocalOrchestrator."""
        pass

    def _exec_ops(self, ops_name: str, operators: dict):
        """execute the operation with specific module name

        Args:
            ops_name (str): ops name to be executed
        """
        ops_spec: datatype.ComponentSpec = operators[ops_name]

        # execute all upstreams first
        upstream_run_ids = {}
        for ops_step_name in ops_spec.upstreams:
            if not operators[ops_step_name].run_id:
                self._exec_ops(ops_step_name, operators)
            upstream_run_ids.update({ops_step_name: operators[ops_step_name].run_id})
        ops_spec.args.update({"upstream_ids": upstream_run_ids})

        # execute current operator
        with ops_spec.include_module():
            if ops_spec.module_file:
                spec = importlib.util.spec_from_file_location(
                    ops_spec.name, ops_spec.module_full_path
                )
                ops_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ops_module)
            elif ops_spec.module:
                ops_module = importlib.import_module(ops_spec.module)

            start = time.time()
            print("\nRunning operation %s..." % (ops_name))
            if ops_spec.run_id is not None:
                print("\nOperation %s has been executed already..." % ops_name)
                return ops_spec.run_id
            _, component_mlflow_run_id = ops_module.run_func(**ops_spec.args)
            end = time.time()
            ops_spec.run_id = component_mlflow_run_id
            print(
                "\nExecution of operator %s took %s seconds" % (ops_name, end - start)
            )

    def run(self, pipeline: pipeline_module.Pipeline) -> None:
        """Runs given logical pipeline locally.

        Args:
            pipeline (pipeline_py.Pipeline): Logical pipeline containing pipeline components.
        """
        try:
            pipeline.mlflow_info.init_mlflow_run()
            for step_ops_name in pipeline.operators.keys():
                self._exec_ops(step_ops_name, pipeline.operators)
            pipeline.log_mlflow()
        finally:
            MlflowUtils.close_active_runs()
