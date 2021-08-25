import time
import importlib.util

import mlflow

from mlops.orchestrators import pipeline as pipeline_module, datatype
from mlops.utils.mlflowutils import MlflowUtils


class LocalDagRunner:
    """Local DAG runner."""

    def __init__(self):
        """Initializes LocalOrchestrator."""
        pass

    def _exec_init_run(self, mlflow_info: datatype.MLFlowInfo):
        MlflowUtils.init_mlflow_client(
            mlflow_info.mlflow_tracking_uri, mlflow_info.mlflow_registry_uri
        )
        pipelinne_mlflow_run = mlflow.start_run(
            experiment_id=MlflowUtils.get_exp_id(mlflow_info.mlflow_exp_id),
            run_name=mlflow_info.name,
        )
        mlflow_info.mlflow_run_id = pipelinne_mlflow_run.info.run_id
        mlflow.end_run()

    def _exec_ops(self, ops_name: str, operators: dict):
        """execute the operation with specific module name

        Args:
            ops_name (str): ops name to be executed
        """
        ops_spec: datatype.ComponentSpec = operators[ops_name]

        if ops_spec.pipeline_init:
            self._exec_init_run(ops_spec.args["mlflow_info"])
        # execute all upstreams first
        upstream_run_ids = {}
        for ops_step_name in ops_spec.upstreams:
            if not operators[ops_step_name].run_id:
                self._exec_ops(ops_step_name, operators)
            upstream_run_ids.update({ops_step_name: operators[ops_step_name].run_id})
        ops_spec.args.update({"upstream_ids": upstream_run_ids})

        # execute current operator
        spec = importlib.util.spec_from_file_location(
            ops_spec.name, ops_spec.module_file
        )
        ops_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ops_module)

        start = time.time()
        print("\nRunning operation %s..." % (ops_name))
        if ops_spec.run_id is not None:
            print("\nOperation %s has been executed already..." % ops_name)
            return ops_spec.run_id
        _, component_mlflow_run_id = ops_module.run_func(**ops_spec.args)
        end = time.time()
        ops_spec.run_id = component_mlflow_run_id
        print("\nExecution of operator %s took %s seconds" % (ops_name, end - start))

    def run(self, pipeline: pipeline_module.Pipeline) -> None:
        """Runs given logical pipeline locally.

        Args:
            pipeline (pipeline_py.Pipeline): Logical pipeline containing pipeline components.
        """
        try:
            for step_ops_name in pipeline.operators.keys():
                pipeline_mlflow_run_id = self._exec_ops(
                    step_ops_name, pipeline.operators
                )
            pipeline.run_id = pipeline_mlflow_run_id
        finally:
            MlflowUtils.close_active_runs()
