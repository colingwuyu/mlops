from mlops.serving import model as mlops_model
from mlops.utils.mlflowutils import MlflowUtils
from examples.iris.logistic_regression import OPS_NAME, ARTIFACT_MODEL


def _assert_ops_type(run_id: str):
    assert MlflowUtils.get_run_name(run_id) == OPS_NAME


def load_model(run_id: str):
    _assert_ops_type(run_id)
    return mlops_model.load_model(f"runs:/{run_id}/{ARTIFACT_MODEL}")
