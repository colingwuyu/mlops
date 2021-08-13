import cloudpickle

from mlops.utils.mlflowutils import MlflowUtils
from examples.iris.feature_transform import OPS_NAME, ARTIFACT_SCALER


def _assert_ops_type(run_id: str):
    assert MlflowUtils.get_run_name(run_id) == OPS_NAME


def load_scaler(run_id: str):
    _assert_ops_type(run_id)
    scaler_art = MlflowUtils.get_artifact_path(run_id, ARTIFACT_SCALER)
    with open(scaler_art, "rb") as scaler_io:
        scaler = cloudpickle.load(scaler_io)
    return scaler
