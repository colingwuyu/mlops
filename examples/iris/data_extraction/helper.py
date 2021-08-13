import pandas as pd

from mlops.utils.mlflowutils import MlflowUtils


def _assert_ops_type(run_id: str):
    assert MlflowUtils.get_run_name(run_id=run_id) == "url_data_extraction"


def load_raw_data(run_id: str):
    _assert_ops_type(run_id)
    raw_data_file = MlflowUtils.get_artifact_path(run_id, "raw_data.csv")
    return pd.read_csv(raw_data_file)
