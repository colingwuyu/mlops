import pandas as pd
import numpy as np

from mlops.utils.mlflowutils import MlflowUtils

from examples.iris.data_gen import (
    OPS_NAME,
    ARTIFACT_TRAIN_X,
    ARTIFACT_TRAIN_Y,
    ARTIFACT_TEST_X,
    ARTIFACT_TEST_Y,
    PARAM_X_HEADERS,
    PARAM_Y_HEADER,
)


def _assert_ops_type(run_id: str):
    assert MlflowUtils.get_run_name(run_id=run_id) == OPS_NAME


def load_train_X(run_id: str) -> pd.DataFrame:
    _assert_ops_type(run_id)
    train_X = pd.read_csv(MlflowUtils.get_artifact_path(run_id, ARTIFACT_TRAIN_X))
    return train_X


def load_train_y(run_id: str) -> pd.DataFrame:
    _assert_ops_type(run_id)
    train_y = pd.read_csv(MlflowUtils.get_artifact_path(run_id, ARTIFACT_TRAIN_Y))
    return train_y


def load_test_X(run_id: str) -> pd.DataFrame:
    _assert_ops_type(run_id)
    test_X = pd.read_csv(MlflowUtils.get_artifact_path(run_id, ARTIFACT_TEST_X))
    return test_X


def load_test_y(run_id: str) -> pd.DataFrame:
    _assert_ops_type(run_id)
    test_y = pd.read_csv(MlflowUtils.get_artifact_path(run_id, ARTIFACT_TEST_Y))
    return test_y


def get_label_header(run_id: str):
    _assert_ops_type(run_id)
    return MlflowUtils.get_parameter(run_id, PARAM_Y_HEADER)


def get_feature_header(run_id: str):
    _assert_ops_type(run_id)
    return eval(MlflowUtils.get_parameter(run_id, PARAM_X_HEADERS))
