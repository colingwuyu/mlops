import tensorflow_data_validation as tfdv
from IPython.display import Image, display

from mlops.utils.mlflowutils import MlflowUtils

from examples.iris.data_validation import (
    OPS_NAME,
    ARTIFACT_SCHEMA,
    ARTIFACT_TRAINSET_STATS,
    ARTIFACT_PLOT,
)


def _assert_ops_type(run_id: str):
    assert MlflowUtils.get_run_name(run_id=run_id) == OPS_NAME


def get_schema(run_id: str):
    _assert_ops_type(run_id)
    schema_txt = MlflowUtils.get_artifact_path(run_id, ARTIFACT_SCHEMA)
    return tfdv.load_schema_text(schema_txt)


def display_schema(run_id: str):
    schema = get_schema(run_id)
    tfdv.display_schema(schema)


def get_trainset_stat(run_id: str):
    _assert_ops_type(run_id)
    trainset_stat_txt = MlflowUtils.get_artifact_path(run_id, ARTIFACT_TRAINSET_STATS)
    return tfdv.load_stats_text(trainset_stat_txt)


def display_trainset_stat(run_id: str):
    train_stat = get_trainset_stat(run_id)
    tfdv.visualize_statistics(train_stat, lhs_name="TRAINING DATA STAT")


def display_scatter_plot(run_id: str):
    _assert_ops_type(run_id)
    plot_path = MlflowUtils.get_artifact_path(run_id, ARTIFACT_PLOT)
    display(Image(plot_path))
