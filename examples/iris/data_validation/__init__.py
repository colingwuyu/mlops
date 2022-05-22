import os
import shutil
import tempfile

import mlflow
import seaborn as sns
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils.display_util import get_statistics_html


from mlops.components import base_component
from examples.iris.data_gen import helper as dg_helper
from examples.iris.data_gen import OPS_NAME as DATA_GEN_OPS_NAME


OPS_NAME = "data_validation"

TRAINING_ENV = "TRAINING"
SERVING_ENV = "SERVING"

ARTIFACT_TRAINSET_STATS = "trainset_stats.txt"
ARTIFACT_TRAINSET_STATS_REPORT = "trainset_stats.html"
ARTIFACT_SCHEMA = "schema.txt"
ARTIFACT_PLOT = "scatter_plot.png"


@base_component
def run_func(upstream_ids: dict, **kwargs):
    data_gen_id = upstream_ids[DATA_GEN_OPS_NAME]

    X_train = dg_helper.load_train_X(data_gen_id)
    Y_train = dg_helper.load_train_y(data_gen_id)
    X_test = dg_helper.load_test_X(data_gen_id)
    Y_test = dg_helper.load_test_y(data_gen_id)
    label_header = dg_helper.get_label_header(data_gen_id)
    feature_header = dg_helper.get_feature_header(data_gen_id)

    dataset_train = X_train.join(Y_train)
    dataset_test = X_test.join(Y_test)
    whole_dataset = dataset_train.append(dataset_test)
    train_stats = tfdv.generate_statistics_from_dataframe(dataset_train)

    # infer schema
    schema = tfdv.infer_schema(statistics=train_stats)
    # All features are by default in both TRAINING and SERVING environments.
    schema.default_environment.append(TRAINING_ENV)
    schema.default_environment.append(SERVING_ENV)

    for feature in feature_header:
        feature_type = tfdv.get_feature(schema, feature)
        feature_type.skew_comparator.jensen_shannon_divergence.threshold = 0.15
        feature_type.drift_comparator.jensen_shannon_divergence.threshold = 0.15

    label_type = tfdv.get_feature(schema, label_header)
    label_type.skew_comparator.infinity_norm.threshold = 0.15
    label_type.skew_comparator.infinity_norm.threshold = 0.15

    # Specify that label feature (Y) is not in SERVING environment.
    tfdv.get_feature(
        schema, label_header).not_in_environment.append(SERVING_ENV)

    artifact_dir = tempfile.mkdtemp()
    stat_file = os.path.join(artifact_dir, ARTIFACT_TRAINSET_STATS)
    tfdv.write_stats_text(train_stats, stat_file)
    schema_file = os.path.join(artifact_dir, ARTIFACT_SCHEMA)
    tfdv.write_schema_text(schema, schema_file)

    sns.set_theme(style="ticks")
    sns_plot = sns.pairplot(whole_dataset, hue=label_header)
    sns_plot.savefig(os.path.join(artifact_dir, ARTIFACT_PLOT))

    with open(os.path.join(artifact_dir, ARTIFACT_TRAINSET_STATS_REPORT), "w") as f:
        f.write(get_statistics_html(train_stats, lhs_name="TRAIN_DATASET"))

    mlflow.log_artifacts(artifact_dir)
    shutil.rmtree(artifact_dir)
