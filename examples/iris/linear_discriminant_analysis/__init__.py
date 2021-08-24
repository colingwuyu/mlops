import os

import mlflow
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from mlops.estimator.skestimator import SkEstimator
from mlops.components import model_training_component
from mlops.serving import model as mlops_model
from mlops.serving.model import (
    MODEL_COMP_SCALER,
    MODEL_COMP_SCHEMA,
    MODEL_COMP_STATS,
    MODEL_COMP_MODEL,
)
from examples.iris.data_gen import helper as dg_helper, OPS_NAME as DATA_GEN_OPS_NAME
from examples.iris.data_validation import (
    helper as dv_helper,
    OPS_NAME as DATA_VAL_OPS_NAME,
)
from examples.iris.feature_transform import (
    helper as ft_helper,
    OPS_NAME as FEATURE_TRANSFORM_OPS_NAME,
)

OPS_NAME = "linear_discriminant_analysis"
OPS_DES = """
# Linear Discriminant Analysis
This is LDA model training operation.
## Task type
- classification
## Upstream dependencies
- Data Gen
- Data Validation
- Feature Transformation
## Parameter
None
## Metrics
- test_accuracy_score
## Artifacts
1. mlops_model: Mlflow Model 
## Helper functions
- `load_model(run_id: str)`
"""

LEARNING_FRAMEWORK = "sklearn"
METRIC_TEST_ACCURACY = "test_accuracy_score"
ARTIFACT_MODEL = "mlops_model"


@model_training_component(name=OPS_NAME, note=OPS_DES, framework=LEARNING_FRAMEWORK)
def run_func(upstream_ids: dict, **kwargs):
    model_comps = {}

    data_gen_id = upstream_ids[DATA_GEN_OPS_NAME]
    X_train = dg_helper.load_train_X(data_gen_id)
    y_train = dg_helper.load_train_y(data_gen_id)
    X_test = dg_helper.load_test_X(data_gen_id)
    y_test = dg_helper.load_test_y(data_gen_id)

    data_val_id = upstream_ids[DATA_VAL_OPS_NAME]
    model_comps[MODEL_COMP_SCHEMA] = dv_helper.get_schema(data_val_id)
    model_comps[MODEL_COMP_STATS] = dv_helper.get_trainset_stat(data_val_id)

    feature_scaler_id = upstream_ids[FEATURE_TRANSFORM_OPS_NAME]
    feature_scaler = ft_helper.load_scaler(feature_scaler_id)
    model_comps[MODEL_COMP_SCALER] = feature_scaler

    X_train = feature_scaler.transform(X_train)
    X_test = feature_scaler.transform(X_test)

    lda = LinearDiscriminantAnalysis()
    clf = lda.fit(X_train, y_train)
    model_estimator = SkEstimator()
    model_estimator.set_model(clf)
    model_comps[MODEL_COMP_MODEL] = model_estimator

    mean_acc = clf.score(X_test, y_test)
    mlflow.log_metric(METRIC_TEST_ACCURACY, mean_acc)

    conda_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../conda.yaml"
    )
    mlops_model.log_model(ARTIFACT_MODEL, model_comps, conda_env=conda_file)
