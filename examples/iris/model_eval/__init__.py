from mlops.model_analysis.utils import load_eval_result_text
import shutil
import tempfile
import os

import mlflow

from mlops.components import base_component
from mlops.utils.mlflowutils import MlflowUtils
import mlops.model_analysis as mlops_ma
import mlops.model_analysis.metrics as mlops_ma_metrics
from mlops.serving import model as mlops_model
import examples.iris.data_gen.helper as dg_helper
from examples.iris.data_gen import OPS_NAME as DATA_GEN_OPS_NAME

OPS_NAME = "model_eval"

OPS_DES = """
# Model Evaluation
This is to evaluate trained models to select the best one for deployment
## Task type
- any
## Upstreaam dependencies
- Model training
## Parameter
compare_metric: performance metrics for model score
## Metrics
- test_accuracy_score
## Artifacts
1. Model
## Helper functions
- `load_model(run_id: str)`
- `display_performance_eval_report(run_id: str, model_name: str)`
- `get_model_metric_name(run_id: str)`
- `get_selected_model_name(run_id: str)`
- `get_selected_model_metric_value(run_id: str)`
"""

PARAM_COMPAR_METRIC = "compare_metric"

TAG_MODEL_NAME = "model.name"
TAG_MODEL_METRIC_NAME = "model.metric.name"
TAG_MODEL_METRIC_VALUE = "model.metric.value"

ARTIFACT_MODEL = "mlops_model"
ARTIFACT_EVAL_REPORTS = "eval_reports"


@base_component(name=OPS_NAME, note=OPS_DES)
def run_func(upstream_ids: dict, **kwargs):
    # artifact directory
    artifact_dir = tempfile.mkdtemp()

    # Use mlops.model_analysis
    eval_config = mlops_ma.EvalConfig()
    eval_config.metric_spec.CopyFrom(
        mlops_ma_metrics.specs_from_metrics(
            [
                (mlops_ma_metrics.confusion_matrix,),
                (mlops_ma_metrics.classification_report,),
                (mlops_ma_metrics.accuracy_score,),
                (mlops_ma_metrics.log_loss,),
                (mlops_ma_metrics.roc_curve_display,),
                (mlops_ma_metrics.roc_auc_score, {"average": "weighted"}),
            ]
        )
    )

    eval_config.model_score.score_name = "classification_report"
    eval_config.model_score.report_column = "f1-score"
    eval_config.model_score.report_row = "macro avg"
    eval_config.model_score.threshold = 0.85

    data_gen_run_id = upstream_ids.pop(DATA_GEN_OPS_NAME)

    eval_config.model_spec.label_keys.append(
        dg_helper.get_label_header(data_gen_run_id)
    )

    eval_X = dg_helper.load_test_X(data_gen_run_id)
    eval_y = dg_helper.load_test_y(data_gen_run_id)

    eval_result_dir = os.path.join(artifact_dir, ARTIFACT_EVAL_REPORTS)
    loaded_models = {}
    for model_name, model_run_id in upstream_ids.items():
        loaded_models[model_name] = mlops_model.load_model(
            f"runs:/{model_run_id}/{ARTIFACT_MODEL}"
        )
        eval_result_output_path = os.path.join(eval_result_dir, f"{model_name}")
        mlops_ma.run_model_analysis(
            model=loaded_models[model_name],
            model_name=model_name,
            data=eval_X.join(eval_y),
            eval_config=eval_config,
            output_path=eval_result_output_path,
            save_report=True,
        )
    selected_model_eval_result = mlops_ma.select_model(
        eval_results=eval_result_dir, eval_config=eval_config
    )
    best_model_name = selected_model_eval_result.model_spec.name
    best_metric = mlops_ma.get_model_score(
        selected_model_eval_result, eval_config.model_score
    )

    conda_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../conda.yaml"
    )
    mlops_model.log_model(
        ARTIFACT_MODEL,
        loaded_models[best_model_name],
        conda_env=conda_file,
        eval_config=eval_config,
    )

    mlflow.set_tags(
        {
            TAG_MODEL_NAME: best_model_name,
            TAG_MODEL_METRIC_NAME: repr(eval_config.model_score),
            TAG_MODEL_METRIC_VALUE: best_metric,
        }
    )

    mlflow.log_artifacts(artifact_dir)
    shutil.rmtree(artifact_dir)
