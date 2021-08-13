from IPython.core.display import display, HTML

from mlops.utils.mlflowutils import MlflowUtils
from mlops.serving.model import load_model as mlops_model_loader
from examples.iris.model_eval import OPS_NAME, ARTIFACT_MODEL, ARTIFACT_EVAL_REPORTS


def _assert_ops_type(run_id: str):
    assert MlflowUtils.get_run_name(run_id) == OPS_NAME


def load_model(run_id: str):
    _assert_ops_type(run_id)
    return mlops_model_loader(f"runs:/{run_id}/{ARTIFACT_MODEL}")


def display_performance_eval_report(run_id: str, model_name: str):
    report_url = MlflowUtils.get_artifact_path(
        run_id, ARTIFACT_EVAL_REPORTS + "/" + model_name + "/report.html"
    )
    with open(report_url, "r") as report_f:
        display(HTML(report_f.read()))


def get_model_metric_name(run_id: str):
    return MlflowUtils.get_tag(run_id, "model.metric.name")


def get_selected_model_name(run_id: str):
    return MlflowUtils.get_tag(run_id, "model.name")


def get_selected_model_metric_value(run_id: str):
    return MlflowUtils.get_tag(run_id, "model.metric.value")
