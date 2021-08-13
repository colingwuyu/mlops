import os
from typing import Text

from IPython.core.display import display, HTML
from jinja2 import Environment, FileSystemLoader
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils.display_util import (
    get_anomalies_dataframe,
    get_statistics_html,
)

from mlops.serving.model import MlopsLoadModel

# from weasyprint import HTML


def display_report(
    model,
    serving_data,
    prev_data,
):
    # Use in notebook
    report_html, _ = view_report(model, serving_data, prev_data, save=False)
    display(HTML(report_html))


def view_report(
    model: MlopsLoadModel,
    serving_data,
    prev_data=None,
    report_save_path: Text = None,
    save: bool = True,
):
    jinjia_env = Environment(
        loader=FileSystemLoader(os.path.dirname(os.path.abspath(__file__)))
    )
    report_template = jinjia_env.get_template("data_validation_report.html")

    train_data_stats = model.trainset_stats
    eval_data_stats = tfdv.generate_statistics_from_dataframe(serving_data)
    if prev_data is not None:
        prev_eval_data_stats = tfdv.generate_statistics_from_dataframe(prev_data)
    else:
        prev_eval_data_stats = None
    data_schema = model.data_schema

    skew_anomalies = tfdv.validate_statistics(
        statistics=train_data_stats,
        schema=data_schema,
        serving_statistics=eval_data_stats,
        environment="SERVING",
    )
    skew_anomalies_df = get_anomalies_dataframe(skew_anomalies)

    drift_anomalies = tfdv.validate_statistics(
        statistics=eval_data_stats,
        schema=data_schema,
        previous_statistics=prev_eval_data_stats,
        environment="SERVING",
    )
    drift_anomalies_df = get_anomalies_dataframe(drift_anomalies)

    stat_plots = [
        (
            "Evaluation Dataset vs. Trainning Dataset",
            get_statistics_html(
                lhs_statistics=eval_data_stats,
                rhs_statistics=train_data_stats,
                lhs_name="EVAL_DATTASET",
                rhs_name="TRAIN_DATASET",
            ),
        )
    ]

    if prev_data is not None:
        stat_plots.append(
            (
                "Evaluation Dataset vs. Previous Evaluation Dataset",
                get_statistics_html(
                    lhs_statistics=eval_data_stats,
                    rhs_statistics=prev_eval_data_stats,
                    lhs_name="EVAL_DATASET",
                    rhs_name="PREV_EVAL_DATASET",
                ),
            )
        )

    # ignore label anomaly
    if skew_anomalies_df.shape[0] > 0:
        skew_anomalies_df = skew_anomalies_df[
            ~skew_anomalies_df["Anomaly long description"].str.contains(
                "not in the environment SERVING"
            )
        ]

    template_vars = {
        "skew_anomaly": skew_anomalies_df.to_html(),
        "drift_anomaly": drift_anomalies_df.to_html(),
        "stat_plots": stat_plots,
    }

    html_out = report_template.render(template_vars)

    # display(HTML(html_out))
    if save:
        if not os.path.exists(report_save_path):
            os.makedirs(report_save_path)
        with open(os.path.join(report_save_path, "report.html"), "w") as f:
            f.write(html_out)
    return html_out, skew_anomalies_df.shape[0] == 0


# if __name__ == "__main__":
#     view_report("examples/iris/serving/performance_analysis")
