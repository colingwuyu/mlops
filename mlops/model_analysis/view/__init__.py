import os
from typing import Text

from IPython.core.display import display, HTML
from jinja2 import Environment, FileSystemLoader
import pandas as pd
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils.display_util import (
    get_anomalies_dataframe,
    get_statistics_html,
)

from mlops.utils.sysutils import path_splitall
from mlops.model_analysis.utils import load_eval_result_text, convert_pandas_df
from mlops.model_analysis.proto import result_pb2
from mlops.model_analysis.consts import (
    FILE_EVAL_DATA_STATS,
    FILE_TRAIN_DATA_STATS,
    FILE_DATA_SCHEMA,
    FILE_PREV_EVAL_DATA_STATS,
)

# from weasyprint import HTML


def display_report(
    eval_result_path: Text,
    report_save_path: Text = None,
):
    # Use in notebook
    display(HTML(view_report(eval_result_path, report_save_path, save=False)))


def view_report(
    eval_result_path: Text,
    report_save_path: Text = None,
    save: bool = True,
):
    if not report_save_path:
        report_save_path = eval_result_path
    jinjia_env = Environment(
        loader=FileSystemLoader(os.path.dirname(os.path.abspath(__file__)))
    )
    report_template = jinjia_env.get_template("performance_report.html")
    eval_result: result_pb2.EvalResult = load_eval_result_text(
        eval_result_path)
    meta_table = pd.Series(name="Meta Data")
    meta_table["Evaluation Date"] = eval_result.eval_date
    meta_table["Model Name"] = eval_result.model_spec.name
    meta_table["Model Version"] = eval_result.model_spec.model_ver
    meta_table = pd.DataFrame(meta_table)

    metric_value_table = pd.Series(name="Result")

    reports = [["Meta Data", meta_table.to_html()]]

    for metric_name in eval_result.metric_results:
        metric_result = eval_result.metric_results[metric_name]
        if metric_result.HasField("report"):
            reports.append(
                [metric_name, convert_pandas_df(
                    metric_result.report).to_html()]
            )
        elif metric_result.HasField("url"):
            if _is_view_for_flask(report_save_path):
                render_img_url_func = _render_img_url_for_flask

            else:
                render_img_url_func = _render_img_url
            reports.append(
                [
                    metric_name,
                    render_img_url_func(
                        os.path.join(eval_result_path, metric_result.url)
                    ),
                ]
            )
        else:
            metric_value_table[metric_name] = metric_result.value

    metric_value_table = pd.DataFrame(metric_value_table)
    metric_value_table.index.name = "Metric Name"

    reports.append(["Other Metric Results", metric_value_table.to_html()])

    train_data_stats = tfdv.load_stats_text(
        os.path.join(eval_result_path, FILE_TRAIN_DATA_STATS)
    )
    eval_data_stats = tfdv.load_stats_text(
        os.path.join(eval_result_path, FILE_EVAL_DATA_STATS)
    )
    prev_eval_data_stats = None
    if os.path.exists(os.path.join(eval_result_path, FILE_PREV_EVAL_DATA_STATS)):
        prev_eval_data_stats = tfdv.load_stats_text(
            os.path.join(eval_result_path, FILE_PREV_EVAL_DATA_STATS)
        )
    data_schema = tfdv.load_schema_text(
        os.path.join(eval_result_path, FILE_DATA_SCHEMA)
    )

    skew_anomalies = tfdv.validate_statistics(
        statistics=train_data_stats,
        schema=data_schema,
        serving_statistics=eval_data_stats,
    )
    skew_anomalies_df = get_anomalies_dataframe(skew_anomalies)

    drift_anomalies = tfdv.validate_statistics(
        statistics=eval_data_stats,
        schema=data_schema,
        previous_statistics=prev_eval_data_stats,
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

    if prev_eval_data_stats:
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

    template_vars = {
        "title": "Performance Report",
        "reports": reports,
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
    return html_out


def _render_img_url(url):
    return f'<img src="{url}" object-fit "cover">'


def _render_img_url_for_flask(url):
    all_paths = path_splitall(url)
    base_folder = all_paths[0]
    assert base_folder == "static"
    url = os.path.join(*all_paths[1:])
    return """<img src="{{ url_for('static', filename='%s') }}">""" % url


def _is_view_for_flask(save_path):
    return path_splitall(save_path)[0] == "templates"


# if __name__ == "__main__":
#     view_report("examples/iris/serving/performance_analysis")
