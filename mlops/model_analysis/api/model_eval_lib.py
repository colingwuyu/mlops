from examples.iris import serving
import json
from mlops.model_analysis import view
import os
from datetime import datetime
from typing import Optional, Text, Union, List

import pandas as pd
import numpy as np
import tensorflow_data_validation as tfdv

from mlops.model_analysis.proto import config_pb2 as config, result_pb2 as result
import mlops.model_analysis.metrics as ma_metrics
from mlops.model_analysis.utils import (
    get_eval_data_stats,
    load_eval_result_text,
    write_eval_result_text,
    get_model_score,
)
from mlops.model_analysis.consts import (
    TIME_FORMAT,
    FILE_DATA_SCHEMA,
    FILE_TRAIN_DATA_STATS,
    FILE_EVAL_DATA_STATS,
    FILE_EVAL_RESULT,
    FILE_PREV_EVAL_DATA_STATS,
    VALIDATION_SUCCESS,
    VALIDATION_FAIL,
)
from mlops.model_analysis.view import view_report


def _generate_eval_result(
    true_label,
    prediction,
    eval_config: config.EvalConfig,
    output_path: Optional[Text] = None,
):
    if not output_path:
        output_path = "."
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    eval_result = result.EvalResult()
    if eval_config.HasField("model_spec"):
        eval_result.model_spec.CopyFrom(eval_config.model_spec)
    for metric_config in eval_config.metric_spec.metrics:
        kwargs = {}
        name = metric_config.metric_name
        if metric_config.config:
            kwargs.update(json.loads(metric_config.config))
        if "display" in name:
            kwargs["output"] = output_path
        metric_func = getattr(ma_metrics, name)
        try:
            metric_result = metric_func(y_true=true_label, y_pred=prediction, **kwargs)
        except BaseException as e:
            print(f"Warning: fail in calculating metric {name}. ", e)
            _eval_result_add_value(eval_result, name, -999)
        else:
            if isinstance(metric_result, pd.DataFrame):
                _eval_result_add_report(eval_result, name, metric_result)
            elif isinstance(metric_result, Text):
                rel_url = os.path.relpath(metric_result, output_path)
                _eval_result_add_url(eval_result, name, rel_url)
            else:
                _eval_result_add_value(eval_result, name, metric_result)
    return eval_result


def validate_model(
    model,
    data: pd.DataFrame,
    prev_data_stat=None,
    model_name: Optional[Text] = None,
    eval_config: Optional[config.EvalConfig] = None,
    output_path: Optional[Text] = None,
):
    run_model_analysis(
        model, data, prev_data_stat, model_name, eval_config, output_path
    )
    eval_result = load_eval_result_text(output_path)
    model_score = get_model_score(eval_result, model.eval_config.model_score)
    if model_score < model.eval_config.model_score.threshold:
        return VALIDATION_FAIL
    else:
        return VALIDATION_SUCCESS


def run_model_analysis(
    model,
    data: pd.DataFrame,
    prev_data_stat=None,
    model_name: Optional[Text] = None,
    eval_config: Optional[config.EvalConfig] = None,
    output_path: Optional[Text] = None,
    save_report: bool = False,
):
    if not eval_config:
        eval_config = model.eval_config
    if model.eval_config and model.eval_config.model_spec.name:
        model_name = model.eval_config.model_spec.name
    if model_name:
        orig_model_name = eval_config.model_spec.name
        eval_config.model_spec.name = model_name
    true_label = data[list(eval_config.model_spec.label_keys)]
    prediction = model.predict(data.drop(columns=eval_config.model_spec.label_keys))

    eval_result = _generate_eval_result(
        true_label, prediction, eval_config, output_path
    )

    eval_data_stats = tfdv.generate_statistics_from_dataframe(data)

    now = datetime.now()
    eval_result.eval_date = now.strftime(TIME_FORMAT)
    write_eval_result_text(eval_result, output_path)
    tfdv.write_stats_text(
        model.trainset_stats, os.path.join(output_path, FILE_TRAIN_DATA_STATS)
    )
    tfdv.write_stats_text(
        eval_data_stats,
        os.path.join(output_path, FILE_EVAL_DATA_STATS),
    )
    tfdv.write_schema_text(
        model.data_schema,
        os.path.join(output_path, FILE_DATA_SCHEMA),
    )
    if prev_data_stat:
        tfdv.write_stats_text(
            prev_data_stat, os.path.join(output_path, FILE_PREV_EVAL_DATA_STATS)
        )
    view_report(output_path, save=save_report)

    if model_name:
        eval_config.model_spec.name = orig_model_name


def analyze_raw_data(
    data: pd.DataFrame,
    eval_config: Optional[config.EvalConfig] = None,
    output_path: Optional[Text] = None,
):
    true_label = data[eval_config.model_spec.label_keys]
    ...


def _eval_result_add_report(
    eval_result: result.EvalResult, metric_name: Text, report: pd.DataFrame
):
    metric_result = eval_result.metric_results[metric_name]
    metric_result.report.columns[:] = report.columns.values
    metric_result.report.index[:] = report.index.values
    metric_result.report.values[:] = report.values.flatten()
    return eval_result


def _eval_result_add_url(eval_result: result.EvalResult, metric_name: Text, url: Text):
    metric_result = eval_result.metric_results[metric_name]
    metric_result.url = url
    return metric_result


def _eval_result_add_value(eval_result: result.EvalResult, metric_name: Text, value):
    metric_result = eval_result.metric_results[metric_name]
    metric_result.value = value
    return metric_result


def select_model(
    eval_results: Union[List[Text], Text], eval_config: config.EvalConfig
) -> result.EvalResult:
    """select the model according to model score

    Args:
        eval_results (Union[List[Text], Text]): [description]
        eval_config (config.EvalConfig): [description]
    """
    eval_results_folders = []
    if isinstance(eval_results, Text):
        for subdir in os.walk(eval_results):
            if FILE_EVAL_RESULT in subdir[2]:
                eval_results_folders.append(subdir[0])
    else:
        for eval_result_folder in eval_results:
            if FILE_EVAL_RESULT not in os.listdir(eval_result_folder):
                print(
                    "Warning: %s does not have evaluation result." % eval_result_folder
                )
            else:
                eval_results_folders.append(eval_result_folder)

    best_model_eval_result = None
    best_model_metric = None
    for eval_result_folder in eval_results_folders:
        eval_result = load_eval_result_text(eval_result_folder)
        model_metric = get_model_score(eval_result, eval_config.model_score)
        if (best_model_metric is None) or (best_model_metric < model_metric):
            best_model_eval_result = eval_result
            best_model_metric = model_metric
    return best_model_eval_result
