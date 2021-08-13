import os
from re import I
from typing import Text, Type

from google.protobuf import text_format
import numpy as np
import pandas as pd
import tensorflow_data_validation as tfdv

from mlops.model_analysis.proto import config_pb2, result_pb2
from mlops.model_analysis.consts import FILE_EVAL_RESULT, FILE_EVAL_DATA_STATS


def write_eval_config_text(eval_config: config_pb2.EvalConfig, output_path: Text):
    if not isinstance(eval_config, config_pb2.EvalConfig):
        raise TypeError(
            "eval_config is of type %s, should be a EvalConfig proto."
            % type(eval_config).__name__
        )

    eval_config_text = text_format.MessageToString(eval_config)
    with open(output_path, "w") as f:
        f.write(eval_config_text)


def load_eval_config_text(input_path: Text) -> config_pb2.EvalConfig:
    eval_config = config_pb2.EvalConfig()
    with open(input_path, "r") as f:
        eval_config_text = f.read()
    text_format.Parse(eval_config_text, eval_config)
    return eval_config


def write_eval_result_text(eval_result: result_pb2.EvalResult, output_path: Text):
    if not isinstance(eval_result, result_pb2.EvalResult):
        raise TypeError(
            "eval_result is of type %s, should be a EvalResult proto."
            % type(eval_result).__name__
        )

    eval_result_text = text_format.MessageToString(eval_result)
    with open(os.path.join(output_path, FILE_EVAL_RESULT), "w") as f:
        f.write(eval_result_text)


def load_eval_result_text(input_path: Text) -> result_pb2.EvalResult:
    eval_result = result_pb2.EvalResult()
    with open(os.path.join(input_path, FILE_EVAL_RESULT), "r") as f:
        eval_result_text = f.read()
    text_format.Parse(eval_result_text, eval_result)
    return eval_result


def convert_proto_df(pandas_df: pd.DataFrame) -> result_pb2.DataFrame:
    proto_df = result_pb2.DataFrame()
    proto_df.columns[:] = pandas_df.columns.values
    proto_df.index[:] = pandas_df.index.values
    proto_df.values[:] = pandas_df.values.flatten()
    return proto_df


def convert_pandas_df(proto_df: result_pb2.DataFrame) -> pd.DataFrame:
    df_columns = proto_df.columns
    df_index = proto_df.index
    df_data = np.array(proto_df.values)
    df_data = np.reshape(df_data, (len(df_index), len(df_columns)))
    return pd.DataFrame(data=df_data, columns=df_columns, index=df_index)


def get_model_score(
    eval_result: result_pb2.EvalResult, model_score: config_pb2.ModelSpec
):
    model_metric = eval_result.metric_results[model_score.score_name]
    if model_metric.HasField("value"):
        return model_metric.value
    elif model_metric.HasField("report"):
        return convert_pandas_df(model_metric.report)[model_score.report_column][
            model_score.report_row
        ]


def get_eval_data_stats(eval_result_path: Text):
    return tfdv.load_stats_text(os.path.join(eval_result_path, FILE_EVAL_DATA_STATS))
