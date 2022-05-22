import os
import json
from typing import Union, Text

import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mlops.serving.estimator import consts as estimator_consts
from mlops.model_analysis.proto import config_pb2

MetricDisplayURL = Text
MetricResult = Union[pd.DataFrame, float, MetricDisplayURL]

ARTIFACT_CONFUSION_MATRIX = "ConfusionMatrix.png"
ARTIFACT_ROC_CURVE = "RocCurve.png"
ARTIFACT_PR_CURVE = "PRCurve.png"


def _convert_one_hot_ids(labels, label_mapping):
    def _class_one_hot_map(class_):
        return label_mapping[class_][estimator_consts.PREDICTION_KEY_CLASS_ONE_HOT_IDS]

    return np.array(list(map(_class_one_hot_map, np.squeeze(labels))))


def confusion_matrix(y_true, y_pred) -> MetricResult:
    y_pred = y_pred[estimator_consts.PREDICTION_KEY_CLASSES]
    labels = sklearn.utils.multiclass.unique_labels(y_true, y_pred)
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, labels)
    return pd.DataFrame(
        cm,
        columns=[f"Prediction {label}" for label in labels],
        index=[f"True {label}" for label in labels],
    )


def confusion_matrix_display(y_true, y_pred, output: Text = ".") -> MetricResult:
    output_file = os.path.join(output, ARTIFACT_CONFUSION_MATRIX)
    y_pred = y_pred[estimator_consts.PREDICTION_KEY_CLASSES]
    labels = sklearn.utils.multiclass.unique_labels(y_true, y_pred)
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred, labels)
    display = sklearn.metrics.ConfusionMatrixDisplay(cm, labels).plot()
    display.ax_.set_title("Confusion Matrix")
    display.figure_.savefig(output_file)
    return display


def classification_report(y_true, y_pred, digits=3) -> MetricResult:
    y_pred = y_pred[estimator_consts.PREDICTION_KEY_CLASSES]
    report = sklearn.metrics.classification_report(
        y_true, y_pred, digits=digits, output_dict=True
    )
    report = (
        pd.DataFrame(report)
        .T.drop(labels=["accuracy"], axis=0)
        .rename(columns={"support": "example counts"})
    )

    return report


def accuracy_score(y_true, y_pred) -> MetricResult:
    y_pred = y_pred[estimator_consts.PREDICTION_KEY_CLASSES]
    return sklearn.metrics.accuracy_score(y_true, y_pred)


def cohen_kappa_score(y_true, y_pred) -> MetricResult:
    y_pred = y_pred[estimator_consts.PREDICTION_KEY_CLASSES]
    return sklearn.metrics.cohen_kappa_score(y_true, y_pred)


def log_loss(y_true, y_pred) -> MetricResult:
    y_pred_proba = y_pred[estimator_consts.PREDICTION_KEY_PROBABILITIES]
    y_true_one_hot_ids = _convert_one_hot_ids(
        y_true, y_pred[estimator_consts.PREDICTION_KEY_ALL_CLASSES_MAPPING]
    )
    return sklearn.metrics.log_loss(y_true_one_hot_ids, y_pred_proba)


def roc_auc_score(
    y_true, y_pred, average: str = "weighted", multi_class: str = "ovo"
) -> MetricResult:
    y_pred_proba = y_pred[estimator_consts.PREDICTION_KEY_PROBABILITIES]
    y_true_one_hot_ids = _convert_one_hot_ids(
        y_true, y_pred[estimator_consts.PREDICTION_KEY_ALL_CLASSES_MAPPING]
    )
    return sklearn.metrics.roc_auc_score(
        y_true_one_hot_ids, y_pred_proba, average=average, multi_class=multi_class
    )


def roc_curve_display(y_true, y_pred, output: Text = ".") -> MetricResult:
    output_file = os.path.join(output, ARTIFACT_ROC_CURVE)
    all_classes = y_pred[estimator_consts.PREDICTION_KEY_ALL_CLASSES]
    num_classes = len(all_classes)

    y_true_one_hot_ids = _convert_one_hot_ids(
        y_true, y_pred[estimator_consts.PREDICTION_KEY_ALL_CLASSES_MAPPING]
    )
    predict_probs = y_pred[estimator_consts.PREDICTION_KEY_PROBABILITIES]

    fig, axes = plt.subplots(1, num_classes, figsize=(8 * num_classes, 8))

    for i, class_ in enumerate(all_classes):
        ax = axes[i]
        i_labels = y_true_one_hot_ids[:, i]
        #     i_labels = np.random.choice([0,1], predict_probs.shape[0])
        i_predict_scores = predict_probs[:, i]
        fpr, tpr, _ = sklearn.metrics.roc_curve(
            i_labels, i_predict_scores, pos_label=1)
        sklearn.metrics.RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax)
        ax.set_title(class_)
    fig.suptitle("ROC Curve")
    plt.savefig(output_file)
    return output_file


def pr_curve_display(y_true, y_pred, output: Text = ".") -> MetricResult:
    output_file = os.path.join(output, ARTIFACT_PR_CURVE)
    all_classes = y_pred[estimator_consts.PREDICTION_KEY_ALL_CLASSES]
    num_classes = len(all_classes)

    y_true_one_hot_ids = _convert_one_hot_ids(
        y_true, y_pred[estimator_consts.PREDICTION_KEY_ALL_CLASSES_MAPPING]
    )
    predict_probs = y_pred[estimator_consts.PREDICTION_KEY_PROBABILITIES]

    fig, axes = plt.subplots(1, num_classes, figsize=(8 * num_classes, 8))

    for i, class_ in enumerate(all_classes):
        ax = axes[i]
        i_labels = y_true_one_hot_ids[:, i]
        #     i_labels = np.random.choice([0,1], predict_probs.shape[0])
        i_predict_scores = predict_probs[:, i]
        prec, recall, _ = sklearn.metrics.precision_recall_curve(
            i_labels, i_predict_scores, pos_label=1
        )
        sklearn.metrics.PrecisionRecallDisplay(precision=prec, recall=recall).plot(
            ax=ax
        )
        ax.set_title(class_)
    fig.suptitle("Precision Recall Curve")
    plt.savefig(output_file)
    return output_file


def specs_from_metrics(metrics) -> config_pb2.MetricSpec:
    """convert spec from list of metrics

    Example
    metrics = [
        (mlops_ma.metrics.confusion_matrix,),
        (mlops_ma.metrics.roc_auc_score, {"average": "macro"})
    ]

    Args:
        metrics (MetricSpec): proto buffer of MetricSpec
    """
    metric_spec = config_pb2.MetricSpec()

    for metric in metrics:
        metric_proto = metric_spec.metrics.add()
        metric_proto.metric_name = metric[0].__name__
        if len(metric) > 1:
            metric_proto.config = json.dumps(metric[1])

    return metric_spec
