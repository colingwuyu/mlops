"""Wrap sklearn Pipeline"""
from collections import OrderedDict

import numpy as np
import sklearn

from mlops.utils.functionutils import call_func
from mlops.types import ModelInput, ModelOutput
import mlops.estimator.consts as consts

# FUNC_PROBABILITIES = "predict_proba"
# FUNC_CLASSES = "predict"
# FUNC_CLASS_IDS = "class_ids"
# FUNC_ALL_CLASS_IDS = "all_class_ids"
# FUNC_ALL_CLASSES = "all_classes"


class SkEstimator(object):
    def __init__(self, sklearn_estimator):
        self._estimator = sklearn_estimator
        self._estimator_type = sklearn_estimator._estimator_type
        if sklearn.base.is_classifier(sklearn_estimator):
            self._all_classes_mapping = OrderedDict()
            self._all_classes_one_hot = np.identity(len(self._estimator.classes_))
            for i, class_ in enumerate(self._estimator.classes_):
                self._all_classes_mapping[class_] = dict(
                    [
                        (consts.PREDICTION_KEY_CLASS_IDS, i),
                        (
                            consts.PREDICTION_KEY_CLASS_ONE_HOT_IDS,
                            self._all_classes_one_hot[i, :],
                        ),
                    ]
                )
            self._all_class_ids = np.array(list(self._all_classes_mapping.values()))
            self._all_classes = np.array(list(self._all_classes_mapping.keys()))
            self._prediction = OrderedDict(
                [
                    (consts.PREDICTION_KEY_PROBABILITIES, "predict_proba"),
                    (consts.PREDICTION_KEY_CLASSES, "predict"),
                    (consts.PREDICTION_KEY_CLASS_IDS, "class_ids"),
                    (consts.PREDICTION_KEY_CLASS_ONE_HOT_IDS, "class_one_hot_ids"),
                    (consts.PREDICTION_KEY_ALL_CLASS_IDS, "all_class_ids"),
                    (consts.PREDICTION_KEY_ALL_CLASSES, "all_classes"),
                    (consts.PREDICTION_KEY_ALL_CLASSES_MAPPING, "all_classes_mapping"),
                ]
            )
        elif sklearn.base.is_regressor(sklearn_estimator):
            ...

    @classmethod
    def convert_to_estimator(cls, sklearn_estimator):
        estimator = cls(sklearn_estimator)
        return estimator

    def predict(
        self, data: ModelInput, predict_keys=None, **predict_params
    ) -> ModelOutput:
        if not predict_keys:
            predict_keys = self._prediction.keys()
        predictions = {}
        for predict_key in self._prediction.keys():
            if predict_key not in predict_keys:
                continue
            pred_func = self._prediction[predict_key]
            if hasattr(self._estimator, pred_func):
                predictions[predict_key] = call_func(
                    self._estimator, pred_func, data, predictions, **predict_params
                )
            else:
                predictions[predict_key] = call_func(
                    self, pred_func, data, predictions, **predict_params
                )
        for predict_key in predictions:
            if predict_key not in predict_keys:
                predictions.pop(predict_key)
        return predictions

    def all_class_ids(self):
        return self._all_class_ids

    def all_classes(self):
        return self._all_classes

    def all_classes_mapping(self):
        return self._all_classes_mapping

    def _map_class(self, data, predictions, prediction_key):
        def _class_id_map(class_):
            return self._all_classes_mapping[class_][prediction_key]

        if consts.PREDICTION_KEY_CLASSES not in predictions:
            predictions[consts.PREDICTION_KEY_CLASSES] = self._estimator.predict(data)
        return np.array(
            list(map(_class_id_map, predictions[consts.PREDICTION_KEY_CLASSES]))
        )

    def class_ids(self, data, predictions):
        return self._map_class(data, predictions, consts.PREDICTION_KEY_CLASS_IDS)

    def class_one_hot_ids(self, data, predictions):
        return self._map_class(
            data, predictions, consts.PREDICTION_KEY_CLASS_ONE_HOT_IDS
        )
