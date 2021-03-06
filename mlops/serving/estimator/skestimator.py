"""Wrap sklearn Pipeline"""
import os
import pickle

from mlops.serving.estimator.estimator import Estimator

# FUNC_PROBABILITIES = "predict_proba"
# FUNC_CLASSES = "predict"
# FUNC_CLASS_IDS = "class_ids"
# FUNC_ALL_CLASS_IDS = "all_class_ids"
# FUNC_ALL_CLASSES = "all_classes"


class SkEstimator(Estimator):
    def save(self, model_path):
        saved_path = os.path.join(model_path, "model.pkl")
        with open(saved_path, "wb") as out:
            pickle.dump(self._estimator, out)
        return saved_path

    def load(self, model_path):
        with open(model_path, "rb") as f:
            estimator = pickle.load(f)
        self.set_model(estimator)
