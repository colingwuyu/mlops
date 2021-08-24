from examples.iris.serving import model
import pathlib

from mlops.estimator.skestimator import SkEstimator
from mlops.estimator.torchestimator import TorchEstimator


class EstimatorLoader:
    def load(model_path):
        model_ext = pathlib.Path(model_path).suffix
        if model_ext == ".pkl":
            model_estimator = SkEstimator()
            model_estimator.load(model_path)
        elif model_ext == ".pth":
            model_estimator = TorchEstimator()
            model_estimator.load(model_path)
        return model_estimator
