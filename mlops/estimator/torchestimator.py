from abc import ABC, abstractmethod
from mlops import estimator
from examples.iris.serving import model
import os
import inspect
from typing import Dict

import torch
from importlib import import_module

from mlops.estimator.estimator import Estimator
from mlops.types import ModelInput, ModelOutput


class TorchModel(ABC):
    def __init__(self, estimator_type, model_cls_name) -> None:
        self._estimator_type = estimator_type
        self.classes_ = []
        self.net: torch.nn.modules = None
        frm = inspect.stack()[1]
        mod = inspect.getmodule(frm[0])
        self._model_module = mod.__name__
        self._model_cls_name = model_cls_name
        super().__init__()

    def is_classifier(self):
        return self._estimator_type == "classifier"

    def is_regressor(self):
        return self._estimator_type == "regressor"

    @abstractmethod
    def predict(self, X: ModelInput) -> ModelOutput:
        pass

    def predict_proba(self, X: ModelInput) -> ModelOutput:
        if self._estimator_type != "classifier":
            return None

    def state_dict(self):
        return {
            "estimator_type": self._estimator_type,
            "classes_": self.classes_,
            "model_module": self._model_module,
            "model_cls": self._model_cls_name,
            "net_dict": self.net.state_dict(),
        }

    @classmethod
    def load_model(self, model_path):
        state_dict: Dict = torch.load(model_path)
        model_cls = getattr(
            import_module(state_dict.pop("model_module")), state_dict.pop("model_cls")
        )
        net_dict = state_dict.pop("net_dict")
        classes_ = state_dict.pop("classes_")
        estimator_type = state_dict.pop("estimator_type")
        model_instance = model_cls(**state_dict)
        model_instance.net.load_state_dict(net_dict)
        model_instance._estimator_type = estimator_type
        model_instance.classes_ = classes_
        return model_instance


class TorchEstimator(Estimator):
    def save(self, model_path):
        saved_path = os.path.join(model_path, "model.pth")
        # torch.save(self._estimator.state_dict(), saved_path)
        torch.save(
            self._estimator.state_dict(),
            saved_path,
        )
        return saved_path

    def load(self, model_path):
        estimator = TorchModel.load_model(model_path)
        self.set_model(estimator)
