import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from mlops.types import ModelInput, ModelOutput
from mlops.components import base_component
from mlops.serving.model import (
    MODEL_COMP_SCALER,
    MODEL_COMP_SCHEMA,
    MODEL_COMP_STATS,
    MODEL_COMP_MODEL,
)
from examples.iris.data_gen import helper as dg_helper, OPS_NAME as DATA_GEN_OPS_NAME
from examples.iris.data_validation import (
    helper as dv_helper,
    OPS_NAME as DATA_VAL_OPS_NAME,
)
from examples.iris.feature_transform import (
    helper as ft_helper,
    OPS_NAME as FEATURE_TRANSFORM_OPS_NAME,
)

OPS_NAME = "mlp_multiclass"
OPS_DES = """
# MLP Multi-Classification
"""


class MlpClassifier(nn.Module):
    def __init__(
        self,
        num_features,
        num_hidden_layers,
        num_nerons,
        num_classes,
    ):
        super(MlpClassifier, self).__init__()
        self.fc_layers = [nn.Linear(num_features, num_nerons)]
        for _ in range(1, num_hidden_layers):
            self.fc_layers.append(nn.Linear(num_nerons, num_nerons))
        self.fc_layers.append(nn.Linear(num_nerons, num_classes))

    def forward(self, X):
        for fc_layer in self.fc_layers:
            X = F.relu(fc_layer(X))
        return X

    def predict(
        self, data: ModelInput, predict_keys=None, **predict_params
    ) -> ModelOutput:
        input = torch.tensor(data.values)


@base_component(name=OPS_NAME, note=OPS_DES)
def run_func(upstream_ids: dict, **kwargs):
    model_comps = {}

    data_gen_id = upstream_ids[DATA_GEN_OPS_NAME]
    X_train = dg_helper.load_train_X(data_gen_id)
    y_train = dg_helper.load_train_y(data_gen_id)
    X_test = dg_helper.load_test_X(data_gen_id)
    y_test = dg_helper.load_test_y(data_gen_id)

    data_val_id = upstream_ids[DATA_VAL_OPS_NAME]
    model_comps[MODEL_COMP_SCHEMA] = dv_helper.get_schema(data_val_id)
    model_comps[MODEL_COMP_STATS] = dv_helper.get_trainset_stat(data_val_id)

    feature_scaler_id = upstream_ids[FEATURE_TRANSFORM_OPS_NAME]
    feature_scaler = ft_helper.load_scaler(feature_scaler_id)
    model_comps[MODEL_COMP_SCALER] = feature_scaler

    X_train = feature_scaler.transform(X_train)
    X_test = feature_scaler.transform(X_test)
